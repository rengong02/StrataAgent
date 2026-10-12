"""The benchmark profile: what one GeoBenchX task is allowed to touch.

The contract this implements is summarized in `README.md` and enforced by the
profile/runner tests shipped in this repository.
Four of its constraints exist because the alternatives do not actually hold:

* **`bash` and `agent` are not exposed.** The workspace boundary only guards the
  structured file tools; a shell can read anything, and a sub-agent gets its own
  budget. Exposing either would make "the run was bounded" untrue.
* **The workspace is per task and absolute.** `set_workspace_root` resolves once,
  so a relative path would be anchored to whatever the process cwd happened to
  be when the first task ran.
* **Tools are a whitelist, not all 24.** A wide tool surface makes the model's
  job harder and the failure attribution muddier, for no gain on a first pass.
* **The run is headless.** GeoBenchX's tools call `plt.show()` from ten sites
  when their state says `visualize`, and the Agent runs a round's tool calls on
  a thread pool -- which macOS' GUI backend rejects outright. Measured, not
  predicted: 15 of the first baseline's 40 tool errors were exactly that.

The budget is per `chat()` (see the contract). `LimitExceeded` is a normal
outcome here, not an error: it is recorded as `truncated` on the trajectory so a
stopped run can never be scored as a finished one.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from corecoder.agent import Agent
from corecoder.limits import LimitExceeded, Limits
from corecoder.tools.base import workspace_root

from .evidence import DEFAULT_SCHEMA_PREVIEW, EvidenceStore, validate_schema_preview
from .tools import build_geobench_tools
from .trajectory import Trajectory

#: The first-pass tool set. Ordered by how often GeoBenchX's own reference
#: solutions use them, so the common task shapes stay solvable; the map tools
#: are kept because the "Merge, Visualize" group cannot be scored without them.
DEFAULT_TOOLS: tuple[str, ...] = (
    "load_data",
    "load_geodata",
    "merge_dataframes",
    "get_unique_values",
    "filter_categorical",
    "filter_numerical",
    "select_features_by_spatial_relationship",
    "calculate_column_statistics",
    "calculate_columns",
    "scale_column_by_value",
    "make_choropleth_map",
    "reject_task",
)


@dataclass
class Profile:
    """Configuration for a benchmark run."""

    tools: tuple[str, ...] = DEFAULT_TOOLS
    limits: Limits = field(default_factory=lambda: Limits(max_rounds=25, deadline_s=600))
    #: Kept under the workspace so a task cannot read the previous task's output.
    keep_workspaces: bool = False
    #: Whether a not-found error names what was available instead.
    #:
    #: **Off by default because the ablation did not find an effect.** Seven
    #: runs per arm on the task that motivated it: failed column lookups 6 vs
    #: 21 raw, but a permutation test puts that at p=0.41 -- the control's total
    #: is one run with 13 failures. Tokens p=0.91, steps p=0.66. So this adds
    #: text to every not-found error for no measurable gain, and is kept only
    #: so a properly powered study can settle it later.
    #:
    #: What the ablation *did* establish is that the incident that motivated the
    #: hint -- 36 steps, 1.16M tokens, 13 bad lookups -- was a sampling outlier.
    #: It also showed errors and tokens are decoupled here: one control run had
    #: 13 failures and 563k tokens, another 2 failures and 1,044k.
    schema_hints: bool = False
    #: Where this run's raw tool results are stored, when evidence is on.
    #:
    #: Setting this *is* how evidence is turned on -- there is no separate flag,
    #: because "addressable" means the path still resolves after the run, and a
    #: store with nowhere to put anything could not honour that. It must not be
    #: inside the task workspace: that is deleted on the way out.
    evidence_dir: Path | None = None
    #: Which bounded schema preview an evidence arm delivers to the model.
    #: `head` reproduces the original prefix-only summary; `head_tail` is the
    #: candidate that preserves both early metadata and late year columns.
    evidence_schema_preview: str = DEFAULT_SCHEMA_PREVIEW

    def __post_init__(self) -> None:
        self.evidence_schema_preview = validate_schema_preview(self.evidence_schema_preview)


def _is_inside(path: Path, root: Path) -> bool:
    """Whether `path` would be removed along with `root`."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def run_task(
    task_text: str,
    llm,
    profile: Profile | None = None,
    task_id: str = "",
    on_tool=None,
) -> Trajectory:
    """Run one GeoBenchX task under the profile and return its trajectory.

    The caller owns the workspace lifetime; `keep_workspaces=False` removes it
    on the way out, so a crashed task cannot leak state into the next one.
    """
    profile = profile or Profile()
    traj = Trajectory(task_id=task_id)

    tokens_before = getattr(llm, "total_prompt_tokens", 0) + getattr(llm, "total_completion_tokens", 0)
    started = time.monotonic()

    workspace = Path(tempfile.mkdtemp(prefix=f"strata-{task_id or 'task'}-")).resolve()
    # `workspace_root` restores whatever was configured before, rather than
    # clearing to "unset" -- clearing would hand the boundary back to
    # CORECODER_WORKSPACE, so a task in a process that had that variable set
    # would finish still bounded by the environment's root.
    with workspace_root(workspace):
        # One fresh state per task, and a headless one: `benchmark_state()` turns
        # GeoBenchX's `visualize` display side effect off, which is what the
        # parallel tool executor needs. See that function for why.
        evidence = None
        if profile.evidence_dir is not None:
            # Refused rather than silently written and deleted: a path inside a
            # workspace that gets removed is not addressable, and claiming it is
            # would be worse than not offering the feature.
            if _is_inside(profile.evidence_dir, workspace):
                raise ValueError(
                    f"evidence_dir {profile.evidence_dir} is inside the task workspace, "
                    "which is deleted when the task ends -- the artifacts would not be "
                    "addressable afterwards"
                )
            evidence = EvidenceStore(profile.evidence_dir / (task_id or "task"))

        tools = build_geobench_tools(
            record=traj.record,
            on_result=traj.record_result,
            schema_hints=profile.schema_hints,
            evidence=evidence,
            evidence_schema_preview=profile.evidence_schema_preview,
        )
        keep = set(profile.tools)
        tools = [t for t in tools if t.name in keep]

        agent = Agent(llm=llm, tools=tools, limits=profile.limits)

        def observe(name, args):
            # A call to a tool the profile does not expose produces an error
            # string and never reaches `traj.record`, so without this it leaves
            # no trace anywhere. Recorded as a diagnostic, never as a step.
            if name not in keep:
                traj.unexposed.append(name)
            if on_tool is not None:
                on_tool(name, args)

        # No bash, no sub-agents: the Agent is constructed from the whitelist, so
        # neither can be reached even if the model asks for it by name.
        try:
            traj.final_response = agent.chat(task_text, on_tool=observe) or ""
            traj.stop_reason = "completed"
        except LimitExceeded as e:
            traj.stop_reason = f"truncated:{e.kind}"
        except KeyboardInterrupt:
            traj.stop_reason = "interrupted"
            raise
        finally:
            # Collected on the failure paths too: a truncated or crashed task is
            # exactly the one whose repeat counts somebody needs to read.
            traj.repeat_failures = agent.failure_repeats
            traj.deadline_overruns = agent.deadline_overruns
            if evidence is not None:
                traj.evidence_refs = [ref.as_dict() for ref in evidence.refs]
                traj.evidence_stats = evidence.stats()

    traj.wall_seconds = time.monotonic() - started
    traj.tokens = (
        getattr(llm, "total_prompt_tokens", 0)
        + getattr(llm, "total_completion_tokens", 0)
        - tokens_before
    )

    if not profile.keep_workspaces:
        shutil.rmtree(workspace, ignore_errors=True)
    return traj


def run_tasks(tasks: list[dict], llm, profile: Profile | None = None, on_tool=None) -> list[Trajectory]:
    """Run a batch, one fresh workspace and one fresh Agent per task.

    A new Agent per task is deliberate: budgets are per `chat()`, and a reused
    transcript would carry the previous task's tool calls into the next one.
    """
    out: list[Trajectory] = []
    for task in tasks:
        traj = run_task(
            task["task_text"],
            llm,
            profile=profile,
            task_id=task.get("task_ID", ""),
            on_tool=on_tool,
        )
        out.append(traj)
    return out
