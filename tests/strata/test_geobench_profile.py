"""GeoBenchX adapter smoke: does a real task produce a scorable trajectory?

This runs the actual GeoBenchX tools against the actual dataset -- no mocks of
the GIS layer -- because the adapter's whole job is to be the seam between two
systems, and a mocked seam proves nothing.

The LLM is scripted rather than real. That is not a shortcut around the hard
part: it makes the run deterministic, so a failure is a failure of the adapter
instead of "the model chose differently today". Swapping in a real client is
the next step, with this as the baseline it has to match.

Skips (rather than fails) when the GeoBenchX checkout or its 1.2 GB of datasets
are absent, so the rest of the suite stays runnable on a machine without them.
"""

from __future__ import annotations

import json

import pytest

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.tools.base import get_workspace_root
from strata.geobench import build_geobench_tools, geobenchx_root
from strata.geobench.profile import DEFAULT_TOOLS, Profile, run_task
from strata.geobench.trajectory import load_reference_tasks

ROOT = geobenchx_root()


def _have_data() -> bool:
    return (ROOT / "benchmark_set" / "tasks_and_reference_solutions.json").exists() and (
        ROOT / "Data" / "GeoData"
    ).is_dir()


requires_geobenchx = pytest.mark.skipif(
    not _have_data(), reason="GeoBenchX checkout or datasets not present"
)


# --- the adapter's shape, which needs no datasets -----------------------------


def test_the_whitelist_is_a_subset_of_the_real_tools():
    """Catches a typo in DEFAULT_TOOLS, which would otherwise silently shrink."""
    from strata.geobench.tools import GEOBENCH_TOOL_NAMES

    unknown = set(DEFAULT_TOOLS) - set(GEOBENCH_TOOL_NAMES)
    assert not unknown, f"profile names tools GeoBenchX does not have: {unknown}"
    assert 8 <= len(DEFAULT_TOOLS) <= 12, "the contract fixes the first profile at 8-12 tools"


def test_bash_and_agent_are_not_in_the_profile():
    """The boundary only guards file tools; a shell would route around it."""
    assert "bash" not in DEFAULT_TOOLS
    assert "agent" not in DEFAULT_TOOLS


def test_evidence_schema_preview_is_explicit_and_validated():
    assert Profile(evidence_schema_preview="head").evidence_schema_preview == "head"
    assert Profile(evidence_schema_preview="head_tail").evidence_schema_preview == "head_tail"
    with pytest.raises(ValueError, match="schema preview"):
        Profile(evidence_schema_preview="middle")


@requires_geobenchx
def test_the_injected_state_parameter_is_stripped():
    """GeoBenchX publishes `state` as required; the model must never see it."""
    tools = build_geobench_tools()
    for tool in tools:
        assert "state" not in tool.parameters["properties"], tool.name
        assert "state" not in (tool.parameters.get("required") or []), tool.name


@requires_geobenchx
def test_every_profile_tool_can_be_built():
    tools = build_geobench_tools(only=DEFAULT_TOOLS)
    assert len(tools) == len(DEFAULT_TOOLS)
    assert {t.name for t in tools} == set(DEFAULT_TOOLS)


@requires_geobenchx
def test_a_missing_tool_fails_loudly():
    """An upstream rename must break here, not quietly shrink the profile."""
    with pytest.raises(AttributeError, match="gone_missing_tool"):
        build_geobench_tools(only=("gone_missing_tool",))


# --- headless: the failure the first baseline actually hit ---------------------
#
# GeoBenchX's tools call `plt.show()` from ten sites, and the Agent runs one
# round's tool calls on a thread pool. On macOS that raised
# `Cannot create a GUI FigureManager outside the main thread` 15 times in the
# first baseline -- every tool error in the one task that drew the most maps.


def test_the_benchmark_state_is_headless():
    """`visualize` is the switch those ten `plt.show()` calls read."""
    from strata.geobench.tools import benchmark_state

    assert benchmark_state()["visualize"] is False


def test_each_task_gets_its_own_state():
    """A shared state dict would let one task read another task's DataFrames."""
    from strata.geobench.tools import benchmark_state

    first, second = benchmark_state(), benchmark_state()
    assert first is not second
    assert first["data_store"] is not second["data_store"]


@requires_geobenchx
def test_the_tools_are_handed_a_headless_state():
    """The default must be headless, not just available on request."""
    tools = build_geobench_tools(only=DEFAULT_TOOLS)
    assert tools, "no tools were built"
    assert all(t._state["visualize"] is False for t in tools), (
        "a tool was handed a state that still asks for a GUI window"
    )


@requires_geobenchx
def test_loading_geobenchx_leaves_matplotlib_non_interactive():
    """The second layer: even a `plt.show()` that slips through must not open a window."""
    import matplotlib

    build_geobench_tools(only=DEFAULT_TOOLS)  # triggers the load
    assert matplotlib.get_backend().lower() == "agg"


@requires_geobenchx
def test_a_writer_and_its_reader_survive_being_asked_for_at_once():
    """GeoBenchX's tools share one `data_store`, so a batch is a dataflow.

    Ask for a column to be computed and then read, in a single response -- which
    is what the model did in the run that exposed this -- and the reader used to
    reach the store first: **27 of 30 such batches failed** with
    `DataFrame 'out' not found`. It was being read as a model mistake. The
    adapter now declares itself unsafe to run beside its siblings, so the batch
    runs in order.
    """
    script = [
        LLMResponse(tool_calls=[ToolCall(id="c0", name="load_data", arguments={
            "dataset": "Forest area (% of land area)",
            "output_dataframe_name": "forest"})]),
        # both in one response, the reader depending on the writer
        LLMResponse(tool_calls=[
            ToolCall(id="c1", name="calculate_columns", arguments={
                "dataframe1_name": "forest", "column1_name": "1990",
                "dataframe2_name": "forest", "column2_name": "2021",
                "operation": "subtract",
                "output_column_name": "loss", "output_dataframe_name": "out"}),
            ToolCall(id="c2", name="calculate_column_statistics", arguments={
                "dataframe_name": "out", "column_name": "loss",
                "output_variable_name": "s", "include_quantiles": False}),
        ]),
        LLMResponse(content="done"),
    ]

    traj = run_task("dummy", ScriptedLLM(script), task_id="writer-reader")

    assert traj.tool_errors == [], (
        "the reader did not see the writer's output; the batch was run out of order"
    )


@requires_geobenchx
def test_a_call_outside_the_profile_is_recorded_but_not_scored():
    """The diagnostic channel the profile had none of.

    A call to a tool the profile does not expose is answered with an error
    string and never reaches `traj.record`, so it showed up in neither `steps`
    nor `tool_errors` -- it left no trace at all. That made "the model never
    needed the other twelve tools" unfalsifiable: every time it reached for one,
    the evidence was dropped.

    It has to stay out of `steps`, though: the judge reads those, and scoring a
    call that was never made would be a different bug.
    """
    script = [
        LLMResponse(tool_calls=[ToolCall(id="c1", name="create_buffer", arguments={
            "geodataframe_name": "g", "distance": 1, "output_geodataframe_name": "b"})]),
        LLMResponse(tool_calls=[ToolCall(id="c2", name="get_unique_values", arguments={
            "dataframe_name": "g", "column_name": "NAME_EN"})]),
        LLMResponse(content="done"),
    ]

    traj = run_task("dummy", ScriptedLLM(script), task_id="unexposed")

    assert traj.unexposed == ["create_buffer"], "the out-of-profile call was dropped"
    assert [s.function_name for s in traj.steps] == ["get_unique_values"]
    assert "create_buffer" not in json.dumps(traj.as_solution()), (
        "an unexposed call leaked into the scored trajectory"
    )


# --- end to end on a real task ------------------------------------------------


def _world_electricity_calls():
    """The reference solution's four steps, as CoreCoder tool calls."""
    return [
        ToolCall(id="c1", name="load_data", arguments={
            "dataset": "Electric power consumption (kWh per capita)",
            "output_dataframe_name": "electricity_data"}),
        ToolCall(id="c2", name="load_geodata", arguments={
            "geodataset": "Countries", "output_geodataframe_name": "world_geo"}),
        ToolCall(id="c3", name="merge_dataframes", arguments={
            "dataframe_name": "electricity_data", "geodataframe_name": "world_geo",
            "statkey": "Country Name", "geokey": "NAME_EN",
            "output_dataframe_name": "electricity_map"}),
        ToolCall(id="c4", name="make_choropleth_map", arguments={
            "dataframe_name": "electricity_map", "mappingkey": "2014",
            "legendtext": "Electric power consumption (kWh per capita)",
            "colormap": "Economics"}),
    ]


@requires_geobenchx
def test_reference_trajectory_is_reproduced_step_for_step():
    """The headline smoke: run the reference plan and get the reference calls.

    If this passes, the adapter runs real tools, keeps their shared state across
    calls, and records what the judge reads. That is the whole contract.
    """
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks
                if t["task_ID"] == "TASK_250309_135125_523121")

    script = [LLMResponse(tool_calls=[c]) for c in _world_electricity_calls()]
    script.append(LLMResponse(content="Map produced."))
    llm = ScriptedLLM(script)

    traj = run_task(task["task_text"], llm, task_id=task["task_ID"])

    assert traj.stop_reason == "completed"
    produced = [(s.function_name, s.arguments) for s in traj.steps]
    expected = [(s["function_name"], s["arguments"])
                for s in task["reference_solutions"][0]["steps"]]
    assert produced == expected, "the recorded trajectory diverges from the reference"


@requires_geobenchx
def test_the_trajectory_is_what_the_judge_reads():
    """Rendering must match GeoBenchX's own `get_solution_code` shape."""
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")
    script = [LLMResponse(tool_calls=[c]) for c in _world_electricity_calls()]
    script.append(LLMResponse(content="done"))

    traj = run_task(task["task_text"], ScriptedLLM(script), task_id=task["task_ID"])
    solution = traj.as_solution()

    assert set(solution) == {"steps"}
    assert all(set(s) == {"function_name", "arguments", "comment"} for s in solution["steps"])
    # arguments survive as data, not as a pre-rendered string
    assert solution["steps"][0]["arguments"]["dataset"].startswith("Electric power")


@requires_geobenchx
def test_the_reference_choropleth_task_runs_without_a_tool_error():
    """The tool has to *run*, not just be planned.

    GeoBenchX passed `fmt` into matplotlib's Colorbar, which takes `format`
    instead. Every choropleth call in the first baseline failed -- 25 times
    across 7 of 9 tasks -- and this file did not notice, because the tests above
    assert the recorded trajectory rather than the tool's return value. This one
    closes that gap, and is what makes losing the compatibility patch in
    `patches/` loud instead of silent.
    """
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")
    script = [LLMResponse(tool_calls=[c]) for c in _world_electricity_calls()]
    script.append(LLMResponse(content="done"))

    traj = run_task(task["task_text"], ScriptedLLM(script), task_id=task["task_ID"])

    assert traj.tool_errors == [], (
        "a whitelisted tool failed. If it is Colorbar's `fmt` TypeError, the "
        "GeoBenchX compatibility patch has not been applied."
    )


# --- the profile's guarantees -------------------------------------------------


@requires_geobenchx
def test_the_workspace_boundary_is_active_during_a_task():
    """A task must not be able to read outside its own scratch directory."""
    seen = {}

    def on_tool(name, args):
        seen["root"] = get_workspace_root()

    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")
    script = [LLMResponse(tool_calls=[_world_electricity_calls()[1]])]
    script.append(LLMResponse(content="done"))

    run_task(task["task_text"], ScriptedLLM(script), task_id=task["task_ID"], on_tool=on_tool)

    assert seen["root"] is not None, "no workspace boundary was set for the task"
    assert seen["root"].is_absolute()
    # and it is released afterwards, so nothing leaks into the next task
    assert get_workspace_root() is None


@requires_geobenchx
def test_a_budget_stop_is_recorded_as_truncated_not_completed():
    """A stopped run must never be scorable as a finished one."""
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")

    # always asks for another call, so the round cap is the only way out
    class Looping:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        model = "looping"
        _n = 0

        def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
            self._n += 1
            return LLMResponse(tool_calls=[ToolCall(
                id=f"c{self._n}", name="get_unique_values",
                arguments={"dataframe_name": "x", "column_name": "y"})])

        @property
        def estimated_cost(self):
            return None

    from corecoder.limits import Limits

    traj = run_task(
        task["task_text"],
        Looping(),
        profile=Profile(limits=Limits(max_rounds=3, deadline_s=120)),
        task_id=task["task_ID"],
    )

    assert traj.stop_reason == "truncated:rounds"
    assert traj.stop_reason != "completed"


@requires_geobenchx
def test_each_task_is_isolated_from_the_last():
    """State and workspace must not carry over between tasks."""
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")

    def run_once():
        script = [LLMResponse(tool_calls=[_world_electricity_calls()[1]])]
        script.append(LLMResponse(content="done"))
        return run_task(task["task_text"], ScriptedLLM(script), task_id=task["task_ID"])

    first = run_once()
    second = run_once()

    assert [s.function_name for s in first.steps] == ["load_geodata"]
    assert [s.function_name for s in second.steps] == ["load_geodata"], (
        "the second task inherited the first task's trajectory or state"
    )


@requires_geobenchx
def test_write_submission_carries_the_reference_fields_through(tmp_path):
    """The judge reads the original task fields alongside the submission."""
    from strata.geobench.trajectory import write_submission

    tasks = load_reference_tasks(ROOT)[:2]
    script = [LLMResponse(content="nothing to do")]
    traj = run_task(tasks[0]["task_text"], ScriptedLLM(script), task_id=tasks[0]["task_ID"])

    out = write_submission([traj], tmp_path / "sub.json", tasks, "scripted")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "scripted"
    assert len(payload["tasks"]) == 2
    assert "reference_solutions" in payload["tasks"][0], "reference fields were dropped"
    assert payload["tasks"][0]["generated_solution"] == {"steps": []}
    assert payload["tasks"][1]["submission_stop_reason"] == "missing"


def test_agent_construction_uses_only_the_whitelist():
    """Belt and braces: even with a scripted LLM, bash is not reachable."""
    tools = build_geobench_tools(only=DEFAULT_TOOLS) if _have_data() else []
    if not tools:
        pytest.skip("GeoBenchX not present")
    agent = Agent(llm=ScriptedLLM([LLMResponse(content="x")]), tools=tools)
    names = {t.name for t in agent.tools}
    assert "bash" not in names and "agent" not in names


@requires_geobenchx
def test_a_request_that_outlives_the_deadline_is_recorded_as_truncated():
    """The end of the chain: an expired request budget must reach the artifact.

    `GeoBenchX` scores a trajectory; a run that stopped because it ran out of
    time must not look like one that finished. `truncated:deadline` says which
    happened, and it has to survive the whole way from the LLM client to
    `Trajectory.stop_reason`.
    """
    from corecoder.limits import Limits
    from corecoder.llm import RequestBudgetExceeded

    class _OutOfTime:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        estimated_cost = None

        def chat(self, messages, tools=None, on_token=None, on_reasoning=None, timeout=None):
            raise RequestBudgetExceeded("out of time")

    traj = run_task(
        "anything",
        _OutOfTime(),
        profile=Profile(limits=Limits(max_rounds=3, deadline_s=10)),
        task_id="deadline",
    )

    assert traj.stop_reason == "truncated:deadline"
    assert traj.stop_reason != "completed"


# --- evidence: addressable artifacts, without touching the scored trajectory ---


def test_the_addressability_guard_knows_what_lives_inside_the_workspace(tmp_path):
    from strata.geobench.profile import _is_inside

    assert _is_inside(tmp_path / "a" / "b", tmp_path)
    assert not _is_inside(tmp_path.parent / "elsewhere", tmp_path)
    assert _is_inside(tmp_path, tmp_path), "a root contains itself"


@requires_geobenchx
def test_evidence_leaves_the_scored_trajectory_exactly_as_it_was(tmp_path):
    """The judge reads `steps`. Filing an artifact is not a tool call.

    If turning evidence on changed what the judge sees, every comparison
    between the arms would be measuring two things at once.
    """
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")

    def run(profile):
        script = [LLMResponse(tool_calls=[c]) for c in _world_electricity_calls()]
        script.append(LLMResponse(content="done"))
        return run_task(task["task_text"], ScriptedLLM(script), profile=profile,
                        task_id=task["task_ID"])

    control = run(Profile())
    treatment = run(Profile(evidence_dir=tmp_path / "evidence"))

    assert treatment.as_solution() == control.as_solution()
    assert control.evidence_refs == [], "the control arm stored something"
    assert treatment.evidence_refs, "evidence was on and nothing was written"


@requires_geobenchx
def test_a_stored_artifact_still_resolves_after_the_task(tmp_path):
    """The point of a stable directory: `path` has to work once the run is over.

    The task workspace is deleted on the way out, so an artifact written there
    would be gone by now -- which is exactly why the store may not live there.
    """
    tasks = load_reference_tasks(ROOT)
    task = next(t for t in tasks if t["task_ID"] == "TASK_250309_135125_523121")
    root = tmp_path / "evidence"

    script = [LLMResponse(tool_calls=[c]) for c in _world_electricity_calls()]
    script.append(LLMResponse(content="done"))
    traj = run_task(task["task_text"], ScriptedLLM(script),
                    profile=Profile(evidence_dir=root), task_id=task["task_ID"])

    assert traj.evidence_refs
    stored = [ref for ref in traj.evidence_refs if ref["path"] is not None]
    assert stored, "nothing was stored at all"
    for ref in stored:
        path = root / task["task_ID"] / ref["path"]
        assert path.exists(), f"{ref['path']} did not survive the run"
        assert path.stat().st_size == ref["size"]

    # The contract is "addressable up to a stated size", not "everything is":
    # a frame that serialises to tens of megabytes is fingerprinted instead, and
    # the record says so rather than quoting a path that resolves to nothing.
    for ref in traj.evidence_refs:
        if ref["path"] is None:
            assert ref["sha256"] and ref["size"] > 0


@requires_geobenchx
def test_evidence_never_makes_a_result_larger(tmp_path):
    """The rule that came out of measuring: a description is only used when it
    costs less than the thing it describes.

    The first version had no such rule and made `load_geodata` results two and a
    half times longer, because those results are short to begin with and the
    block added a head of the original on top of a summary.
    """
    from strata.geobench.evidence import EvidenceStore
    from strata.geobench.tools import build_geobench_tools

    def state():
        return {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}

    control = {t.name: t for t in build_geobench_tools(state=state())}
    treatment = {t.name: t for t in build_geobench_tools(
        state=state(), evidence=EvidenceStore(tmp_path / "evidence"))}

    calls = [
        ("load_data", {"dataset": "Forest area (sq. km)", "output_dataframe_name": "f"}),
        ("load_geodata", {"geodataset": "Countries", "output_geodataframe_name": "c"}),
    ]
    for name, args in calls:
        raw = control[name].execute(**args)
        enriched = treatment[name].execute(**args)
        assert len(enriched) <= len(raw), (
            f"{name}: evidence turned {len(raw)} characters into {len(enriched)}"
        )


@requires_geobenchx
def test_a_high_cardinality_result_is_cut_hard(tmp_path):
    """Where the block earns its place: a column with hundreds of thousands of
    distinct values, which is the shape that appeared in a real run."""
    from strata.geobench.evidence import EvidenceStore
    from strata.geobench.tools import build_geobench_tools

    def state():
        return {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}

    control = {t.name: t for t in build_geobench_tools(state=state())}
    treatment = {t.name: t for t in build_geobench_tools(
        state=state(), evidence=EvidenceStore(tmp_path / "e"))}

    load = {"geodataset": "Earthquakes occurences and magnitude March 15- February 14 2025",
            "output_geodataframe_name": "q"}
    control["load_geodata"].execute(**load)
    treatment["load_geodata"].execute(**load)

    ask = {"dataframe_name": "q", "column": "time"}
    raw = control["get_unique_values"].execute(**ask)
    enriched = treatment["get_unique_values"].execute(**ask)

    assert len(raw) > 100_000, f"expected a huge result, got {len(raw)}"
    assert len(enriched) * 20 < len(raw), (
        f"{len(enriched)} vs {len(raw)}: the summary did not cut a huge result"
    )
