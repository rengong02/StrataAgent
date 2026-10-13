#!/usr/bin/env python
"""Run GeoBenchX tasks with a real model, under the core-12 benchmark profile.

Two things this deliberately does *not* do:

* **It does not score.** Producing trajectories and judging them are separate
  steps with separate failure modes; running both in one command makes "the
  model did badly" and "the judge did badly" hard to tell apart.
* **It does not report a single headline number.** It reports the eligible
  denominator alongside the results, because the twelve whitelisted tools cannot
  solve 50 of the 202 tasks, and folding those zeros into a model score would
  understate the model rather than describe it.

An eligible task has **at least one**
reference solution uses only whitelisted tools, since one reachable plan is
enough for the model to have a chance.

Usage:
    python scripts/run_geobench_tasks.py --provider deepseek --model deepseek-flash --limit 5
    python scripts/run_geobench_tasks.py --provider deepseek --model deepseek-flash \
        --labels "Merge, Visualize" --out evals/solver_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corecoder.limits import Limits
from corecoder.llm import LLM
from strata.geobench.evidence import DEFAULT_SCHEMA_PREVIEW, SCHEMA_PREVIEWS
from strata.geobench.judge import JudgeError, load_env
from strata.geobench.profile import DEFAULT_TOOLS, Profile, run_task
from strata.geobench.provenance import geobenchx_provenance
from strata.geobench.tools import geobenchx_root, load_geobenchx_tools_module
from strata.geobench.trajectory import Trajectory


def classify(task: dict, whitelist: set[str]) -> str:
    """Which capability a task actually exercises.

    A single number over a mixed pool says nothing: answering 3 control questions
    and rejecting 86 unsolvable ones is not the same skill as planning a
    multi-step spatial query. The classification of the 202 tasks must
    keep them apart.
    """
    solutions = task.get("reference_solutions") or []
    names = [{s["function_name"] for s in sol["steps"]} for sol in solutions]
    reachable = [n for n in names if n <= whitelist]
    if not reachable:
        return "unreachable"              # needs a tool outside the whitelist
    if any(n and n != {"reject_task"} for n in reachable):
        return "positive"                 # a real multi-step tool plan
    if any(not n for n in reachable):
        return "control"                  # answerable from knowledge, no tools
    return "reject_only"                  # correctly refusing is the whole task


def eligible(tasks: list[dict], whitelist: set[str]) -> list[dict]:
    """Tasks where at least one reference plan is entirely reachable."""
    out = []
    for task in tasks:
        for solution in task.get("reference_solutions") or []:
            names = {step["function_name"] for step in solution["steps"]}
            if names <= whitelist:
                out.append(task)
                break
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--labels", default=None, help="only tasks carrying this label")
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated task IDs; overrides --limit and file order. "
                         "Sampling in file order mixes kinds together.")
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--deadline", type=float, default=600.0)
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "solver_runs.jsonl")
    ap.add_argument("--evidence-dir", type=Path, default=None,
                    help="persist raw tool results and structured evidence under this directory. "
                         "The directory must survive after the run.")
    ap.add_argument("--evidence-schema-preview", choices=SCHEMA_PREVIEWS,
                    default=DEFAULT_SCHEMA_PREVIEW,
                    help="schema columns delivered by evidence: head reproduces the "
                         "legacy prefix-only arm; head_tail keeps both ends.")
    ap.add_argument("--schema-hints", action="store_true",
                    help="name the available columns on a not-found error. Off by default: "
                         "seven runs per arm found no effect (p=0.41 on the failure count, "
                         "0.91 on tokens), so this is an ablation arm rather than a default.")
    ap.add_argument("--allow-unapplied-patches", action="store_true",
                    help="run even though a patch in patches/ is not applied. Only for a "
                         "deliberate diagnostic run -- the artifact carries a known upstream bug.")
    return ap


def make_row(task: dict, traj: Trajectory, solver: str,
             whitelist: set[str], provenance: dict) -> dict:
    """Serialize one trajectory without mixing diagnostics into scored steps."""
    return {
        "task_ID": task["task_ID"],
        "task_text": task["task_text"],
        "labels": task.get("task_labels"),
        "kind": classify(task, whitelist),
        "solver": solver,
        "stop_reason": traj.stop_reason,
        "steps": traj.as_solution()["steps"],
        "step_count": len(traj.steps),
        "tokens": traj.tokens,
        "wall_seconds": round(traj.wall_seconds, 1),
        "final_response": traj.final_response,
        "tool_errors": traj.tool_errors,
        "unexposed_tool_calls": traj.unexposed,
        "repeat_failures": traj.repeat_failures,
        "deadline_overruns": traj.deadline_overruns,
        "evidence_refs": traj.evidence_refs,
        # How often the block engaged and what it saved. "Evidence is on" is not
        # a result; this is what a paired comparison is allowed to quote.
        "evidence_stats": traj.evidence_stats,
        "reference_step_counts": [
            len(s["steps"]) for s in task.get("reference_solutions") or []
        ],
        "provenance": provenance,
    }


def main() -> int:
    args = build_parser().parse_args()

    load_env()
    load_geobenchx_tools_module()

    prov = geobenchx_provenance(ROOT)
    unapplied = sorted(name for name, applied in prov["patches"].items() if not applied)
    print("provenance:")
    for key, value in prov.items():
        print(f"  {key:20s} {value}")
    print()
    if unapplied and not args.allow_unapplied_patches:
        # Refusing by default: a run without the compatibility patch produces
        # artifacts full of a failure that is already fixed, and the first time
        # that happened nothing in the output said so.
        print(f"refusing to run: not applied -> {unapplied}", file=sys.stderr)
        print("apply them, or pass --allow-unapplied-patches for a diagnostic run",
              file=sys.stderr)
        return 2

    tasks_path = geobenchx_root() / "benchmark_set" / "tasks_and_reference_solutions.json"
    all_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]

    pool = eligible(all_tasks, set(DEFAULT_TOOLS))
    if args.labels:
        pool = [t for t in pool if args.labels in (t.get("task_labels") or [])]
    if args.task_ids:
        wanted = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        by_id = {t["task_ID"]: t for t in pool}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            print(f"not in the eligible pool: {missing}", file=sys.stderr)
            return 2
        selected = [by_id[t] for t in wanted]
    else:
        selected = pool[: args.limit]
    if not selected:
        print("no eligible task matches; check --labels", file=sys.stderr)
        return 2

    print(f"solver  : {args.provider}:{args.model}")
    print(f"profile : {len(DEFAULT_TOOLS)} tools, no bash, no agent, per-task workspace")
    print(f"pool    : {len(pool)} eligible of {len(all_tasks)} total  (labels={args.labels or 'any'})")
    print(f"running : {len(selected)} task(s)")
    print()

    llm = LLM(model=args.model, api_key=_solver_key(args.provider), base_url=_solver_url(args.provider))
    evidence_dir = args.evidence_dir.expanduser().resolve() if args.evidence_dir else None
    profile = Profile(
        limits=Limits(max_rounds=args.max_rounds, deadline_s=args.deadline),
        schema_hints=args.schema_hints,
        evidence_dir=evidence_dir,
        evidence_schema_preview=args.evidence_schema_preview,
    )
    # The arm is part of what produced the artifact. Two runs that differ only
    # by this flag are otherwise indistinguishable in the output.
    prov["schema_hints"] = profile.schema_hints
    prov["evidence_dir"] = str(evidence_dir) if evidence_dir is not None else None
    prov["evidence_schema_preview"] = (
        args.evidence_schema_preview if evidence_dir is not None else None
    )
    print(f"arm     : schema_hints={profile.schema_hints}")
    print(f"evidence: {evidence_dir or 'off'}"
          f"{'' if evidence_dir is None else f' ({args.evidence_schema_preview})'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Truncated once here because rows are appended as they finish; without
    # this a re-run would stack onto the previous one's output.
    args.out.write_text("", encoding="utf-8")

    rows = []
    crashes: list[tuple[str, str]] = []
    started = time.time()
    for i, task in enumerate(selected, 1):
        try:
            traj = run_task(task["task_text"], llm, profile=profile, task_id=task["task_ID"])
        except Exception as e:  # noqa: BLE001
            # One task dying must not discard the whole run's evidence: three
            # runs were lost that way, each eight minutes of real work. The
            # task is recorded as `crashed`, never as `completed`, so it cannot
            # be mistaken for a finished one -- and the exit code still fails.
            crashes.append((task["task_ID"], f"{type(e).__name__}: {e}"[:200]))
            traj = Trajectory(task_id=task["task_ID"], stop_reason=f"crashed:{type(e).__name__}")
        row = make_row(
            task,
            traj,
            f"{args.provider}:{args.model}",
            set(DEFAULT_TOOLS),
            prov,
        )
        rows.append(row)
        # Flushed per task, not at the end: a run that is killed part-way still
        # leaves the tasks it finished, which is the difference between a
        # partial result and no result at all.
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {i}/{len(selected)}  {task['task_ID'][-6:]}  "
              f"steps={row['step_count']}  stop={row['stop_reason']}  "
              f"tokens={row['tokens']:,}  [{row['kind']}]")

    # Already written, one row at a time. This block used to run only after the
    # loop, which meant a run killed by an external timeout -- as one was, at
    # twenty-five minutes -- left nothing on disk and had to be paid for twice.

    truncated = [r for r in rows if r["stop_reason"].startswith("truncated")]
    errored = [r for r in rows if r["tool_errors"]]
    print(f"\n{'=' * 64}")
    print(f"solver {args.provider}:{args.model}   {len(rows)} task(s)")
    print(f"{'=' * 64}")
    print("  by kind (a mixed pool has no single meaningful score):")
    for kind in ("positive", "reject_only", "control"):
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        steps = sorted(r["step_count"] for r in group)
        print(f"    {kind:12s} n={len(group):>3}   median steps={steps[len(steps) // 2]:>3}   "
              f"tokens={sum(r['tokens'] for r in group):,}")
    print(f"  truncated (budget) : {len(truncated)}   {sorted({r['stop_reason'] for r in truncated})}")
    print(f"  tasks with tool errors: {len(errored)}   "
          f"{sorted({name for r in errored for name, _ in r['tool_errors']})}")
    # Not scored and not in `steps`: this is how often the model reached for a
    # tool the profile does not expose. Discarding it would make "the model did
    # not need the other twelve" unfalsifiable.
    reached = [n for r in rows for n in r["unexposed_tool_calls"]]
    print(f"  calls to unexposed tools: {len(reached)}   {sorted(set(reached))}")
    # The deterministic cost of trial and error. Independent of the judge, so a
    # change here is a real change even when the score cannot say so.
    repeated = {k: v for r in rows for k, v in r["repeat_failures"].items() if v > 1}
    wasted = sum(v - 1 for v in repeated.values())
    print(f"  repeated identical failures: {len(repeated)} call(s), "
          f"{wasted} repeat(s) beyond the first   {sorted({k.split(':', 1)[0] for k in repeated})}")
    # GeoBenchX's tools are in-process functions and cannot be interrupted, so
    # the deadline is cooperative for them. Reporting the overruns is the honest
    # alternative to claiming they cannot happen.
    ev = [r["evidence_stats"] for r in rows if r["evidence_stats"]]
    if ev:
        before = sum(e["bytes_before"] for e in ev)
        after = sum(e["bytes_after"] for e in ev)
        replaced = sum(e["replaced"] for e in ev)
        calls = sum(e["calls"] for e in ev)
        print(f"  evidence           : {replaced}/{calls} call(s) replaced, "
              f"{before:,} -> {after:,} chars ({after / max(before, 1):.1%} kept), "
              f"{sum(e['artifacts'] for e in ev)} artifact(s)")
    overruns = [(r["task_ID"][-6:], n, s, m)
                for r in rows for n, s, m in r["deadline_overruns"]]
    print(f"  tool calls past the deadline: {len(overruns)}   "
          f"{sorted({(n, m) for _, n, _, m in overruns})}")
    print(f"  total tokens       : {sum(r['tokens'] for r in rows):,}")
    print(f"  elapsed            : {time.time() - started:.0f}s")
    print(f"\nwrote {args.out}")
    print("Nothing is scored here by design; run the judge as a separate step.")
    if crashes:
        # The artifacts are still written -- a partial run is evidence, and the
        # crashed rows are marked -- but the exit code must not say "fine".
        print(f"\n{len(crashes)} task(s) CRASHED and were recorded as such:", file=sys.stderr)
        for task_id, why in crashes:
            print(f"  {task_id[-6:]}: {why}", file=sys.stderr)
        return 1
    return 0


def _provider_entry(provider: str):
    """Look up (base_url, env var) from the judge's table.

    One table knows the endpoints and variable names, so a provider added for
    judging is immediately usable for solving.
    """
    from strata.geobench.judge import _OPENAI_COMPATIBLE

    entry = _OPENAI_COMPATIBLE.get(provider.lower())
    if entry is None:
        raise JudgeError(
            f"{provider!r} is not OpenAI-compatible; this runner only drives those. "
            "Known: " + ", ".join(_OPENAI_COMPATIBLE)
        )
    return entry


def _solver_key(provider: str) -> str:
    import os

    _, env = _provider_entry(provider)
    key = os.getenv(env)
    if not key:
        raise JudgeError(f"{env} is not set, so {provider} cannot be the solver")
    return key


def _solver_url(provider: str) -> str | None:
    return _provider_entry(provider)[0]


if __name__ == "__main__":
    raise SystemExit(main())
