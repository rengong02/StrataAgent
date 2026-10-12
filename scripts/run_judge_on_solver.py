#!/usr/bin/env python
"""Score a solver run with the calibrated judge.

This is the second half of the two-step split `run_geobench_tasks.py` describes:
that script produces trajectories and deliberately does not score them, because
"the model did badly" and "the judge did badly" are different failures and one
command cannot tell them apart.

It exists as a script rather than a shell one-liner because the first scoring of
a solver run *was* a one-liner: the numbers reached a report while nothing in the
repo could regenerate them. That is the same "a calibration that cannot be re-run
is an anecdote" problem the calibration runner's docstring names.

Nothing here reads or prints a key value; a missing one is reported by variable
name alone.

Usage:
    python scripts/run_judge_on_solver.py --provider deepseek --model deepseek-flash \
        --solver evals/solver_positive9_toolfix.jsonl \
        --out evals/judge_positive9_toolfix.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata.geobench.judge import Judge, JudgeError, build_judge_config, load_env, majority_score
from strata.geobench.tools import geobenchx_root, load_geobenchx_tools_module


def load_solver_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["task_ID"]: row for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--solver", type=Path, required=True, help="a JSONL written by run_geobench_tasks.py")
    ap.add_argument("--out", type=Path, required=True, help="where the judge's own log goes")
    ap.add_argument("--repeats", type=int, default=1, metavar="N",
                    help="score each task N times and report the majority. The judge is a "
                         "reasoning model that is not deterministic even at temperature 0: "
                         "on the 9-task ablation, 8 of 9 tasks scored differently across four "
                         "runs of the SAME trajectories, totals ranged 8-12 of 18, and pairwise "
                         "agreement was about 39 percent. A single run is not a measurement.")
    args = ap.parse_args()

    load_env()
    load_geobenchx_tools_module()
    from geobenchx.dataclasses import Task

    if not args.solver.exists():
        print(f"solver run not found: {args.solver}", file=sys.stderr)
        return 2

    rows = load_solver_rows(args.solver)
    tasks_path = geobenchx_root() / "benchmark_set" / "tasks_and_reference_solutions.json"
    references = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]

    # Only tasks this run actually produced a trajectory for. Scoring the whole
    # 202 would silently mix "not attempted" with "attempted and failed".
    selected = [t for t in references if t["task_ID"] in rows]
    if not selected:
        print(f"no task in {args.solver} matches the reference set", file=sys.stderr)
        return 2

    try:
        cfg = build_judge_config(args.provider, args.model)
    except JudgeError as e:
        print(f"cannot build the judge: {e}", file=sys.stderr)
        return 2

    print(f"judge  : {cfg.describe()}")
    print(f"solver : {args.solver}")
    print(f"tasks  : {len(selected)}   (of {len(references)} in the reference set)")
    print(f"log    : {args.out}")
    print()

    judge = Judge(cfg, log_path=args.out)
    per_task: list[tuple[str, list[int], str]] = []
    failures: list[tuple[str, str]] = []
    started = time.time()

    for i, reference in enumerate(selected, 1):
        row = rows[reference["task_ID"]]
        kind = row.get("kind", "?")
        scores: list[int] = []
        for _ in range(max(1, args.repeats)):
            # The judge reads `generated_solution`; the reference fields must
            # stay, because the rubric hands both the candidate and the
            # reference plans to the judge. A submission that dropped them
            # would score nothing useful.
            entry = dict(reference)
            entry["generated_solution"] = {"steps": row["steps"]}
            entry["match_reasoning_LLM"] = None
            entry["match_score_LLM"] = None
            try:
                _, score = judge.score(Task(**entry))
                scores.append(score)
            except JudgeError as e:
                failures.append((reference["task_ID"], str(e)[:160]))
        if scores:
            per_task.append((reference["task_ID"], scores, kind))
        print(f"  {i}/{len(selected)}  {reference['task_ID'][-6:]}  "
              f"scores={''.join(str(s) for s in scores) or '-':>6}  "
              f"steps={row['step_count']:>3}  [{kind}]")

    if not per_task:
        print("\nnothing was scored; see the failures above", file=sys.stderr)
        return 1

    settled = [(task_id, majority_score(scores), scores, kind) for task_id, scores, kind in per_task]
    n = len(settled)
    total = sum(s for _, s, _, _ in settled)
    exact = sum(1 for _, s, _, _ in settled if s == 2)

    print(f"\n{'=' * 64}")
    print(f"judge  {cfg.describe()}")
    print(f"{'=' * 64}")
    print(f"  scored        : {n}   ({args.repeats} judge run(s) each)")
    print(f"  total         : {total} / {2 * n}   ({100 * total / (2 * n):.1f}%)   [majority]")
    print(f"  exact (==2)   : {exact} / {n}   [majority]")
    print(f"  judge failures: {len(failures)}")
    for task_id, why in failures:
        print(f"    {task_id[-6:]}: {why}")

    # Hoisted out of the branch below: the summary is written either way, and a
    # one-run summary still has a total to report.
    repeats = max(1, args.repeats)
    runs = [sum(s[i] for _, _, s, _ in settled if len(s) > i) for i in range(repeats)]
    unstable = [t for t, _, s, _ in settled if len(set(s)) > 1]

    if repeats > 1:
        # The spread is the whole reason for repeating: reporting only the
        # majority would hide that the number moves at all.
        print(f"  per-run totals: {runs}   spread={max(runs) - min(runs)}")
        print(f"  unstable tasks: {len(unstable)}/{n} scored differently across runs")
    print(f"  elapsed       : {time.time() - started:.0f}s")

    # A sidecar, because the log is one row per judge call: with repeats, nine
    # tasks become twenty-seven rows and the only way to recover "which score
    # did this run report" is to infer it from row order. The summary states it.
    summary = {
        "judge": cfg.describe(),
        "solver": str(args.solver),
        "tasks": n,
        "repeats": repeats,
        "per_task_scores": {task_id: scores for task_id, _, scores, _ in settled},
        "majority_total": total,
        "max_total": 2 * n,
        "exact": exact,
        "per_run_totals": runs,
        "spread": max(runs) - min(runs),
        "unstable_tasks": unstable,
        "judge_failures": [{"task_id": t, "error": e} for t, e in failures],
        "metric": "GeoBenchX tool-plan match score, not end-to-end GIS accuracy",
    }
    sidecar = args.out.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"wrote {sidecar}")
    print("This is the GeoBenchX tool-plan match score, not end-to-end GIS accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
