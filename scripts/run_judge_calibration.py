#!/usr/bin/env python
"""Score GeoBenchX's 50 human-annotated tasks with a judge of your choosing.

This exists because the first calibration result was produced by a hand-typed
command that lived in nobody's shell history: the numbers reached a report, but
nothing in the repo could regenerate them. A calibration that cannot be re-run
is an anecdote.

Prints the report GeoBenchX's own taxonomy asks for: exact agreement first, then
the confusion matrix, then within-one as a supplement. Nothing here reads or
prints a key value -- a missing one is reported by variable name.

Usage:
    python scripts/run_judge_calibration.py --provider deepseek --model deepseek-flash
    python scripts/run_judge_calibration.py --provider openai --model gpt-4.1 \
        --out evals/judge_tuning50_openai.jsonl

Requires the relevant key in the environment or in `.env` (found by walking up
from the cwd).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata.geobench.judge import (
    Judge,
    JudgeError,
    build_judge_config,
    calibration_report,
)
from strata.geobench.tools import geobenchx_root, load_geobenchx_tools_module


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval.

    50 tasks is a small sample: two judges four points apart can easily have
    overlapping intervals, and saying "90% vs 94%" without them invites a
    significance claim the data does not support.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, help="openai | deepseek | dashscope | siliconflow | anthropic")
    ap.add_argument("--model", required=True, help="model id available to your account")
    ap.add_argument("--tuning-set", type=Path, default=None, help="defaults to GeoBenchX's")
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "judge_calibration.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N (for a smoke run)")
    args = ap.parse_args()

    load_geobenchx_tools_module()
    from geobenchx.dataclasses import Task

    tuning = args.tuning_set or (geobenchx_root() / "benchmark_set" / "evaluator_tuning_set.json")
    if not tuning.exists():
        print(f"tuning set not found: {tuning}", file=sys.stderr)
        return 2
    tasks_raw = json.loads(tuning.read_text(encoding="utf-8"))["tasks"]
    if args.limit:
        tasks_raw = tasks_raw[: args.limit]

    try:
        cfg = build_judge_config(args.provider, args.model)
    except JudgeError as e:
        print(f"cannot build the judge: {e}", file=sys.stderr)
        return 2

    print(f"judge   : {cfg.describe()}")
    print(f"tasks   : {len(tasks_raw)} from {tuning}")
    print(f"log     : {args.out}")
    print()

    judge = Judge(cfg, log_path=args.out)
    ours, human, panel, failures = [], [], [], []
    started = time.time()

    for i, raw in enumerate(tasks_raw, 1):
        task = Task(**raw)
        human.append(task.match_score_Human.value)
        panel.append(task.match_score_LLM.value)
        try:
            _, score = judge.score(task)
            ours.append(score)
        except JudgeError as e:
            failures.append((task.task_ID, str(e)[:120]))
            ours.append(None)
        if i % 10 == 0 or i == len(tasks_raw):
            print(f"  {i}/{len(tasks_raw)}  ({time.time() - started:.0f}s)")

    paired = [(o, h) for o, h in zip(ours, human) if o is not None]
    if not paired:
        print("\nno task was scored; see the failures above", file=sys.stderr)
        return 1

    js = [o for o, _ in paired]
    hs = [h for _, h in paired]
    report = calibration_report(js, hs)

    panel_hits = sum(1 for p, h in zip(panel, human) if p == h)
    ours_hits = sum(1 for a, b in zip(js, hs) if a == b)
    lo, hi = wilson(ours_hits, report["n"])
    plo, phi = wilson(panel_hits, len(human))

    print(f"\n{'=' * 60}")
    print(f"judge: {cfg.describe()}   vs   {report['n']} human-annotated tasks")
    print(f"{'=' * 60}")
    print(f"  exact agreement : {report['exact_agreement']:.1%}  ({ours_hits}/{report['n']})")
    print(f"    95% Wilson CI : [{lo:.1%}, {hi:.1%}]")
    print(f"  within one      : {report['within_one']:.1%}")
    print(f"  GeoBenchX panel : {panel_hits}/{len(human)} = {panel_hits / len(human):.1%}   CI [{plo:.1%}, {phi:.1%}]")
    print("  (row = human, column = this judge)")
    for h in (0, 1, 2):
        cells = "  ".join(f"{report['confusion'][f'{h}->{j}']:>3}" for j in (0, 1, 2))
        print(f"    human {h} |{cells}")
    print(f"  parse failures  : {len(failures)}")
    for task_id, err in failures[:5]:
        print(f"    {task_id}: {err}")
    print(f"  tokens          : in={judge.usage.get('input_tokens', 0):,} "
          f"out={judge.usage.get('output_tokens', 0):,}")
    print(f"  elapsed         : {time.time() - started:.0f}s")
    print()
    print("The intervals overlap unless stated otherwise -- do not claim this judge")
    print("beats or loses to another on this sample size alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
