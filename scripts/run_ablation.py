#!/usr/bin/env python
"""Run one task under several arms, N times each, and report the difference.

Built because the P1 ablation was done by hand: fourteen artifacts sat in
`/tmp`, and the permutation test lived in a shell history that nobody could
re-run. **A comparison whose analysis is not in the repo is an anecdote**, which
is the same failure the judge-calibration runner's docstring already names.

Two things it does that a loop of `run_geobench_tasks.py` does not:

* every run is written under the artifact directory with its arm in the
  filename, and each row already carries its arm in `provenance`;
* the summary tests matched, completed tasks rather than pooling truncated
  runs and discarding the task pairing. Repeats are averaged within each task.

The metric that matters is chosen per experiment, not averaged blindly: P1a
reported failed `get_unique_values` calls, steps and tokens separately, because
they turned out not to move together.

Usage:
    python scripts/run_ablation.py --task-ids TASK_... --repeats 7 \
        --arm schema_hints=off --arm schema_hints=on \
        --out-dir evals/ablation/p1a_schema_hints
    python scripts/run_ablation.py --analyze-only --out-dir evals/ablation/p1a_schema_hints
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata.geobench.profile import Profile
from strata.geobench.stats import paired_permutation_p

#: Seconds a single whole run may take. Measured: a nine-task control run takes
#: ~600s, and an evidence run took past 1500s -- the summarised results change
#: what the model does, and one arm being slower is a result, not a failure. Set
#: above the slowest arm so a slow arm gets measured instead of killed.
RUN_TIMEOUT = 3600

#: The fields a report is allowed to quote. `tool_errors` is split by tool
#: because "13 errors" and "13 errors, all the same wrong column" are different
#: findings.
METRICS = ("step_count", "tokens", "wall_seconds")


def run_metrics(path: Path) -> list[dict]:
    """Every task row in one artifact file, as its own observation.

    One file holds a whole task set, so reading only the first row would report
    the first task and call it the arm -- which is exactly what happened on the
    first multi-task run here: nine tasks of control were summarised as a single
    four-step task.
    """
    repeat = _repeat_number(path)
    return [_row_metrics(json.loads(line), path.name, repeat)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _repeat_number(path: Path) -> int | None:
    match = re.fullmatch(r"run_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _row_metrics(row: dict, artifact: str, repeat: int | None = None) -> dict:
    counts: dict[str, int] = {}
    for name, _ in row["tool_errors"]:
        counts[name] = counts.get(name, 0) + 1
    provenance = row.get("provenance") or {}
    preview = provenance.get("evidence_schema_preview")
    arm = f"evidence:{preview}" if provenance.get("evidence_dir") and preview else (
        "evidence" if provenance.get("evidence_dir") else "control"
    )
    return {
        "task_id": row["task_ID"],
        "task": row["task_ID"][-6:],
        "artifact": artifact,
        "arm": arm,
        "repeat": repeat,
        "stop_reason": row["stop_reason"],
        "steps": row["step_count"],
        "tokens": row["tokens"],
        "wall_seconds": row["wall_seconds"],
        "tool_errors": len(row["tool_errors"]),
        "failed_calls_by_tool": counts,
        "repeat_failures": row.get("repeat_failures") or {},
        "unexposed_calls": len(row.get("unexposed_tool_calls") or []),
        # Cooperative tools cannot be stopped mid-call, so how often they ran
        # past the deadline is a measured property of the arm, not a promise.
        "deadline_overruns": row.get("deadline_overruns") or [],
        # For an evidence arm this is the point: how often the block engaged,
        # and how many characters it saved.
        "evidence_stats": row.get("evidence_stats") or {},
    }


def arm_of(path: Path) -> str:
    """The directory name under the out-dir is the arm.

    The row also carries its arm in `provenance`, which is what a reader should
    trust; the directory is what makes two arms distinguishable on disk.
    """
    return path.parent.name


def _metric_test(pairs: list[tuple[str, dict, dict]], getter: Callable[[dict], float]) -> dict:
    """Test task-mean differences; retain completed-run totals for diagnostics."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for task_id, left, right in pairs:
        grouped.setdefault(task_id, []).append((getter(left), getter(right)))
    left_values = [statistics.mean(left for left, _ in rows) for rows in grouped.values()]
    right_values = [statistics.mean(right for _, right in rows) for rows in grouped.values()]
    diff, probability = paired_permutation_p(left_values, right_values) if pairs else (None, None)
    return {
        "mean_diff": diff,
        "p": probability,
        "left_total": sum(getter(left) for _, left, _ in pairs),
        "right_total": sum(getter(right) for _, _, right in pairs),
        "right_lower": sum(right < left for left, right in zip(left_values, right_values)),
        "equal": sum(right == left for left, right in zip(left_values, right_values)),
        "right_higher": sum(right > left for left, right in zip(left_values, right_values)),
    }


def _paired_summary(per_arm: dict[str, list[dict]]) -> dict:
    if len(per_arm) != 2:
        return {}

    left_name, right_name = sorted(per_arm)
    grouped: dict[tuple[str, int | None], dict[str, list[dict]]] = {}
    for arm, rows in per_arm.items():
        for row in rows:
            key = (row["task_id"], row.get("repeat"))
            grouped.setdefault(key, {}).setdefault(arm, []).append(row)

    pairs = []
    excluded = []
    exclusions = []
    pair_keys = []
    for (task_id, repeat), arms in sorted(grouped.items()):
        pair_key = task_id if repeat is None else f"{task_id}#run_{repeat}"
        left_rows = arms.get(left_name, [])
        right_rows = arms.get(right_name, [])
        if len(left_rows) != 1 or len(right_rows) != 1:
            excluded.append(pair_key)
            reason = "duplicate_rows" if len(left_rows) > 1 or len(right_rows) > 1 else "missing_arm"
            exclusions.append({"pair": pair_key, "reason": reason,
                               "row_counts": {left_name: len(left_rows), right_name: len(right_rows)}})
            continue
        left, right = left_rows[0], right_rows[0]
        if left["stop_reason"] != "completed" or right["stop_reason"] != "completed":
            excluded.append(pair_key)
            exclusions.append({"pair": pair_key, "reason": "not_completed",
                               "stop_reasons": {left_name: left["stop_reason"], right_name: right["stop_reason"]}})
            continue
        pair_keys.append(pair_key)
        pairs.append((task_id, left, right))

    tests = {}
    for label, getter in (
        ("tokens", lambda row: row["tokens"]),
        ("steps", lambda row: row["steps"]),
        ("failed calls", lambda row: sum(row["failed_calls_by_tool"].values())),
    ):
        tests[label] = _metric_test(pairs, getter)

    tool_names = sorted({name for _, left, right in pairs
                         for row in (left, right) for name in row["failed_calls_by_tool"]})
    by_tool = {
        name: _metric_test(pairs, lambda row, tool=name: row["failed_calls_by_tool"].get(tool, 0))
        for name in tool_names
    }
    totals = {}
    for arm, side in ((left_name, 1), (right_name, 2)):
        rows = [pair[side] for pair in pairs]
        totals[arm] = {
            "n": len(rows),
            "steps": sum(row["steps"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "failed_calls": sum(row["tool_errors"] for row in rows),
        }
    tasks = sorted({task_id for task_id, _, _ in pairs})

    return {
        "arms": [left_name, right_name],
        "n": len(pairs),
        "n_tasks": len(tasks),
        "tasks": tasks,
        "pairs": pair_keys,
        "excluded": excluded,
        "exclusions": exclusions,
        "totals": totals,
        "tests": tests,
        "failed_calls_by_tool": by_tool,
    }


def analyze(out_dir: Path) -> dict:
    runs = sorted(out_dir.glob("*/run_*.jsonl"))
    if not runs:
        raise SystemExit(f"no artifacts under {out_dir}")

    per_arm: dict[str, list[dict]] = {}
    for path in runs:
        per_arm.setdefault(arm_of(path), []).extend(run_metrics(path))

    print("all runs (diagnostic totals; includes truncated outcomes):")
    print(f"{'arm':<12} {'n':>2}  {'steps med':>10}  {'tokens med':>12}  "
          f"{'failed calls':>12}  {'late':>8}   by tool")
    stats: dict[str, dict] = {}
    for arm, rows in sorted(per_arm.items()):
        steps = [r["steps"] for r in rows]
        tokens = [r["tokens"] for r in rows]
        by_tool: dict[str, int] = {}
        for r in rows:
            for name, n in r["failed_calls_by_tool"].items():
                by_tool[name] = by_tool.get(name, 0) + n
        failed = sum(by_tool.values())
        late = sum(len(r["deadline_overruns"]) for r in rows)
        stats[arm] = {"n": len(rows), "steps": steps, "tokens": tokens,
                      "failed_calls": failed, "failed_calls_by_tool": by_tool,
                      "tool_error_events": sum(r["tool_errors"] for r in rows),
                      "deadline_overruns": late,
                      "stop_reasons": dict(Counter(row["stop_reason"] for row in rows))}
        # The per-tool split is printed because "7 failures" and "7 failures, all
        # one wrong column" are different findings, and a cut by tool changes the
        # p-value. A report must say which cut it means.
        detail = ", ".join(f"{k}={v}" for k, v in sorted(by_tool.items())) or "-"
        print(f"{arm:<12} {len(rows):>2}  {statistics.median(steps):>10}  "
              f"{statistics.median(tokens):>12,}  {failed:>12}  {late:>8}   {detail}")

    paired = _paired_summary(per_arm)
    tests = paired.get("tests", {})
    if paired:
        left_name, right_name = paired["arms"]
        print(f"\ncompleted pairs: {paired['n']} across {paired['n_tasks']} tasks; "
              f"excluded: {len(paired['excluded'])}")
        print(f"paired sign-flip tests ({left_name} minus {right_name}, "
              "two-sided, exact; task means across repeats):")
        for label, result in tests.items():
            if result["p"] is None:
                print(f"  {label:<14} no completed pair available")
                continue
            print(f"  {label:<14} {result['left_total']:>12,} -> {result['right_total']:>12,}"
                  f"   mean diff {result['mean_diff']:>12,.2f}   p = {result['p']:.6f}")
        for name, result in paired["failed_calls_by_tool"].items():
            print(f"  errors/{name}: {result['left_total']} -> {result['right_total']}, "
                  f"p = {result['p']:.6f}")
        for exclusion in paired["exclusions"]:
            print(f"  excluded {exclusion['pair']}: {exclusion['reason']}")
        print("\n  The primary metric must be chosen before running. Other metrics "
              "are exploratory; completed tool plans are not verified GIS outcomes.")
    return {"schema_version": 2,
            "analysis": {"method": "paired_sign_flip", "alternative": "two_sided",
                         "population": "completed_pairs", "inference_unit": "task_mean"},
            "arms": stats, "tests": tests,
            "paired": paired,
            "runs": [m for rows in per_arm.values() for m in rows]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-ids", default=None, help="comma-separated; required unless --analyze-only")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--arm", action="append", default=None, metavar="SETTING=VALUE",
                    help="profile setting and value, e.g. schema_hints=on or "
                         "evidence_schema_preview=head. Repeat for each arm.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-derive the summary from artifacts already on disk")
    args = ap.parse_args()

    if not args.analyze_only:
        if not args.task_ids or not args.arm:
            print("--task-ids and at least one --arm are required unless --analyze-only",
                  file=sys.stderr)
            return 2
        for spec in args.arm:
            name, _, value = spec.partition("=")
            if not name or not value:
                print(f"bad --arm {spec!r}; expected SETTING=VALUE. "
                      "Boolean settings use on/off; enum settings use their explicit value.",
                      file=sys.stderr)
                return 2
            if name not in Profile.__dataclass_fields__:
                print(f"Profile has no setting {name!r}", file=sys.stderr)
                return 2

            if name == "evidence_schema_preview":
                if value not in ("head", "head_tail"):
                    print(f"bad --arm {spec!r}; evidence_schema_preview must be head or head_tail",
                          file=sys.stderr)
                    return 2
            elif value not in ("on", "off"):
                print(f"bad --arm {spec!r}; boolean settings must be on or off",
                      file=sys.stderr)
                return 2

            arm_dir = args.out_dir / value
            arm_dir.mkdir(parents=True, exist_ok=True)
            if name == "evidence_schema_preview":
                flags = [
                    "--evidence-dir", str(arm_dir / "_evidence"),
                    "--evidence-schema-preview", value,
                ]
            elif name == "evidence_dir":
                # Not a boolean flag: a directory *is* the switch, because
                # "addressable" means the path still resolves afterwards.
                flags = (["--evidence-dir", str(arm_dir / "_evidence")]
                         if value == "on" else [])
            else:
                flags = ["--" + name.replace("_", "-")] if value == "on" else []
            for i in range(1, args.repeats + 1):
                out = arm_dir / f"run_{i}.jsonl"
                if out.exists():
                    print(f"  {out.name} exists, skipping")
                    continue
                cmd = [sys.executable, "scripts/run_geobench_tasks.py",
                       "--provider", args.provider, "--model", args.model,
                       "--task-ids", args.task_ids, "--out", str(out), *flags]
                print(f"  {name}={value} run {i}/{args.repeats}", flush=True)
                try:
                    subprocess.run(cmd, cwd=ROOT, timeout=RUN_TIMEOUT, check=False,
                                   capture_output=True, text=True)
                except subprocess.TimeoutExpired:
                    print(f"    TIMED OUT after {RUN_TIMEOUT}s -- no artifact")

    summary = analyze(args.out_dir)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
