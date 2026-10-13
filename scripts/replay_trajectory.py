#!/usr/bin/env python
"""Replay a saved solver trajectory's tool calls, offline, and keep what they make.

**This is a replay of a stored trajectory, not a new agent run.** No model is
called: the plan (the `steps` a solver already produced) is read back from an
artifact and each tool call is executed again against the current tool source.
So the steps are the old ones and only the tools are fresh — which is exactly
what makes the artifacts worth having, because it shows what those calls
actually produce on the data, without paying for another run and without the
model's stochasticity in the way.

Why it exists: a run's `steps` record the arguments but not what came back.
For a task whose whole point is a map, the trajectory text is not the
deliverable — the map is, and nothing in the p2c artifacts is openable. This
writes the real outputs (PNG, per-step row counts, missing-value counts) and a
step-by-step record next to them.

Two things it deliberately keeps apart:

* the **original model response** is copied verbatim to `original_response.md`
  and never edited — a corrected explanation must not overwrite history;
* the **replay record** says what the tools returned when re-run, which may
  differ from what the model saw.

The state is rebuilt in order, so a later step sees what earlier steps stored.

    python scripts/replay_trajectory.py \
        --task-id TASK_250309_135125_702855 \
        --from evals/ablation/p2c_headtail_vs_raw/head_tail/run_1.jsonl \
        --out-dir evals/replays/head-tail/702855
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Headless before pyplot is imported anywhere: a replay must not open a window.
os.environ.setdefault("MPLBACKEND", "Agg")

from corecoder.tool_errors import is_error_result
from strata.geobench.tools import benchmark_state, build_geobench_tools

#: A tool's return value is model-facing prose and can run to thousands of
#: characters (a frame description). The record keeps the head and flags it,
#: so the file stays readable without pretending the rest did not exist.
RESULT_HEAD = 400


def find_row(source: Path, task_id: str) -> dict:
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_ID") == task_id:
            return row
    raise SystemExit(f"{task_id} not found in {source}")


def frame_table(state: dict) -> dict[str, dict]:
    """Every stored frame, by name, with its shape and missing-value count.

    Missing values are counted because "793 stations" is a claim about rows
    that survived a join, and a join that silently drops rows looks identical
    to a filter that legitimately selected fewer.
    """
    table: dict[str, dict] = {}
    for name, value in (state.get("data_store") or {}).items():
        entry: dict[str, Any] = {"type": type(value).__name__}
        shape = getattr(value, "shape", None)
        if shape is not None:
            entry["rows"], entry["cols"] = int(shape[0]), int(shape[1])
        if hasattr(value, "isna"):
            try:
                entry["null_cells"] = int(value.isna().sum().sum())
            except (TypeError, ValueError):
                pass
        table[name] = entry
    return table


def new_keys(before: dict, after: dict) -> list[str]:
    return sorted(set(after) - set(before))


def png_size(path: Path) -> tuple[int, int] | None:
    """Width/height from the IHDR chunk, so "the artifact opens" is checked.

    Read here rather than via an image library: the point is to confirm the
    bytes are a well-formed PNG of a sane size, and that needs the header only.
    """
    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--from", dest="source", type=Path, required=True,
                    help="the JSONL artifact holding the trajectory to replay")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--start-at", type=int, default=0,
                    help="record from this step index; earlier steps still run to rebuild state")
    ap.add_argument("--answer", default=None,
                    help="what the replay's own numbers amount to, in the operator's "
                         "words. Written to its own section; it never touches "
                         "original_response.md, which stays verbatim.")
    ap.add_argument("--answer-file", type=Path, default=None,
                    help="same, read from a Markdown file (easier for long text)")
    args = ap.parse_args()

    if args.answer_file is not None:
        args.answer = args.answer_file.read_text(encoding="utf-8")

    row = find_row(args.source, args.task_id)
    planned_steps = row.get("steps") or []
    if args.start_at < 0 or args.start_at > len(planned_steps):
        ap.error(f"--start-at must be between 0 and {len(planned_steps)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.out_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    state = benchmark_state()
    tools = {t.name: t for t in build_geobench_tools(state=state)}

    steps: list[dict] = []
    executed_steps: list[dict] = []
    for index, step in enumerate(planned_steps):
        name = step.get("function_name")
        arguments = step.get("arguments") or {}
        tool = tools.get(name)

        before_tables = frame_table(state)
        images_before = len(state.get("image_store") or [])
        started = time.perf_counter()
        if tool is None:
            result, failed = f"tool {name!r} is not in the current profile", True
        else:
            try:
                result = tool.execute(**arguments)
                failed = is_error_result(result)
            except Exception as exc:  # noqa: BLE001 - a replay records, it does not raise
                result, failed = f"{type(exc).__name__}: {exc}", True
        elapsed = time.perf_counter() - started

        after_tables = frame_table(state)
        created = new_keys(before_tables, after_tables)
        images_after = len(state.get("image_store") or [])

        executed_steps.append({
            "index": index,
            "tool": name,
            "arguments": arguments,
            "ok": not failed,
            "seconds": round(elapsed, 4),
            "created": {k: after_tables[k] for k in created},
            "images_added": images_after - images_before,
            "result_head": str(result)[:RESULT_HEAD],
            "result_chars": len(str(result)),
            "result_truncated": len(str(result)) > RESULT_HEAD,
        })

        if index >= args.start_at:
            steps.append(executed_steps[-1])

    # Images last, so the record above can be written even if decoding fails.
    written: list[dict] = []
    for i, image in enumerate(state.get("image_store") or []):
        description = image.get("description") or f"image_{i}"
        path = artifacts_dir / f"{i:02d}_{description[:40].replace('/', '_').replace(' ', '_')}.png"
        try:
            path.write_bytes(base64.b64decode(image["base64"]))
        except Exception as exc:  # noqa: BLE001
            written.append({"path": str(path.relative_to(args.out_dir)), "error": str(exc)})
            continue
        size = png_size(path)
        written.append({
            "path": str(path.relative_to(args.out_dir)),
            "bytes": path.stat().st_size,
            "description": description,
            "png_size": list(size) if size else None,
            "opens": size is not None,
        })

    record = {
        "what": "offline replay of a saved solver trajectory -- NOT a new LLM run",
        "task_id": args.task_id,
        "task_text": row.get("task_text"),
        "plan_source": str(args.source),
        "plan_from_arm": (row.get("provenance") or {}).get("evidence_schema_preview") or "off",
        "steps_planned": len(planned_steps),
        "steps_replayed": len(executed_steps),
        "steps_recorded_from": args.start_at,
        "steps_recorded": len(steps),
        "steps_failed": sum(1 for s in executed_steps if not s["ok"]),
        "steps_failed_recorded": sum(1 for s in steps if not s["ok"]),
        "artifacts": written,
        "final_tables": frame_table(state),
        "steps": steps,
    }
    (args.out_dir / "replay.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    original = row.get("final_response") or ""
    (args.out_dir / "original_response.md").write_text(
        "# 原始模型回答（未修改）\n\n"
        f"> 来源：`{args.source}`，task `{args.task_id}`。\n"
        "> 这是**当时模型写下的原文**，逐字保留，没有做任何更正。\n\n"
        "---\n\n" + original + "\n", encoding="utf-8")

    lines = [
        f"# 轨迹重放：{args.task_id}",
        "",
        "> **这是保存轨迹的离线重放，不是重新运行 LLM Agent。** 计划（工具调用参数）来自已有产物，",
        "> 工具调用按当前版本重新执行，以下记录包含对应观测与产物。",
        "",
        f"- 任务原文：{row.get('task_text')}",
        f"- 计划来源：`{args.source}`（arm `{record['plan_from_arm']}`）",
        (f"- 计划步数 {record['steps_planned']}，执行 {record['steps_replayed']}，"
         f"记录从第 {record['steps_recorded_from']} 步开始，共记录 {record['steps_recorded']} 步，"
         f"失败 {record['steps_failed']}"),
        "",
        "## 逐步记录",
        "",
        "| # | 工具 | 参数 | 结果 | 新建对象（行×列，空值） |",
        "|---|---|---|---|---|",
    ]
    for s in steps:
        args_text = json.dumps(s["arguments"], ensure_ascii=False)
        if len(args_text) > 110:
            args_text = args_text[:110] + "…"
        args_text = args_text.replace("|", "\\|")
        created = "、".join(
            f"`{k}` {v.get('rows', '?')}×{v.get('cols', '?')}"
            + (f"/空 {v['null_cells']}" if "null_cells" in v else "")
            for k, v in s["created"].items()) or "—"
        mark = "✅" if s["ok"] else "❌"
        lines.append(f"| {s['index']} | `{s['tool']}` | `{args_text}` | {mark} {s['seconds']}s | {created} |")

    lines += ["", "## 产物", ""]
    if written:
        for w in written:
            if w.get("opens"):
                lines.append(f"- `{w['path']}` —— {w['bytes']:,} bytes，PNG {w['png_size'][0]}×{w['png_size'][1]}，**可打开**")
            else:
                lines.append(f"- `{w['path']}` —— ⚠️ 未能解码或不是有效 PNG：{w.get('error', 'header check failed')}")
    else:
        lines.append("- 本次重放没有产生图像产物。")

    if args.answer:
        lines += [
            "",
            "## 重放口径下的说明（操作者撰写，非模型输出）",
            "",
            args.answer,
            "",
        ]

    lines += [
        "",
        "## 原始模型回答",
        "",
        "见 `original_response.md`（逐字保留，含其中的错误披露，未更正）。",
        "",
    ]
    (args.out_dir / "replay.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"steps {record['steps_replayed']}  recorded_from {record['steps_recorded_from']}  "
          f"failed {record['steps_failed']}  "
          f"artifacts {len(written)}")
    for w in written:
        print(f"  {w['path']}  {w.get('bytes', 0):,} bytes  opens={w.get('opens')}")
    print(f"wrote {args.out_dir}/replay.md and replay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
