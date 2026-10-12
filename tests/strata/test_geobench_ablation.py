from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_ablation import analyze


def _solver_row(task_id: str, evidence_dir: str | None = None) -> dict:
    return {
        "task_ID": task_id,
        "step_count": 2,
        "tokens": 10,
        "tool_errors": [],
        "stop_reason": "completed",
        "wall_seconds": 0.1,
        "provenance": {"evidence_dir": evidence_dir},
        "repeat_failures": {},
        "unexposed_tool_calls": [],
        "deadline_overruns": [],
        "evidence_stats": {},
    }


def test_analyze_ignores_judge_jsonl_and_keeps_both_solver_arms(tmp_path: Path):
    for arm, evidence_dir in (("off", None), ("on", "/tmp/evidence")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "run_1.jsonl").write_text(
            json.dumps(_solver_row("task-1", evidence_dir)) + "\n",
            encoding="utf-8",
        )
        (arm_dir / "judge.jsonl").write_text(
            json.dumps({"task_id": f"task-{arm}", "score": 2}) + "\n",
            encoding="utf-8",
        )

    summary = analyze(tmp_path)

    assert set(summary["arms"]) == {"off", "on"}
    assert summary["arms"]["off"]["n"] == 1
    assert summary["arms"]["on"]["n"] == 1
    assert set(summary["tests"]) == {"tokens", "steps", "failed calls"}
    assert summary["paired"]["n"] == 1
    assert set(summary["paired"]["tests"]) == {"tokens", "steps", "failed calls"}


def test_analyze_pairs_repeated_runs_by_task_and_run_number(tmp_path: Path):
    for arm, evidence_dir in (("head", "/tmp/head"), ("head_tail", "/tmp/head-tail")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        for repeat, tokens in ((1, 10), (2, 20)):
            row = _solver_row("task-1", evidence_dir)
            row["tokens"] = tokens if arm == "head" else tokens + 1
            (arm_dir / f"run_{repeat}.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

    summary = analyze(tmp_path)

    assert summary["paired"]["n"] == 2
    assert summary["paired"]["excluded"] == []
    assert summary["paired"]["pairs"] == ["task-1#run_1", "task-1#run_2"]
    assert summary["paired"]["tests"]["tokens"]["mean_diff"] == -1


def test_analyze_defaults_to_completed_paired_statistics(tmp_path: Path, capsys):
    counts = [3, 2, 1, 0, 1, 0, 0, 1]
    for arm in ("head", "head_tail"):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        rows = []
        for number, count in enumerate(counts):
            row = _solver_row(f"task-{number}", str(arm_dir / "_evidence"))
            if arm == "head":
                row["tool_errors"] = [["calculate_column_statistics", "Error: missing column"]] * count
                if number == 0:
                    row["tool_errors"][0] = ["make_choropleth_map", "Error: missing column"]
            rows.append(row)
        truncated = _solver_row("truncated-task", str(arm_dir / "_evidence"))
        truncated["stop_reason"] = "truncated:deadline"
        truncated["tokens"] = 1_000 if arm == "head" else 1
        rows.append(truncated)
        (arm_dir / "run_1.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    summary = analyze(tmp_path)

    assert summary["tests"]["failed calls"]["mean_diff"] == 1
    assert summary["tests"]["failed calls"]["p"] == pytest.approx(0.0625)
    assert summary["tests"]["tokens"]["mean_diff"] == 0
    assert summary["analysis"]["population"] == "completed_pairs"
    assert summary["paired"]["totals"]["head"]["failed_calls"] == 8
    assert summary["paired"]["totals"]["head_tail"]["failed_calls"] == 0
    by_tool = summary["paired"]["failed_calls_by_tool"]
    assert by_tool["calculate_column_statistics"]["left_total"] == 7
    assert by_tool["calculate_column_statistics"]["right_total"] == 0
    assert by_tool["calculate_column_statistics"]["p"] == pytest.approx(0.0625)
    assert summary["arms"]["head"]["stop_reasons"] == {"completed": 8, "truncated:deadline": 1}
    assert summary["paired"]["exclusions"][0]["reason"] == "not_completed"
    output = capsys.readouterr().out
    assert "completed pairs" in output
    assert "non-paired" not in output


def test_repeated_runs_do_not_inflate_the_number_of_independent_tasks(tmp_path: Path):
    for arm in ("head", "head_tail"):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        for repeat in range(1, 6):
            row = _solver_row("task-1", str(arm_dir / "_evidence"))
            if arm == "head":
                row["tool_errors"] = [["calculate_column_statistics", "Error: missing column"]]
            (arm_dir / f"run_{repeat}.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

    summary = analyze(tmp_path)

    assert summary["paired"]["n"] == 5
    assert summary["paired"]["n_tasks"] == 1
    assert summary["analysis"]["inference_unit"] == "task_mean"
    assert summary["tests"]["failed calls"]["p"] == 1


def test_analyze_explains_missing_and_duplicate_pairs(tmp_path: Path):
    for arm in ("head", "head_tail"):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        rows = [_solver_row("duplicate-task")]
        if arm == "head":
            rows.append(_solver_row("duplicate-task"))
            rows.append(_solver_row("missing-task"))
        (arm_dir / "run_1.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    summary = analyze(tmp_path)

    assert summary["paired"]["n"] == 0
    assert {row["reason"] for row in summary["paired"]["exclusions"]} == {
        "duplicate_rows", "missing_arm"
    }
    assert all(test["p"] is None for test in summary["tests"].values())
