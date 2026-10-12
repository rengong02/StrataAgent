from pathlib import Path

from scripts.run_geobench_tasks import build_parser, make_row
from strata.geobench.trajectory import Trajectory


def test_runner_exposes_a_persistent_evidence_directory(tmp_path: Path):
    args = build_parser().parse_args([
        "--provider", "deepseek",
        "--model", "deepseek-flash",
        "--evidence-dir", str(tmp_path / "evidence"),
    ])

    assert args.evidence_dir == tmp_path / "evidence"


def test_runner_exposes_the_evidence_schema_preview_mode(tmp_path: Path):
    args = build_parser().parse_args([
        "--provider", "deepseek",
        "--model", "deepseek-flash",
        "--evidence-dir", str(tmp_path / "evidence"),
        "--evidence-schema-preview", "head",
    ])

    assert args.evidence_schema_preview == "head"


def test_runner_row_keeps_evidence_references_outside_scored_steps():
    trajectory = Trajectory(
        task_id="task-1",
        evidence_refs=[{"path": "raw.txt", "sha256": "abc", "size": 3, "kind": "text"}],
    )
    task = {
        "task_ID": "task-1",
        "task_text": "demo",
        "task_labels": [],
        "reference_solutions": [{"steps": []}],
    }

    row = make_row(task, trajectory, "deepseek:deepseek-flash", set(), {})

    assert row["steps"] == []
    assert row["evidence_refs"] == trajectory.evidence_refs


def test_runner_row_preserves_evidence_mode_in_provenance():
    task = {
        "task_ID": "task-1",
        "task_text": "demo",
        "task_labels": [],
        "reference_solutions": [{"steps": []}],
    }
    row = make_row(
        task,
        Trajectory(task_id="task-1"),
        "deepseek:deepseek-flash",
        set(),
        {"evidence_dir": "/tmp/evidence", "evidence_schema_preview": "head"},
    )

    assert row["provenance"]["evidence_schema_preview"] == "head"
