"""The F1 pipeline: template -> real execution -> reference check -> sample.

These tests are about the seams, because the seams are where the data can go
wrong quietly:

* the executor must keep the tool's **actual** return, and keep enough of it for
  validation to compare against -- a truncation that loses the difference is
  worse than no truncation, because it looks like agreement;
* a failure must be recorded as a step, never raised away -- the recovery trace
  is the reason failure samples exist at all;
* the reference check must catch a template whose "wrong" guess is a real
  column, since that template would emit a failure that is not one.
"""

from __future__ import annotations

import json
from pathlib import Path

from strata.train.build import (
    build_f1_batch,
    digest_of_frame,
    digest_of_runtime,
    fresh_tools,
)
from strata.train.executor import (
    RENDERER_VERSION,
    execute_plan,
    render_visible,
)
from strata.train.reference import check_f1_gold, confusable_columns
from strata.train.sample import validate_sample
from strata.train.templates import F1_CASES, F1Case, build_f1_plan


def test_expanded_synthetic_sample_records_its_generator_provenance():
    from strata.train.synth import build_synth_f1_batch
    from strata.train.synth_families import FAMILY_VERSION, f1_family_cases

    samples, report = build_synth_f1_batch(f1_family_cases("year_role", count=1))

    assert report[0]["gold_ok"]
    generator = samples[0]["source_provenance"]["generator"]
    assert generator["module"] == "strata.train.synth_families"
    assert generator["version"] == FAMILY_VERSION


def test_runtime_digest_covers_the_declared_synthetic_generator():
    from strata.train.synth import SYNTH_GENERATOR_VERSION
    from strata.train.synth_families import FAMILY_VERSION

    legacy = digest_of_runtime(
        "synthetic_generator", generator_module="strata.train.synth",
    )
    expanded = digest_of_runtime(
        "synthetic_generator", generator_module="strata.train.synth_families",
    )

    assert SYNTH_GENERATOR_VERSION != FAMILY_VERSION
    assert legacy != expanded


def test_runtime_digest_covers_the_local_geobench_adapter(monkeypatch):
    from strata.train.build import digest_of_runtime

    before = digest_of_runtime("geobenchx_catalog")
    adapter = Path(__file__).resolve().parents[2] / "strata" / "geobench" / "tools.py"
    original_read_bytes = Path.read_bytes

    def changed_read_bytes(path):
        content = original_read_bytes(path)
        return content + b"\nchanged for digest test\n" if path == adapter else content

    monkeypatch.setattr(Path, "read_bytes", changed_read_bytes)

    assert adapter.exists()
    assert digest_of_runtime("geobenchx_catalog") != before




def test_expanded_synthetic_sample_replays_with_its_declared_runtime():
    from strata.train.replay import replay_all
    from strata.train.synth import build_synth_f1_batch
    from strata.train.synth_families import f1_family_cases

    samples, _ = build_synth_f1_batch(f1_family_cases("year_role", count=1))
    report = replay_all(samples)

    assert report["replayed_equal"] == 1, report["reports"]
    assert report["tools_changed"] == 0, report["reports"]
    assert report["stale"] == 0, report["reports"]


class _FakeTool:
    """Stands in for a GeoBench tool: returns what it is told, or raises."""

    def __init__(self, result=None, raises: Exception | None = None):
        self.result, self.raises = result, raises
        self.calls: list[dict] = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.result


# --- executor -----------------------------------------------------------------


def test_the_raw_observation_is_kept_in_full():
    """Validation compares against raw, so truncating it there would compare
    two views of the same shortening and always agree."""
    long_text = "x" * 5000
    execution = execute_plan([("load_data", {"dataset": "d"})],
                             {"load_data": _FakeTool(long_text)})
    step = execution.steps[0]
    assert len(step.raw) == 5000
    assert len(step.visible) < len(step.raw)


def test_truncation_is_versioned_and_says_what_it_dropped():
    visible = render_visible("y" * 3000)
    assert RENDERER_VERSION in visible
    assert "truncated" in visible


def test_short_results_are_not_annotated():
    """A marker on a result that was not truncated is noise the model has to
    learn to ignore."""
    assert render_visible("short") == "short"


def test_a_failing_call_is_a_step_not_an_exception():
    execution = execute_plan([("get_unique_values", {"column": "NAME"})],
                             {"get_unique_values": _FakeTool("Error: Column 'NAME' not found")})
    assert len(execution.steps) == 1
    assert execution.steps[0].failed


def test_a_tool_that_raises_is_recorded_rather_than_propagated():
    """Some upstream tools raise instead of returning an error string. Both must
    land in the trace, or the sample silently loses the hard failures."""
    execution = execute_plan([("x", {})],
                             {"x": _FakeTool(raises=ValueError("cannot unpack non-iterable"))})
    assert execution.steps[0].failed
    assert "ValueError" in execution.steps[0].raw


def test_steps_are_paired_into_messages_by_id():
    execution = execute_plan([("a", {}), ("b", {})],
                             {"a": _FakeTool("ok-a"), "b": _FakeTool("ok-b")})
    messages = [m for s in execution.steps for m in s.as_messages()]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_1", "call_2"]


# --- reference ----------------------------------------------------------------


def test_a_near_miss_column_is_flagged_even_though_it_exists():
    """The gap: `Country Code` is real, well-formed, and the wrong answer."""
    confusable = confusable_columns("Country Name", ["Country Name", "Country Code", "Year"])
    assert confusable == ["Country Code"]


def test_a_wrong_guess_that_is_a_real_column_is_a_problem():
    """This template would emit a 'failure' that succeeded -- teaching the model
    that a correct call is an error. It must not build."""
    case = F1Case(case_id="bad", loader="load_data", dataset="Forest area (% of land area)",
                  frame="forest", role="country_name", question="?",
                  correct_column="Country Name", wrong_guesses=["Indicator Name"])
    tools, state = fresh_tools()
    check = check_f1_gold(case, tools, state)
    assert not check.ok
    assert any("is a real column" in p for p in check.problems)


def test_a_missing_correct_column_is_a_problem():
    case = F1Case(case_id="bad2", loader="load_data", dataset="Forest area (% of land area)",
                  frame="forest", role="country_name", question="?",
                  correct_column="Not A Column", wrong_guesses=["nope"])
    tools, state = fresh_tools()
    assert not check_f1_gold(case, tools, state).ok


# --- the assembled sample -----------------------------------------------------


def test_the_real_batch_builds_and_validates():
    """Against the shipped cases and the real datasets, not a fixture."""
    samples, report = build_f1_batch(F1_CASES)
    assert len(samples) == len(F1_CASES)
    for row in report:
        assert row["gold_ok"], (row["sample_id"], row["gold_problems"])
        assert row["validate_ok"], (row["sample_id"], row["blockers"])


def test_every_case_carries_a_real_failure_and_a_recovery():
    """The point of the family. A batch of all-successful traces could not
    support any claim about error recovery."""
    samples, _ = build_f1_batch(F1_CASES)
    for sample in samples:
        failed = [s for s in sample["steps"] if s["failed"]]
        assert failed, f"{sample['sample_id']} has no failure trace"
        assert sample["steps"][-1]["failed"] is False, "the trace must end recovered"


def test_the_observations_are_the_tools_own_output():
    """A spot check that nothing wrote prose into an observation: the real
    column list is in there, which only the tool could have produced."""
    samples, _ = build_f1_batch([F1_CASES[1]])
    joined = json.dumps(samples[0]["steps"], ensure_ascii=False)
    assert "Country Name" in joined and "Country Code" in joined


def test_the_sample_records_the_versions_and_digests_it_was_built_with():
    """A sample whose tools have changed is not the sample that was validated,
    so the digests travel with it."""
    samples, _ = build_f1_batch([F1_CASES[0]])
    sample = samples[0]
    assert sample["tool_source_digest"].startswith("sha256:")
    assert sample["dataset_digest"].startswith("sha256:")
    assert sample["renderer_version"] and sample["comparator_version"]


def test_a_fresh_frame_digest_differs_between_different_frames():
    tools, state = fresh_tools()
    tools["load_data"].execute(dataset="GDP per capita (current US$)",
                               output_dataframe_name="g")
    tools["load_data"].execute(dataset="Forest area (% of land area)",
                               output_dataframe_name="f")
    store = state["data_store"]
    assert digest_of_frame(store["g"]) != digest_of_frame(store["f"])


def test_the_plan_has_one_loader_and_ends_on_the_right_column():
    case = F1_CASES[0]
    plan = build_f1_plan(case)
    assert plan[0][0] == "load_geodata"
    assert plan[-1][1]["column"] == case.correct_column
    assert all(name == "get_unique_values" for name, _ in plan[1:])


def test_a_case_without_wrong_guesses_has_no_fabricated_failure():
    """No observed failure means no failure trace -- inventing one would put a
    difficulty in the data that nothing actually exhibited."""
    case = F1Case(case_id="clean", loader="load_data", dataset="Forest area (% of land area)",
                  frame="forest", role="country_name", question="?",
                  correct_column="Country Name", wrong_guesses=[])
    tools, state = fresh_tools()
    sample = __import__("strata.train.build", fromlist=["build_f1_sample"]).build_f1_sample(
        case, tools, state)
    assert sample["steps"][-1]["failed"] is False
    assert validate_sample(sample).ok


# --- F2: the join family ------------------------------------------------------


def test_f2_gold_reproduces_the_measured_fill_rates():
    """The numbers this case was designed around, from the real data:

        Country Code <-> ISO_A3 -> 74.1% filled
        Country Name <-> ISO_A3 ->  0.0% filled

    If the reference does not reproduce them, the case is not testing what the
    design says it tests.
    """
    from strata.train.build import build_f2_batch
    from strata.train.templates import F2_CASES

    _, report = build_f2_batch([F2_CASES[0]])
    row = report[0]
    assert row["gold_ok"], row["gold_problems"]
    assert abs(row["correct_fill_rate"] - 0.741) < 0.01
    assert max(row["wrong_fill_rates"]) < 0.01


def test_f2_wrong_join_is_a_silent_failure_not_an_error():
    """The point of the family, and the reason it needs its own check.

    `merge_dataframes` is a left join on mismatched keys: it does not raise, and
    it returns the same 251x96 frame as the correct join. Nothing about the
    call's *shape* says it went wrong -- so the wrong attempt is not marked
    `failed`, and a checker looking at shape or row count would see no
    difference at all.
    """
    from strata.train.build import build_f2_batch, fresh_tools
    from strata.train.templates import F2_CASES

    case = F2_CASES[0]
    tools, state = fresh_tools()
    tools["load_data"].execute(dataset=case.data_dataset,
                               output_dataframe_name=case.data_frame)
    tools["load_geodata"].execute(geodataset=case.geo_dataset,
                                  output_geodataframe_name=case.geo_frame)
    bad = tools["merge_dataframes"].execute(
        dataframe_name=case.data_frame, geodataframe_name=case.geo_frame,
        statkey="Country Name", geokey="ISO_A3", output_dataframe_name="bad")
    tools["merge_dataframes"].execute(
        dataframe_name=case.data_frame, geodataframe_name=case.geo_frame,
        statkey=case.data_key, geokey=case.geo_key, output_dataframe_name="good")

    assert "Error" not in bad, "the wrong join reports no error at all"
    store = state["data_store"]
    assert store["bad"].shape == store["good"].shape, "shape cannot tell them apart"

    samples, _ = build_f2_batch([case])
    assert all(not s["failed"] for s in samples[0]["steps"]), "no step raises"


def test_f2_observation_carries_the_signal_the_model_must_read():
    """The recovery has to be *visible*, or the trace teaches nothing: the merge
    result lists per-column non-null counts, and the wrong join's are zero."""
    from strata.train.build import build_f2_batch
    from strata.train.templates import F2_CASES

    samples, _ = build_f2_batch([F2_CASES[0]])
    bad_obs = samples[0]["steps"][2]["observation_raw"]
    assert "non-null" in bad_obs
    assert " 0 non-null" in bad_obs, "the empty column should be visible as 0 non-null"


def test_every_built_sample_replays_to_itself():
    """V2 end to end: stored observations vs a fresh run of the same calls."""
    from strata.train.build import build_f1_batch, build_f2_batch
    from strata.train.replay import replay_all
    from strata.train.templates import F1_CASES, F2_CASES

    f1, _ = build_f1_batch(F1_CASES)
    f2, _ = build_f2_batch(F2_CASES)
    report = replay_all(f1 + f2)

    assert report["samples"] == len(f1) + len(f2)
    assert report["replayed_equal"] == report["samples"], report["reports"]
    assert report["stale"] == 0


def test_synthetic_samples_replay_with_their_declared_runtime():
    from strata.train.replay import replay_all
    from strata.train.synth import (
        build_synth_f1_batch,
        build_synth_f2_batch,
        f1_synth_cases,
        f2_synth_cases,
    )

    f1, _ = build_synth_f1_batch(f1_synth_cases(2))
    f2, _ = build_synth_f2_batch(f2_synth_cases(2))
    report = replay_all(f1 + f2)

    assert report["replayed_equal"] == 4, report["reports"]
    assert report["stale"] == 0


def test_a_sample_from_changed_tools_is_stale_only_if_the_change_shows():
    """`stale` is not `tools_changed`: a digest move that does not alter what the
    sample sees is not a reason to rebuild it."""
    from strata.train.build import build_f1_batch
    from strata.train.replay import ReplayReport
    from strata.train.templates import F1_CASES

    sample = build_f1_batch([F1_CASES[0]])[0][0]
    sample["tool_source_digest"] = "sha256:something-else"
    report = ReplayReport(sample_id="x", tool_digest_before=sample["tool_source_digest"],
                          tool_digest_now="sha256:current", steps_replayed=1,
                          comparison={"equal": True})
    assert report.tools_changed and not report.stale
