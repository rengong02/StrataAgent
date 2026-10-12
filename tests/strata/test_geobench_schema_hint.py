"""A not-found error should name what was available.

GeoBenchX answers a bad column with `Column 'X' not found in DataFrame` and
nothing else, so the model's only move is to guess one name per call. On the
task that motivated this it tried nine names on a 27-column frame and reached
`NAME_EN` ninth -- a column that frame has.

The hint is bounded and factual: it does not correct the model, it stops it
choosing blind.
"""

from __future__ import annotations

import json

import pytest

from strata.geobench.tools import (
    MAX_HINTED_COLUMNS,
    _with_schema_hint,
    geobenchx_root,
)

ROOT = geobenchx_root()
requires_geobenchx = pytest.mark.skipif(
    not (ROOT / "benchmark_set" / "tasks_and_reference_solutions.json").exists(),
    reason="GeoBenchX checkout not present",
)


class _Frame:
    def __init__(self, columns):
        self.columns = list(columns)


def _state(**frames) -> dict:
    return {"data_store": {name: _Frame(cols) for name, cols in frames.items()}}


COLUMN_ERROR = "Error: Column 'NAME' not found in DataFrame"


# --- the pure part ------------------------------------------------------------


def test_a_bad_column_gets_the_columns_that_exist():
    state = _state(countries=["OBJECTID", "NAME_EN", "CONTINENT", "geometry"])
    out = _with_schema_hint(COLUMN_ERROR, {"dataframe_name": "countries"}, state)

    assert out.startswith(COLUMN_ERROR), "the original error must stay readable"
    assert "NAME_EN" in out and "CONTINENT" in out


def test_only_the_frames_the_call_named_are_listed():
    state = _state(countries=["NAME_EN"], rainfall=["station", "mm"])
    out = _with_schema_hint(COLUMN_ERROR, {"dataframe_name": "countries"}, state)

    assert "NAME_EN" in out
    assert "station" not in out, "an unrelated frame's columns are noise"


def test_the_frame_the_call_is_about_to_create_is_not_mistaken_for_input():
    """`output_dataframe_name` may still be in the store from an earlier call."""
    state = _state(countries=["NAME_EN"], leftovers=["should_not_appear"])
    out = _with_schema_hint(
        COLUMN_ERROR,
        {"dataframe_name": "countries", "output_dataframe_name": "leftovers"},
        state,
    )

    assert "should_not_appear" not in out


def test_a_column_argument_is_not_read_as_a_frame_name():
    """`column_name` holds a column, and a column can share a frame's name."""
    state = _state(countries=["NAME_EN"], NAME=["collision"])
    out = _with_schema_hint(
        COLUMN_ERROR, {"dataframe_name": "countries", "column_name": "NAME"}, state
    )

    assert "collision" not in out


def test_a_wide_frame_is_truncated_rather_than_dumped():
    """The World Bank frames have 69 columns; the hint must stay bounded."""
    state = _state(wide=[f"c{i}" for i in range(MAX_HINTED_COLUMNS + 20)])
    out = _with_schema_hint(COLUMN_ERROR, {"dataframe_name": "wide"}, state)

    assert "+20 more" in out
    assert f"c{MAX_HINTED_COLUMNS}" not in out


def test_a_missing_frame_gets_the_frames_that_exist():
    state = _state(countries=[], gdp_per_capita=[])
    out = _with_schema_hint(
        "Error: DataFrame 'countrie' not found in data store", {}, state
    )

    assert "countries" in out and "gdp_per_capita" in out


def test_the_unnamed_frame_error_also_gets_the_list():
    """`calculate_columns` says "One or both DataFrames not found" and names neither."""
    state = _state(countries=[], quakes=[])
    out = _with_schema_hint(
        "Error: One or both DataFrames not found in data store", {}, state
    )

    assert "countries" in out and "quakes" in out


@pytest.mark.parametrize("out", [
    "Generated map visualizing POP_EST with Population legend",
    "Error: something else entirely went wrong",
    "",
])
def test_other_results_are_left_alone(out):
    state = _state(countries=["NAME_EN"])
    assert _with_schema_hint(out, {"dataframe_name": "countries"}, state) == out


def test_an_empty_store_changes_nothing():
    assert _with_schema_hint(COLUMN_ERROR, {"dataframe_name": "x"}, {"data_store": {}}) == COLUMN_ERROR
    assert _with_schema_hint(COLUMN_ERROR, {"dataframe_name": "x"}, {}) == COLUMN_ERROR


# --- through a real tool ------------------------------------------------------


@requires_geobenchx
def test_the_hint_reaches_the_model_and_the_record():
    """What the model saw is what the recorder kept.

    A hint the model received but the record does not show would make the next
    run's numbers unexplainable.
    """
    from strata.geobench import build_geobench_tools
    from strata.geobench.profile import DEFAULT_TOOLS

    seen: list[tuple[str, str]] = []
    # Explicit: the hint is off by default because the ablation found no effect,
    # so a test of what it does has to ask for it.
    by = {t.name: t for t in build_geobench_tools(
        only=DEFAULT_TOOLS, on_result=lambda n, r: seen.append((n, r)), schema_hints=True)}

    by["load_geodata"].execute(geodataset="Countries", output_geodataframe_name="countries")
    out = by["get_unique_values"].execute(dataframe_name="countries", column="ADMIN")

    assert out.startswith("Error: Column 'ADMIN' not found")
    assert "NAME_EN" in out, "the frame has NAME_EN; guessing should not be necessary"
    assert seen[-1][1] == out, "the recorded result diverged from what the model got"


@requires_geobenchx
def test_a_valid_call_is_not_decorated():
    """The hint must not attach itself to working calls."""
    from strata.geobench import build_geobench_tools
    from strata.geobench.profile import DEFAULT_TOOLS

    by = {t.name: t for t in build_geobench_tools(only=DEFAULT_TOOLS)}
    by["load_geodata"].execute(geodataset="Countries", output_geodataframe_name="countries")
    out = by["get_unique_values"].execute(dataframe_name="countries", column="NAME_EN")

    assert not out.startswith("Error")
    assert "Columns actually available" not in out
    assert json.dumps(out)  # a plain string result, nothing appended


@requires_geobenchx
def test_the_hint_can_be_switched_off_for_an_ablation():
    """One arm of a comparison is a sample, not a measurement.

    P1's first claim rests on running the same task with hints on and off, so
    the off arm has to be the untouched upstream error -- not "the same error
    with the hint trimmed", which would not be a control.
    """
    from corecoder.demo import ScriptedLLM
    from corecoder.llm import LLMResponse, ToolCall
    from strata.geobench import build_geobench_tools
    from strata.geobench.profile import DEFAULT_TOOLS, Profile, run_task

    off = Profile(schema_hints=False, tools=DEFAULT_TOOLS)
    assert off.schema_hints is False

    script = [
        LLMResponse(tool_calls=[ToolCall(id="c1", name="load_geodata", arguments={
            "geodataset": "Countries", "output_geodataframe_name": "countries"})]),
        LLMResponse(tool_calls=[ToolCall(id="c2", name="get_unique_values", arguments={
            "dataframe_name": "countries", "column": "ADMIN"})]),
        LLMResponse(content="done"),
    ]
    traj = run_task("dummy", ScriptedLLM(script), profile=off, task_id="control")

    assert traj.tool_errors, "the control arm should still fail"
    assert traj.tool_errors[0][1].strip() == "Error: Column 'ADMIN' not found in DataFrame", (
        "the control arm's error must be upstream's, unmodified"
    )

    by = {t.name: t for t in build_geobench_tools(only=("load_geodata",))}
    assert by  # the hint-on path is exercised by the tests above
