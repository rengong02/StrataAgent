"""A spatial selection selects *features*, not join rows.

`select_features_by_spatial_relationship` wrote `gpd.sjoin`'s output into the
state, and `sjoin` emits one row per (feature, reference) **pair**. A country
containing two earthquakes therefore appeared twice, with every left column
repeated. Nothing downstream could tell: the count was of pairs, and an average
over the frame weighted the duplicated country twice.

Reproduced before the fix, on the two-country example below: three rows and a
mean of 166.67 where two countries averaging 200 was correct. On the real
GeoBenchX input it selected **7,715 rows for 45 distinct countries**.

These tests call the tool the way the adapter does, because that is the path a
benchmark run takes.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point, box

from strata.geobench.tools import build_geobench_tools, load_geobenchx_tools_module

TOOL = "select_features_by_spatial_relationship"


def _tools(state: dict):
    tools = build_geobench_tools(state=state, only=(TOOL, "calculate_column_statistics"))
    return {t.name: t for t in tools}


def _state(features, references) -> dict:
    return {"data_store": {"features": features, "references": references},
            "image_store": [], "html_store": [], "visualize": False}


def _two_countries(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {"name": ["country_a", "country_b"], "gdp": [100.0, 300.0]},
        geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2)], crs=crs,
    )


def _three_events(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {"event": ["first", "second", "third"]},
        geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(4, 1)], crs=crs,
    )


# --- the defect ---------------------------------------------------------------


def test_two_events_in_one_country_do_not_duplicate_the_country():
    """The headline case: a one-to-many hit must yield one row, not two."""
    state = _state(_two_countries(), _three_events())
    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    selected = state["data_store"]["selected"]
    assert len(selected) == 2, f"expected 2 distinct countries, got {len(selected)}"
    assert sorted(selected["name"]) == ["country_a", "country_b"]


def test_a_duplicated_country_does_not_skew_the_statistics():
    """Why it matters: the pair rows silently weight one country twice.

    Real figure from before the fix: mean 166.67 for GDPs of 100 and 300.
    """
    state = _state(_two_countries(), _three_events())
    tools = _tools(state)
    tools[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )
    tools["calculate_column_statistics"].execute(
        dataframe_name="selected", column_name="gdp", output_variable_name="stats",
    )

    assert state["data_store"]["stats"]["summary"]["mean"] == 200.0


def test_the_reported_count_is_of_features_not_of_pairs():
    state = _state(_two_countries(), _three_events())
    message = _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert "Selected 2 features" in message


# --- the shapes that make de-duplication non-obvious --------------------------


def test_two_predicates_are_still_an_or_over_features():
    """A feature matching both predicates must appear once, not twice."""
    state = _state(_two_countries(), _three_events())
    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains", "intersects"],
        output_geodataframe_name="selected", plot_result=False,
    )

    assert len(state["data_store"]["selected"]) == 2


def test_features_that_share_an_index_are_not_merged():
    """De-duplicating by index would silently delete a real feature.

    Two genuinely different countries can carry the same index -- a filtered
    frame keeps its original labels -- and they must both survive.
    """
    features = gpd.GeoDataFrame(
        {"name": ["country_a", "country_b"], "gdp": [100.0, 300.0]},
        geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2)], crs="EPSG:4326",
        index=[7, 7],
    )
    references = gpd.GeoDataFrame(
        {"event": ["first", "second", "third"]},
        geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(4, 1)], crs="EPSG:4326",
    )
    state = _state(features, references)

    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    selected = state["data_store"]["selected"]
    assert len(selected) == 2, "a shared index collapsed two distinct features"
    assert sorted(selected["gdp"]) == [100.0, 300.0]


def test_features_with_identical_geometry_are_not_merged():
    """Identical geometry is not proof of the same row -- only position is."""
    features = gpd.GeoDataFrame(
        {"name": ["a", "b"], "gdp": [100.0, 300.0]},
        geometry=[box(0, 0, 2, 2), box(0, 0, 2, 2)], crs="EPSG:4326",
    )
    references = gpd.GeoDataFrame({"event": ["x"]}, geometry=[Point(1, 1)], crs="EPSG:4326")
    state = _state(features, references)

    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert len(state["data_store"]["selected"]) == 2


def test_input_order_and_source_fields_are_kept():
    state = _state(_two_countries(), _three_events())
    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    selected = state["data_store"]["selected"]
    assert list(selected["name"]) == ["country_a", "country_b"], "input order changed"
    assert "gdp" in selected.columns, "a source field was dropped"


@pytest.mark.parametrize("source_has_position_column", [False, True])
def test_position_marker_avoids_columns_from_both_inputs(source_has_position_column):
    features = _two_countries()
    references = _three_events()
    reference_column = "__strata_source_position__"
    if source_has_position_column:
        features[reference_column] = [19, 23]
        reference_column += "_"
    references[reference_column] = [31, 37, 41]
    state = _state(features, references)

    message = _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert "Selected 2 features" in message
    assert_geodataframe_equal(state["data_store"]["selected"], features)


# --- the edges ----------------------------------------------------------------


def test_no_hits_is_still_an_explicit_no_match():
    far = gpd.GeoDataFrame({"name": ["x"], "gdp": [1.0]},
                           geometry=[box(50, 50, 51, 51)], crs="EPSG:4326")
    state = _state(far, _three_events())

    message = _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert "No features found" in message
    assert "selected" not in state["data_store"]


def test_the_inputs_are_not_modified():
    """The CRS contract reprojects the *output*; the stored inputs stay put."""
    state = _state(_two_countries(), _three_events())
    before = state["data_store"]["features"].copy()

    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert state["data_store"]["features"].crs == before.crs
    assert len(state["data_store"]["features"]) == len(before)


def test_a_crs_mismatch_is_reprojected_rather_than_failing():
    """Reference in a projected CRS, features in geographic: still two countries."""
    references = _three_events().to_crs("EPSG:3857")
    state = _state(_two_countries(), references)

    _tools(state)[TOOL].execute(
        features_geodataframe_name="features", reference_geodataframe_name="references",
        spatial_predicates=["contains"], output_geodataframe_name="selected",
        plot_result=False,
    )

    assert len(state["data_store"]["selected"]) == 2
    assert state["data_store"]["selected"].crs == references.crs


# --- the real data ------------------------------------------------------------


def test_the_real_input_yields_distinct_countries_not_join_rows():
    """Measured before the fix: 7,715 rows for 45 distinct countries."""
    module = load_geobenchx_tools_module()
    state = {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}

    tools = build_geobench_tools(state=state, only=("load_geodata", TOOL))
    by = {t.name: t for t in tools}
    by["load_geodata"].execute(
        geodataset="Earthquakes occurences and magnitude March 15- February 14 2025",
        output_geodataframe_name="earthquakes")
    by["load_geodata"].execute(geodataset="Countries", output_geodataframe_name="countries")

    message = by[TOOL].execute(
        features_geodataframe_name="countries", reference_geodataframe_name="earthquakes",
        spatial_predicates=["contains"], output_geodataframe_name="hit",
        plot_result=False)

    selected = state["data_store"]["hit"]
    assert len(selected) == 45, f"got {len(selected)} rows; the input has 45 matching countries"
    assert selected["NAME_EN"].is_unique, "a country was selected more than once"
    assert "Selected 45 features" in message
    assert module is not None
