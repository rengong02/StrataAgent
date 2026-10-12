#!/usr/bin/env python
"""Where a spatial selection's wall time goes, measured offline.

No API, no key, no model -- local tool calls on the real datasets only.

Why this exists: one full task ran past its 600s deadline and took over 1000s of
wall time, while the selection itself measures in milliseconds. The plotting
path is the obvious suspect, and the pre-fix selector made it far worse than it
looks: it wrote `gpd.sjoin`'s output into the state, and `sjoin` emits one row
per (feature, reference) **pair**, so the 45 countries matching the earthquake
input arrived as 7,715 rows -- and every one of them was drawn.

This times each stage separately. It times the tool's own plotting sequence
twice: once on the 45 rows the fixed selector produces, once on the 7,715 the
old one did. The pre-fix frame is **rebuilt here** with the same `sjoin` call
the old code made. It is a reconstruction of the old selector's output, not a
saved artifact, and it is plotted with the same calls the tool makes rather than
by reverting the fix.

What this is not: an agent speed benchmark, a stable performance number, or a
full attribution of that 1000s task. The model is not involved, so nothing here
bounds how long a *task* takes -- only how long these local stages do.

Each stage runs under SIGALRM, so a slow plot reports a timeout instead of
holding the terminal.

    python scripts/measure_spatial_select.py
    python scripts/measure_spatial_select.py --out evals/spatial_select_timing/timing.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Before anything imports pyplot: a headless backend, so this never opens a
# window and never hits the macOS GUI-thread crash the harness already fixed.
os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt

from strata.geobench.tools import benchmark_state, build_geobench_tools

COUNTRIES = "Countries"
EARTHQUAKES = "Earthquakes occurences and magnitude March 15- February 14 2025"
PREDICATES = ["contains"]

DEFAULT_STAGE_LIMIT = 180.0


class StageTimeout(RuntimeError):
    pass


@contextmanager
def bounded(seconds: float):
    """Abort the stage rather than let it run away.

    `signal.setitimer` needs the main thread; this script is single-threaded by
    design, which is also why the number it produces is comparable at all.
    """
    def _fire(signum, frame):
        raise StageTimeout(f"stage exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def timed(label: str, fn, limit: float) -> dict:
    start = time.perf_counter()
    try:
        with bounded(limit):
            note = fn()
        return {"stage": label, "seconds": round(time.perf_counter() - start, 4), "note": note}
    except StageTimeout as exc:
        return {
            "stage": label,
            "seconds": round(time.perf_counter() - start, 4),
            "note": str(exc),
            "outcome": "timeout",
        }


def plot_like_the_tool(reference_gdf, selected, predicates: list[str]) -> dict:
    """The plotting block `select_features_by_spatial_relationship` runs.

    Copied call for call from `geobenchx/tools.py` -- grey reference, red
    selection, legend, title, PNG into a buffer at dpi=150 -- so timing it on a
    different frame measures the frame, not a different drawing.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    reference_gdf.plot(ax=ax, alpha=0.5, color="grey", label="Reference")
    selected.plot(ax=ax, color="red", label="Selected")
    ax.legend()
    plt.title(f'Features matching predicates: {", ".join(predicates)}')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    size = buf.tell()
    plt.close(fig)
    return {"png_bytes": size, "plotted_rows": len(selected)}


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def provenance() -> dict:
    geobenchx = ROOT.parent / "GeoBenchX"
    return {
        "python": platform.python_version(),
        "geopandas": gpd.__version__,
        "matplotlib": matplotlib.__version__,
        "matplotlib_backend": matplotlib.get_backend(),
        "geobenchx_tools_sha256": sha256_of(geobenchx / "geobenchx" / "tools.py"),
        "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="write the raw result as JSON here as well as to stdout")
    parser.add_argument("--stage-limit", type=float, default=DEFAULT_STAGE_LIMIT,
                        help="abort a stage after this many seconds (default: 180)")
    parser.add_argument("--skip-duplicate-plot", action="store_true",
                        help="skip the reconstructed 7,715-row plot, the slowest stage")
    args = parser.parse_args()

    state = benchmark_state()
    tools = {t.name: t for t in build_geobench_tools(
        state=state, only=("load_geodata", "select_features_by_spatial_relationship"))}
    load = tools["load_geodata"]
    select = tools["select_features_by_spatial_relationship"]

    stages: list[dict] = []

    stages.append(timed("load_countries", lambda: load.execute(
        geodataset=COUNTRIES, output_geodataframe_name="countries"), args.stage_limit))
    stages.append(timed("load_earthquakes", lambda: load.execute(
        geodataset=EARTHQUAKES, output_geodataframe_name="earthquakes"), args.stage_limit))

    countries = state["data_store"]["countries"]
    earthquakes = state["data_store"]["earthquakes"]

    def select_only():
        message = select.execute(
            features_geodataframe_name="countries", reference_geodataframe_name="earthquakes",
            spatial_predicates=PREDICATES, output_geodataframe_name="hit", plot_result=False)
        return {"rows_selected": len(state["data_store"]["hit"]), "message": message}

    stages.append(timed("select_only", select_only, args.stage_limit))
    hit = state["data_store"]["hit"]

    stages.append(timed("plot_fixed_selection", lambda: plot_like_the_tool(
        earthquakes, hit, PREDICATES), args.stage_limit))

    # The tool's own end-to-end call with plotting on, for the same frame: this
    # is the number a task actually pays for this step.
    stages.append(timed("tool_end_to_end_with_plot", lambda: {
        "message": select.execute(
            features_geodataframe_name="countries", reference_geodataframe_name="earthquakes",
            spatial_predicates=PREDICATES, output_geodataframe_name="hit2", plot_result=True),
        "rows_selected": len(state["data_store"]["hit2"]),
    }, args.stage_limit))

    if not args.skip_duplicate_plot:
        def prefix_rows():
            # Exactly what the old selector stored: sjoin's pair rows.
            joined = gpd.sjoin(countries, earthquakes, predicate="contains", how="inner")
            return {"rows": len(joined), "distinct_countries": joined["NAME_EN"].nunique()}

        stages.append(timed("rebuild_prefix_frame", prefix_rows, args.stage_limit))
        joined = gpd.sjoin(countries, earthquakes, predicate="contains", how="inner")
        stages.append(timed("plot_prefix_frame", lambda: plot_like_the_tool(
            earthquakes, joined, PREDICATES), args.stage_limit))

    result = {
        "what": "spatial selection stage timings, real GeoBenchX inputs, no API",
        "caveat": ("local stage timings only; not an agent benchmark, not a stable "
                   "performance figure, and not a full attribution of the observed "
                   "1000s task wall time"),
        "stage_limit_seconds": args.stage_limit,
        "provenance": provenance(),
        "stages": stages,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
