#!/usr/bin/env python
"""How much of a tool result the evidence block replaces, measured offline.

No API, no key, no model: the same calls are run twice against real GeoBenchX
datasets, once with the control (raw text) and once with evidence on, and the
two lengths are compared.

This exists because the first version of the block was **not** smaller. Built
with an unconditional head of the original, it turned `load_geodata` results
from ~1,800 characters into ~4,700 -- two and a half times worse -- for results
that are already short. Measuring it is what produced the rule now in the code:
file the artifact unconditionally, hand over whichever text is shorter.

    python scripts/measure_evidence.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata.geobench.evidence import EvidenceStore
from strata.geobench.tools import build_geobench_tools

#: Real calls, chosen to cover the shapes that exist: a wide World Bank frame, a
#: narrow shapefile, an image, and -- the one that matters -- a column with
#: hundreds of thousands of distinct values.
CALLS: list[tuple[str, dict]] = [
    ("load_data", {"dataset": "Forest area (sq. km)", "output_dataframe_name": "f"}),
    ("load_data", {"dataset": "GDP per capita (current US$)", "output_dataframe_name": "g"}),
    ("get_unique_values", {"dataframe_name": "f", "column": "Country Name"}),
    ("load_geodata", {"geodataset": "Countries", "output_geodataframe_name": "c"}),
    ("get_unique_values", {"dataframe_name": "c", "column": "NAME_EN"}),
    ("load_geodata", {"geodataset": "Power stations in Africa", "output_geodataframe_name": "p"}),
    ("load_geodata", {"geodataset": "Earthquakes occurences and magnitude March 15- February 14 2025",
                      "output_geodataframe_name": "q"}),
    ("get_unique_values", {"dataframe_name": "q", "column": "time"}),
    ("get_unique_values", {"dataframe_name": "q", "column": "place"}),
    ("calculate_column_statistics", {"dataframe_name": "f", "column_name": "2021",
                                     "output_variable_name": "s", "include_quantiles": True}),
]


def _state() -> dict:
    return {"data_store": {}, "image_store": [], "html_store": [], "visualize": False}


def main() -> int:
    control = build_geobench_tools(state=_state())
    treatment = build_geobench_tools(
        state=_state(), evidence=EvidenceStore(Path(tempfile.mkdtemp()) / "evidence")
    )
    by_name = {"control": {t.name: t for t in control},
               "evidence": {t.name: t for t in treatment}}

    print(f"{'call':<48}{'control':>10}{'evidence':>11}{'kept':>8}")
    print("-" * 77)

    totals = {"control": 0, "evidence": 0}
    worst = 0.0
    for name, args in CALLS:
        lengths = {}
        for arm, tools in by_name.items():
            try:
                lengths[arm] = len(tools[name].execute(**args))
            except Exception as e:  # noqa: BLE001
                print(f"{name:<48}  {arm} failed: {str(e)[:28]}")
                lengths = {}
                break
        if not lengths:
            continue
        label = f"{name}({str(next(iter(args.values())))[:30]})"
        kept = lengths["evidence"] / max(lengths["control"], 1)
        worst = max(worst, kept)
        totals["control"] += lengths["control"]
        totals["evidence"] += lengths["evidence"]
        print(f"{label:<48}{lengths['control']:>10,}{lengths['evidence']:>11,}{kept:>7.1%}")

    print("-" * 77)
    print(f"{'TOTAL':<48}{totals['control']:>10,}{totals['evidence']:>11,}"
          f"{totals['evidence'] / max(totals['control'], 1):>7.1%}")
    print()
    print(f"  worst case (largest fraction kept): {worst:.1%}")
    print("  A result is never made larger: the block is only used when it is")
    print("  shorter than the text it would replace. Where it engages, the size")
    print("  it saves is the point; where it does not, the run is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
