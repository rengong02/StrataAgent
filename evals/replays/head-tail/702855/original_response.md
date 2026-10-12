# 原始模型回答（未修改）

> 来源：`evals/ablation/p2c_headtail_vs_raw/head_tail/run_1.jsonl`，task `TASK_250309_135125_702855`。
> 这是**当时模型写下的原文**，逐字保留，没有做任何更正。

---

Here's the world map displaying net migration patterns:

**What I did:**
1. **Loaded** the "Net migration" statistical dataset (266 rows, yearly values 1960–2023) and the "Countries" geodataset (251 country polygons).
2. **Merged** them by country code (`Country Code` ↔ `ISO_A3`), matching 217 countries to their geometries.
3. **Visualized** the **2023** values (the most recent year) using a **divergent colormap** — appropriate since net migration has both positive values (net immigration) and negative values (net emigration).

**Reading the map:**
- The divergent color scale centers on zero, distinguishing countries gaining population through migration from those losing it.
- Countries in gray indicate missing/NaN data (countries where the merger found no match).

If you'd like, I can also:
- Map a different year (e.g., 2020, or a multi-year average)
- Normalize by population (net migration per 1,000 people) to make small and large countries more comparable
- Show the top gainers and losers as a ranked list
