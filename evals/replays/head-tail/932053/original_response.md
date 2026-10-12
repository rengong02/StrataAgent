# 原始模型回答（未修改）

> 来源：`evals/ablation/p2c_headtail_vs_raw/head_tail/run_1.jsonl`，task `TASK_250309_135125_932053`。
> 这是**当时模型写下的原文**，逐字保留，没有做任何更正。

---

## Answer

**793 power stations** in Africa are located in countries with significant forest loss.

### Method
- **"Significant forest loss"** defined as a net decrease in forest area of **>5 percentage points** (share of land area) between **1990 and 2021**, computed from the World Bank *Forest area (% of land area)* dataset (`1990 − 2021`).
- Merged country-level forest loss onto the 3,034 *Power stations in Africa* features, then counted stations in qualifying countries (handling World Bank vs. common naming differences: e.g., *Congo, Dem. Rep.* → "Congo (Kinshasa)", *Cote d'Ivoire* → "Côte d'Ivoire", *Sao Tome and Principe* → "São Tomé and Príncipe").

### Breakdown
**20 African countries** exceeded the threshold:

| Group | Countries | Stations |
|---|---|---|
| Exact name match | Mozambique, Nigeria, Ghana, Angola, Botswana, Uganda, Tanzania, Senegal, Equatorial Guinea, Benin, Burkina Faso, Malawi, Liberia, Sierra Leone, Gambia, Guinea-Bissau, Comoros | **693** |
| Name mismatch (added manually) | Congo (Kinshasa) 70, Côte d'Ivoire 19, São Tomé and Príncipe 11 | **100** |
| **Total** | | **793** |

The largest contributors are Nigeria, Mozambique, DR Congo, and Ghana.

**Note:** "Significant" is a judgment call. I used a fixed **>5 pp** threshold (a robust, round cutoff near the top quartile of losses). Relaxing to the **top-quartile cutoff (>4.64 pp)** adds ~5 more countries, so the count would rise slightly; tightening it lowers the count. Let me know if you'd prefer a different threshold.
