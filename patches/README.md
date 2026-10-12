# 上游兼容补丁

这里放的是**对上游仓库的最小修改**。它们不是本项目的原创点，是让评测能跑起来的前置修复——
在报告和简历里必须分开标注；项目 README、`README.md` 和 `README.md` 记录了本仓库内的口径。

工作副本在 `work/GeoBenchX`（`main` 分支，基线 `0edf610`）。

## 为什么要单独存成补丁

`work/GeoBenchX` 是 clone 出来的第三方仓库，不是我们的代码。改动直接落在它的工作区里，
一旦被 `git checkout -- .` / 重新 clone 就会静默消失——**而症状是模型分数变低，不是报错**。
把 diff 存进本仓库 + 配一条回归测试，丢失就变成红色失败而不是悄悄变差。

## 应用方式

```bash
cd work/GeoBenchX
git apply ../CoreCoder/patches/geobenchx-0001-drop-colorbar-fmt.patch
git apply ../CoreCoder/patches/geobenchx-0002-spatial-select-distinct-features.patch
```

两个补丁**互不重叠**，可单独应用；runner 的启动前检查会逐条验证已应用状态。

## 清单

| 补丁 | 目标 | 症状 | 回归测试 |
|---|---|---|---|
| `geobenchx-0002-spatial-select-distinct-features.patch` | `geobenchx/tools.py` `select_features_by_spatial_relationship` | 选择结果返回**关联行**而非**独立源要素**（7,715 行 vs 45 国）；污染计数/均值，并增加重复绘图。临时行位置字段同时避开两侧输入字段，防止 join 后缀导致误报无命中 | `tests/strata/test_geobench_spatial_select.py`（13 条） |
| `geobenchx-0001-drop-colorbar-fmt.patch` | `geobenchx/tools.py:1061` `make_choropleth_map` | `Colorbar.__init__() got an unexpected keyword argument 'fmt'`——**首次 baseline 里 25 次，7/9 条任务**，调用永久失败且模型会反复重试 | `tests/strata/test_geobench_profile.py::test_the_reference_choropleth_task_runs_without_a_tool_error` |

### 0002 的判定依据

`select_features_by_spatial_relationship` 把 `gpd.sjoin` 的输出直接写进 state，
而 sjoin **每个 (左要素 × 右要素) 命中产出一行**。含两个地震的国家因此出现两次，
左表所有列被复制。下游无从分辨：**计数数的是「对」，均值被「命中几次」加权**。

| 探针（无绘图、无 API） | 修复前 | 修复后 |
|---|---|---|
| 1 国含 2 个事件点 | Selected 2 features | **1** |
| 两国 GDP 100/300，命中 2/1 个事件 → 均值 | **166.67**（错） | **200.0** |
| 真实数据：国家 contains 9,176 个地震 | **7,715 行** | **45 国**（`NAME_EN` 唯一） |

**附带成本**：绘图。同一段绘图调用，45 国 **0.19s**，7,715 个重复多边形 **74.99s** —— **385 倍**。
选择本身只要 **0.018s**，完整 selector 含绘图 **0.31s**。**所以重复要素不只是算错数，它让绘图慢了近三个数量级**——
而重复那张图**反而更小**（152,323 B vs 182,403 B），浪费在产物里完全看不出来。

> ⚠️ 这**不足以单独解释**观察到的 1000s 任务墙钟 —— 完整归因仍需在真实 run 上分段计时。
> 也不要把「改用空间索引」写成原创加速：原实现**已经在用** `sjoin`。

**修法**：按**源行位置**收集命中（临时列，不依赖索引），再按输入顺序取回原行。
**不按索引、也不按几何去重** —— 筛选后的帧保留原标签，两个不同要素可以共用一个索引，
也可以共用几何；按它们去重会**静默删掉真实要素**。修复在写入 state、计数与绘图**之前**生效。

### 0001 的判定依据

三条都是一手实测，不是推测：

1. `Colorbar.__init__` 的签名里**没有** `fmt`、**有** `format`（matplotlib 3.11.2，`inspect.signature`）。
2. geopandas 的 `plot(legend_kwds=...)` 把字典原样转发给 `Colorbar`，所以这是直接 TypeError。
3. 该函数在 `legend_kwds` 之后**三行**就自己设了 formatter（`cbar.xaxis.set_major_formatter`），
   而那正是处理 `>1e6` 缩放的地方——**所以删掉 `fmt` 是等价改动，不是行为改变**。

最小的独立复现（不需要 GeoBenchX 数据集）：

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, geopandas as gpd
from shapely.geometry import Point
g = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(0, 0), Point(1, 1)])
g.plot(column="v", legend=True, legend_kwds={"fmt": "{:.1f}", "label": "x"})
# TypeError: Colorbar.__init__() got an unexpected keyword argument 'fmt'
```

> 0001 是可以单独提给上游的一行兼容修复；0002 是独立要素选择的正确性修复。只有对应修复都已纳入所使用的上游版本、回归通过并同步 runner/provenance 后，才考虑移除补丁。

### 计时口径

数字来自 `scripts/measure_spatial_select.py`，原始结果存在
`evals/spatial-select-timing/timing.json`（含 Python/GeoPandas/Matplotlib 版本、后端 `Agg`、
以及本轮 `tools.py` 的 SHA256）。两次绘图走的是**同一段代码**——从
`select_features_by_spatial_relationship` 里逐句抄出的 plot 块；修复前的帧在脚本内用旧代码
同样的 `sjoin` 调用**重建**，不是留存的旧产物。

它仍然只是**本地单段计时**：不是 agent 端到端，不是稳定性能指标，也不是那个 1000s 墙钟的完整归因
（脚本不跑模型）。写简历时只能说「同一绘图调用下，重复要素使该阶段慢约两个半数量级」，
不能写成整体加速比。
