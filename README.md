# StrataAgent

面向地理空间分析任务的 Agent：读任务、调 GIS 工具，并留下别人能复核的执行记录。

运行时来自 [CoreCoder](https://github.com/he-yufeng/CoreCoder)（MIT）——一个极简 Agent harness。
本仓库在它之上加了 GIS 工具层、证据契约、工具失败恢复的训练管线，以及测量这些的评测。

## 为什么做这些

四个问题都来自 GeoBenchX 真实任务的实测，不是设想：

- **工具会用"没有报错"的方式返回错误结果。** 空间选择本应返回独立源要素，实际返回 join 关联行：
  1 个国家命中 2 个事件就变成 2 行，全程不抛异常，直到下游统计和出图全错。
- **失败观测要能回流给模型**，否则它只能把同一个调用再试一遍。
- **工具共用一个 `data_store`**，并发调用会互相踩。
- **产物没有身份**，"这个数字是哪一版数据算出来的"无从回答。

## 这个 fork 在 harness 上改了什么

在 CoreCoder v0.7.0 之上改动 **17 个文件 / +1070 −78 行**，新增 **9 个测试文件**。

- **run limit 是具名类型**（`corecoder/limits.py`）。上游返回字符串
  `(reached maximum tool-call rounds)`，丢掉了*哪个* limit 触发；调用方一旦漏检返回值，
  被截断的运行会被记成跑完的运行。现在触发时抛 `LimitExceeded`，带 kind 和触发值。
- **子进程按进程组终止**（`corecoder/process.py`）。只杀直接子进程会留下
  `npm run dev` 的 node 占着端口、MCP server 调起的 GDAL；孤儿进程比发起它的调用活得久。
- **失败判定收敛到一处**（`corecoder/tool_errors.py`），使 recorder 与重试计数口径一致。
  它刻意不是框架级自动重试：静默重试会让"模型自己重复"和"框架替你重试"在记录里变成同一件事，
  而这两者要改的地方相反。
- 另有 workspace 边界、工具 `concurrency_safe` 与 `deadline_mode`（cooperative/enforced）、
  请求级 deadline 预算，以及 thinking 模型的 `reasoning_content` 回传。

## GIS 工具层与证据契约

接入 GeoBenchX 目录下的 25 项工具（24 个 GIS 操作加 `reject_task`），统一工具调用、错误观测、
任务状态与终止结果，共享同一个 DataFrame/GeoDataFrame 状态。
`strata/geobench/tools.py` 提供工具档案与证据契约：产物登记为可寻址证据并附类型化摘要，
每条样本记录数据快照、工具源码 digest 与 renderer/schema 版本。

## 训练线

`strata/train/` 离线构造"工具失败 → 恢复"的训练轨迹，并在写盘前跑完全部门禁。
它的起点是一次错误结论：首版数据的错误调用是人为构造的，用 `__missing` 后缀自我标注，
而且**不是推理期真正发生的失败**；那一版 SFT 在 validation 上比不微调还差。
数字、推理与口径见 [RESULTS.md](RESULTS.md)。

## 跑起来

```bash
python -m pytest tests -q                 # 764 passed
```

GeoBenchX 是外部依赖，不在本仓库内。它是 GIS 工具层与全部真实数据评测的前提，
缺了它 `tests/strata/test_geobench_*.py` 连收集都会失败：

```bash
git clone https://github.com/Solirinai/GeoBenchX.git
cd GeoBenchX && git checkout 0edf610
git apply <此仓库>/patches/geobenchx-0001-*.patch
git apply <此仓库>/patches/geobenchx-0002-*.patch
```

放在**与本仓库同级**的目录（代码按 `../GeoBenchX` 查找）。补丁对上游的修改已逐字节验证过。

仓库自带 `.venv` 的解释器没有 GIS 依赖，且 `pytest` 在其中会挂起而非报错；
跑测试请用带 geopandas 的解释器，并设 `MPLCONFIGDIR=/tmp/corecoder-mpl`。

复现评测：

```bash
# 合成线的配对消融候选（门禁含与基线的配对性校验）
python scripts/build_recovery_candidate.py --out-dir <新目录>

# 真实数据线候选（快照随仓库携带，构建时校验 SHA256）
python scripts/build_real_candidate.py --out-dir <新目录>

# 干跑：校验候选、实体池、源哈希与工具档案；不加载模型、不占 GPU
python scripts/run_real.py --stage base --check-only \
    --bundle-dir evals/candidates/real-48 --run-root <专用目录>
```

`run_real.py` 的 bundle 与 run root **没有默认值**，两个都要显式给：
合成线的 runner 默认指向它自己的 run root，第二次实验沿用就会把结果写进第一次的目录。

## 仓库结构

```
corecoder/     Agent 运行时（上游引擎）
strata/
  geobench/    工具层、证据契约、provenance、判分与统计
  train/       离线样本管线、配对消融、真实数据子线
scripts/       构建、打包、运行与门禁脚本
tests/         测试
evals/
  candidates/  冻结的候选数据（manifest 与四份报告）
  runs/        服务器跑出的冻结报告，RESULTS.md 的数字都出自这里
  sources/     真实数据线的两张公开表快照
  ablation/    P2 消融的 summary
  replays/ reviews/  评测与重放产物
patches/       对 GeoBenchX 的上游补丁
```

仓库不含任何研发时间节点的叙事：目录与文件名只说明用途，冻结报告里也不记录运行时间。

## 局限

- 合成线只有 **96 条不同样本、两个 seed**；真实线只有 48 条、单 seed。足以支撑配对消融要回答的问题，
  不足以声称稳定性能。`disjoint_root` 家族的泛化问题未解决。
- 判分器在 temperature 0 下 4 次重复得到 8/12/10/11，因此评测以确定性指标为主证据，不以判分分数为准。
- 两条线的数字**不得合并**；真实线只作窄任务 schema/join-key 学习的评估。

## 来源与许可

- 运行时：[CoreCoder](https://github.com/he-yufeng/CoreCoder)，MIT。本仓库是其 fork，
  harness 上的改动见上文与 `git log`。
- 评测数据：[GeoBenchX](https://github.com/Solirinai/GeoBenchX)，以补丁形式适配。
- 真实数据线来源：World Bank WDI（CC BY 4.0）、Natural Earth（public domain），
  快照随仓库携带，URL、许可证与 SHA256 记录在 `strata/train/real_data.py` 的 `SOURCES`。
