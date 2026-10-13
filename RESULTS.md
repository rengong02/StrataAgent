# Results

本页记录仓库中冻结 candidate 和运行报告对应的结果。评分使用确定性的状态 schema；每个分数都以样本数为分母，并区分不同 seed 的重复运行。

## 1. GIS Agent 组件

| 项目 | 结果 |
|---|---:|
| 工具 catalog | 25 项：24 个 GIS 操作和 `reject_task` |
| 空间选择：独立源要素 | 45 个国家 |
| 空间选择：未去重的 join 结果 | 7,715 行 |
| GDP 均值：修复前 / 修复后 | 166.67 / 200.00 |
| 同一绘图阶段：修复前 / 修复后 | 74.9861 s / 0.1946 s |

空间选择层按源要素位置恢复输入行，避免一个源要素命中多个关联对象后被重复写入状态。385 倍是这段绘图阶段的单次测量，不是 Agent 端到端加速比。

## 2. 合成恢复数据线

candidate 为 144 条样本，train、validation、test 各 48 条；三类错误各 48 条，并在每个 split 中保持均衡。paired ablation 与 control 共享 sample id、world、数据摘要和初始提示，只改变 `recovery_error_kind`。

| 运行 | 样本 / 运行次数 | 正确 | 准确率 |
|---|---:|---:|---:|
| Base（恢复 candidate） | 144 | 0 | 0% |
| SFT validation，seed 7 | 48 | 48 | 100% |
| SFT held-out test，seed 7 | 48 | 42 | 87.5% |
| SFT validation，seed 17 | 48 | 48 | 100% |
| SFT held-out test，seed 17 | 48 | 48 | 100% |

两个 seed 在同一组 48 个 test 样本上运行，合计是 90/96 次运行正确，而不是 96 个独立 test 样本。validation 的合计是 96/96 次运行正确，同样不增加独立样本数。

在 seed 7 的 test 中，6 个失败全部属于 `disjoint_root` family。把同一冻结 checkpoint 的 `max_turns` 从 6 调到 12 后仍为 42/48，失败样本和结果不变，重复工具调用增加，因此剩余问题是错误键选择后的泛化与恢复策略，而不是回合预算不足。

## 3. 真实数据线

真实 candidate 有 48 条样本，train、validation、test 各 16 条。数据来自 World Bank WDI 与 Natural Earth 的固定快照；共有国家键分到三个互不相交的实体池，来源 SHA256、许可证和实体标识写入每条样本。

| 运行 | 样本 | 正确 | 格式失败 | 报告错误 |
|---|---:|---:|---:|---:|
| Base | 48 | 0 | 18 | 29 |
| Recovery SFT validation | 16 | 16 | 0 | 0 |
| Recovery SFT held-out test | 16 | 16 | 0 | 0 |

真实数据线的 SFT 轨迹是直接选择正确工具和参数，没有测量推理期“先失败再恢复”的在线成功率。它支持的是窄 schema 和 join-key 任务上的学习结果。

## 4. 训练和评测配置

- 模型：Qwen3-0.6B。
- 训练：full-parameter BF16 SFT，48 steps，learning rate `5e-6`，合成 seed 7/17、真实 seed 7，最大序列长度 4096。
- 解码：greedy，`max_new_tokens=1024`，默认 `max_turns=6`。
- 监督策略：屏蔽上下文和失败工具调用，只监督正确恢复和合法终止。
- 评分：`v2.1-status-schema`，不依赖 judge model。
- 原始报告：`evals/runs/recovery-ablation/`、`evals/runs/recovery-ablation-seed17/`、`evals/runs/real-seed7/`。

## 5. 其他消融

在 9 个完成配对任务上的 head-tail 与 raw evidence 对比中，head-tail 的 token 总量减少 32.9%，双侧 sign-flip 检验 `p=0.1171875`。这项结果作为上下文预算观察保留，不能表述为显著性能提升。

## 6. 限制

- 合成线只有 96 个不同样本，两个 seed 主要用于观察运行稳定性。
- 真实线只覆盖一个窄的表连接任务族，不能外推到通用 GIS Agent。
- `disjoint_root` 的泛化失败仍未解决。
- 真实数据快照只代表记录的公开数据版本；结果依赖工具 schema、prompt、模型和解码配置。
空间选择数据与阶段计时见 [timing.json](evals/spatial-select-timing/timing.json)，完整 selector 的独立计时见 [timing_corrected.json](evals/reviews/head-tail/timing_corrected.json)。GDP 100/300 的均值对照来自 [空间选择回归](tests/strata/test_geobench_spatial_select.py)。证据消融的两臂总 token 为 995,882 / 1,484,308，原始统计见 [summary.json](evals/ablation/p2c_headtail_vs_raw/summary.json)。

配对 control 使用单一缺失键错误轨迹，其 seed 7 validation 为 0/48；三类错误轨迹的 seed 7 validation 为 48/48。两者共用 144 个 world 和初始提示，Base 的逐样本结果一致。该对照限定于当前配方与任务分布。

## 原始报告索引

下表 SHA256 标识仓库内的报告文件，链接可查看逐样本结果与运行配置。

| 报告 | SHA256 |
|---|---|
| [真实 Base](evals/runs/real-seed7/base/report.json) | `98f86682ee932bd485de456f10b0703fdb9a976be73319fc324d54834b071d38` |
| [真实 SFT test](evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/test_report.json) | `858c8afd6be2b0f03721abcfa18f8495a7a34e00e327f456e63f1970ff6b145b` |
| [真实 SFT validation](evals/runs/real-seed7/sft_recovery16_epoch1_lr5e-6/validation_report.json) | `d1df5378b9d113145eda982438f5400614aafc949f63e1d8c38882a94d358dd8` |
| [合成 Base](evals/runs/recovery-ablation/base/report.json) | `e57fd1c7f15d9af0d2d683d94cb414ef61c44249dc2a98bce7fdeeee478f12b0` |
| [合成 SFT seed 7 test](evals/runs/recovery-ablation/sft_seed7/test_report.json) | `afd6cf6bbd35d669796ac8edf1248a0649b217a4d0a96bb4a15d9166eb9cd94d` |
| [合成 SFT seed 7 validation](evals/runs/recovery-ablation/sft_seed7/validation_report.json) | `a73c036dcfd01d20afaaa332f64fe3ae8758ac9eca23cb0a415839e6f4a0ff07` |
| [合成 SFT seed 17 test](evals/runs/recovery-ablation-seed17/sft_seed17/test_report.json) | `bd8fd474f2bb190ba3e5bca29312f872729cdf1f639ae9101504580e896e5855` |
| [合成 SFT seed 17 validation](evals/runs/recovery-ablation-seed17/sft_seed17/validation_report.json) | `f335aefc08227b5398f31c2b445cb1f13a12186e8de163c46b9e18413af08e5c` |
| [12 回合诊断](evals/runs/recovery-ablation/sft_seed7/test_report_maxturns12.json) | `e5179c116412577ae0393b27c2ce2371b664af7070c12d910ca2fb86f027c318` |
| [配对 control Base](evals/runs/recovery-baseline/base/report.json) | `ed72e4e044793af8a8a44bba060d8a1c0a1de5e4fb6f11f90112a3d30c969fb3` |
| [配对 control SFT validation](evals/runs/recovery-baseline/sft_seed7/validation_report.json) | `53801e5b97b419a71c0e5b97286d580698e42c83329637673a847c63da4b5679` |

真实数据的冻结报告使用输入摘要 `sha256:79e62b311240654dd7abcab577b1b27a9dcad61fcd59e2d025b023f7d253d185`；仓库候选的 manifest 使用 `sha256:e6794199c3684a1fbd1f77fcd8f7fcf9e302c9eec11dab64d37d0d4355159509`。两者的任务、实体、工具观测和 gold 相同，候选摘要差异来自来源获取时间字段的省略；报告保留运行时输入身份。

重建候选会记录当前工具源码摘要，源码变化时 `samples_digest` 会相应变化。相同样本集合还需要核对 sample id、split 和数据摘要；归档文件的字节哈希不用于识别跨环境重建的数据。
