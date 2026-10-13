# GeoBenchX patches

本目录保存 StrataAgent 运行所需的两个 GeoBenchX 兼容补丁。GeoBenchX 是外部依赖，补丁以 `git apply` 的形式保留，便于在固定版本上重新部署。

## 应用

如果 GeoBenchX 与本仓库同级：

```bash
git -C ../GeoBenchX checkout 0edf610
git -C ../GeoBenchX apply "$PWD/patches/geobenchx-0001-drop-colorbar-fmt.patch"
git -C ../GeoBenchX apply "$PWD/patches/geobenchx-0002-spatial-select-distinct-features.patch"
```

也可以把 `GEOBENCHX_ROOT` 指向其他 checkout，并在该目录中执行相同的 `git apply` 命令。

## 补丁

| 文件 | 作用 | 回归覆盖 |
|---|---|---|
| `geobenchx-0001-drop-colorbar-fmt.patch` | 修复 Matplotlib 新版本中 choropleth colorbar 的参数兼容性 | `test_geobench_profile.py` |
| `geobenchx-0002-spatial-select-distinct-features.patch` | 按源要素位置恢复空间选择结果，避免 sjoin 关联行重复进入状态 | `test_geobench_spatial_select.py` |

第二个补丁同时修正下游统计和绘图的输入基数。它不改变空间索引的使用方式，也不把局部绘图计时当作 Agent 端到端性能指标。

补丁应用状态会被 GeoBenchX provenance 检查；未应用时，评测 runner 默认拒绝启动。
