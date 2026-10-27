# 论文文件导航

## 正文的阅读方式

[DRAFT.md](DRAFT.md) 是英文论文初稿，独立给出问题定义、模型与信息条件、理论陈述及证明、数值实验设置与必要结果、文献定位和局限。阅读它不要求先打开推导笔记或结果分析文件。

[RELATED_WORK.md](RELATED_WORK.md) 是中文文献对照，独立定义风险比较器与首次越界事件，说明直接近邻、经典工具和贡献边界。它也不依赖初稿或其它笔记才能理解。两份正文可以包含外部文献链接；不把仓库内其它文件作为理解正文的前提。

正文之间允许重复必要定义。文献对照不代替论文的证明，论文初稿也不代替逐篇文献比较。

## 推导与原始材料

[notes/entropy_geometry.md](notes/entropy_geometry.md) 保留分布几何、确定性安全时长及正漂移区间 SGD 的原始详细推导。[notes/critical_sgd.md](notes/critical_sgd.md) 保留临界状态依赖 SGD、离散时间分布与微小容忍度的后续推导。它们用于追溯论证来源，各自的局部编号不与初稿混用；初稿已经写入阅读所需证明。

[notes/archive/](notes/archive/) 保存导入的重复副本和历史检索记录。[references/](references/) 集中保存参考文献 PDF。

## 结果材料

初稿包含支撑论述所需的实验设置和摘要数值；逐项结果分析、原始汇总、轨迹及图片保留在各自结果位置：

- [基础 pilot 分析](../experiments/RESULTS.md)。
- [几何与安全时长分析](../experiments/results/geometry/analysis.md)。
- [临界 SGD 分析](../experiments/results/critical/analysis.md)。

这些结果材料用于追溯数值证据，不是理解两份正文的必读前置文件。文档修改本身不代表重新运行实验。

当前研究状态、已完成工作和下一步安排统一在根目录 [STATUS.md](../STATUS.md)。工作纪律见根目录 [AGENTS.md](../AGENTS.md)，代码位置与运行方式见 [实验 README](../experiments/README.md)；本页不维护运行入口或脚本清单。
