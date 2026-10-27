# 熵适应的安全时域与风险回升

研究固定表征、固定斜率的二分类阈值模型中，混合位置与先验偏移下的熵更新、风险反弹和首次越界时间。

- [当前研究状态、证据与下一步](STATUS.md)：唯一状态入口。
- [工作纪律](AGENTS.md)。
- [论文初稿](paper/DRAFT.md)：统一模型、理论主线与证明索引。
- [分布几何与确定性安全时长](paper/notes/entropy_geometry.md)：详细证明。
- [临界 SGD 与容忍度标度](paper/notes/critical_sgd.md)：详细证明。
- [相关工作](paper/RELATED_WORK.md)：文献对照与新颖性边界。
- [实验代码与统一运行方式](experiments/README.md)。
- [几何历史结果](theory_progress_reproducibility/README.md) 与 [临界 SGD 历史结果](critical_sgd_reproducibility/critical_sgd/README.md)：原始输出及分析导航。
- [历史记录](paper/notes/archive/)：导入的前序笔记副本与历史检索记录。

从仓库根目录运行基础合成实验：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_pilot.py --phase compute
```

其他复现命令见对应实验 README。脚本可能覆盖各自已有输出；仅做文档整理时无需重新运行。
