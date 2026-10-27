# 熵阈值实验

代码位于 [code/entropy/](code/entropy/)。当前研究结果、采用的证据和下一步统一见 [STATUS.md](../STATUS.md)，工作纪律见 [AGENTS.md](../AGENTS.md)。本页只说明实验用途、位置和运行方式。

## 运行约定

直接运行相应文件，不带 `--phase`、`--output` 等参数，也不需要统一入口。实验参数与输出路径在代码顶部修改；不同参数使用独立输出目录，避免覆盖已有结果。计算与绘图分别运行，绘图和检查复用对应计算模块的路径。

基础数值计算使用 NumPy、SciPy，绘图另需 Matplotlib。冻结视觉特征提取另需 PyTorch、torchvision、Transformers 和本地数据、预训练权重。正式训练和完整实验在服务器运行，本机仅做小规模合成验证与结果分析。

## 合成实验

| 计算文件 | 用途 | 绘图文件 |
|---|---|---|
| [entropy_pilot.py](code/entropy/entropy_pilot.py) | 基础扫描与合成轨迹 | [pilot_plot.py](code/entropy/pilot_plot.py) |
| [entropy_geometry.py](code/entropy/entropy_geometry.py) | 几何、安全时长和正漂移 SGD | [geometry_plot.py](code/entropy/geometry_plot.py) |
| [entropy_critical.py](code/entropy/entropy_critical.py) | 临界 SGD、AR 对照及有限参数修正 | [critical_plot.py](code/entropy/critical_plot.py) |
| [entropy_mixed.py](code/entropy/entropy_mixed.py) | 混合平移/先验抵消扫描、真实 SGD 左右联合退出 CDF 与预测消融 | [mixed_plot.py](code/entropy/mixed_plot.py) |

[entropy_check.py](code/entropy/entropy_check.py) 独立进行数值检查。[entropy_prediction.py](code/entropy/entropy_prediction.py) 是共用的预测和含删失 CDF 计算模块。

从仓库根目录运行，例如：

```powershell
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/entropy_mixed.py
& H:/anaconda3/envs/py312/python.exe experiments/code/entropy/mixed_plot.py
```

临界 SGD 计算完成后会进行有限参数几何修正。对已有轨迹单独重算修正，可直接调用该模块的 `refine_saved_results()`，随后另行运行绘图文件；无需重新模拟。

混合偏移实验比较平衡极限、真实参数局部线性、完整共轭，以及分别去掉峰值、曲率、噪声和风险端点修正的预测。参数独立于退出观测计算，不用退出曲线拟合时间常数。

## 冻结视觉特征实验

按以下顺序准备和运行：

1. 在 [visual_features.py](code/entropy/visual_features.py) 中设置 `DATA`、`MODEL_NAME`、`DEVICE` 和 `OUT`。默认权重为 Hugging Face 的 [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18)，也可将 `MODEL_NAME` 设为本地 `save_pretrained` 目录。使用配套 `AutoImageProcessor` 预处理，提取 `ResNetModel.pooler_output` 展平后的 512 维特征，不使用 ImageNet 分类头。`LOCAL_FILES_ONLY=True` 默认只读本地或已缓存权重；服务器需要首次获取权重时可显式设为 `False`。CIFAR-10 始终从 `DATA` 读取官方 train/test，不自动下载，不训练。
2. 在 [entropy_visual.py](code/entropy/entropy_visual.py) 中设置 `CACHE`、`OUT` 及实验常量。官方 train 提供固定投影与独立于目标退出轨迹的经验几何校准，官方 test 提供受控混合采样和完整加权经验风险。类别对预先指定，不新增划分。校准使用标签，是离线 oracle 机制验证，不是无标签部署方法。
3. 计算后单独运行 [visual_plot.py](code/entropy/visual_plot.py)，读取保存结果绘图。

视觉实验比较固定阈值、阈值与 log-scale 联合更新，以及 LA 对照。LA 的估计先验来自未调整预测的类别计数平方根与 EMA（alpha=.1、tau=1，数值先验下限 1e-6），并单列已知真实先验的 oracle LA；只在熵目标中调整 logits，风险用未调整预测评价。固定斜率预测用于联合尺度或 LA 时仅作为适用条件压力检查。

## 输出位置与解释

以下路径相对仓库根目录：

| 实验 | 默认位置 |
|---|---|
| pilot | `experiments/` |
| 几何与安全时长 | `experiments/results/geometry/` |
| 临界 SGD | `experiments/results/critical/` |
| 混合偏移 | `experiments/results/mixed_shift/` |
| 冻结特征缓存 | `experiments/results/visual_features/` |
| 视觉预测与适应 | `experiments/results/visual_shift/` |

混合偏移、特征提取和视觉计算发现输出目录已存在时会提示更换目录。较早的合成计算可能重写同名文件，运行前应设置独立输出位置。绘图读取计算结果并写入对应目录；更改计算输出位置后，绘图代码复用计算模块的路径常量。

首次越界相对原冻结模型风险加容忍度定义，不等于最佳停止时间。新实验中 `times=-1` 表示预算内未观测到越界；完整 CDF 保留这些轨迹的分母，同时记录数值失效路径。视觉结果另保存有限评估集的风险分辨率与不确定性诊断。

已有结果与分析保持原位，当前采用哪些证据由 [STATUS.md](../STATUS.md) 统一说明。绘图与数值诊断不是额外的实验登记或运行门槛。
