# 冻结视觉特征实验

本实验检验独立校准的方向与时间预测，并比较阈值、尺度和 LA 更新。当前结果与待办见 [STATUS.md](../../../STATUS.md)，其它实验见 [实验导航](../../README.md)。数值计算依赖 NumPy、SciPy，绘图依赖 Matplotlib；特征提取另需 PyTorch、torchvision、Transformers。

## 提取冻结特征

[visual_features.py](visual_features.py) 读取本地 CIFAR-10 官方 train/test，使用 ImageNet-1k 预训练的 [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) 及配套 `AutoImageProcessor`，保存池化层展平后的 512 维特征。不训练、不新增划分，CIFAR-10 不自动下载。

从仓库根目录运行，示例中的相对路径替换为实际位置：

```shell
python experiments/code/frozen_visual/visual_features.py --data-root data/cifar10 --model models/resnet18 --output experiments/results/visual_features --device cuda
```

| 参数 | 含义 |
|---|---|
| `--data-root` | 必填，本地 CIFAR-10 根目录 |
| `--model` | 本地 `save_pretrained` 模型目录或 HF ID；默认 `microsoft/resnet-18` |
| `--cache-dir` | 可选，HF 权重缓存目录 |
| `--allow-download` | 显式允许获取模型文件；默认只读本地或已缓存模型 |
| `--output` | 特征输出目录；默认 `experiments/results/visual_features` |
| `--device` | 默认 `cuda`，也可指定 `cpu` |
| `--batch-size` | 特征提取 batch；默认 128 |

`--model microsoft/resnet-18 --allow-download` 可在服务器首次获取权重；`--cache-dir` 控制缓存位置。模型和数据路径均通过参数提供，不需要修改源码。

## 计算与绘图

[entropy_visual.py](entropy_visual.py) 读取特征缓存，官方 train 提供固定投影与独立于目标退出轨迹的经验几何校准；官方 test 提供受控混合采样和完整加权经验风险。标签用于离线校准、控制类别先验及风险评价，不进入原始熵梯度。这是有标签离线机制实验，不是无标签部署方法。

```shell
python experiments/code/frozen_visual/entropy_visual.py --features experiments/results/visual_features/features.npz --output experiments/results/visual_shift --modes threshold
python experiments/code/frozen_visual/visual_plot.py --input experiments/results/visual_shift
```

[visual_plot.py](visual_plot.py) 在独立进程中读取结果并将图片写入同一结果目录。

| 计算参数 | 含义与默认值 |
|---|---|
| `--features` | 输入 NPZ；默认 `experiments/results/visual_features/features.npz` |
| `--output` | 结果目录；默认 `experiments/results/visual_shift` |
| `--pairs` | 正类:负类；默认 `3:5 1:9 4:7 0:8` |
| `--batches` | 默认 `32 128 512` |
| `--priors` | 正类先验；默认 `0.5 0.4 0.3` |
| `--offsets` | 平移偏移；默认 `-2 -1 0 1 2` |
| `--modes` | 更新模式列表，见下文 |
| `--paths` | 每组轨迹数；默认 600 |
| `--max-steps` | 最大更新次数；默认 2000 |
| `--seed` | 随机种子；默认 20261027 |

更新模式包括 `threshold`、`joint_scale`、`la_threshold`、`la_joint_scale`、`oracle_la_threshold`，这五项是默认列表；另支持 `oracle_la_joint_scale`。不同模式或预算使用独立输出目录。例如只做尺度对照时：

```shell
python experiments/code/frozen_visual/entropy_visual.py --features experiments/results/visual_features/features.npz --output experiments/results/visual_scale --modes threshold joint_scale --batches 128 --priors 0.4 --offsets -1 0 1
python experiments/code/frozen_visual/visual_plot.py --input experiments/results/visual_scale
```

LA 的估计先验来自未调整预测的类别计数平方根与 EMA（alpha=.1、tau=1，数值先验下限 1e-6）；oracle LA 单独使用已知目标先验。只在熵目标中调整 logits，风险用原始预测评价。固定斜率预测用于尺度或 LA 模式时仅作为适用条件压力检查。

输出包含 `results.json` 和 `samples.npz`，计算目录已存在时会提示更换目录，不覆盖。首次越界相对原冻结模型风险加容忍度定义；`times=-1` 保留在 CDF 分母中，数值失效另行记录。结果同时记录有限评估集的风险分辨率、不确定性及几何适用条件失效，不根据目标退出曲线拟合参数。共用退出分布工具位于 `../common/entropy_prediction.py`。

## 已有轨迹的几何诊断

[visual_diagnosis.py](visual_diagnosis.py) 读取已有特征、结果和轨迹，分别替换峰值、整个熵向量场、风险边界和非线性共轭，不训练或模拟新 SGD 路径。[diagnosis_plot.py](diagnosis_plot.py) 单独读取诊断结果绘图。

```shell
python experiments/code/frozen_visual/visual_diagnosis.py --features experiments/results/visual_features/features.npz --results experiments/results/visual_shift/results.json --samples experiments/results/visual_shift/samples.npz --output experiments/results/visual_shift/diagnosis
python experiments/code/frozen_visual/diagnosis_plot.py --input experiments/results/visual_shift/diagnosis
```

计算提供 `--features`、`--results`、`--samples` 和 `--output`；绘图提供 `--input`。计算默认输出到原视觉结果目录下的 `diagnosis/`，已有诊断目录不覆盖。输入原始特征与轨迹保持原样。输出 `results.json` 和 `curves.npz`，图片写入同目录。

`source_*` 使用原训练校准，`target_peak_only` 仅将局部线性预测的峰换成目标峰；`target_field_only` 替换局部向量场参数、保留原风险端点；`target_field_conjugate` 使用完整目标向量场与共轭、仍保留原风险端点；`target_boundary_only` 只替换风险端点；`target_linear` 和 `target_conjugate` 同时使用目标向量场和目标风险边界。峰不在相应安全区内或安全区单侧无界时记录不适用，不强行生成双侧预测。不同模型的汇总比较需使用共同适用组。

目标经验几何使用了原适应特征池，是事后 oracle 诊断；所有模型参数直接从特征求得，不拟合退出时间，但也不能将目标几何结果称为独立预测。当前分析与采用的证据统一见根目录 STATUS.md。

## 独立无标签校准流

[visual_calibration.py](visual_calibration.py) 从原目标经验混合分布独立有放回抽取校准分数，仅使用分数与重复计数估计熵场，再对已有 SGD 路径评分。校准随机数独立于适应路径，初始化保持原值；校准流与适应流共享固定的官方测试特征池，但不新增官方划分。不训练模型，不重跑 SGD，也不从退出时间拟合参数。

```shell
python experiments/code/frozen_visual/visual_calibration.py --features experiments/results/visual_features/features.npz --results experiments/results/visual_shift/results.json --samples experiments/results/visual_shift/samples.npz --diagnosis experiments/results/visual_shift/diagnosis/results.json --output experiments/results/visual_calibration
python experiments/code/frozen_visual/calibration_plot.py --input experiments/results/visual_calibration
```

参数包括 `--features`、`--results`、`--samples`、`--diagnosis`、`--output`，以及：

| 参数 | 默认值 |
|---|---|
| `--budgets` | `2048 16384 131072 1048576` 个校准流观测 |
| `--repeats` | 20 次独立校准重复 |
| `--pairs` | 原结果中的全部预选类别对；可传入 `3:5 1:9` 等 |
| `--priors` | `0.5` |
| `--batches` | `32 128 512` |
| `--offsets` | `0`，对应原实验初始化，不按新校准峰重新居中 |
| `--seed` | 20261028 |

同次运行的不同预算使用嵌套校准前缀；重复之间随机数独立，同次校准复用于各个 batch 的已有轨迹。重复计数通过 multinomial 精确表示所有有放回抽样的权重；校准预算是流观测次数，不是不同图片数。

输出的 `source_risk` 使用估计的目标熵场和原训练风险端点；`oracle_risk` 使用同一估计熵场和目标标签提供的风险端点，后者仅用于隔离场估计误差。目标标签只用于构造受控类别混合、oracle 诊断和评价，场估计器不读取标签。原精确目标几何预测作为条件 oracle 基线。

单个校准重复的 CDF 误差分别评分，再汇总平均；`curves.npz` 另保存有效重复的平均预测曲线。失败校准及每组有效重复比例也保存在 JSON 中，不将只在有效重复上计算的平均误差解释为全部重复的成功率。相同 SGD 路径被重复用于评分，因此不能将校准重复当作新增的独立 SGD 测试轨迹。

根标准误及 Wald 区间为一阶渐近诊断，方向区间仅传播根误差，不是完整风险或退出时间置信保证。已有初始化不依赖这批校准，因此代码不将校准方差加进实际 SGD 种子方差。数值几何计算由 [field_geometry.py](field_geometry.py) 复用，绘图由 [calibration_plot.py](calibration_plot.py) 独立读取保存结果。
