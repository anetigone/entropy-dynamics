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
