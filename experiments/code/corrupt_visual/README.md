# CIFAR-10-C 独立事前预测

冻结编码器与标量阈值下，检验源端基准迁移、目标无标签校准和未来独立 SGD 的风险退出方向。当前状态与下一步见 [STATUS.md](../../../STATUS.md)，已有实验见[实验导航](../../README.md)。

## 数据

使用官方 CIFAR-10-C，保留原始测试划分和严重度块，不另拆校准/评估图片。用户提供的 [Hugging Face 仓库](https://huggingface.co/datasets/galilai-group/cifar10-c)只含旧式加载脚本，实际下载地址指向 [官方 Zenodo](https://zenodo.org/records/2535967)。本代码直接读取官方解包后的 NPY，不依赖远程 dataset script 或 `trust_remote_code`。

Zenodo 较慢时，可使用 [TorchUncertainty 的 Hugging Face 镜像](https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/tree/main)，其中直接提供约 2.92 GB 的 `CIFAR-10-C.tar`，无需运行数据集加载脚本。下面是服务器 shell 命令；本地 PowerShell 中将 `curl`写为 `curl.exe`。镜像速度取决于运行机器网络，未承诺提速。

若先前遇到 Xet/CAS reconstruction 401，可沿用禁用 `hf-xet` 客户端的下载方式。服务器中在启动 `hf` 前设置环境变量，下载到独立目录：

```shell
HF_HUB_DISABLE_XET=1 hf download torch-uncertainty/CIFAR-C CIFAR-10-C.tar --repo-type dataset --local-dir data/cifar10_c_download
tar -xf data/cifar10_c_download/CIFAR-10-C.tar -C data
```

`HF_HUB_DISABLE_XET=1`是 [Hugging Face 官方开关](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet)。Python 中使用时，应在导入 `huggingface_hub`之前设置；已导入的交互式会话应重新启动。此设置禁用 Xet 客户端，不保证链路速度，也不表示服务端不再使用 Xet 存储。

也可直接使用 curl：

```shell
mkdir -p data
curl -L --fail --retry 3 -C - -o data/cifar10_c_hf.tar https://huggingface.co/datasets/torch-uncertainty/CIFAR-C/resolve/main/CIFAR-10-C.tar
tar -xf data/cifar10_c_hf.tar -C data
```

使用独立文件名保存镜像下载，不在 Zenodo 的未完成文件上拼接续传；重跑下载命令仅续传同一镜像文件。下载本身由使用者执行，脚本不自动启动。

手动下载官方包或镜像中的 `CIFAR-10-C.tar`（不是 CIFAR-10-P），解包为：

```text
data/CIFAR-10-C/
  labels.npy
  gaussian_noise.npy
  defocus_blur.npy
  contrast.npy
```

每个腐蚀数组形状为 `(50000,32,32,3)`，连续五块分别对应严重度 1–5，每块 10000 张。`labels.npy` 支持 10000 个标签或五次重复的 50000 个标签，按严重度正确取块。读取全部官方块；不去重，不重划分。脚本不会下载图片数据。

## 1. 提取腐蚀特征

[corrupt_features.py](corrupt_features.py)使用与干净缓存相同的冻结 ResNet-18 权重；必须提供同一个 checkpoint。图像处理器按干净缓存 `protocol.json` 中的类型及完整预处理配置重建，不采用 checkpoint 的默认预处理。默认只读本地/已缓存模型，`--allow-download`仅允许获取模型文件。重建后核对尺寸、插值、裁剪、缩放和归一化参数；浮点参数允许序列化精度差异，真实差异报出双方值，不计算哈希。模型路径可以不同，使用者应确保选用同一组权重。

```shell
python experiments/code/corrupt_visual/corrupt_features.py --data-root data/CIFAR-10-C --model models/resnet-18 --clean-protocol experiments/results/visual_features/protocol.json --output experiments/results/corrupt_features --device cuda
```

默认提取 `gaussian_noise`、`defocus_blur`、`contrast` 的严重度 1、3，共六个完整测试块；参数 `--corruptions`、`--severities`可替换。`--batch-size`默认 128，`--cache-dir`控制模型缓存。GPU 特征提取不更新网络或 BN；使用既有 512 维池化特征约定。干净 train/test 缓存直接复用，不重复提取。

每环境输出 `{corruption}_s{severity}.npz`，含 `test_features`、`test_labels`；`protocol.json`保存模型、预处理、环境参数和完成标记。目录存在则拒绝覆盖；中断保留完成的文件，不支持续跑。只要求后续计算使用的环境文件已存在。

## 2. 独立预测计算

[corrupt_prediction.py](corrupt_prediction.py)直接运行真实非线性熵 SGD：

```shell
python experiments/code/corrupt_visual/corrupt_prediction.py --clean-features experiments/results/visual_features/features.npz --corrupt-features experiments/results/corrupt_features --output experiments/results/corrupt_prediction --workers 4
```

默认四类对，clean 加六种腐蚀条件，共 28 环境；n=128，M=2048/16384/131072，每环境 1024 条参考路径、每预算 256 次校准重复、4000 步固定上限、eta=.1、epsilon=.02。汽车/卡车同样保留 4000 步。数据与模型路径不写死；预算、种子、类对、环境、batch、步长及容忍度均由命令行参数指定。随机种子是整数标识，不代表日期。

干净训练特征确定投影、尺度、源端熵根和一次性初始化。所有环境保留相同初始化；目标图片不再倾斜加权。源根不适用时固定使用源投影中点零初始化，源端预测以 .5 回退，全部环境照常计算并记录原因。使用干净与腐蚀缓存相邻的 `protocol.json`核对预处理。

七列预测为：

| 名称 | 输入与用途 |
|---|---|
| constant | 常数 .5 |
| source_root | 源端根和尺度直接迁移；初始化就在源根，因此与常数 .5 相同，不视为额外信息基线 |
| target_root | 独立目标无标签分数估计根与尺度，预测 `Phi(sqrt(n)*(x0-qhat)/shat)` |
| source_moment | 用源端影响函数直接估计种子坐标，再取 Phi |
| source_clipped | 将同一坐标裁剪到固定窗口再取 Phi |
| gaussian_transfer | 对同一坐标应用固定 Gaussian 阶梯规则，仅为迁移消融 |
| oracle_root | 完整目标池给出的精确经验根与尺度，离线诊断；仍是局部方向近似，不保证性能上界 |

默认阶梯规则固定 `(K,r)=(1,1)`，整个实验不重新优化，所有预算共用同一规则。它通常不匹配实际相对校准噪声；保存源端噪声比例、匹配标记、逐重复坐标越窗情况，以及精确目标标准化位移的窗口诊断。目标几何只进入离线诊断列及评价，不进入其他预测器；不存在目标重居中或按适用性筛选环境。规则 JSON 复用 `../frozen_visual/gaussian_rules.json`，随源码一起保留；自定义规则可用 `--rules`指定。

根估计在源分数范围向两侧扩展 `--scan-margin`（默认 2）的固定 513 点网格扫描，随后 Brent 细化。固定扫描域不使用完整目标池极值；估计器仅使用观测分数及计数。扫描非唯一、未找到根或根处尺度无效时预测 .5，记录原因、失败率并进入完整 MAE。有限扫描不证明全实线唯一性；目标诊断则按完整目标分数范围扫描。

参考与校准随机流独立；同一环境五种估计及两种对照共享校准重复，不同 M 使用嵌套前缀。校准通过 multinomial 计数精确表示独立有放回抽样。所有流共享固定官方测试图片池，不代表新增独立图像；标签用于预设二分类环境和实际风险评价，不进入校准估计器。每个环境使用实际经验标签风险，以初始化风险加 epsilon 定义退出，保留分量退出与实际风险退出及其方向。

每完成环境保存 `cell_*.npz` 和更新 `results.json`。NPZ 保留参考退出时间/方向和逐重复预测；JSON 保留协议、源/目标诊断、根拟合及失败。中断保留完成环境，不支持续跑；新参数使用新输出目录。`--workers`只改变进程调度，不改变既定参数下的随机流和结果。服务器无需默认限制线程；本机解释器及线程限制见 AGENTS.md。

## 3. 汇总与绘图

[corrupt_plot.py](corrupt_plot.py)在独立进程中读取完整计算，不生成新轨迹：

```shell
python experiments/code/corrupt_visual/corrupt_plot.py --input experiments/results/corrupt_prediction
```

输出 `summary.json`、`analysis.md`、`budget.png` 和 `direction.png`。报告每环境/预算的七种 MAE、根失败、三项配对改善、删失及几何不适用；聚合按全部环境等权，clean 对照也计入且逐环境独立列出。`budget.png`比较每类对平均误差与 M；`direction.png`在最大 M 显示参考概率同时区间和各规则平均预测，平均预测不是评分对象，MAE 始终按逐重复预测计算。

默认 alpha=.05，联合控制全部环境的参考和校准误差。参考为删失兼容的精确单侧二项式界；规则损失与配对差用条件经验 Bernstein 界。每设置七项风险、三项配对差共同分配校准误差；确定性预测直接传播参考区间。配对项为 source_root−target_root、source_moment−gaussian_transfer、source_clipped−gaussian_transfer，正值表示后者改善。失败以 .5 回退纳入全部评分，不仅报告成功重复；不声称连续窗口最优性、一般未知基准保证或完整 Tent 性能。
