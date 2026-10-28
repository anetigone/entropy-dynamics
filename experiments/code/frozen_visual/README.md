# 冻结视觉特征实验

本实验检验独立校准的方向与时间预测，并比较阈值、尺度和 LA 更新。当前结果与待办见 [STATUS.md](../../../STATUS.md)，其它实验见 [实验导航](../../README.md)。数值计算依赖 NumPy、SciPy，绘图依赖 Matplotlib；特征提取另需 PyTorch、torchvision、Transformers。

## 预测性停止闭环

[stopping.py](stopping.py) 复用原初始化与协议，用新独立种子模拟逐步阈值和风险。默认四类对、p=.5、n=128、offset=-2/0/2，每组128路径、1500步；无标签场预算2048/131072，各4次校准重复。全部参数预定，不从退出数据拟合安全余量。

```shell
python experiments/code/frozen_visual/stopping.py --features experiments/results/visual_features/features.npz --results experiments/results/visual_shift/results.json --diagnosis experiments/results/visual_shift/diagnosis/results.json --output experiments/results/stopping_pilot
python experiments/code/frozen_visual/stopping_plot.py --input experiments/results/stopping_pilot
```

计算支持 `--pairs`、`--batches`、`--priors`、`--offsets`、`--paths`、`--steps`、`--alphas`、`--budgets`、`--repeats`、`--fixed-steps`、`--seed`。筛选限原 results.json 中存在的threshold case，eta=.1、w=2。新参数须指定新输出目录。输出 results.json 逐case逐重复决策和 paths.npz 逐步阈值/风险。每完成一个case保存，中断保留部分结果但不支持续跑。绘图进程要求全部完成，生成 summary.json、analysis.md、tradeoff.png。

对照包括 frozen、固定步数、全预算风险最低点oracle checkpoint、精确无标签经验边际的候选分歧保护、精确目标几何oracle预测，以及独立无标签场配源/oracle端点。不可用预测冻结在0步并纳入全部case。校准与SGD随机数独立；同一校准复用于batch/offset。M为固定池重复观测次数，不是新图像。分歧保护先检查候选再部署，拒绝后冻结上一状态。

预测时长是近似退出CDF不超过alpha的最后安全前缀时刻，初始预测已超标则冻结。它测试安全上限作为停止点的效果，不含收益优化，也不是已证明的概率保证。源端点是未经保证的迁移消融；oracle端点读取目标标签。分歧保护使用精确池边际，不是有限样本估计；已发表风险监测基线尚未实现。

报告路径/最终违规率、收益、相对hindsight收益损失、平均停止步数、冻结率与可用率。先在case内平均校准重复，再对case等权；不把共享路径当作独立重复。平均停止步数是策略长度，不是本次完整轨迹生成的实测成本。当前结果见根目录STATUS.md。

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

## 扩展理论对应的验证

### 长尾噪声消融

[tail_noise.py](tail_noise.py) 用固定目标经验场对照四种过程：`real_sgd` 为原始SGD；`frozen_root` 保留真实平均漂移、全程注入根处的中心化batch噪声；`near_noise` 只在距目标峰不超过 `--radius` 的位置保留状态依赖噪声；`far_noise` 只在其它位置保留。后三者是噪声修改后的机制诊断，不是原SGD或新TTA方法。每个设置在四模式间共享随机batch；保存分量与风险退出、跳跃、确定性轨迹及一阶沿程方差诊断。

均值漂移采用8193点PCHIP插值，根显式入网格；记录全部中点的最大漂移误差。真实SGD直接使用样本梯度，不依赖漂移插值。输入、输出、设置选择、预算均可通过参数指定。先验证offset=-2的单侧主导设置，用两个半径检验分区敏感性；不把多分支设置的一阶时间标准差解释成完整分布保证。

```shell
python experiments/code/frozen_visual/tail_noise.py --offsets -2 --batches 32 128 512 --paths 600 --max-steps 4000 --radius 0.5 --workers 3 --output experiments/results/tail_noise
python experiments/code/frozen_visual/noise_plot.py --input experiments/results/tail_noise
python experiments/code/frozen_visual/tail_noise.py --offsets -2 --batches 32 128 512 --paths 600 --max-steps 4000 --radius 0.25 --workers 3 --output experiments/results/tail_inner
python experiments/code/frozen_visual/noise_plot.py --input experiments/results/tail_inner
```

中断后原计算命令加 `--resume`。保存单元为一个设置的全部四种模式；未保存设置会重算。参数与输入路径保持不变，workers允许调整。每个设置的NPZ先写临时文件，JSON最后原子写入，作为完成标记；最终汇总为results.json。若实际漂移插值误差或分区敏感性较大，应在新输出目录调整 `--grid-points` 复核，再解释噪声消融。几何不适用或数值错误会直接报错，已完成设置仍可续跑。绘图独立生成noise_ablation.png。

校准支持 `--workers` 多进程（默认1），每个任务是一个类别对、先验和校准重复，内部预算保持嵌套前缀。网格漂移和共轭逆迭代已批量计算；节点数、网格密度、截止半径及实验预算不变。建议先用4个worker；服务器不默认设置OMP/MKL/OpenBLAS线程限制；本地运行前将这三个库的线程数设为1，按CPU与内存情况调整worker数，不承诺线性加速。

每个重复完成后保存到输出目录的 `repeats/`；JSON最后以原子替换写入，表示该重复已经完整保存。中断后用同一命令、同一输出目录加 `--resume`，仅重算尚未保存的重复，worker数可改变。续跑直接比较输入路径与实验参数，不计算文件哈希；不要在原路径替换输入内容。未保存的正在计算的重复需要重算。Ctrl+C会取消尚未启动的任务并等待正在运行的worker结束；必要时终止进程后，已写出的重复仍可复用。全部重复保存后按固定顺序汇总，进程调度及续跑不改变校准编号与汇总顺序。旧版本只在结尾写结果，没有这些重复文件，不能恢复其打印过的进度。

```shell
python experiments/code/frozen_visual/visual_calibration.py --output experiments/results/calibration_parallel --repeats 20 --priors 0.5 0.4 0.3 --offsets -2 -1 0 1 2 --workers 4
# 中断后：其余参数保持一致
python experiments/code/frozen_visual/visual_calibration.py --output experiments/results/calibration_parallel --repeats 20 --priors 0.5 0.4 0.3 --offsets -2 -1 0 1 2 --workers 4 --resume
python experiments/code/frozen_visual/calibration_plot.py --input experiments/results/calibration_parallel
```

校准入口支持全部已有先验与offset；新一批参数应使用新目录。每次拟合增加 `theoretical_ratio`（目标经验场下的 R）、`target_standardized_initialization` 和匹配的局部方向参考。汇总保留跨校准重复的方向预测10/50/90%分位和标准差。绘图另生成 `calibration_scale.png`，比较CDF误差与理论尺度；方向图的 arctan(R)/π 曲线仅适用于目标临界居中，原训练峰 offset=0 不代表目标临界居中，不能将所有点都视为该极限的验证。逐重复预测在 `fits` 中保存；校准根失败在 `calibrations` 中保留，报告成功率时包含这些失败。

```shell
python experiments/code/frozen_visual/visual_calibration.py --output experiments/results/calibration_grid --repeats 20 --priors 0.5 0.4 0.3 --offsets -2 -1 0 1 2
python experiments/code/frozen_visual/calibration_plot.py --input experiments/results/calibration_grid
```

[exit_probe.py](exit_probe.py) 读取原特征和threshold结果参数，检查目标初始安全分量、含一步overshoot余量的区间、网格向外漂移、映射导数、峰曲率/噪声及标准化初始化。`--geometry-only` 不模拟路径；网格检查只是数值诊断，不能作为区间证明，也不能从单个有限n验证渐近初始化条件。不适用的设置写入JSON。

```shell
python experiments/code/frozen_visual/exit_probe.py --geometry-only --pairs 3:5 1:9 4:7 0:8 --priors 0.5 0.4 0.3 --batches 32 128 512 --offsets -2 -1 0 1 2 --output experiments/results/exit_geometry
```

不加 `--geometry-only` 时，从固定池新采样独立SGD，逐步保留初始安全分量退出和真实风险退出、两种退出位置、跨分量跳跃标记及删失。方向统一相对目标峰定义；原批次方向相对初始化定义，比较时须区分。所有路径在退出分量后继续运行至预算，以观察后续风险退出；CDF始终保留全部路径。该入口只做固定斜率threshold更新，不用于LA或尺度模式。

```shell
python experiments/code/frozen_visual/exit_probe.py --pairs 1:9 --priors 0.5 --batches 32 128 512 --offsets -2 0 2 --paths 600 --max-steps 4000 --output experiments/results/exit_tail
python experiments/code/frozen_visual/exit_plot.py --input experiments/results/exit_tail
```

共同路径上的总CDF和左右联合CDF差异应不超过预算内跳跃比例，代码直接检查这一关系；跳跃比例的标准误只描述有限池的路径抽样。全输出为 `results.json` 和 `samples.npz`，绘图独立生成 `exit_probe.png`。只检查几何的输出不运行退出CDF绘图。

上述入口均可用 `--features`、`--results`、`--output` 指定位置；校准入口还需要 `--samples` 和 `--diagnosis`，默认沿用本README已有路径。新入口的 `--grid-points` 默认1025、`--seed` 默认20261029；`--paths` 默认600、`--max-steps` 默认4000。不覆盖已有目录；新批次更换输出目录，不修改历史结果。不需要GPU且耗时可接受的数值实验可直接在本地执行；本地解释器与线程设置见根目录AGENTS.md。
