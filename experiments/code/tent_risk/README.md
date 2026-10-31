# 真实 Tent 有限时域风险预测

从固定当前状态预测未来H步内分类错误率相对当前增加超过epsilon的概率。完整网络与BN批次实现复用 [BN小试](../bn_probe/README.md)，当前状态、结果判断及下一步见 [STATUS.md](../../../STATUS.md)。本目录是新的开发候选，不改变已有投影实验或其结果。

## 固定任务与信息

默认：同一CIFAR-10 ResNet-18，完整官方CIFAR-10-C测试块；三种开发腐蚀，severity=5，SGD学习率.001及.01，启动步数0及100，每个组合2次启动抽样、每启动4条独立未来，H=100、epsilon=.02。共24个启动任务、96条未来。前者为与原小试一致的学习率，后者单列为压力配置，不称常规Tent性能。默认网格是开发设计，不保证能产生恶化。一次完整测试面板评估需要10000张图像，默认未来约9600次更新和对应面板评估，不能当作快速smoke。

所有初始状态从相同checkpoint重置后在同一个环境预适配得到；零步启动虽有多次探针抽样，但参数状态相同，不称不同初值。非零步启动独立抽样，不从同一长路径抽取多个相关启动点。可用参数控制腐蚀、严重度、学习率、启动步数和预算；开始前限定网格，不按未来结果删除安全单元。使用完整官方划分，不新增图像训练/测试划分。

特征不使用标签、环境名称或真实当前风险：包含学习率、启动步数、H、epsilon，以及当前输出熵、置信度、前两类间隔、平均软类别分布平方和的批次均值/标准差。16个独立完整批次更新均从同一状态开始，不推进真实轨迹。在固定4批无标签面板上测每次更新前后上述信号的变化，取变化均值及标准差。该标准差条件于测量面板，尚非总体风险噪声；同批样本受BN耦合，不按单样本梯度估计方差。未加入PCA轴，不预设不稳定根。

未来批次与预适配、探针随机流独立。每条未来重新抽取完整测试块的评估分组，并在该未来的0..H步固定。每一步评价，无事件删失。概率平均未来适配批次及评估分组，风险是指定批量分组下的完整有限面板错误率。独立抽样流共用官方图像池，不代表独立图像或总体风险。

## 运行入口

正式GPU运行与服务器同步由用户负责。先采开发任务，再检查coverage；初轮无事件时保留负结果，不运行无意义确认。所有输出目录必须不存在；不同设计使用独立目录，不覆盖结果。

```bash
python experiments/code/tent_risk/collect_tasks.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18/pytorch_model.bin --output experiments/results/tent_development
python experiments/code/tent_risk/fit_forecast.py --development experiments/results/tent_development --output experiments/results/tent_predictor
```

拟合会先保存按腐蚀、严重度、学习率及启动步数分组的coverage.json。事件全0或全1时停止，不写predictor.json。出现两类后拟合三个固定正则逻辑模型，目标为每个启动任务的未来事件比例：static仅使用状态特征，drift加入探针平均变化，dynamics再加入变化标准差。常数基线仅使用开发事件率；不作类别重加权，以保持概率目标。默认正则.1，任何调参仅限开发；按腐蚀留一的开发评分不等于确认。

### Contrast 后续开发

基于首轮观察到的contrast局部转折，限定一次新的开发试验：只用contrast、severity=5，学习率.001（对照）及.01（压力配置），均预适配100步；每个配置6个独立启动、每启动4条未来，H=500，epsilon=.02，seed=31415。共12个启动任务、48条未来、24000次未来更新和完整面板逐步评估。checkpoint、归一化、batch=64、eval batch=200和探针预算保持原值。已有采集脚本直接支持，无需新运行入口。

```bash
python experiments/code/tent_risk/collect_tasks.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18/pytorch_model.bin --corruptions contrast --severity 5 --learning-rates 0.001 0.01 --start-steps 100 --starts 6 --futures 4 --horizon 500 --epsilon 0.02 --seed 31415 --output experiments/results/contrast_followup
python experiments/code/tent_risk/summarize_tasks.py --tasks experiments/results/contrast_followup --output experiments/results/contrast_followup/coverage.json
```

启动状态重新独立生成，不是首轮异常路径的定点续跑；两种学习率同时用于各自预适配和未来，因此对照比较的是完整适配配置，不能解释成相同状态下仅改变未来学习率的因果效应。使用新随机流防止重复已有路径。H由100改到500后，预测任务也改变，不混合两个时域的事件率。

先完成全部预定任务并按启动状态、学习率报告事件数、最早超阈值时间、实际风险增量和反弹；不因中途出现事件提前停止或只保留失效路径。若仍全零，保留本轮负结果，不自动增加预算；若仅一个启动出现事件，也不立即拟合并宣布可预测。首要判断是恶化是否在多个独立启动上复现，再决定模型训练和新的确认设计。这是看过首轮结果后的定向开发，不是冻结方法的独立确认，不能合并首轮样本提高名义确认精度。

### 独立确认示例

模型冻结后，下面是新随机流、留出腐蚀的独立确认示例。它不是必须立即运行的第二轮预算；确认设计应在开发结束后固定。同一checkpoint、归一化、批量、H、epsilon及探针预算保持一致。若模型以H=500开发，确认采集也需显式传入`--horizon 500`，不能沿用下面H=100的原始示例。

```bash
python experiments/code/tent_risk/collect_tasks.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18/pytorch_model.bin --corruptions shot_noise motion_blur brightness --seed 2718 --output experiments/results/tent_confirmation
python experiments/code/tent_risk/score_forecast.py --tasks experiments/results/tent_confirmation --predictor experiments/results/tent_predictor/predictor.json --output experiments/results/tent_scores
python experiments/code/tent_risk/forecast_plot.py --result experiments/results/tent_scores
```

评分只对features应用冻结系数，标签仅用于评分。与开发相同seed的任务会被拒绝；不根据确认标签更新模型或阈值。确认也可先使用相同腐蚀的新随机流，但不能称留出环境迁移。报告保留全部条件，并按学习率分列，不能把压力配置的收益表述为常规配置收益。

每个task文件保存启动BN状态、预先固定的特征、各独立未来的初始绝对风险及完整风险增量、事件、探针/未来耗时及参数；不保存全部高维未来状态。初始绝对风险用于区别“不会进一步恶化”与“起点本来就好”，不输入预测器。protocol记录设计及完成状态，部分运行可保留但不进入拟合。不提供自动续跑或统一调度器。

已有冻结模型时，可对一个保存的启动状态重新作独立无标签探针并直接输出预测。这个入口不打开labels.npy，也不加载task中的事件或未来数组；它只取启动状态和任务参数。必须使用与开发一致的checkpoint和预处理。

```bash
python experiments/code/tent_risk/forecast_start.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18/pytorch_model.bin --state experiments/results/tent_confirmation/task_0000.npz --predictor experiments/results/tent_predictor/predictor.json --seed 3141 --output experiments/results/tent_start
```

本地合成验证入口（不下载数据或训练真实网络）：

```bash
python experiments/code/tent_risk/test_forecast.py
```

验证使用小型BN网络和合成事件，检查批次探针重置、概率学习、独立确认及退化任务处理；不验证真实数据事件覆盖或预测效力。采集与数值拟合保持分进程，避免混合本机Torch和数值拟合的运行库。

## 证据与限制

已有任务可直接重算事件覆盖、初始风险、最终变化与路径反弹；不拟合模型，不运行网络。最大反弹与相对启动风险的阈值事件分别报告。

```bash
python experiments/code/tent_risk/summarize_tasks.py --tasks experiments/results/tent_development --output experiments/results/tent_development/coverage.json
```

此处coverage.json是完整风险曲线汇总；拟合目录中的coverage.json是拟合入口保存的条件事件计数，两者位置不同。

比较常数、static、drift、dynamics的Brier，以及dynamics相对各基线的配对改善。区间重采样整个启动任务，不把重复未来当作独立启动点；只描述选定环境下任务抽样的不确定性，不提供跨环境总体置信保证。可靠性分箱为描述性检查；4次未来只能粗略估计单状态概率，独立确认预算需另行确定。

static与dynamics共用同一批静态测量和上下文，但dynamics额外使用反向传播与更新后前向，尚不是等计算预算比较。正式方法证据还需相同额外预算的静态测量对照及耗时报告。部署无标签，训练使用带标签开发未来；不能称完全无监督方法。

这是直接学习事件概率的首个基线与动力学消融接口，未实现学习风险坐标后的多步动力学模拟，也没有真实网络理论保证。首要问题是开发网格有没有非退化事件、完整批次动态信号相对静态信号是否有独立预测增益。已有标量minimax定理不能直接为该逻辑预测器背书。
