# BN/Tent 投影小试

在完整十分类 CIFAR ResNet-18 上检查 BN 仿射参数投影、整批更新动力学与有限时域分类风险的联系。当前安排与判断统一见 [STATUS.md](../../../STATUS.md)。这是新实验入口，不复用冻结特征分类器。

## 输入和固定设计

- 本地官方 CIFAR-10-C `gaussian_noise.npy`、`defocus_blur.npy`、`contrast.npy` 和 `labels.npy`；默认 severity=3，使用完整官方 10,000 张测试图像块，不添加数据划分。
- 在官方 CIFAR-10 训练集上训练的十分类 torchvision ResNet-18 checkpoint：3×3 stride-1 stem、无 maxpool、10 类输出。支持原始 state dict 或 `state_dict` 字段及 `module.` 前缀。不会下载或训练模型。ImageNet 1000 类权重不能直接使用。
- 已检查服务器现有 `models/resnet-18`：它是 Hugging Face `ResNetModel` backbone，权重中没有分类头；采用 ImageNet 归一化和224像素预处理。它不能传给当前 `--checkpoint`，也不能只换一个随机十分类头就用于风险实验。当前入口需要另行取得上述已训练的 CIFAR 十分类模型。若改为现有 backbone 加源训练分类头，需要同时修改加载器、预处理和源端训练协议。
- 默认归一化 mean=(.4914,.4822,.4465)，std=(.2470,.2435,.2616)；必须按 checkpoint 的实际训练预处理传参。
- BN 配置遵循 [官方 Tent](https://github.com/DequanWang/tent/blob/master/tent.py)：只更新所有 BN2d 的 weight/bias，移除 running mean/variance，始终使用当前批次统计。优化器为固定 lr=.001、无 momentum 的 SGD，每批一步，批量64；H=50，epsilon=.02。此配置不代表官方所有 benchmark 的默认优化器。

## 信息与目标

每个独立未来轨迹都从同一个 checkpoint 重置，适配批次从完整腐蚀块中有放回抽样。开发与确认使用不同随机流，共用图像池；独立性针对抽样随机性，不表示新增独立图像。

当前批次 BN 的风险依赖批次分组。每条轨迹在启动前独立随机排列完整测试块，分成固定200张评估批次；该分组在此轨迹的0..H步保持不变。评估无梯度且不更新 BN 状态。事件为这些完整测试面板上的分类错误率相对初值在任一步增加超过 epsilon。它是明确的有限面板、批次分组风险口径，不是精确总体风险；概率同时平均未来适配批次和独立评估分组的随机性。只评估固定检查点会漏掉中间事件，因此实际每一步均评估。

预测器允许接收：开发阶段固定的投影、带标签开发轨迹拟合的二次风险映射、各环境开发事件率，以及启动前独立无标签整批梯度探针。开发标签构成额外监督信息，应在论文中明示。预测器不读取确认标签、确认轨迹或它们的结果。当前所有确认运行从相同状态启动，故每环境检验一个启动状态下的概率；校准重复不代表不同的真实概率。

## 运行

在服务器运行真实 GPU 计算。三个脚本分别进行开发、独立确认和绘图，不根据确认结果调整投影或规则。下面两个计算命令必须保持公共参数一致；改变预算或配置使用新的输出目录。

```bash
python experiments/code/bn_probe/projection_fit.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18.pt --output experiments/results/bn_development
python experiments/code/bn_probe/projection_check.py --data-root data/CIFAR-10-C --checkpoint models/cifar_resnet18.pt --development experiments/results/bn_development --output experiments/results/bn_confirmation
python experiments/code/bn_probe/projection_plot.py --result experiments/results/bn_confirmation
```

开发默认每环境12条轨迹。对相对初值的 BN 位移做 SVD，冻结一个全局单位投影，使用全部开发状态拟合无截距二次风险增量映射；奇偶运行估计方向的绝对余弦作为稳定性诊断。投影局部探针半径为开发投影坐标标准差。

确认默认每环境32条未来轨迹、16次校准。每次校准在投影坐标 -offset、0、+offset 处分别抽取32个完整批次更新，拟合仿射漂移，使用去中心的合并投影增量作噪声，以4096条标量模拟路径预测有限时域风险事件。另在初值处以 batch=32/64/128 直接测量投影方差、全协方差迹及均值的投影外分量。没有强制假设方差严格按1/batch缩放。

确认保存：投影路径、风险增量、遗漏方向范数、独立轨迹估计轴与冻结轴的余弦、一步漂移残差、预测概率及全部探针系数。绘图汇总报告风险映射误差、局部模型误差、超出开发探针半径的比例、批量方差、Brier及相对开发常数的改善。Brier平均所有独立校准预测与未来事件组合；区间分别重采样校准轴和未来轨迹轴，不把组合数当作独立样本量。

## 解释和后续决策

这是探索性小试。bootstrap 区间是描述性区间，不是预先保证覆盖率的同时确认；每环境一个概率不足以验证跨环境普适校准。常数基线只使用开发事件率。应同时看预测偏差、事件率和Brier，全部事件为0或1时不能据此主张预测困难已经解决。

正漂移斜率只说明所测切片局部扩张；还需查看推断平衡点是否落在探针范围、斜率重复波动及真实路径残差，才能判断是否存在可辨认的不稳定动力学。SVD方向稳定不能单独证明有效降维。仿射漂移、独立合并残差和二次风险映射都是待检验模型，未获得原标量定理的保证。确认中估计的诊断轴只用于比较，绝不替换冻结预测轴。

第一轮重点判断三个关系：投影与局部扩张是否稳定；投影模型是否解释独立轨迹；投影是否关联有限时域风险。任一关系失败均保留结果，再决定修改模型或停止一维路线。后续若改变预测器，需要新的独立确认，不能重新评分本轮并称作确认成功。
