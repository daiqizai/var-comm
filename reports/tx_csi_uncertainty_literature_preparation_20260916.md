# 发送端CSI不确定性：转题前原文核查与最小验证草案

日期：2026-09-16（UTC+8）。状态：**文献核查与未执行草案，不是已经确定新题或启动新实验。** 当前固定m7混合路线已完成三权重收尾并结束，见`hybrid_weight_closure_convergence_20260916.md`；CSI候选问题仍缺足够清楚的技术差异。本文件不修改任何既有实验结论，也不自动授权新的训练。

## 1. 一个可测需求，以及目前不能声称的创新

候选需求是：每张图只有一次固定信道资源与总能量的发送机会，不重传；发送时的实际信道质量不等于发送端得到的估计。能否在同样的信息条件下，降低图像未达到预定质量要求的比例，同时避免过度保守造成大量源细节损失？

旧系统只验证了理想共享名义SNR；**尚无本项目在CSI误差、量化或滞后下的性能结果**。新问题必须重新定义发送端可观测量，不得把旧的正确SNR输出重命名为错误CSI测试。

三篇已发表原文已经分别覆盖：不确定SINR下鲁棒链路适配；分层图像、源/信道联合适配与过期CSI；不依赖显式SNR的学习式图像通信及反馈失配。因此，“增加CSI误差”“把层数和FEC一起选”“去掉SNR输入”都不能直接作为本项目首次贡献。**目前没有足够清楚的机制差异支持直接确定一个新论文题目。** 下述最小验证是决定是否值得立项的计划，不是以模块增加来补创新。

这里选择三篇与问题直接相关的正规同行评审期刊论文；未声称它们是截至2026年9月的全部最新工作，也没有将期刊杜撰为CCF C类会议。

## 2. 三篇原文及其真正边界

| 原文 | 已核实的信息假设与做法 | 对本项目的约束 |
|---|---|---|
| [1] Ugrasen Singh、Olav Tirkkonen，*Robust Link Adaptation in Multiantenna URLLC Systems With Flashlight Interference*，IEEE Communications Letters 28(10):2432–2436，2024年10月，DOI `10.1109/LCOMM.2024.3451018` | 正文II节假设期望链路CSI与CQI反馈理想；不确定性来自测量后干扰基站波束变化。利用条件干扰/SINR统计与backoff满足outage要求，关注不可重传的时延约束场景。III–IV节给出分析及Monte Carlo验证。 | 条件分布、分位数或保守SNR回退并不是新机制。其特定干扰模型不能冒充任意估计误差模型；本文outage-capacity分析也不是已测有限码长图像系统。 |
| [2] Ahmed Hadji等，*Shaping duo binary turbo-coded BICM scheme for JPWL image transmission using a link adaptation strategy over wireless channels*，EURASIP Journal on Wireless Communications and Networking，2021:64，DOI `10.1186/s13638-021-01949-9` | 依据CSI联合选择JPWL质量层、源/信道码率、调制、功率和译码迭代，使用UEP/UPA。3.2.5节专门比较完美CSI与每20/100个OFDM符号更新CSI，报告PSNR变化。 | “图像源率＋保护适配＋过期CSI”已有直接图像通信先例。其变化功率/多参数QoS优化不等于我们的每图固定N/E；但仅把约束固定，也不能自动获得足够技术新颖性。 |
| [3] Hongjie Yuan、Weizhang Xu、Yuhuan Wang、Xingxing Wang，*Channel-Blind Joint Source–Channel Coding for Wireless Image Transmission*，Sensors 24(12):4005，2024-06-20，DOI `10.3390/s24124005` | 网络不要求显式SNR条件，训练覆盖−5至20dB；4.2节在AWGN/慢Rayleigh下对比ADJSCC，并包含反馈SNR失配。数字参照包括JPG/BPG搭配capacity假设，不是本项目有限码长FEC实测。 | 合理学习参照至少应包括CSI-blind模型，以及在同样CSI误差分布训练的条件模型；不能只给完美CSI训练的网络喂错条件，再宣布数字或新模型鲁棒。原文不构成已证明胜过真实有限资源数字系统。 |

逐份读取了公开正式PDF，而非只看搜索摘要。特别校正了初始资料中的两处书目信息：[1]第一作者是**Ugrasen**、DOI尾号是**3451018**；[3]作者是**Yuan/Xu/Wang/Wang**。Aalto封面元数据写的01/01不作为实际卷期发表日期。

补充阅读了Wang等IEEE Wireless Communications Letters 2024的*Wireless Adaptive Image Transmission Over OFDM Channels Based on Entropy Model*：其III节明确不需要实际算术编码，熵模型用于OFDM重要性匹配。它不宜冒充本项目“实际熵码＋整帧FEC”参照，故不替代上表更直接的过期CSI图像工作。

## 3. 与现有工程能形成的最小区别——尚待检验

可检验区别不是网络，而是**在不确定发送端CSI下，以所有发送后图像的后果而非只以正确帧/BLER评价有限模式选择**。候选模式必须在同一个每图N/E账本内，失败仍输出协议允许的候选图。

若记实际SNR为gamma、TX观测为z、模式为a，可以在校准上估计两类量：

- 链路侧：`E[BLER(a, gamma) | z]`或条件SNR下分位数对应的成功率/有效源比特。
- 图像侧：`E[image_loss(a, gamma) | z]`或预先规定图像质量违约概率。

这是标准风险决策框架，**不是本项目原创公式**。只有相对充分重新校准的鲁棒链路策略仍有稳定、实际可解释的质量/资源取舍，才值得进一步寻找特定机制与技术差异。若最优策略只是一个普通backoff，没有额外价值，应停止此立项，而不是再加一个策略网络。

## 4. 最小验证方案（仅计划，未执行）

### 第一步：先只分离一种信息缺陷

首轮可只研究有噪TX标量SNR，不同时叠加估计误差、量化、时延、快衰落与反馈丢包。明确生成`p(gamma,z)`和TX可见z；允许RX理想知道实际gamma时必须标成隔离TX误差的理想接收端假设，不能称真实CSI系统。未来若改成量化或滞后，应分别登记量化器/时序相关模型，并完整核算pilot、CSI反馈和模式信令。

最便宜的开题检查可在已有五SNR校准表上定义一个预登记的离散观测混淆矩阵，计算冻结策略的**期望回放**。这只是在现有离散点上的决策检查，不是新的波形部署测试，不通过在五SNR之间插值制造实际测得性能。100张development已反复使用，仅能决定是否继续，不能重新称独立测试。

### 第二步：数字强参照必须重新校准

使用相同可行源率/FEC集合、观测z、N/E、类别计费和失败输出。至少包括：

1. 校准最优固定保守模式；原理想SNR策略直接用z只作失配诊断，不是唯一主对照。
2. 分位数/backoff或条件BLER约束的鲁棒最大源率、有效源比特策略；BLER目标和回退大小都允许用同一校准数据重新确定。
3. 条件分布下的全发送图像风险策略，与2共享同一误差模型和校准机会，不额外知道实际gamma或误码位置。

完整raw与实际算术数字链均保留；不能以不给数字基线校准、把所有失败强行清零、或用不计码流开销的capacity近似制造收益。若以后更换更强FEC，各策略都应得到同一候选集合；当前收尾实验不做这种更换。

### 第三步：学习参照和结论门槛

若离线数字检查不能区别2和3，不应先启动新训练。只有发现稳定作用范围，才考虑经明确授权运行CSI-blind和误差分布匹配训练的条件DeepJSCC，配对训练预算，并给数字策略相同重新校准权利。旧Deep系统使用SNR条件，不能直接复用其正确条件重建图来冒充错误TX条件结果；需要分别定义TX的z和RX允许使用的信息。

主指标应在校准登记：例如图像LPIPS要求的违约率、全部发送均值LPIPS以及PSNR辅助约束；具体阈值不能看development后临时放宽。报告均值与源图级配对区间，也报告失败尾部、发送端/接收端计算、CSI与控制开销。DINO如继续使用则仅作辅助描述，披露其既往暴露。有限样本不足以验证URLLC极低outage，不能借用[1]的目标可靠度包装小样本图像实验。

当前不存在新CSI实验结果、获胜模型或独立holdout结论。下一步是否执行这个最小验证，由本轮收敛结论和明确的问题价值决定；本文件没有自动训练入口。

## 5. 原文归档与复核入口

目录：`outputs/HYBRID-WEIGHT-CLOSURE-20260916/literature_preparation/`，PDF与`pdftotext -layout`文本均保留。

| 文献 | 本地PDF | SHA256 |
|---|---|---|
| [1] | `CL2024_robust_link.pdf` | `0a8b1d198fec71dcb954146ca592d4f0ffce9d2eee94f80e1b49f792dae209b4` |
| [2] | `EURASIP2021_JPWL_CSI.pdf` | `9aebd3175ee09fb59e25959baea85e72466a716c1259cf7b5eb1cb589fcefe78` |
| [3] | `Sensors2024_CBJSCC.pdf` | `9e388ebac7cb9b17a8a2177a606d8f63cb0c1ea810562493bbfc92bc5a92c2bf` |

官方出处分别为Aalto作者机构的published PDF、Springer论文页和MDPI出版社论文PDF：

- [1] `https://acris.aalto.fi/ws/portalfiles/portal/161685824/Robust_Link_Adaptation_in_Multiantenna_URLLC_Systems_With_Flashlight_Interference.pdf`
- [2] `https://link.springer.com/article/10.1186/s13638-021-01949-9`
- [3] `https://www.mdpi.com/1424-8220/24/12/4005`

未运行论文代码、下载模型权重或声明完成性能复现；也未将本轮检索误称穷尽的新颖性保证。
