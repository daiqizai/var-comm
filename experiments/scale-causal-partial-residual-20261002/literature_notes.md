# 文献核查与研究边界

核查日期：2026-10-02。本文依据论文原文及作者官方代码；固定引用所读 arXiv 版本。代码链接指向核查日的公开版本。没有将综述、论文作者的性能声明或一次检索未发现先例，当作本方案的新颖性证明。

## 结论

目前能明确排除的宽泛首创表述是：预测熵选择传输 token、数字基本信息配模拟残差、生成先验与信道似然结合、预训练生成器上的免新增模型训练观测引导。这些方向已有公开先例。

本次可检验的具体问题是：冻结 VAR 和共同解码器后，只从收发双方一致的较粗尺度上下文计算预测熵，能否用不发送位置索引的部分尺度协议获得有效增益；以及在实际付费的模拟残差观测下，固定前向映射的似然引导能否改善同预算重建。两者的新颖性、收益和计算代价都待实证，不宣称首创或最优。

## 一手来源与核查事实

### 1. VAR：同尺度并行分布不等于同一个分布

[VAR 论文 v2](https://arxiv.org/html/2404.02905v2)，2024-06-10，§3.2 式(6)，以整个尺度为自回归单位。当前尺度所有位置的分布由较粗尺度前缀和位置嵌入并行产生。[官方 `models/var.py`](https://github.com/FoundationVision/VAR/blob/main/models/var.py) 的 `autoregressive_infer_cfg` 在同一尺度先计算全部 logits，再一次采样全部位置；训练掩码按尺度分块，允许同尺度输入特征互相注意。

因此，不应称“同尺度每个位置分布相同”或“真实 token 相互独立”。原接口也不会在收到本尺度某个真 token 后，按 raster AR 方式重新更新本尺度其他位置的概率。保留所选真 token、补全其余位置后再更新下一尺度，是与原尺度接口一致的使用方式；改变尺度内条件结构需要另作算法说明。

### 2. HDA-DeepSC：数字基本信息与模拟残差已有明确公式

[Hybrid Digital-Analog Semantic Communications v1](https://arxiv.org/html/2405.12580v1)，2024-05-21，§III-A 式(16)明确模拟部分为 `z_A = z - z_tilde`；式(17)把数字重建和模拟残差相加。Algorithm 1 包含语义编解码器和混合收发器训练，数字分支在相应训练阶段采用无误传输以保持梯度。

它证明混合残差的基本结构已有先例；它没有证明本次冻结 VAR、固定 PCA 映射、无新增参数训练的接收算法有效。其训练及数字可靠性假设需要与本次实际 CRC 失败、总符号和总能量预算区分。

### 3. DiffJSCC：预训练扩散与推理引导已有先例，但整体需要训练

[DiffJSCC 原始论文 v1](https://arxiv.org/html/2404.17736v1)，2024-04-27；[官方仓库](https://github.com/mingyuyng/DiffJSCC)明确包含 JSCC 收发器和条件扩散模块训练。

[官方 `inference_cldm.py`](https://github.com/mingyuyng/DiffJSCC/blob/main/inference_cldm.py) 把 JSCC 重建图载入引导目标；[`model/cond_fn.py`](https://github.com/mingyuyng/DiffJSCC/blob/main/model/cond_fn.py) 实现可选 `MSEGuidance`。这个推理期引导无需重新训练模型，但它是向第一阶段重建图或其 latent 靠近，不应直接等同于本次对原始模拟信道残差的物理似然。

### 4. JSCGC：生成式接收已有先例，不能误称免训练方案

[JSCGC 六月论文 v1](https://arxiv.org/html/2606.12858v1)，2026-06-11，§III-A 是编码器和生成器联合训练；§IV-A 的实现采用 MambaJSCC、Z-Image 及通信适配器，先联合训练再按目标信道微调。其收到信号控制生成、互信息与感知约束的基本主张已有公开文本。

[一月论文 2601.12808v1](https://arxiv.org/abs/2601.12808v1)，2026-01-19，是另一条 arXiv 记录，标题为 *Joint Source-Channel-Generation Coding: From Distortion-oriented Reconstruction to Semantic-consistent Generation*。不能把两个编号当作同一记录的版本号，或把六月方案写成免新增训练接收器。

### 5. Ada-TokenCom：固定顺序的前缀截断与尾部生成

[Ada-TokenCom v1](https://arxiv.org/html/2608.28086v1)，2026-08-28，§II-A 区分真实 token 的自信息和预测分布的熵；采用预先确定的 token 顺序，默认 raster scan。§II-B/II-C 使用前缀算术编码、尾部生成；§III 联合选择前缀长度和 MCS，并计入 Type-I ARQ 的期望符号消耗。§IV-A 的量化实验主干是 LlamaGen-L 的 16×16 token grid。

这不是已经验证的“VAR 同尺度预测熵重排、零位置索引开销”方案。其平均预算与重传假设也不能直接充当本次单次固定 N、E 预算基线。

### 6. MaskGIT 视觉 IoT：位置掩码有收费，原文不是每个 token 10 bit 位置

[Semantic-Aware Generative Image Transmission for Resource-Constrained Visual IoT Systems v1](https://arxiv.org/html/2606.28398v1)，2026-06-24，§III-B 用预测熵、局部复杂度和实例语义评分选择 token；原文链接[作者代码](https://github.com/zzzccy1/Semantic-Aware-Generative-Image-Transmission-for-Resource-Constrained-Visual-IoT-Systems)。

§III-C 的码本 K=16384，每个量化索引为14 bit；其预算估算把冗余计为20 bit/保留 token，再付24×24网格的576 bit二进制 mask，总量为 `20*N_keep + 576` bit。300–450 bit 压缩 mask 只作为可能值讨论，主计算仍用576 bit。这个估算不能替代本次实际封包、信道编码、CRC和符号计账。

### 7. 共享 MLM 的预测熵 mask 与似然检测：直接相关先例

[Shin 等，Context-Aware Wireless Token Communication via Joint Token Masking and Detection v1](https://arxiv.org/html/2605.02123v1)，2026-05-04，§V-A 式(34)/(35)按预测熵逐步 mask 可推断 token；§IV 将 MLM 条件先验与信道似然相乘进行近似 MAP 检测。

但其 Tx 熵计算条件含真实未 mask 文本（式(33)），对 Rx 已检测序列采用理想化近似（§V-A）。共享模型不意味着双方拥有同样的当前上下文。所读方法未给出能直接迁移为本次“尺度因果、免位置索引”的完整同步封包证明。

### 8. 免新增训练的测量引导：DPS 是更早的通用先例

[Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)，初稿2022-09-29，ICLR 2023；[官方 `condition_methods.py`](https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py) 对已知测量前向算子计算 `measurement - operator(x_hat)` 的误差，并以梯度修正扩散采样。

这支持“预训练生成先验＋已知噪声模型＋测量一致性”已有先例。DPS 的近似后验方法并不自动保证离散 VAR token、非线性能量归一化和数字残差条件下的精确后验或最优估计。

### 9. 所谓2026年9月综述确实存在

[From Semantic to Token Communication: The Next Paradigm for Large-Model-Driven 6G Intelligent Connectivity v1](https://arxiv.org/html/2609.10714v1)，2026-09-09。Table 9 区分预测熵 mask、任务价值和 VAR 的尺度顺序；引用[127]即上面的 Shin 论文，§5/§6 讨论 Ada-TokenCom 和上下文先验检测。

综述可作相关文献入口，不能据此证明本次具体组合无先例。其表中 VAR 的免费尺度顺序也不是免费数据依赖位置选择的证明。

## 本次设计必须明确的边界

以下是基于上述接口和信道模型的设计推论，不是引用论文已证明的性能结论。

1. **零位置索引的条件。** 两端必须只使用一致的较粗尺度上下文、相同先验、条件类别、精度、熵定义及确定性 tie-break 来生成位置顺序。K和尺度边界须预先固定或在付费 header 中可恢复。CRC无效或上下文分歧应触发已登记失败路径；发送端不能悄悄用接收端没有的真实 omitted token。这样最多支持零显式位置索引，仍需计入 header、类别、CRC、FEC和失败帧。
2. **预测熵不是 oracle 自信息。** `-sum_v q(v) log q(v)` 可以在当前 token 尚未知时计算；`-log q(t_true)` 需要真实值。两种排序必须分开。源图驱动 oracle mask 的位置信息必须收费，不作为免索引方案成绩。
3. **熵不保证任务价值或质量最优。** 高熵位置不一定最值得传。每SNR只能在 calibration 选择K，包含K=0；同K固定顺序和随机顺序对照才能隔离排序收益。development 不参与选K。
4. **似然必须匹配实际观测。** M2 的前向模型须包含既定 residual 定义、PCA中心和投影、空间变换、能量归一化及实际噪声。投影/resize 会丢失信息；P decoder 后的误差不能未经验证当作独立同方差 AWGN。若源依赖增益不可由 Rx 推断，须使用已登记固定缩放或付费传递。
5. **数字残差条件要处理失败。** Tx 从其数字基本信息构造残差，而 Rx 若数字解码错误就没有同一基本信息。CRC失败不能免费获得真前缀、真label或真mask；失败策略及其质量必须计入完整分母。
6. **免训练的含义。** 本次指无新增模型参数梯度训练；calibration 的 PCA、统计量、K和引导强度拟合仍是数据使用，需注册。接收器优化输入变量或概率不等于模型参数训练，也不代表没有算力和时延成本。
7. **比较与结论。** 只和同N、同E、同源/噪声、同共同解码器的实际P与数字基线比较；报告质量、失败率、波形预算及接收时间。没有证据时不宣称优于全部JSCC、最优、精确后验、或已建立新颖性。

本文仅完成文献和设计边界核查。没有新增实验、模型训练或历史结果重算。
