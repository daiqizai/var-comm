# NeST-Com：论文图表重画规范、图例字典与数据入口

版本：v1 · 2026-10-12  
核对仓库：`daiqizai/var-comm`  
核对快照：`7fbb85555f682f25d13db1c2d3d91951eb942684`  
用途：交给绘图执行端统一重画。此文件不是新实验授权，也不表示下列全部备选图都必须进入主文。原始数据、原图像、已发布统计区间和历史图文件只读。

---

## 0. 先固定论文的图表主线

**整体问题：利用逐尺度生成先验，在固定无线预算内配置实际传输与接收端生成。**

图表按以下顺序服务这一主线：

1. 系统图说明“传什么、生成什么、如何穿过实际数字链路”。
2. 外部性能图说明整个系统具有怎样的质量取舍。
3. 同先验机制图说明部分尺度这个具体设计增加了什么价值。
4. 重建图说明哪些内容保留了、哪些仍然出错。
5. 成本表说明这些收益的在线计算代价。

建议WCL第一版采用 **4张主图＋1张小表**。这是本项目的版面建议，不是WCL对图数的规定。WCL公开要求为总计不超过5页，包含图表与参考文献；最终以完整论文编译结果决定是否再压缩图数。备选图先独立导出，放项目附录、学位论文或投稿系统允许的补充材料，不能默认用附件绕过主文篇幅要求。[R1]

### 主文默认组合

| 工作编号 | 建议论文编号 | 内容 | 推荐布局 | 当前准备度 |
|---|---|---|---|---|
| M01 | Fig. 1 | 系统流程＋边界尺度恢复 | 通栏、横向紧凑，两个子模块可拆开 | 可据冻结实现绘制 |
| M02 | Fig. 2 | common500：raw主方法与三个外部基线 | 2×2通栏；另导出四张单指标图 | 已有全部数值和区间，只重画 |
| M03 | Fig. 3 | 部分尺度在raw/熵编码内部的增量 | 两个紧凑面板：ΔPSNR、ΔDINOv2-L；四指标完整版另存 | 已有new100配对结果，只重画 |
| M04 | Fig. 4 | 最终模型的固定重建比较 | 通栏2–3行×4列；四行完整版另存 | 原图和数字结果已有；选中样例的P100k/Swin200k图像需确认或补保存 |
| MT01 | Table I | 在线收发处理成本 | 紧凑表，正文取一个明确工作点，全档另存 | 六方法统一计时仍待实际完成；不得填旧设备时间 |

**空间不足时先减少M04的样例行数，或将成本详细分解移至附录。不要缩小到看不清的字号，也不要删掉M03后只剩外部模型比较。**

---

## 1. 统一方法命名：正文、图例、照片列名和表格必须一致

### 1.1 当前主方法与外部基线

| 中文含义 | 正文首次出现的英文名称 | 统一图例 | 照片列名，可换行 | 身份说明 |
|---|---|---|---|---|
| 本文固定长度token实现 | NeST-Com with fixed-length token coding | **NeST-Com (raw)** | NeST-Com\n(raw) | 每个token采用固定12 bit表示；raw不是原始RGB图像 |
| 本文VAR条件熵编码实现 | NeST-Com with VAR-conditional entropy coding | **NeST-Com (entropy-coded)** | NeST-Com\n(entropy-coded) | 发送端需要VAR概率计算；不能沿用raw的低发送端计算说明 |
| 潜空间连续JSCC | Perceptually trained latent-space joint source-channel coding | **Latent-space JSCC** | Latent-space\nJSCC | 主文只绑定P100k；这是感知训练基线，不是MSE-only |
| 作者Swin模型适配基线 | Adapted SwinJSCC | **SwinJSCC (adapted)** | SwinJSCC\n(adapted) | 主文只绑定Swin200k；训练1–19 dB，预算停止，未宣称收敛 |
| 传统数字强基线 | Adaptive-resolution BPG with LDPC coding | **Adaptive-resolution BPG + LDPC** | Adaptive BPG\n+ LDPC | 必须是自适应分辨率版本，不能用原生256版本的性能或计时替代 |
| 原始图像 | Original image | 不作性能曲线 | **Original** | 全部方法使用同一源图像素 |

**主文性能图里可以不在每个图例重复100k/200k，但图注或实验设置必须明确检查点。凡同时出现新旧版本，步骤数必须写进图例。**不再使用P、H、MAIN、B0、EC等项目代号作为对外方法名称。

### 1.2 同先验机制对照

| 中文含义 | 统一图例 | 图注中必须解释 |
|---|---|---|
| raw整尺度前缀＋VAR | **Whole-scale prefix + VAR (raw)** | 只允许K=0；whole-scale prefix不是传完全部十尺度 |
| VAR条件熵编码整尺度＋VAR | **Whole-scale prefix + VAR (entropy-coded)** | 与熵编码部分尺度使用相同熵模型、恢复器和相应协议 |
| 固定统计熵编码整尺度＋VAR | **Whole-scale prefix + VAR (static entropy)** | 概率来自固定统计，而非在线VAR；它不是NeST-Com熵编码增强版 |
| raw部分尺度、不做生成补全 | **Partial-scale, no completion (raw)** | 与配对有补全分支使用相同实际接收token和Dc；缺失残差贡献置零，不是码本索引置0 |

为了图例不太长，四方法图可将括号部分换到第二行，但英文词语不变。不要把同先验熵编码整尺度对照命名为ARPC：它不是ARPC原生复现。

### 1.3 历史与诊断版本

| 使用场景 | 统一名称 |
|---|---|
| 旧P结果 | **Latent-space JSCC (40k)** |
| 新旧P对比中的最终版 | **Latent-space JSCC (100k)** |
| 旧Swin结果 | **SwinJSCC (adapted, 80k)** |
| 新旧Swin对比中的最终版 | **SwinJSCC (adapted, 200k)** |
| 历史HiFi图像 | **HiFi-DiffCom (adapted)**；图注写清以Swin80k接收观测恢复 |
| 原生256×256 BPG历史分支 | **BPG + LDPC (native resolution)** |
| 免费完整连续latent参考 | **Full-latent reference** |
| P保持通信瓶颈、去掉噪声 | **Latent-space JSCC (noise removed)** |
| P相同条件实际19 dB | **Latent-space JSCC (19 dB)** |
| 冻结原图像解码器与适配图像解码器对照 | **Original image decoder**／**Adapted image decoder**，用D0/Dc作为括号数学符号 |

`P1024`、`Swin`等没有版本的历史键不能自动映射到最新模型。特别是目前选中的狗、西兰花、糕点、购物车拼图：P是40k，Swin是80k，不能只改列名成100k/200k。[R3]

---

## 2. 统一指标、坐标轴与数学符号

### 2.1 绝对指标

| 内部字段 | 对外坐标轴 | 方向 | 不允许的简写或解释 |
|---|---|---|---|
| `psnr_db` | **PSNR (dB)** | ↑ | 不把逐图PSNR均值改成平均MSE取对数 |
| `lpips_alex` | **LPIPS** | ↓ | 不为了让全部曲线向上而擅自画1−LPIPS |
| `dinov2_vitl14_cosine` | **DINOv2-L similarity** | ↑ | 图注注明ViT-L/14 cosine；不可混入早期其他DINO |
| `convnext_top1_source_prediction` | **Prediction agreement (%)** | ↑ | 图注说明ConvNeXt与原图预测一致；不是Classification accuracy或语义正确率 |

顺序固定为：**PSNR → LPIPS → DINOv2-L → Prediction agreement**。2×2图中左上PSNR、右上LPIPS、左下DINO、右下一致率。

### 2.2 配对差值与资源坐标

- `ΔPSNR (dB)`：方法减参考；正数更好。
- `ΔLPIPS`：方法减参考；负数更好，不翻转已发布统计的符号。
- `ΔDINOv2-L similarity`：方法减参考；正数更好。
- `ΔPrediction agreement (pp)`：方法减参考，以百分点表示，均值和区间同时×100。
- `SNR (dB)`：1、4、7、10、13、19使用真实数值间距；不能均匀排6个类别。
- `Complex channel uses per image, N`：复符号次数，明确N是总预算还是正文预算。
- `Transmitted prefix tokens`：计划实际装入源码的token数，不写Correctly delivered tokens，除非确实统计了正确接收数。
- `Encoded source bits per image`：真实源码bit长度；不等于信道编码bit或无线符号数。
- `Effective code rate, k/n`：采用实际码块数值，不用名义码率标签冒充。
- `Transmitter processing time (ms)`、`Receiver processing time (ms)`、`Software end-to-end time (ms)`：软件处理时间，不叫实际空口时延。
- `Actual frame energy / (2N)`：实测每帧能量相对标称值，不宣称每帧精确等能量。
- `Body CRC rejection rate (%)`：不能自动解释为灰图率、语义中断率或图像重建失败率。

只统一显示名称，不更改数据字段计算。不要新加FID雷达图、综合分数或人为归一化后的“总胜率”。

---

## 3. 主图逐张规范

### M01 — System overview and prefix-conditioned reconstruction

**回答的问题：什么真实传输，什么由先验生成，如何满足无线预算？**

推荐两个可拆模块：

**(a) System overview**

```text
Input image
  → Frozen visual encoder
  → Multiscale residual quantization
  → Prefix selection (m, K)
  → Source coding: raw / optional VAR-conditional entropy coding
  → Packetization and CRC
  → LDPC coding and rate matching
  → QAM mapping and frame assembly
  → AWGN channel
  → Header and payload decoding
  → Protocol-specific error handling / source decoding
  → VAR completion
  → Frozen image decoder Dc
  → Reconstructed image
```

旁路只画真实存在的依赖：

- `Offline configuration selection`依据校准集建立表；部署时由`N, SNR`查询，控制前缀与LDPC–QAM。
- Header单独的保护路径，以及接收端读取profile的路径。
- 可选熵编码的发送端VAR概率计算用虚线支路标示，raw不经过它。
- raw正文CRC拒收仍按KEEP使用实际硬判决token；不要把全系统画成CRC失败统一丢弃。
- 熵码失败走该熵编码协议实际冻结的失败规则；不把损坏熵码当正确码流继续解码。

**(b) Boundary-scale recovery**

```text
Received coarse-scale prefix
  → Predict boundary scale
  → Overwrite received boundary positions
  → Update accumulated latent
  → Predict finer scales
```

收到的边界token直接改变累计latent，并影响之后尺度；不会在当前实现中重算同尺度其余位置。错误接收情形下不能把硬判决token标成Ground-truth tokens。

颜色的角色标签：`Received tokens`、`Generated tokens`、`Untransmitted positions`。若该局部图只解释无噪声机制，明确`Correctly received prefix (illustration)`，不暗示所有运行都正确。

十尺度边长：**1, 2, 3, 4, 5, 6, 8, 10, 13, 16**。

| 完整前缀 | token累计数 | raw源bit |
|---|---:|---:|
| m6 | 91 | 1092 |
| m7 | 155 | 1860 |
| m8 | 255 | 3060 |
| m9 | 424 | 5088 |
| m10 | 680 | 8160 |

第8尺度为10×10，第9尺度为13×13，第10尺度为16×16。19 dB的raw示例可用`m=8, K=142`；13 dB历史样例为`m=8, K=9`。使用具体配置时核对对应工作点，不把二者混用。

帧结构用`N_header + N_body + N_idle = N`。raw模式头部68复符号的标注只适用于对应raw协议，不自动搬到熵编码接口。补齐槽不是额外图像信息；raw实现中的已知补齐符号带能量，不能画成零能量不发射。

图注必须限定位：发送端已知用于查表的SNR；前缀在发送前选定；这不是空口任意时间打断码字仍能独立交付尺度的协议。冻结不等于所有模型从未训练；Dc是此前适配后冻结的组件。

建议尺寸：宽6.8–7英寸、高约2.2–2.8英寸，具体随正文排版调整。字体以最终尺寸为准。框图不必强套matplotlib曲线模板；继续使用现有可编辑矢量框图路线即可。

### M02 — Reconstruction quality under a fixed channel-use budget

**固定条件：common500、N=1024、SNR=[1,4,7,10,13,19]，每来源3噪声。**

四条曲线按以下顺序：

1. `NeST-Com (raw)`
2. `Latent-space JSCC`
3. `SwinJSCC (adapted)`
4. `Adaptive-resolution BPG + LDPC`

2×2四指标；均值曲线＋已有逐点95%来源图bootstrap区间。失败帧保留。不用区间重叠与否代替配对检验。

**直接数据入口：**

```text
paper/figures/final_paper_comparisons_20261012/plot_data.csv
paper/figures/figures4papers_trial_20261012/plot_data.csv
```

这两张绘图输入已经包含96个方法×指标×SNR行；重画优先用绑定过来源的现有绘图表。[R2]

底层需要复核时读：

```text
results/main_raw64_20261007/final_common500_r6/summary.csv
results/baseline_strength_20261011/paired_supplement_v1/common500/summary.csv
results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1/comparison_summary.csv
```

**不得加入：**仅16图的HiFi、100图的熵编码部分尺度、旧P40k/Swin80k混作最终方法。历史版本学习曲线另画。

建议通栏宽7英寸、高约4.0–4.6英寸，顶部独立2列图例。另导出四个单栏版本，不能直接将总图缩一半。

### M03 — Benefit of partial-scale transmission under two source-coding interfaces

**目的：用直接对照说明具体传输设计，不以这张图替代整体系统主线。**

默认主文版两个子图：

- (a) `ΔPSNR (dB)`。
- (b) `ΔDINOv2-L similarity`。

横轴为4、10、19 dB。两条线／两组误差棒的图例不是方法名，而是**对照条件**：

- **Raw coding**：`NeST-Com (raw) − Whole-scale prefix + VAR (raw)`。
- **VAR-conditional entropy coding**：`NeST-Com (entropy-coded) − Whole-scale prefix + VAR (entropy-coded)`。

使用`results/entropy_partial_new100_20261011/paired.csv`中的已发布配对均值与区间，不相减两个单方法区间；100来源、每来源3噪声。画y=0细灰线。零增量、跨零区间、不利结果全部保留。

四指标完整2×2版作为S03的配套；主文不必把同一信息用曲线、柱图、表格再重复三遍。不得把raw和熵编码的增量相加：它们的包格式与错误规则也不同，不是纯正交两因素消融。[R4]

### M04 — Qualitative reconstruction comparison

**主文默认列序：**

```text
Original | SwinJSCC (adapted) | Latent-space JSCC | NeST-Com (raw)
```

有同源、同工作点、最终配置的实际图像时，可加入`Adaptive BPG + LDPC`；有新熵编码图像时，可另出增强版，但不能通过占位图、AI生成、锐化或旧图改标签补齐。

当前选中四个来源：**21、92、95、75**，对应白色长毛狗、西兰花、糕点、购物车。13 dB历史缓存已存在；原选取是看过16图后的人为示例选择，不能改称随机代表样本。[R3]

执行顺序：

1. 固定这四个来源，不因新版基线恢复较好就重新换图。
2. 优先确认是否已有P100k/Swin200k在同来源、同13 dB、固定第一噪声下的重建缓存。
3. 没有才登记少量实际新模型推断，生成自己的发射波形、辅助信息和接收观测；不能给新模型复用旧接收y。
4. 数字主方法不变且缓存身份一致时复用。第一噪声按原规则：连续2001、数字6201；同噪声分布不等于同一接收观测。
5. 主文排2–3行，完整4行保留附录。不做每方法最优噪声选择。不做超分、去噪、颜色增强。

**历史HiFi处理：**现有HiFi基于Swin80k接收观测。它留在明确标注的历史比较图；如将Swin换成200k又要保留“共享同一个接收观测”的HiFi比较，必须真正重新运行相应HiFi恢复。当前重画不自动授权这项高成本计算。[R3]

若改用19 dB说明高SNR特点，必须让每列全部来自19 dB；不能只替换本文一列。若没有适当失败样例，不凭空生成；正文说明局限，附录展示已有失败图。

### MT01 — Online computational cost

六行方法对应当前登记的六方法计时：

```text
NeST-Com (raw)
NeST-Com (entropy-coded)
Whole-scale prefix + VAR (raw)
Whole-scale prefix + VAR (entropy-coded)
Latent-space JSCC
SwinJSCC (adapted)
```

主文可取19 dB一档，明确这一取点不是各方法各选最快档。列建议：

`Method | Transmitter (ms) | Receiver (ms) | Software end-to-end (ms)`。

完整4/10/19 dB结果和分位数放附录。发送端要计入真实图像编码、熵概率计算、所有实际长度回退尝试和编码工作。磁盘审计、文件I/O等是否排除依照冻结计时协议，并在方法中说明。

**本快照中实际六方法统一计时未完成。**可准备布局，但不得填旧4090D与新H800的拼接数字，也不得用训练墙钟时间除图片数冒充在线时延。[R5]

---

## 4. 补充材料／长稿可用图：完整候选清单

这些是“能支持某个论点时可采用”的素材库，不是全部必做，更不是额外训练清单。

### S01 — Reconstruction from different scale prefixes（优先级高）

- 说明整体逐尺度思路，建议m6、m7、m8、m9、m10横向排列。
- 照片两行：`Direct prefix decoding`／`VAR completion`，另外放Original；所有列相同来源、相同冻结解码器。
- 可配两曲线：横轴`Transmitted prefix tokens`或真实源码bit，纵轴LPIPS／DINOv2-L。
- 无噪声参考明确`Noiseless source-interface diagnostic`，不是无线可靠性结果。
- 入口：`reports/token_channel_efficiency_20260923_source_A.md`及其绑定的逐来源表。
- **需先核对旧Source A的VAR推断、Dc和指标版本。早期字段DINO不能重命名成DINOv2-L。**若只用旧原图示意，明示来源；若要与当前定量结果比较，需要对应同版本数据。
- 主文可把小型前缀序列并入Fig.1(b)，完整版单独存。

### S02 — Whole-scale versus partial-scale transmission (raw)

- common500六档SNR，四指标2×2。
- 图例：`Whole-scale prefix + VAR (raw)`、`NeST-Com (raw)`。
- 入口：`results/wcl_evidence_closure_20261009/T5_figures/fig_t5_same_prior_N1024.*`及绑定数据；底层也见原raw500汇总。
- 这张是原主机制全范围图，不与M03的新100均值混合。

### S03 — Raw and entropy-coded transmission: four-method comparison

- 同一new100、N1024、4/10/19 dB；四指标2×2。
- 四方法：raw整尺度、raw部分尺度、条件熵编码整尺度、条件熵编码部分尺度。
- 入口：`results/entropy_partial_new100_20261011/summary.csv`。
- 同时导出M03的四指标配对差值完整版；保留4 dB一致率未确定改善、19 dB熵码两分支一致率差值为0。

### S04 — Entropy-coded NeST-Com versus continuous baselines

- 同一new100；曲线：`NeST-Com (entropy-coded)`、`Latent-space JSCC`、`SwinJSCC (adapted)`。
- 默认不加入没有该图集结果的BPG或HiFi。
- 入口：`results/baseline_strength_20261011/paired_supplement_v1/new100/`，原熵编码锚点在`results/entropy_partial_new100_20261011/summary.csv`。
- 使用相同来源的现有行与区间。两方法差值图用该目录的`paired.csv`。

### S05 — Effect of VAR completion with identical received tokens

- 比较`Partial-scale, no completion (raw)`与`NeST-Com (raw)`。
- 数值曲线与照片可以分别导出；两个方法要使用相同实际接收token和Dc。
- 照片列序：`Original | Direct decoding | VAR completion`。
- 对现有13 dB样例，这比用它强调K=9的细小增益更直接。
- 入口：原raw500核心消融与`paper/figures/user_selected_examples_20261012/`；参考`selection_and_integrity.json`中的接收状态绑定。
- 不把“发送相同token”误写成“收到相同token”；要核对实际接收状态。

### S06 — Dependence on the channel-use budget

- N2048确认集：100来源、4/10/19 dB，3个数字方法。
- 方法：`Whole-scale prefix + VAR (raw)`、`NeST-Com (raw)`、`Whole-scale prefix + VAR (entropy-coded)`。
- 当前这组没有条件熵编码部分尺度N2048结果，不自动添加第四条线。
- 入口：`results/wcl_evidence_closure_20261009/T6_budget2048/v1/summary.csv`、`paired.csv`、`selected_configurations.csv`。
- 跨预算图优先画每个预算内部的`Partial − Whole`差值，不直接用N1024的500图绝对均值减N2048的100图绝对均值。
- N2048/19 dB各数字方案输出相同、差值为0，要保留，不强行让所有点K>0。[R6]

### S07 — Actual source-code lengths and payload fit

- 图一可用码长ECDF：横轴`Encoded source bits per image`，纵轴`Empirical CDF`。
- 图例：`Fixed-length token coding`、`Static entropy coding`、`VAR-conditional entropy coding`。
- 在同一完整前缀（例如m9）比较，不能把不同m的曲线当同内容压缩率。
- 容量线为扣除该协议必要信息字段和CRC后的实际源码容量；总信道N=1024不能直接当1024 bit容量。
- 入口：`results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/source_lengths.csv`及相应协议报告。
- 图二可用不同m的真实均值／P95码长。真实整数码流包含终止；交叉熵估计不能替代实际码长。

### S08 — Selected transmission configurations and resource accounting

分成两种图，不堆在一个双纵轴上：

1. **配置表／热图**：每个SNR选中的`m`、`K`、调制、实际`k/n`。类别色块用QPSK/16QAM/64QAM，不把调制名称当连续数。
2. **堆叠资源条**：`Protected header`、`Coded payload`、`Known padding`，纵轴复符号次数，合计为N。

- raw与熵编码单独从各自协议读Nheader/Nbody/Nidle，不共用一个头长常数。
- 字节、源bit、信道编码bit和复符号必须分层画，不能放到同一单位堆叠条。
- 平均实际帧能量可另作小表或箱线图。只保存了汇总统计时不得伪造箱线图分位数。
- 入口：`results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/`、冻结配置以及新熵码配置登记。

### S09 — Baseline training and checkpoint diagnostics

- P与Swin分别画图。横轴`Training updates (k)`；主纵轴分别`Calibration objective`、`Calibration RGB MSE`。
- 训练损失使用已经登记的500更新非重叠均值，标清；校准保留全部点，不能只保留好看区间。
- 标出学习率变化、选中检查点；P100k为经验平台，Swin200k为预算停止，不能统一写Converged。
- 四指标里程碑另外导出：P 40/60/80/100k；Swin 80/120/160/200k；图例用SNR值，各图保持顺序。
- 入口：`results/baseline_strength_20261011/training_figures_v1/`及`diagnostic_delivery_v1/`。
- 全校准诊断不是holdout；P和Swin损失定义不一样，不能直接比较纵轴绝对值。

### S10 — Bottleneck diagnostic of latent-space JSCC

- 点图依次：`Full-latent reference`、`Latent-space JSCC (40k, noise removed)`、`Latent-space JSCC (40k, 19 dB)`、对应100k两点。
- 保留无噪声通信路径的N1024瓶颈和19 dB接收器条件。
- 入口：`results/baseline_strength_20261011/automatic_followthrough_v1/bottleneck/`及`diagnostic_delivery_v1/`。
- 免费完整latent参考不叫`N1024 upper bound`或`Optimal JSCC`。

### S11 — Sensitivity to transmitter-side SNR mismatch

- 真实SNR=4/7/10 dB；查表SNR为各自−3/0/+3 dB。
- 横轴`Tx SNR offset (dB)`，定义为查表dB值减真实dB值；纵轴优先`PSNR change from matched configuration (dB)`。
- 两个方法：raw整尺度前缀、raw部分尺度；按真实SNR拆面板。
- 也可画9格热图，但未测的格子必须空白；不得补插值成连续鲁棒性区域。
- 原common500前100图的事后诊断，不是new100确认集；接收端仍使用真实噪声方差。
- 入口由`experiments/entropy-partial-mismatch-20261010/reports/`中已完成mismatch统计报告指向实际CSV。
- KEEP正文CRC拒收不等于灰图输出；不把此图画成“已经提出鲁棒SNR策略”。

### S12 — Frozen-policy transfer to Kodak crops

- 明确`Kodak 256×256 crops`，不称原生高分辨率Kodak传输。
- 只使用该实验真实测过的工作点与方法；当前新P/Swin如果未在Kodak补测，图上仍标历史版本，不能改标签。
- 入口从`results/paper_supplement_20261008/`中的Kodak交付报告定位；本轮未重新读取全部底层行，绘图端必须检查版本与图像处理。
- 用独立图／表，不与ImageNet数值合并成一条曲线。

### S13 — Interpretation of perceptual and representation metrics

- 参照组可命名为`Unrelated image`、`Same-class image`、`Mild degradation`、`Identical image`，仅画实际执行过的组。
- 每个指标一面板；不要把这些参照值命名为普遍“语义安全线”。
- 如没有逐图数据，只画实际均值／已有区间，不凭汇总数字生成小提琴图。
- 先检查`experiments/external-comparison-20261004/metric_reference_preregistration.json`及对应完成记录；登记不是完成。本清单不自动新增评分。

### S14 — Online quality–computation trade-off

- 只有统一计时完成后才能画。
- 横轴TX或RX软件处理时间，纵轴DINO或LPIPS；同一SNR比较，多个SNR分面板。
- 使用MT01六方法；发送端和接收端时间不要在同一条连线中混用。
- 质量如果来自500/100图、计时来自16图，要明示两种统计样本不同；不能解释成逐图质量—时间相关关系。
- 没有数学证明时不标`Global Pareto frontier`，可写`Measured quality–time trade-off`。

### S15 — Source-paired differences against final baselines

- common500：NeST-Com(raw)分别减P100k、Swin200k，四指标，已有配对区间。
- new100：NeST-Com(entropy-coded)分别减相同两新基线，单独画。
- 使用`results/baseline_strength_20261011/paired_supplement_v1/{common500,new100}/paired.csv`。
- 图例写`vs. Latent-space JSCC`、`vs. SwinJSCC (adapted)`；标题明确被减参考，保留不利方向。
- 主文M02已有绝对曲线；这张更适合附录说明差值精度，不强行两张都占正文大版面。

### S16 — Full qualitative gallery and historical HiFi comparison

- 保留16来源候选与最终4来源选择的完整图库。
- 五列旧图：`Original | SwinJSCC (adapted, 80k) | Latent-space JSCC (40k) | HiFi-DiffCom (adapted) | NeST-Com (raw)`。
- 图注标`Historical development examples`，HiFi与旧Swin共享接收观测。图像源来自现有`paper/figures/user_selected_examples_20261012/`和`additional_candidates_20261012/`。
- 可画7列全素材供内部检查，但不建议7列×4行直接压到WCL正文。

### S17 — Additional frozen ablations（仅在有匹配数据时）

可能标签：

- `NeST-Com (raw, QPSK/16QAM only)`：去掉64QAM后重新在校准集合合法选择的已有分支。
- `NeST-Com (raw, convolutional coding)`：已有受控信道码对照。
- `NeST-Com (raw, CRC-failed payload discarded)`：仅当除失败规则外其他设置匹配。
- `Raster-order partial prefix`／`Entropy-ranked partial prefix`：仅使用已经完成、排序开销记账相同的对照。

不能用早期旧M1/B0协议的单点直接改名成最终主系统消融，也不能因为曾经做过一个分支，就自行填齐六档曲线。先读实际配置差异；不匹配时作为历史探索，不进入当前机制归因。

---

## 5. 学位论文可以保留，但WCL不默认新增的图

这些方向已有历史讨论／部分结果，是否绘制取决于具体章节，不从本清单自动启动实验：

| 历史方向 | 建议对外标签 | 必须保留的限定 |
|---|---|---|
| 不等差错保护 | Equal error protection / Scale-aware unequal error protection | 区分已知SNR与统计CSI；不同协议分别画 |
| 反馈与两段式传输 | One-shot transmission / ACK-based progressive transmission / Probe-then-transmit | 反馈、探测、重传和时延开销不能省略 |
| 内容自适应C-REAL | Content-adaptive prefix selection (online reconstruction) | 发送端在线重建成本、选择目标、实际实现与oracle分开 |
| 按图最优参考 | Per-image oracle allocation | Oracle必须显式出现，不称部署方法 |
| 接收端适配器 | Frozen receiver / Receiver-adapted reconstruction | 真实训练数据与代价；失败结果保留 |
| 早期混合数字＋连续 | Digital-prefix transmission / Hybrid digital–continuous transmission | 旧协议、预算、训练范围不能混入当前主图 |
| 接收端生成参数旋钮 | Deterministic completion / Stochastic completion | 只有实际做过才能画；不可从想法生成数值 |

不再将未实施的ARPC、Ada-TokenCom原生方法画成性能曲线；相关工作机制表可列它们，但注明与自己的内部匹配对照不同。

---

## 6. 全部图的视觉规范：沿用figures4papers，但以最终版面为准

### 6.1 方法颜色与线型：跨图不换身份

完整可机读字典见`figure_registry.json`。建议保持现在的主方法深蓝、连续基线暖红、Swin青色、BPG紫色：

| 统一图例 | 色值 | 线型 | 标记 |
|---|---|---|---|
| NeST-Com (raw) | `#0F4D92` | 实线 | 圆点 |
| NeST-Com (entropy-coded) | `#3775BA` | 实线 | 菱形 |
| Latent-space JSCC | `#B64342` | 虚线 | 方形 |
| SwinJSCC (adapted) | `#42949E` | 点划线 | 上三角 |
| Adaptive-resolution BPG + LDPC | `#9A4D8E` | 点线 | 下三角 |
| Whole-scale prefix + VAR (raw) | `#767676` | 虚线 | 空心圆 |
| Whole-scale prefix + VAR (entropy-coded) | `#4D4D4D` | 点划线 | 空心菱形 |
| Whole-scale prefix + VAR (static entropy) | `#8A735A` | 点线 | 空心方形 |
| Partial-scale, no completion (raw) | `#B07A25` | 点线 | × |
| HiFi-DiffCom (adapted) | `#8A5E9B` | 点划线 | 五边形 |

灰度仍靠线型和标记可辨。旧模型与新模型同族颜色，历史版本降低饱和度或用空心标记，图例明确步数。配置“接收/生成token”颜色是机制角色色，不得暗示性能曲线方法身份。[R7]

### 6.2 字体与尺寸

- 最终单栏宽约3.4–3.5英寸；通栏宽约6.8–7英寸，按实际模板的`columnwidth/textwidth`校验。
- 轴标签和刻度目标8–9 pt，图例7.5–8.5 pt，子图标题9–10 pt。数字是本项目建议，不是IEEE统一强制字号。
- 曲线约1.4–1.7 pt、标记3.5–4.5 pt；不要让本文线比所有对照粗两三倍。
- 白底、浅水平网格；顶部/右侧轴脊关闭；图例独立留白，少于5条优先2列。
- 使用Arial/Helvetica可用字体回退到DejaVu Sans。不得为用户打包字体文件。
- 不复制skill的大画布24–36 pt默认后再缩图；当前论文版面的最终字号优先。

### 6.3 数值与区间

- 同一坐标范围内比较曲线，不对每种方法单独选择y轴。
- 允许紧凑y范围，但不能裁掉负结果、失败点或置信区间；不做误导性断轴。
- 曲线只连已测点，不用样条平滑制造额外趋势。
- 增量图明确参考；配对区间只能来自已有配对结果，不能由两个边际区间相减。
- 线性刻度的SNR不等距，4/10/19也不能当三格等距。
- 常规显示：PSNR两位，LPIPS/DINO三位，百分数一位；小差值必要时四位。后台保存完整精度。

### 6.4 导出与检查

每张图输出：`PDF`、`SVG`、`PNG`、`plot_data.csv`、`caption.tex`、`provenance.json`。

PDF中的曲线、文字矢量；照片保留真实栅格像素。SVG文字可编辑。PNG为600 dpi预览，但不宣称256×256重建照片因排版导出变为高分辨率重建。

检查：实际论文尺寸、灰度预览、图例遮挡、上下标、边界尺度编号、正确方向、百分比/百分点、模型步数、数据集标签。最后必须放入论文PDF实际看一遍；“脚本成功”不等于排版合格。

---

## 7. 数据身份与允许复用的范围

| 数据集合 | 可以支持的比较 | 禁止混用 |
|---|---|---|
| common500 | 当前raw主方法、P100k、Swin200k、自适应BPG及已匹配整尺度对照 | 不加入new100熵码均值或旧cal100快照 |
| new100 | raw/熵码四方案；后续固定P100k/Swin200k比较 | 不称所有方法都在从未使用过的新盲测上同时验证 |
| N2048确认100 | 此预算内部三数字方案与配对差值 | 不与另一集合绝对均值直接做预算提升差 |
| calibration1000 | 选策、训练轨迹、里程碑与有界搜索诊断 | 不当成最终泛化性能 |
| development固定样例 | 示意图、实际重建现象、统一计时 | 不当成500图统计代表；选择后的样例不可称随机 |
| Kodak固定裁剪 | 冻结策略的特定跨集合验证 | 不改称原生分辨率评测 |

同一个方法跨数据集合可保持颜色与名字，但数据集、N、SNR和版本必须由每图明确。采样/噪声规则来自原manifest，不从文件名猜。若表中缺列或缺方法，抛出明确错误，不补0、不用旧模型顶替。

---

## 8. 执行分批与停止边界

**第一批：不调用任何神经模型，只重画已经有完整数据的图。**

- M01系统图；M02主性能图；M03配对机制图。
- S02/S03/S04/S06/S09/S10/S15按现有结果导出。
- 对其他备用图先定位文件；找不到完成记录就标`NEEDS_SOURCE_CHECK`，不自动跑新科学实验。

**第二批：只处理实际缺口。**

- M04的最终P100k/Swin200k样例若没有缓存，登记最小范围新推断；已有数字输出不重跑。
- MT01/S14等统一计时真正完成后再填数值；现在仅生成表头／版式说明，不能生成貌似完成的数值图。
- Source A若版本不匹配，写明缺口，先采用机制示意或已匹配图；不自动补全所有m、K、SNR。

**最终主文选图：**M01、M02、M03、M04、MT01。S01的小型前缀恢复可并入M01；S04可依据重点作为替换候选，但不是所有图同时塞进去。

### 可直接给Codex的启动指令

```text
依据 NeST_Com_Figure_Redraw_Spec_20261012.md 与 figure_registry.json，
在新的 paper/figures/redraw_v1/ 输出，不覆盖历史图。
先只做数据读取与版式：优先M01/M02/M03，其次数据已齐的附录图。
沿用figures4papers统一字体、方法颜色、线型与导出，但以真实论文尺寸为准。
对外图例使用指定英文名，不再出现P/H/MAIN/B0/EC等内部代号。
主文P只能用100k、Swin只能用200k；旧照片不能靠改标签变成新模型。
各图严格绑定cohort、N、SNR、模型、噪声与指标版本。
使用原始均值和已发布区间，不重新bootstrap，不平滑，不挑噪声、不做图像增强。
缺少最终基线重建或实际计时的图单独列阻塞项，不自动重训或执行未知规模的推断。
统一导出PDF/SVG/600dpi PNG、绘图数据、图注和来源；实际尺寸及灰度逐图检查。
最终给出主文4图+1表的排版候选，以及独立的备用图目录。
```

---

## 9. 来源核对入口

下列R编号用于区分已确认事实与本文件提出的排版建议。仓库路径均固定到上文核对快照，绘图执行端可在确认新提交后另行更新，不改写旧数据身份。

- **R1：WCL官方篇幅规则**：`https://www.comsoc.org/publications/journals/ieee-wcl/policies-guidelines`（本轮已核查）。
- **R2：现有figures4papers主性能图与数据**：`paper/figures/figures4papers_trial_20261012/README.md`；`paper/figures/final_paper_comparisons_20261012/README.md`；`scripts/plot_paper_figures4papers.py`。
- **R3：用户选择的历史样例、版本和接收状态绑定**：`paper/figures/user_selected_examples_20261012/README.md`；`selection_and_integrity.json`；`provenance.csv`；`scripts/plot_paper_user_selected_examples.py`。
- **R4：四数字方案new100结果**：`results/entropy_partial_new100_20261011/REPORT.md`、`summary.csv`、`paired.csv`。
- **R5：最终基线比较与统一计时边界**：`results/baseline_strength_20261011/formal_delivery_v1/REPORT.md`；`experiments/baseline-delivery-20261012/README.md`；`RESEARCH_STATUS.md`。
- **R6：熵编码、候选、资源、N2048交付索引**：`results/wcl_evidence_closure_20261009/README.md`及其明确链接的完成报告。
- **R7：figures4papers风格参考**：仓库`ChenLiu-1996/figures4papers`中的`scientific-figure-making/SKILL.md`与`references/design-theory.md`；其中示例尺寸不是WCL强制值。
- **R8：源码无噪声参考**：`reports/token_channel_efficiency_20260923_source_A.md`；数据兼容性仍须按本文件S01核对。

