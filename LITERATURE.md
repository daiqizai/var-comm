# VAR 通信相关工作

## 2026-09-16：CSI不确定性转题前核查（仅准备）

`reports/tx_csi_uncertainty_literature_preparation_20260916.md`核对三篇正式原文：Singh/Tirkkonen，IEEE Communications Letters 2024（DOI `10.1109/LCOMM.2024.3451018`）；Hadji等，EURASIP JWCN 2021:64（`10.1186/s13638-021-01949-9`）；Yuan等，Sensors 2024:4005（`10.3390/s24124005`）。PDF、文本与SHA留在本轮`literature_preparation/`。前者已有条件SINR/backoff，第二篇已有分层图像联合适配与CSI更新延迟，第三篇已有channel-blind学习链和SNR反馈失配；不能把“加入CSI误差”本身当新贡献。

当前尚无足够明确的新机制差异。只登记有限资源、无重传、失败图像计入的最小验证草案；数字鲁棒策略需同等重新校准，学习对照需CSI-blind或误差分布匹配训练。固定混合三权重已完成并结束当前主线，见`reports/hybrid_weight_closure_convergence_20260916.md`；但这不自动将CSI草案变成已定稿的新题。不强制VAR，不启动新训练，也不声称穷尽最新文献或完成论文复现。

## 2026-09-15：数字生成基础＋源残差混合通信的直接先例

原文核对见`reports/hybrid_related_work_20260915.md`：HDA-DeepSC `2405.12580v2`、DiffCom/HiFi `2406.07390v2`、RDP-JSCC `2408.14127v1`，HTML与SHA在新实验`literature/`归档。数字基础＋连续残差、受接收观测约束的生成以及失真—感知取舍都已有先例。本轮固定分配/真实CRC错误/同参数可靠度控制仅是待验证设计，不宣布首创或超越这些原系统；没有运行其代码/权重。

## 2026-09-15：数字VAR候选主系统的收敛定位

本次实际阅读ARPC的ICLR 2026正式论文（作者公开PDF，SHA `b7a981b071e5da06d2ec762f6d5610d7cf3f4fa4a1f5619ecd8d5ac0a2b53eac`）、作者demo/sender/reciever公开源码，以及Ada-TokenCom `2608.28086v1`全文。详细差异、具体章节、代码边界与资料见`reports/convergence_report_20260915.md`第八/十一节及`outputs/CONVERGENCE-AUDIT-20260915/literature/`。

结论是现有前缀生成、next-scale熵编码、源率/MCS自适应已有直接先例，不能当作本项目首创。我们的单次固定N/E、明确header和所有失败计入图像质量，是需要继续验证的协议/问题差异，不自动等同新颖性。ARPC的GM是group-masked；Ada区分成功图像质量与含重传CBR，不能说它未计CRC或失败资源。固定commit本地归档遇网络超时，记录保留；未执行作者代码或下载模型，不宣称完整复现或穷尽新颖性审查。

交付前原文复核进一步确认：**Ada-TokenCom自身使用AR概率算术编码；固定14-bit是其vanilla TokenCom对照**。不把两者混淆。该点已修正进收敛报告，旧初稿标为superseded留存，不改任何实验数据。

2026-09-06节是从旧工作区归集的历史核查，不代表本次重新检索或完成论文复现。
后续VAR通信文献只在本文件更新，不再追加到旧项目的LITERATURE.md。

## 2026-09-14：HJSCC源码接入资格，而非性能复现

本次核对arXiv当前v5及作者仓库提交`bea5c6c7ec42ea62e5afac307d37f9e20129cc6d`，只下载8个小型源文件并核对Git blob/SHA256，不运行作者代码或下载权重。具体收发边界、变长CBR、TX归一化尺度、示例固定10 dB与实际RGB导出要求见`reports/hjscc_source_protocol_review_20260914.md`。

作者原文已讨论长度边信息及有/无反馈两种系统；现有示例并非本项目严格每图3060-use协议的即插即用基线。后续须做原口径回放和明确适配，不免费传布局/幅度信息，也不零样本剪裁成弱对照。HJSCC继续保留为待接入强系统，未声称本项目胜过它；这是小范围资格核查，不是全面最新文献审计。

## 接收重编码与创新量：原论文边界核查

作者信息更正：ISEC原论文作者为Changwoo Lee、Xiao Hu、Hun-Seok Kim（AISTATS 2023，PMLR 206），不是此前误写的Xiao Ma、Ashwin Ashok。已重新核对PMLR原始元数据`https://proceedings.mlr.press/v206/lee23c.html`及作者仓库；原文`https://proceedings.mlr.press/v206/lee23c/lee23c.pdf`。这是文献署名勘误，不修改任何冻结实验、源码SHA或性能结果。
§3.3–3.5已讨论重编码后的似然、codeword先验与迭代更新，并指出仅有重编码一致性不能纠正所有源误差。因此“y与重编码结果比较”“MAP＋先验”“反复修正”不是本项目原创。

DeepJSCC-f原文`https://arxiv.org/abs/1911.11174`研究输出反馈下的逐次精化；不能把其发送端可见的反馈信息免费放入当前无反馈系统。

当前新候选只在通信源特征域调用冻结小Encoder，视觉Decoder留在最终输出，检验固定阶段、多尺度/非多尺度及同计算预测特征控制。它尚未证明优于ISEC或任何强系统，不把这次局部文献核查当完整最近工作审计。另有检索题名与arXiv落地页不符的结果，未采用。

## 2026-09-12：HJSCC完整原文与强对照边界

本次核对arXiv当前版本记录（v5，2025-02-27），读取`https://arxiv.org/html/2408.16340v5`及作者仓库`https://github.com/zhang-guangyi/HJSCC`的README；不是性能复现、完整新颖性检索或源码/许可冻结。

- HJSCC已联合层级VAE、逐层条件JSCC与源/信道信息，研究有反馈和无反馈两种系统；“层级＋条件映射”不能作为本项目首次贡献。
- 原文§IV-A明确讨论码率指示/边信息费用；变长平均CBR不能直接当本轮每图固定3060 uses。后续对照还需完整N/E、控制开销和真实噪声口径。
- 原文§IV-D区分发送端完美信道输出反馈；当前无反馈链不得免费复用这项信息。作者提供预训练资源，但本轮未下载权重、训练或移植HJSCC。

它应作为后续合理强对照之一；先完成当前同WeTok训练与归因，不用论文表中的PSNR替代本地同协议测量。

## 2026-09-12：接收硬选择的ST近似核查

本次直接读取Bengio、Léonard、Courville的原论文 *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*，arXiv:1308.3432v1，2013-08-15，重点为第4节。检索工具未返回内容后，从arXiv原始PDF取得正文；PDF SHA256为`509eb5e5294d6a5cb730a239022a6bd3589ed54cd24c1d7482c1edf6237ed863`。

论文明确把identity直通视为有偏估计，也研究过乘sigmoid导数的变体；**其论文场景中不乘该导数反而结果更好**。因此当前有界均值-ST不能写成文献已证明优于identity，更不是本项目原创。我们这里只在已经观察到的WeTok接收logits膨胀与图像训练退化下，做同权重、同前向、配对训练的独立检验。

当前±1接口对应有界均值`tanh(a/2)`，native有界-ST仍是近似；连续分支则真正改变训练/评测Decoder输入，不能称为同一个hard-token模型或免费上界。具体边界见`experiments/wetok-comm-v2-20260912/docs/interface_protocol.md`。本次核查不是最新相关工作的完整审计，也没有证明接口修订具有通信新颖性。

## 2026-09-10：新方向的近邻工作与骨干接口核查

本次只读原论文、作者仓库和官方文档，服务于研究问题定位。没有下载权重、运行模型、训练、租卡或复现新性能；不是系统综述或新颖性认证。

| 原始工作/来源 | 本次核实的相关点 | 对当前方向的约束 |
|---|---|---|
| Kurka、Gündüz，*Successive Refinement of Images with Deep Joint Source-Channel Coding*；`https://arxiv.org/abs/1903.06333` | 已研究多层逐次精化及有/无反馈设置 | 渐进传输或层级发送本身不是本项目新颖性 |
| 同作者，*Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding*；`https://arxiv.org/abs/2009.12480` | 单一编码器、多个带宽预算、AWGN及慢衰落 | 不能把带宽自适应或prefix发送的存在直接当贡献 |
| Zhang等，*Learned Image Transmission with Hierarchical Variational Autoencoder*；`https://arxiv.org/html/2408.16340v1` | HJSCC联合层级表示与通信，利用先验信息确定传输维度；反馈情形独立定义，长度信息单列费用 | 保留近邻强对照；TX不免费读取RX状态，控制/反馈须计费 |
| XQ-GAN；`https://arxiv.org/html/2412.01762v1`；作者库`https://github.com/lxa9867/ImageFolder` | 作者库有MSVR10P2-4096与配套VAR-d17，标注362M；公开结果以rFID/gFID为主 | 是配套替代候选，不是本地PSNR保证；同骨干内部证明通信增量 |
| 作者配置`https://raw.githubusercontent.com/lxa9867/ImageFolder/main/configs/MSVR10P2-4096.yaml` | `product_quant: 2`、末级121 latent、网格`[1,1,2,3,3,4,5,6,8,11]`、`dinov2_enc_dec`与`semantic_guide: true` | 不能照搬原680 token/8160 bit接口；DINO存在骨干训练暴露，须补其他独立证据 |
| PyTorch官方DDP/FSDP2说明；`https://docs.pytorch.org/tutorials/beginner/ddp_series_theory`；`https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html` | DDP复制模型；FSDP可分片状态 | 多张卡显存不自动合成单副本容量；租赁规模由完整训练路径实测决定 |

XQ配置/README来自当前远端main，只用于接口调查，不能冒充已冻结的可复现实验版本。后续选定时还需锁定代码、tokenizer和生成器的确切SHA。
本次没有做显存/吞吐测速或租赁报价调查，不给出预计几天、一定几卡的承诺，也不把论文预训练设备规模直接换算成本地耗时。
当前候选是“在N/E/CSI约束下，条件接收与保护分配是否共同改善通信表现”；尚无证据将这个组合写成已确认的新论文贡献。
具体设计见`reports/multiscale_joint_direction_2026-09-10.md`，旧m8两阶段续训继续暂停。

## 2026-09-07：整帧有限前缀建议的文献待查登记

该建议后续已授权并完成本地原型，结果见`reports/whole_frame_prior_result_2026-09-07.md`。
以下保留当时文献待查登记，不把完成本地实验等同于完成原文或新颖性核查。
本轮web检索没有取得可用原文，终端直连arXiv和作者站点均连接被拒绝；不得登记成再次核实了论文或已读全文。
现阶段仅沿用下方已有VAR/LLM-Viterbi历史记录，不把它们当作本次重新核查。
待补：普通CRC辅助列表/S-LVA的原始文献定位、候选枚举与复杂度定义、误接受/未找到的协议区别。
附件中的“arXiv”未提供可辨认的具体引用，本轮不填写未经确认的CRC论文题名/作者或性能数字。
工程上仍按用户要求建立普通列表强对照，并实测完整搜索账本；本节不构成新颖性认证。

## 2026-09-07：逐尺度先验辅助译码的小范围核查

本机日期Asia/Shanghai。仅核对原始论文摘要和官方代码接口，不是系统综述或论文复现。

- **LLM-Viterbi: Semantic-Aware Decoding for Convolutional Codes**，Zhengtong Li等，
  arXiv:2604.19035v1，首次提交2026-04-21；原始来源`https://arxiv.org/abs/2604.19035v1`。
  摘要明确描述多候选Viterbi路径、周期性可靠性评估、微调ByT5，以及联合信道和语言似然的贝叶斯评分。
  因此“信道似然＋源先验”本身不能当作本项目原创；本项目应验证可信尺度之间更新上下文、
  尺度内复用概率表的实现价值。摘要中的增益属于其文本/卷积码设置，不能转移成VAR性能结论。
- **官方VAR**：`https://github.com/FoundationVision/VAR/blob/main/models/var.py`。
  `get_logits`与后续`sample_with_top_k_top_p_`分开；本轮读取完整logits，不调用采样截断。
  官方teacher-forced路径还提供直接的因果性对照。本次实际复现版本以运行snapshot的本地源码SHA为准，
  不用远端main冒充冻结版本，也不用上层VAR-MAP-GATE0仓库HEAD冒充上游VAR commit。
- **ARPC**作者仓库`https://github.com/Joanna-0421/ARPC`的说明再次确认next-scale生成概率用于熵编码。
  因此“同概率熵编码＋FEC”应是后续系统强对照，而不只是原始12-bit token＋弱FEC。
  这次没有下载、训练或复现ARPC，也没有证明双方预算或源码实现已对齐。

第一关本地log-loss结果只判断是否值得设计译码器，不是新颖性证明；后续仍需全文级机制和预算对照。

同日后续执行状态：已用冻结VAR-d16概率实现整token卷积MAP及实际算术编码＋FEC对照，
完成同观测与完整3060-use开发比较。机制门槛通过，但未胜过强整帧自适应，也未证明优于熵编码。
见`reports/next_scale_decoding_result_2026-09-07.md`。这不是LLM-Viterbi或ARPC的完整复现，
不能把本地两级64×64实现或CRC结果直接包装为已确认的新颖性。

## 2026-09-06：VAR接收端cell约束候选的相关工作核查

本轮只做小规模无训练机制gate，不能把“VAR能补全”或“自适应前缀/FEC”继续当作独占新意。
已直连作者仓库和arXiv API核对下列工作的标题与摘要，不下载模型或数据；没有运行这些论文的复现。

| 工作 | 原始来源 | 与本轮的关系 |
|---|---|---|
| ARPC, *Autoregressive-based Progressive Coding for Ultra-Low Bitrate Image Compression* | 作者库 `https://github.com/Joanna-0421/ARPC`，已读README摘要和训练说明；仓库的paper link仍为TODO | 确认多尺度token选择传输、next-scale补全及熵编码已有；其GM-BMSRQ和Infinity-2B微调不等于本项目官方VAR-d16/原tokenizer，不能声称直接完整复现 |
| Ada-TokenCom（2026-08-28） | `https://arxiv.org/abs/2608.28086` | 摘要确认AR token压缩/生成、源率与调制编码自适应和Lyapunov优化；具体ARQ、成功帧统计和资源预算口径仍须原文逐项核对，不先声称已统一协议 |
| SSG, *Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation*（2026-02-05） | `https://arxiv.org/abs/2602.05534` | 无训练生成侧guidance，强调相对粗尺度prior的高频semantic residual；若本轮先有独立收益，才值得补这个强control |
| AID-VAR, *Adversarial Error Correction for Visual Autoregressive Generation*（2026-05-24） | `https://arxiv.org/abs/2605.24843` | 使用判别器和轻量guidance injector修正跨尺度生成；“VAR推理纠错/跨尺度一致性”本身也不是空白，但不等同固定接收token的无训练cell约束 |
| DiffCom, *Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling* | `https://arxiv.org/abs/2406.07390` | 摘要确认接收观测驱动posterior sampling与confirming约束；用户方案所引“只降低一致性loss的风险”未在本轮做全文逐句核查，不作为已核实原文引语 |

官方VAR量化实现核查入口：本地冻结 `VAR-MAP-GATE0/third_party/VAR/models/quant.py`，
对应 `https://github.com/FoundationVision/VAR/blob/main/models/quant.py`。
本轮运行snapshot记录本地源码SHA，不能把远端main视为可复现版本。
以上只确认相关机制已有先例，不据摘要宣称完整复现、性能支配或正式新颖性审查完成。
