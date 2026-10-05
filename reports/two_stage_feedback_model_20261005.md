# N1024 两阶段反馈表模型：完整参考集合 R1

本轮修正 FIXED 和 QCSI 的候选范围。正式四方法表采用完整参考集合；原报告、原结果和原两阶段/C8策略全部保留。本报告仍是冻结质量表与物理概率表推导的期望值，没有新信道译码、图像生成或模型训练。

## 为什么需要修正

原两阶段方法枚举全部现有首轮 PHY 与后缀组合，而原 FIXED/QCSI 仅使用原目录及补充布局。两边候选范围不同，旧差值可能混入候选集合的影响。R1 为两个参考加入全部 m4/m5/m6 首轮与合法后缀组合，保留原目录；使用同样的精确物理曲线、资源预算及允许的头部格式。

QCSI 在发送前获得付费的量化 SNR；两阶段只把同样的信息放到首轮后获得。在同一 h、理想 CSI、没有 ACK 的登记假设下，两阶段本身不增加信息。完整参考必须能够重现两阶段的公共动作树，因此检查每格构造集 QCSI 不低于 TWO_STAGE。这不要求独立 200 源的复核均值逐格单调。

C8 使用源图信息在冻结公共树字典中选择；其差值需要与完整 QCSI 同预算比较，不能沿用旧参考下的收益结论。

## 登记主点

N1024、导频 16、反馈 16、第二头部 68 符号、measured_headers。反馈 bit 数和首轮 m 仍由原 800 源 TWO_STAGE 均值选择；两阶段/C8的首轮、阈值、字典与逐源树 ID 不改。下面是相同 200 源的复核均值。

|SNR|反馈 bit|首轮 m|方法|完整参考 DINOv2-L|原口径 DINOv2-L|
|---:|---:|---:|---|---:|---:|
|4|3|4|FIXED|0.444166|0.374481|
|4|3|4|QCSI|0.489568|0.454334|
|4|3|4|TWO_STAGE|0.442022|0.442022|
|4|3|4|TWO_STAGE_C8|0.498496|0.498496|
|7|3|4|FIXED|0.457956|0.416154|
|7|3|4|QCSI|0.551031|0.525480|
|7|3|4|TWO_STAGE|0.485244|0.485244|
|7|3|4|TWO_STAGE_C8|0.523532|0.523532|
|10|3|4|FIXED|0.484075|0.467920|
|10|3|4|QCSI|0.616939|0.598615|
|10|3|4|TWO_STAGE|0.532850|0.532850|
|10|3|4|TWO_STAGE_C8|0.562736|0.562736|
|13|3|4|FIXED|0.549658|0.544296|
|13|3|4|QCSI|0.673066|0.660894|
|13|3|4|TWO_STAGE|0.584007|0.584007|
|13|3|4|TWO_STAGE_C8|0.611744|0.611744|

主点配对差值；区间仅描述表模型中的源图变化：

|SNR|比较|差值|源级 95% 区间|未知事件贡献范围|
|---:|---|---:|---|---|
|4|QCSI − FIXED|0.045402|[0.035619, 0.055638]|[0.045402, 0.045402]|
|4|TWO_STAGE − QCSI|-0.047546|[-0.056991, -0.038584]|[-0.047546, -0.047546]|
|4|TWO_STAGE_C8 − TWO_STAGE|0.056474|[0.047534, 0.066080]|[0.056474, 0.056474]|
|4|TWO_STAGE_C8 − QCSI|0.008928|[0.000152, 0.017863]|[0.008928, 0.008928]|
|7|QCSI − FIXED|0.093076|[0.082215, 0.104061]|[0.093076, 0.093076]|
|7|TWO_STAGE − QCSI|-0.065787|[-0.074548, -0.057568]|[-0.065787, -0.065787]|
|7|TWO_STAGE_C8 − TWO_STAGE|0.038288|[0.031465, 0.045711]|[0.038288, 0.038288]|
|7|TWO_STAGE_C8 − QCSI|-0.027499|[-0.036213, -0.018667]|[-0.027499, -0.027499]|
|10|QCSI − FIXED|0.132865|[0.122063, 0.143597]|[0.132865, 0.132865]|
|10|TWO_STAGE − QCSI|-0.084089|[-0.093292, -0.075178]|[-0.084089, -0.084089]|
|10|TWO_STAGE_C8 − TWO_STAGE|0.029885|[0.024789, 0.035307]|[0.029885, 0.029885]|
|10|TWO_STAGE_C8 − QCSI|-0.054204|[-0.064015, -0.044875]|[-0.054204, -0.054204]|
|13|QCSI − FIXED|0.123408|[0.116590, 0.130104]|[0.123408, 0.123408]|
|13|TWO_STAGE − QCSI|-0.089059|[-0.094452, -0.083662]|[-0.089059, -0.089059]|
|13|TWO_STAGE_C8 − TWO_STAGE|0.027737|[0.023747, 0.031913]|[0.027737, 0.027737]|
|13|TWO_STAGE_C8 − QCSI|-0.061322|[-0.066831, -0.055665]|[-0.061322, -0.061322]|

## 计费、头部与选择口径

- 四种方法均支付同一档导频费用；0 导频仅作诊断，8/16/32 仅改变预算，仍假设理想接收/反馈 CSI。没有有限导频估计误差结果。
- FIXED 不支付反馈，QCSI 与两阶段支付登记的同一档反馈符号数；QCSI 的反馈只计费一次。
- 所有首头部计费 68 符号。measured_headers 使用原 68 符号概率曲线；0 第二头由公共配置确定，16 第二头没有实测曲线。
- ideal_controls 中所有付费头部（包括首头）都假设可靠，仅作控制开销比较。
- 参考布局允许无第二头或当前格合法的第二头格式；每条路径都逐项核算，不能通过剪短码字或缩放 BLER 补出新物理曲线。
- 反馈只有 SNR bin，没有 ACK/NACK。第一轮拒收后，第二轮的实际发送仍付费；接收端只使用连续可信前缀。
- C8 整棵树的 ID 已在第一头部付费，先于未知反馈 bin 选定；800 源构造字典，200 源只在冻结字典内作允许的源端选择。

## 完整网格与覆盖审计

完整网格 10368 个方法行；按原 TWO_STAGE 构造集规则选 m 后 3456 行。288 行保持缺资产；参考扩展没有给缺失的 16 符号物理头部补造数值。

grid_summary.csv 与 selected_summary.csv 为最终完整参考口径。reference_changes.csv 给出每格两个参考的新旧差值；reference_dominance_checks.csv 记录 QCSI 构造集覆盖检查。candidate_coverage.json 保留候选计数、逐路径预算与原两阶段数组一致性审计。

matched_axis_effects.csv 单独改变导频、反馈位数、反馈费用或第二头费用，其他控制相同；每格仍允许在 800 源重新优化，所以不是单独的估计误差效应。完整网格全部保留，不以 200 源挑赢家。

## 区间、数值误差与实际复现缺口

200 源配对 bootstrap 共 5000 次，种子 2026100504。每源分数已对信道表积分；没有实际信道 Monte Carlo。区间不覆盖 BLER 表有限样本、插值、尾部、理想 CSI 或选模误差，不作实际链路或多重比较显著性声明。

未评分的错误接受等事件使用登记的 DINO 点值 0，并单列 [-1,1] 贡献范围；不能当成已测恢复质量。原两阶段/C8的数值积分复核见 primary_numerical_diagnostic.json；新 FIXED/QCSI 的同策略复核独立记录在 primary_reference_numerical_diagnostic.json。两者都不重新选策略，也不验证 BLER 插值假设。

部分尺度的熵序位置缓存仍可能缺失。本轮仅闭合表模型中的候选范围，不声称补齐实际发射和接收资产。C8 的源图相关质量评估需要发送端计算；sender_cost.csv 给出所需状态数量，在线时延没有测量。

## 冻结主点的缓存辅助指标

沿冻结策略和逐源树 ID 重加权原 PSNR、LPIPS、DINO、CLIP、DISTS、DreamSim、MS-SSIM 与 ResNet-50 指标。除原主指标 DINOv2-L 外，这些辅助指标不选择策略或树；没有新增神经评测。

auxiliary_primary_metrics.csv 报已评分状态贡献及未评分概率。仅 U=0 时给完整均值；分类指标未知范围 [0,1]，余弦 [-1,1]，其他指标不假设边界。辅助配对区间仍是表模型的源级差异。ConvNeXt、F 恢复误差在登记缓存中缺失，不填值。

- clip: OpenAI CLIP ViT-L/14 (224px), image-image cosine
- dists: DISTS official learned alpha/beta; torchvision VGG16 IMAGENET1K_V1
- resnet50: torchvision ResNet50_Weights.IMAGENET1K_V2
- dinov2_vitl14: Meta DINOv2 ViT-L/14 standard LVD-142M backbone, no registers, 1024-d x_norm_clstoken cosine
- dreamsim: DreamSim ensemble v0.2.0-checkpoints: dino_vitb16,clip_vitb16,open_clip_vitb16 + LoRA
- ms_ssim: VainF/pytorch-msssim MS-SSIM, RGB, data_range=1, five scales
- existing_dino: Retain registered dino_cosine=DINOv2 ViT-S/14 unchanged; dinov2_vitl14_cosine is a separate added column
- backbone_relationships: Communication encoder/VAR/Dc are separate from these evaluation backbones. DreamSim includes DINO and CLIP families; evaluation metrics are not mutually independent. D_C classification is label-conditioned and descriptive only.

## 交付与停止边界

原搜索、原报告、原源码保持不变。正式四方法结论以此 R1 完整参考报告为准；原主点单独保留为 historical_primary_summary.csv。这项候选闭包修正不改变旧 UEP/F1 停止边界或 A1/A2队列，也不启动真实 PHY、GPU、训练或新评测模型。
