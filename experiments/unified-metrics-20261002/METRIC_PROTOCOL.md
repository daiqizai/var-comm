# 冻结结果的统一补充指标

## 范围与运行顺序

本轮只重建封存方法的最终图像并补评价，不训练通信模型，不改 checkpoint、m/K、lambda、alpha、旁路、PHY、N/E 或原校准门槛。评分读取封存策略，不把新增指标反馈到选模选参。原 PSNR、LPIPS、DINO 与 F 误差逐帧字段保留，replay 误差另列。

覆盖原 N512/N1024 最终全部方法，包括 P、无类别 D_U、有类别 D_C、接收插件、Dc/D0 参考及已有诊断；方法一覆盖 development 政策、同 m/K 控制和实际源编码率曲线；方法二把 clean oracle、noisy oracle 与实际链路分开。正常有噪组是同一100源×3噪声×SNR1/4/7/13/19。clean oracle 与源编码率曲线均只有每源一次 seed0，独立登记，不拼作第四次噪声或伪造成100×3。实际链路若封存为 gate 未通过，则保留明确跳过回执，不编造评分组。

源编码率曲线记为 `M1_RATE`，`N=''`、`phy_family='source_only'`、`snr_db='clean'`、`noise_seed=0`，方法为 `rate_curve_<order>`，projection 为 `m{m}_K{q}`。每个冻结动作100源×1，`is_main_conclusion=false`。它比较源端同一m/K下的entropy−raster/random/oracle，不代表已支付头、FEC及信道代价的链路表现；oracle顺序只是参考，也不是严格图像质量上界。

**调度边界：** 当前原两方法的全部绑定源文件不热改。新增目录先保持未跟踪，因原 `common.source_bindings()`会纳入 `git ls-files` 的仓库 Python/C++/配置；提前提交会改变未来阶段登记集合。等待原两方法 completion 和 supervisor COMPLETE、GPU所有者释放后，再提交新源码、冻结新登记并评分。新评价只能使用自己的回执和结果目录。当前不增加 holdout，不恢复旧队列。

## 评价模型与输入

所有新模型使用同一原图及同一重建浮点 RGB；不先转8bit PNG再评分。FP32、eval、无梯度、无 AMP。确切权重 full SHA256、官方实现 pin、每个实现 Python SHA、软件版本与图像变换登记在 metric modelmanifest 中。下载和资格检查与正式评分分开，评分阶段不自动下载或换模型。

| 指标 / CSV 字段 | 骨干与口径 | 方向 |
|---|---|---|
| `clip_image_cosine` | OpenAI CLIP ViT-L/14，224分辨率；图像 embedding 余弦，不是文本相似度 | 高 |
| `dinov2_vitl14_cosine` | DINOv2 ViT-L/14标准无registers，1024维normalized CLS token余弦；与原S/14独立列 | 高 |
| `dists` | 官方 DISTS 的学习 alpha/beta、VGG16 ImageNet1K V1，原生256 RGB | 低 |
| `dreamsim` | 官方 default ensemble、已登记适配权重；DINO/CLIP/OpenCLIP ViT-B/16，224直接resize | 低 |
| `ms_ssim` | 登记的 pytorch-msssim 实现，data_range1，RGB，五尺度默认参数 | 高 |
| `resnet50_top1_label` | torchvision ResNet50 IMAGENET1K_V2；预测与真实0–999标签一致的0/1值 | 高 |
| `resnet50_top1_source_prediction` | 相同分类器对重建和原图的预测一致的0/1值 | 高 |

CLIP 使用短边 bicubic antialias resize224、center crop224与官方均值/方差；ResNet50 V2 使用官方短边232、bilinear及224 crop；DISTS 使用256输入及官方内部归一化；DreamSim使用其固定外层变换和内部归一化。tensor 路径未宣称与先量化再 PIL resize 逐位一致。具体实现与官方出处见 [MODEL_SOURCES](MODEL_SOURCES.md)，正式结果以实际 manifest 为准。已注册模型加载失败必须报错；未注册模型明确 `NOT_EVALUATED`，不静默填数。

已有 `dino_cosine` 精确标为 **DINOv2 ViT-S/14**，原值及 `dino_mismatched` 保留，不用其他 DINO 规格替换。新增 **DINOv2 ViT-L/14** 记为 `dinov2_vitl14_cosine`，是标准无registers模型、1024维normalized CLS token余弦。两个DINO列严格使用同一预处理：浮点RGB直接224×224方形bicubic、`align_corners=False`，ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`；L/14权重及实现pin/full SHA另登记，不复用S/14权重或把CLIP ViT-L/14当成DINO。原 `lpips_alex` 保持已有 LPIPS-Alex 及其注册权重。

## 批量评分资格与加速

原重建器的 batch、随机状态和数值 flags 不变。旧重建模型与全部新评价器加载后，正式源图评分前，使用一张解析生成的参考图及16张固定非恒定合成重建，检验 batch 1/2/4/8/16。这里不读取源图或 development 图像，也不选择通信方法或参数。

每个候选先完整预热一次，再完整重复计时三次，按每图耗时的中位数选最快合格者；相同耗时选较小 batch。五项浮点新指标均须相对逐张评分绝对误差不超过 `2e-5`；分类预测、两种准确率标志、原图分类标志与标签条件标志必须完全一致。CUDA OOM、进程峰值 reserved 显存超过显卡总容量的88%、或数值检查失败，都排除该 batch。显存统计包含仍在本进程中的旧模型。batch1必须合格，才能作为回退及剩余帧的基础单位。所有模型仍使用FP32，不启用AMP，不改指标公式；这不承诺不同 batch 的浮点值逐位相同。

`metric_batch_qualification.json` 标明 `synthetic_images=true`、`scientific_result=false`，记录候选、差值、三次耗时、显存、选中及合格 batch，并绑定评价器身份、modelmanifest SHA、资格检查及评价器源码 SHA、硬件、旧重建输入和数值 flags。登记与评分完成回执记录 `metric_batch_size`、`metric_qualified_batch_sizes`、`metric_batch_qualification_sha256`；结果目录保存回执的原字节副本。恢复执行必须匹配同一绑定，不能静默重测后换 batch。

每源的参考图特征及原图分类基线只准备一次。按精确RGB、参考图和评价器身份去重，最多缓存选中 batch 数量的待评分RGB；重复行保留原顺序，并在同一次评分完成后取得相同的新指标。已完成的标量评分在该源内持续缓存。最后不足整批时，只使用资格检查合格且不大于选定 batch 的2次幂批次拆分。正式评分异常直接停止，不临时改变登记设置；源的全部行与原重建一致性检查完成后，才写恢复检查点。

## 独立性与标签污染

“未共享骨干”指评价网络的实际参数和前向特征没有作为通信编码器、收发模块、VAR生成器或 Dc 解码器使用。它不等于所有指标互相独立，也不排除相同训练数据与模型家族的偏差。

- 旧连续训练的封存损失含像素 MSE、0.1 LPIPS-Alex 和0.01标准化 F 误差。LPIPS-Alex是优化目标，不能称独立验证。
- DINO 未作为上述训练骨干/损失；但原DINOv2 ViT-S/14参与部分既有插件校准约束，以及方法二的lambda/real-link gate保底约束。因此不能声称 DINO 从未用于模型或政策选择。
- 本轮新 CLIP、DINOv2 ViT-L/14、DISTS、ResNet50、DreamSim、MS-SSIM不参与原选模选参。新增DINOv2 ViT-L/14与保留的S/14属于同一模型家族，DreamSim自身包含DINO/CLIP家族；其初代DINO ViT-B/16不同于两种DINOv2，仍不能称全部评价器互相独立。
- D_C接收端已收到真实类别标签。即使类别位已支付传输开销，它的真实标签分类准确率也有此信息条件；每行 `label_conditioned=true`，`is_main_conclusion=false`。D0及参考输出同样不作为主要新方法结论。无类别方法是分类保存的主要比较。
- 原图分类正确率用100张原图各一次预测计算，并保留 `true_class_index`、原图预测和0/1正确标志。重建与原图预测一致不等同于真实标签正确。

不以新增某个分数最高选胜者；按封存方法、范围、预算、SNR列出全部指标和源配对差。免费 oracle 与实际付费链路不合并为一个性能表结论。

## 数据与身份接口

分析入口：`python analysis.py --results-dir <sidecar结果目录>`。它只依赖标准库与 NumPy，不导入旧实验、重建网络或评分模型。

输入：

- `metrics_per_frame.csv`：原值、新值和每帧身份。分组字段为 `experiment,scope,N,phy_family,snr_db,method,projection,control,output_role,decoder_id,label_conditioned`；无关字段用空字符串。
- 每帧另有 `source_id,source_index,preprocessing_id,noise_seed,image_sha256,reference_sha256,is_main_conclusion`。`modelmanifest_sha256`与`replay_parity_passed`可逐帧给出或由登记中的同一值覆盖全体。
- 分类字段为 `resnet50_prediction,resnet50_source_prediction,resnet50_top1_label,resnet50_top1_source_prediction,resnet50_source_top1_label`，不把真实类别传给重建器。
- `source_baseline.csv`：100个唯一原图；字段 `source_id,source_index,preprocessing_id,reference_sha256,true_class_index,resnet50_source_prediction,resnet50_source_top1_label`。
- `metrics_registration.json`：有序100源、完整 `expected_groups`、新指标 `metric_availability`、`modelmanifest_path/sha256`、原表 `input_tables`、封存策略 `frozen_policy_bindings`、原pipeline完成回执 `pipeline_completion_bindings`（后三项均路径→SHA）、`original_pipeline_complete=true`、`training_updates=0`。
- `metric_batch_qualification.json`：批量评分的合成资格收据；登记另记录其路径、SHA、选定与全部合格 batch，评分完成回执绑定同一原件及结果副本。
- 可在新增评分前登记 `contrasts`，给出 `group_A/group_B/name`。未给出时只按方法身份构建 P/whole/unguided 与熵同K控制等固定比较，不读取指标来筛选比较。

每组严格验证100源与预定三噪声/clean0，不接受缺行、重复、换原图/预处理、换模型manifest或未通过replay的行。范围以评分前冻结的 `expected_groups` 为准：方法二实际链路的 gate、不同SNR的projection可能只准入部分组，不要求把未获准的范围补成五SNR网格。登记本身必须来自封存政策及完成回执，不能看新指标后删组。灰图、CRC失败等真实输出也完整评分；这些帧不能因低质量被删除。模型未注册属于整个指标的明确状态，不用逐帧缺值筛掉失败图。

`source_baseline` 的分类正确标志必须与真实标签及原图预测一致；重建的两个0/1标志必须与重建预测、真实标签、同一原图预测一致。原表和封存策略在分析前再次核验 SHA。

## 统计口径

先对每个有噪组、每个源平均三次登记噪声。clean oracle及源编码率曲线只含每源一次seed0。然后对同一有序100源抽样10,000次，bootstrap seed20261002；所有指标/方法共用同一源索引抽样。配对差是 A 的源均值减 B 的源均值，并保留各指标高/低方向。原图分类基线也是按100源计算，不按重复帧扩大分母。

配对保证同一源图，并不自动保证跨旧研究相同物理噪声或波形。只有原登记的同观测控制才能另声称相同观测；本分析表不会凭相同seed编号作此断言。区间是重复使用development上的描述性95%百分位区间，不包含训练seed不确定性，未做多重比较校正。

输出 `metrics_summary.csv`、`metrics_source_means.csv`、`metrics_paired_intervals.csv`、`source_classification_baseline.json`、`METRICS_REPORT.md` 和含输入/输出SHA的 `metrics_analysis_completion.json`。统计不改变原主报告；统一补充表和协议作为独立结果交付。

## 分布指标与验收

**KID = DEFERRED_HOLDOUT；FID = NOT_EVALUATED。** 用户已要求把KID推迟至更多独立holdout。本轮只有100张反复使用的源；100×3次重建不是300个独立图像样本。这里是范围与样本解释的决定，不声称KID在数学上不能对小样本计算。

先通过独立CPU身份/网格/统计检查和真实权重评分资格；每张重建图通过封存replay parity后才能正式评分。CPU合成夹具必须显式 `--allow-synthetic` 且回执 `synthetic=true`，不作为科学质量。新源码在旧pipeline结束后正常提交并冻结，再评分、核对大表与报告、正常推送验证远端SHA。没有新校准搜索、重训练或新holdout。
