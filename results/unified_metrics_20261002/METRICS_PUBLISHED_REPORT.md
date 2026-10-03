# 冻结结果的统一补充指标

原图ResNet50 ImageNet1K V2 top1准确率：0.7800，源配对bootstrap95%区间 [0.7000, 0.8600]。分母为100张原图，每图只计一次。

| 研究 | 完整方法/范围组 | 重建评分帧 |
|---|---:|---:|
| M1 | 194 | 58200 |
| M1_RATE | 108 | 10800 |
| M2_ACTUAL | 100 | 30000 |
| M2_ORACLE | 180 | 48000 |
| N1024 | 150 | 45000 |
| N512 | 150 | 45000 |

有噪组先平均同一源的三次噪声，再对100个源配对抽样10,000次，seed20261002。clean oracle及M1源编码率曲线都按100源×一次seed0独立分组，不复制成三噪声。300次重建不是300张独立图像。区间是重复使用这100张development图上的描述性区间，不含训练种子不确定性，也未做多重比较校正。

PSNR、LPIPS-Alex和原dino_cosine（DINOv2 ViT-S/14）来自封存原表。新增dinov2_vitl14_cosine独立使用DINOv2 ViT-L/14标准无registers、1024维CLS余弦，不替换原S/14数值；两者使用同一224方形bicubic、align_corners=False及ImageNet归一化。补充指标没有反向选择m/K/lambda/alpha、checkpoint或策略。逐帧表保留原始字段及replay误差，分析只读取评分。

| 补充指标 | 方向 | 说明 |
|---|---|---|
| OpenAI CLIP ViT-L/14 image cosine | ↑ | 重建图与原图的图像embedding余弦，不是文本相似度 |
| DINOv2 ViT-L/14 image cosine | ↑ | 新增标准无registers模型的1024维CLS余弦；原DINOv2 ViT-S/14单列保留 |
| DISTS | ↓ | 以冻结VGG16特征度量结构与纹理差异 |
| DreamSim default ensemble | ↓ | 实际manifest登记DINO/CLIP/OpenCLIP ViT-B/16与适配权重 |
| MS-SSIM | ↑ | 多尺度结构相似度，使用登记的数据范围与实现 |
| ResNet50 true-label top1 | ↑ | 重建预测与真实ImageNet标签一致的比例 |
| ResNet50 source-prediction agreement | ↑ | 重建预测与原图预测一致，不能代替真实标签准确率 |

D_C显式标为label_conditioned，不纳入主要结论；D0是参考解码器结果。M1_RATE源编码率曲线在同一m/K下比较entropy与raster/random/oracle，属于源端参考，不代表付费有噪链路。M2免费oracle与实际付费链路按scope分开，oracle结果不能证明实际链路收益。header失败的灰图也完整评分，不按协议成功或新增指标有效性挑帧。

评测骨干不是全部相互独立：新增DINOv2 ViT-L/14与原S/14属同一模型家族，DreamSim复用DINO/CLIP家族；DISTS的VGG16与旧LPIPS的Alex不同。它们没有与编码、通信或重建共享骨干；但原DINOv2 ViT-S/14参与部分既有校准/门槛约束，不能称为从未用于参数选择的独立验证。新增CLIP/DINOv2 ViT-L/14/DISTS/ResNet/DreamSim/MS-SSIM没有进入本轮选模选参。旧连续模型使用LPIPS-Alex训练损失，因此该指标也不能当作完全独立的验证。实际关系与权重/变换见模型manifest和METRIC_PROTOCOL；多指标一致仍不能排除全部评测偏差。

**KID：DEFERRED_HOLDOUT。FID：NOT_EVALUATED。** 目前只有100张反复使用的源图；三噪声不会增加独立图像数。分布指标留待更多独立holdout，当前不以同源重复重建扩充分布样本。

完整输出：[每组汇总](metrics_summary.csv)、[逐源均值](table_shards/metrics_source_means_6e0f7a4f2087.index.md)、[源配对差](table_shards/metrics_paired_intervals_78f90a32e5c4.index.md)、[原图分类基线](source_classification_baseline.json)。

模型与预处理manifest SHA256：`851a92682162e2b2be6e4e53498c50c7c3a9a1d2ea9e559b13cb9215f658ea9f`。
