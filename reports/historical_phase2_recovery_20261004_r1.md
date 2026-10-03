# 历史已完成方法：补充独立指标

本登记队列已完成 12 个研究、201 个方法/资源组、56700 行评测。

本恢复版本保留七项R2和四项R3完整研究，Phase2前81源经逐项审计后只读继承，仅新增后19源855行。Phase2纯连续旧实现与continuous-grid的浮点数值差异显式披露；新指标均来自实际grid图像，严格grid重放容差不变，未平移任何新指标。

**本报告完成用户2026-10-03筛选后的范围：必补项目先完成，再补第5、6、8项的指定工作点。**
完整范围以冻结 coverage_manifest 为准。第7、10、12–16、18–20、22项及新低带宽外部模型训练已排除；不能把这些项目写成已补评。

## 覆盖与结果

[全部指标与区间](../results/historical_phase2_recovery_20261004_r1/summary.csv) · [源图均值](../results/historical_phase2_recovery_20261004_r1/per_source_means.csv) · [逐组覆盖](../results/historical_phase2_recovery_20261004_r1/coverage.csv)

[质量—资源曲线与逐点来源](../results/historical_phase2_recovery_20261004_r1/figures/curve_provenance.json)

| 研究 | 方法数 | 资源/条件组 | 源图数 | 原始结果行数 |
|---|---:|---:|---:|---:|
| FINAL_DIGITAL_16QAM | 6 | 30 | 100 | 9000 |
| FINAL_DIGITAL_QPSK | 6 | 30 | 100 | 9000 |
| FINAL_P2048_P3060 | 2 | 10 | 100 | 3000 |
| FINAL_P4084_SELECTED_SEEDS | 2 | 10 | 100 | 3000 |
| FINAL_PHASE2_N4084_DIGITAL | 2 | 10 | 100 | 3000 |
| LEGACY_N3060_FINAL | 3 | 15 | 100 | 4500 |
| OPTIONAL_H6_13DB | 2 | 2 | 100 | 600 |
| OPTIONAL_PHASE2_MAIN | 3 | 15 | 100 | 4500 |
| OPTIONAL_RX_STEP2_A | 4 | 8 | 100 | 2400 |
| OPTIONAL_RX_STEP2_B | 4 | 8 | 100 | 2400 |
| SELECTED_EXTERNAL_AUTHORS | 9 | 45 | 100 | 13500 |
| SELECTED_NOISELESS_REFERENCES | 18 | 18 | 100 | 1800 |

## 比较口径

所有旧模型、策略、源图、噪声及原始行保留；新指标列使用 new_ 前缀。物理信道 SNR 和等效 latent SNR 分组，不能把数值相同当作同一信道条件。
区间先在每张源图内平均原始噪声重复，再按源图重采样 10,000 次。配对比较只执行登记的对照，要求源图和参考像素完全一致。区间不包括训练种子不确定性。
分类主结论限无类别、非 oracle、非参考的主结果；带真实类别、D0、oracle 及诊断输出保持标记。
新增指标为 DINOv2 ViT-L/14、OpenAI CLIP ViT-L/14 图像余弦、DISTS、DreamSim、MS-SSIM，以及独立 ResNet-50 V2 的真实标签准确率和原图预测一致率。原有DINO是DINOv2 ViT-S/14，LPIPS是AlexNet v0.1，版本记录在每行元数据中。
已有 F 恢复误差和错配特异性指标在存在时纳入汇总；缺失项保持缺失。

无信道参考每种输出只有100张源图，不复制三份噪声。m8无类别补全为本轮新增确定性推断，单列其来源；其余旧指标逐帧保留并核验。
历史N2048/N3060/N4084数字最终策略为付费类别D_C；早期N3060使用D0。它们不能标成当前同Dc、无类别D_U曲线。SwinJSCC的作者假设边信息与共同计费口径分开。

## 原六项研究索引

N512、N1024、M1、M1_RATE、M2_ORACLE、M2_ACTUAL 的原评测见 [原补充指标报告](../results/unified_metrics_20261002/METRICS_REPORT.md)。本报告仅保存其文件与校验值索引，不重算、不假合并结果。

本轮训练更新为 0，策略选择更新为 0；未启用新 holdout。KID 留待最终 holdout，FID 未评估。
