# 数据版本清单

本批仅重绘图2 A/B。起始 HEAD 为 `9e59f62e6b803b79564d57beb905d59dbe1e1ece`，
与 T1-C 正式交付一致；没有更晚的科学结果需要合并。
已读取当前 `RESEARCH_STATUS.md`，其中较旧的“未运行/等待”段落按历史记录处理。
每个实际输入的路径、字节数和 SHA-256 在 `input_manifest.json`。

| 数据/范围 | 本轮采用版本 | 处理与限制 |
|---|---|---|
| common500：19 dB，熵码完整/部分尺度 | `results/t1c_m10_test_20261016_v1/delivery/summary_19dB.csv` | 两家族各7指标，共14个均值和原区间替换；主图只展示部分尺度的4项 |
| common500：4/10 dB，原四指标；其他方法19 dB | `results/entropy_main_common500_20261013/analysis_v1/summary_all_methods.csv` | 冻结结果与区间原样复用，不重评分、不重算区间 |
| common500：DISTS/DreamSim/MS-SSIM | `results/quality_metrics_supplement_20261014_v1/common500/summary.csv` | 已完成补评分；两个熵码家族19 dB的三项由T1-C覆盖 |
| common500逐源一致性 | 原 `analysis_v1/source_means.csv`、补充 `source_noise_means.csv`、T1-C `source_means_19dB.csv` | 每个方法/工作点/指标保留500源、每源原3噪声；仅验证现成逐源均值与汇总相符 |
| T1-C实际终点/容量 | T1-C `actual_fallback_counts.csv`、`m10_source_lengths_test600.csv` | 文件已读且绑定；m10可装common500为6/500，new100为1/100；不根据均值猜测分布 |
| new100 | T1-C `summary_19dB.csv`、`source_means_19dB.csv`中的new100行 | 保持独立100源、原四指标；未混入本次common500图，未补new100三项指标 |
| cal1000 | `results/t1b_m10_calibration_20261016_v1/delivery/` | 冻结策略与校准证据；不作为测试集性能或主图数据 |
| Kodak24 | `results/generalization_kodak_final_20261013/analysis_v1/` | 单独24个固定RGB256中心裁剪、4/10/19 dB；不混入common500 |
| fixed16 | `results/submission_closure_20261016_v1/fixed16/`，原 `results/hifi_final_fixed16_20261012/` | 独立development16、N1024、13 dB；后续图3仍使用raw列，本批不制图库 |
| 共享H800计时 | `results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/` | 已完成的旧版本计时；不得标为m10修正版成本，本批不重画表II |

## 图2的显示与统计

主图仅五方法：NeST-Com (entropy-coded)、NeST-Com (raw)、Latent-space JSCC、
SwinJSCC (adapted)、Adaptive BPG + LDPC。内部代码与显示名在 `methods.json`；
P100k/Swin200k放图注，不放图例。熵码主方法明确对应PARTIAL，不混用WHOLE。

每点是500源的均值，先按原协议在每源内平均3次噪声，含全部失败/灰图分母。
误差棒为已经发布的逐源bootstrap点态95%区间（10,000次，seed2026100701）。
本轮bootstrap调用为0；不把两条边际区间相减当配对区间。若后续画配对图，使用
已存 `paired_19dB.csv` 或对应原配对表，不从本图反推。

Prediction agreement 是重建与原图的ConvNeXt top-1预测一致率，原比例乘100显示。
它不是有标签分类Accuracy。LPIPS保持原LPIPS-Alex；DreamSim保持已登记ensemble
及其预处理/权重，未改分支。4/10/19 dB使用真实横坐标，只连接实测点。

## 缺口与解释边界

- 图2的60个均值及120个区间端点全部存在，无补零或虚构。
- RAW_WHOLE没有已完成DISTS/DreamSim/MS-SSIM；数据表中保留缺失，不能用于填表I。
- new100的三项新增指标未在T1-C执行；本轮不扩展它们。
- 原common500 WHOLE码流来自较早4090环境，与现H800冻结实现存在字节差异。
  旧版到修正版的变化不能全部归因于m10；旧结果完整保留。
- 19 dB两家族修正版是独立的新结果版本；已有H800计时不随图中点自动变成新版计时。
- 图1/图3/表I/表II/补充图仍等待样式选择；本次不以预览充当它们已经交付。
