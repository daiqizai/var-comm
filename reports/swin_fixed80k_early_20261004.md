# SwinJSCC 80k 检查点评测：先行发布

已完成 N1024/N2048、1/7/13 dB、100 张源图 × 三个噪声的 1,800 帧 SwinJSCC 统一评测。包含 DINOv2 ViT-L/14 等全部 13 项指标及源图 bootstrap 区间、原登记 16 张图的 96 个重建单元。

[完整指标、区间和不同信噪比样例](../results/external_comparison_20261004/fixed80k_revision/swin_early_release/evaluation/PUBLISHED_REPORT.md)

模型固定为用户指定的第 80,000 步；实际训练在 81,551 步安全暂停，预算截断，不宣称收敛。这是 SwinJSCC 的先行发布；HiFi-DiffCom 和完整五方法对照尚未完成，不能据此宣称方法间优劣。
