# 感知重建主线

摘要与引言：固定数字帧下的源图感知重建；next-scale 是表示选型，尺度内真实码长装入和边界 token 双重作用是本文设计；保留 VAR 原任务能力与通信效果的区别。

II：表示与数字帧 → 尺度内端点及独立终止实际码长 → 前缀约束重建。校准仍为真实 DINOv2 目标，不新增感知优化模块。当前付费 NULL 及历史 N1024 差异有补充定位。

III.A：六系统、实际训练预算/预训练、独立校准人群和目标；LPIPS/DreamSim/PSNR/CLIP-I；源内三噪声平均与来源配对。

III.B：WHOLE/PARTIAL 的感知收益和容量 → 固定 100 源粒度 → 有序残差/条件干预。Table I 同步 LPIPS。Eighth 的 DreamSim 区间跨零不代表等价或最优；计时未完成明确说明。

III.C：Fig.2 LPIPS/DreamSim 与 Table II PSNR/CLIP-I 完整展示六系统取舍。Ada、Latent 的优势集中报告；Fig.3 为固定 development examples。

III.D：N512 同模型感知收益 → Ada 比较 → 实际失败和共同成功条件诊断；总体均值仍含灰图。Table III 给 ΔLPIPS/ΔDreamSim。

结论：保留与正确使用更多源 token 的同模型感知证据；不宣称全面 SOTA、人类偏好最优、Quarter 最优或已验证 SNR 鲁棒。既有 E1/E2、N2048、失配和成本细节保留补充材料。
