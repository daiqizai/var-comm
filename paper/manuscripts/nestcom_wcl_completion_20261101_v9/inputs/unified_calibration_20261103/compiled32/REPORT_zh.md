# 快速 Ada 的质量与对应计时

同一32源、N1024/10 dB。实际计时输出全部从保存的接收信号恢复并逐哈希匹配；新旧已收到的 token 完全相同。五次计时重复是同一事件，不按五个独立质量观测统计。

编译实现的码流变化会改变由码流派生的尾部种子，因此同时报告实际计时输出和固定旧种子的控制。后者仅改变接收数值实现；不是额外无线事件。原顺序实现继续用于本轮主校准，32源结果不替换common500。

| 对比 | 指标 | 源均值差 | 95%来源配对区间 |
|---|---|---:|---|
| compiled_timed-original_timed | psnr_db | -0.0383915 | [-0.4706207, 0.3921917] |
| compiled_timed-original_timed | lpips_alex | -0.0016766 | [-0.0052352, 0.0017737] |
| compiled_timed-original_timed | dreamsim | -0.0034898 | [-0.0083666, 0.0008154] |
| compiled_timed-original_timed | clip_image_cosine | -0.0010325 | [-0.0061383, 0.0037948] |
| compiled_timed-original_timed | dists | -0.0017546 | [-0.0044421, 0.0009006] |
| compiled_fixed_old_seed-original_timed | psnr_db | 0.0000000 | [0.0000000, 0.0000000] |
| compiled_fixed_old_seed-original_timed | lpips_alex | 0.0000000 | [0.0000000, 0.0000000] |
| compiled_fixed_old_seed-original_timed | dreamsim | 0.0000000 | [0.0000000, 0.0000000] |
| compiled_fixed_old_seed-original_timed | clip_image_cosine | 0.0000000 | [0.0000000, 0.0000000] |
| compiled_fixed_old_seed-original_timed | dists | 0.0000000 | [0.0000000, 0.0000000] |
| compiled_timed-compiled_fixed_old_seed | psnr_db | -0.0383915 | [-0.4706207, 0.3921917] |
| compiled_timed-compiled_fixed_old_seed | lpips_alex | -0.0016766 | [-0.0052352, 0.0017737] |
| compiled_timed-compiled_fixed_old_seed | dreamsim | -0.0034898 | [-0.0083666, 0.0008154] |
| compiled_timed-compiled_fixed_old_seed | clip_image_cosine | -0.0010325 | [-0.0061383, 0.0037948] |
| compiled_timed-compiled_fixed_old_seed | dists | -0.0017546 | [-0.0044421, 0.0009006] |

对应计时仅与 compiled_timed 质量配对：tx=114.55 ms, rx=2949.76 ms, total=3064.30 ms。这些为既有合格计时结果，本轮没有重跑计时。

差值方向：PSNR/CLIP-I越大越好；LPIPS/DreamSim/DISTS越小越好。完整有利、不利和跨零区间均保留。
