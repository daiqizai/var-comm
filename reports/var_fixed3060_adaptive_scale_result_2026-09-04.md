# 固定 3,060 complex uses 的自适应 VAR scale/FEC 结果（2026-09-04）

## 设计

所有模式固定使用 `6,120 coded-bit slots = 3,060 QPSK complex channel uses`，复符号功率严格为2，并在每个 `(image,SNR)` 上共享同一 AWGN realization。只改变真实传输的 VAR 截止尺度和卷积码 rate matching：

| 模式 | information bits | 有效码率 `B/6120` | rate matching |
|---|---:|---:|---|
| m6 | 1,092 | 0.17843 | rate-1/2 mother code 后均匀 repetition，soft combining |
| m7 | 1,860 | 0.30392 | rate-1/2 mother code 后均匀 repetition，soft combining |
| m8 | 3,060 | 0.50000 | 原生 rate-1/2 mother code |
| m9 | 5,088 | 0.83137 | rate-1/2 mother code均匀 puncturing，puncture 位按 zero evidence 解码 |

映射采用 deterministic half-bin 分配，不发送 mode side information。正式协议实现时必须把少量 mode/class side information 一并计费；当前实验用于验证核心机制。

正式输出：`outputs/analysis/ANALYSIS-VAR-FIXED3060-ADAPTIVE-SCALE-IMAGENET100-001/`

## 可靠性结果

| SNR | m6 BLER/BER | m7 BLER/BER | m8 BLER/BER | m9 BLER/BER |
|---:|---:|---:|---:|---:|
| 1 | 0.01 / 9.16e-6 | 0.17 / 5.16e-4 | 1.00 / 3.78e-2 | 1.00 / 4.79e-1 |
| 4 | 0 / 0 | 0 / 0 | 0.02 / 1.63e-5 | 1.00 / 1.96e-1 |
| 7 | 0 / 0 | 0 / 0 | 0 / 0 | 0.31 / 7.47e-4 |
| 13 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 19 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

结果符合预期机制：强 repetition 使 m6/m7 在低 SNR 很稳；m8 在 1 dB 出现 cliff；高码率 punctured m9 在 1/4 dB 完全不可用，7 dB 已接近可用，13 dB 起无误码。

## Oracle envelope 与冻结候选策略

以“最高 DINO → 最低 LPIPS → 最高 PSNR → 较小 m”作为预注册 primary 规则，四项指标在每个测试 SNR 上恰好选择同一个模式：

| SNR | 最优 m | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 18.0823 | 0.45278 | 0.23471 | 0.84796 |
| 4 | 8 | 19.6883 | 0.51218 | 0.17733 | 0.89050 |
| 7 | 9 | 21.2057 | 0.56931 | 0.13681 | 0.92082 |
| 13 | 9 | 21.2330 | 0.56974 | 0.13599 | 0.92359 |
| 19 | 9 | 21.2330 | 0.56974 | 0.13599 | 0.92359 |

m 随 SNR 单调为 `[7,8,9,9,9]`，且至少使用两个模式，预注册 gate 通过。由测试点中点得到的开发集阈值策略是：

- `SNR < 2.5 dB → m7`
- `2.5 ≤ SNR < 5.5 dB → m8`
- `SNR ≥ 5.5 dB → m9`

这是从当前100图开发集导出的候选 policy，不是已经在独立测试集验证的最终控制器。

## 相对固定 m8 的 paired 收益

差值为 `selected policy−fixed m8`：

| SNR | 选择 | ΔPSNR dB | ΔLPIPS | ΔDINO | 代表性 win rate |
|---:|---:|---:|---:|---:|---|
| 1 | m7 | +1.2357 `[+0.6897,+1.8389]` | −0.02994 `[-0.04744,-0.01364]` | +0.06554 `[+0.03492,+0.09930]` | PSNR/DINO 58/100，LPIPS 55/100 |
| 4 | m8 | 0 | 0 | 0 | 100 ties |
| 7 | m9 | +1.5181 `[+1.3887,+1.6551]` | −0.04053 `[-0.04409,-0.03710]` | +0.03016 `[+0.02297,+0.03766]` | 99/99/90 wins |
| 13 | m9 | +1.5454 `[+1.4204,+1.6750]` | −0.04135 `[-0.04469,-0.03803]` | +0.03293 `[+0.02606,+0.04048]` | 100/100/92 wins |
| 19 | m9 | +1.5454 `[+1.4215,+1.6736]` | −0.04135 `[-0.04464,-0.03808]` | +0.03293 `[+0.02597,+0.04056]` | 100/100/92 wins |

1 dB 的 SSIM 差 `+0.00712` CI 跨0，但 PSNR、LPIPS、DINO 均显著优于固定 m8。7 dB 即使 m9 BLER 为31%，平均 bit/token error 很小，仍在四项指标上成为最佳；这也说明只看 BLER 会过度悲观，必须同时看错误数量与图像后果。

## 核验与判断

- 4,000 行、4,000 个唯一 `(image,SNR,m,arm)` 键，100 个唯一 image ID。
- 全部模式均为6,120 coded slots、3,060 complex uses、复符号功率2。
- 每个 `(image,SNR)` 的四个模式共用同一个 noise SHA。
- 所有无误码 Digital/Oracle 重建的最大像素差为0。
- 逐图 CSV 独立重算全部 stored means 的最大误差为0。
- `per_image_metrics.csv` SHA-256：`b70337866ac491923567966c7cac49e63563237acd9b5c3a615af70667b899c4`
- `paired_selected_policy_vs_fixed_m8.csv` SHA-256：`2bf3dcb9661d0017e2a1f55562fa7ffb7b92576ed87c7ff7aee32526e4c83684`

本关明确支持“固定物理带宽下，根据 SNR 在真实尺度数量和 FEC 强度之间自适应分配”的机制。m6 在当前五个点从未最优；下一阶段不应继续扫更多开发参数，而应将 `[m7,m8,m9]`、两个阈值和更现实的 rate-compatible LDPC/Polar 实现冻结后，在至少1,000张新 ImageNet 样本上一次性验证。
