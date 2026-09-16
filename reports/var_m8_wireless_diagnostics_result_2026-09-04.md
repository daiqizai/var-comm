# VAR-m8 正式无线诊断结果（2026-09-04）

## 实验合同

- 总体：冻结的 ImageNet-100 开发集，SNR `[1,4,7,13,19] dB`。
- VAR-m8 source payload：255 个 token × 12 bit = `3,060 information bits`。
- PHY：K=7 rate-1/2 `(171,133)` 卷积码，零初态、开放终态、无 tail bits；严格得到 `6,120 coded bits`，QPSK 后占 `3,060 complex channel uses`。
- DeepJSCC：`6,120 real coordinates = 3,060 complex channel uses`。
- 标准复 AWGN：`SNR=E|X|²/E|W|²`；双方每实坐标均方为1、每复符号平均功率为2、每实噪声方差为 `1/gamma`。
- 每个 `(image,SNR)` 的 VAR 与 DeepJSCC 共用同一 6,120 维标准正态噪声 realization 和同一指标实现。
- 三条曲线：VAR-m8 Oracle、VAR-m8 Digital、DeepJSCC-3060。

正式输出：`outputs/analysis/ANALYSIS-VAR-M8-WIRELESS-DIAGNOSTICS-IMAGENET100-001/`

## 图像质量

| SNR | 方法 | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | DINO ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | VAR Oracle | 19.6876 | 0.51213 | 0.17733 | 0.89066 |
| 1 | VAR Digital | 17.4749 | 0.45345 | 0.25570 | 0.79646 |
| 1 | DeepJSCC-3060 | 22.8294 | 0.61537 | 0.42286 | 0.31059 |
| 4 | VAR Oracle | 19.6876 | 0.51213 | 0.17733 | 0.89066 |
| 4 | VAR Digital | 19.6873 | 0.51215 | 0.17727 | 0.89063 |
| 4 | DeepJSCC-3060 | 24.4311 | 0.68039 | 0.34782 | 0.46995 |
| 7 | VAR Oracle/Digital | 19.6876 | 0.51213 | 0.17733 | 0.89066 |
| 7 | DeepJSCC-3060 | 25.5909 | 0.72118 | 0.30734 | 0.57976 |
| 13 | VAR Oracle/Digital | 19.6876 | 0.51213 | 0.17733 | 0.89066 |
| 13 | DeepJSCC-3060 | 26.8745 | 0.75977 | 0.27595 | 0.69174 |
| 19 | VAR Oracle/Digital | 19.6876 | 0.51213 | 0.17733 | 0.89066 |
| 19 | DeepJSCC-3060 | 27.3497 | 0.77178 | 0.26649 | 0.72449 |

DeepJSCC 在所有 500 个 `(image,SNR)` 上都获得更高 PSNR。相反，VAR Digital 相对 DeepJSCC 在 LPIPS 上胜 `443/500`，在 DINO 上胜 `492/500`。因此当前固定 m8 的结论是清楚的 distortion–perception/semantic tradeoff，而不是某一方法全面支配。

## PHY 与 token 诊断

| SNR | post-Viterbi BER | BLER | token error rate | 首错尺度摘要 |
|---:|---:|---:|---:|---|
| 1 | 0.0397810 | 1.00 | 0.106078 | 100 帧均错；首错主要在 scale 2–6 |
| 4 | 0.00005882 | 0.04 | 0.0001961 | 4 个失败包全部首错 scale 8 |
| 7 | 0 | 0 | 0 | 无错误 |
| 13 | 0 | 0 | 0 | 无错误 |
| 19 | 0 | 0 | 0 | 无错误 |

1 dB 下 Digital−Oracle 为 `−2.2127 dB PSNR / +0.07837 LPIPS / −0.09420 DINO`，95% bootstrap CI 均不跨零，说明损失来自真实 PHY 误码。4 dB 只有 4 个失败包且错误只落在 scale 8，整体 Digital 与 Oracle 的四项均值差均接近0。7 dB 起全部无误码，Digital 与 Oracle 的重建像素最大差严格为0。

BLER 分层进一步确认：4 dB 的 96 个正确包中 Digital 与 Oracle 完全一致；4 个错误包的 VAR Digital 仍为 `20.5150 dB / 0.18299 LPIPS / 0.90844 DINO`，没有发生灾难性平均语义崩溃。1 dB 则是典型数字 cliff，所有帧都有错误，但 VAR prior 仍使平均 DINO 保持0.7965。

## 判断

1. **高 SNR 回放链正确。** 无误码时 `VAR Digital = VAR Oracle`，不存在 bit/token 序列化或回放错误。
2. **低 SNR 损失来自 PHY。** 1 dB 的质量下降与 BER/BLER/TER 同时出现，不是 VAR completion 自身随机变化。
3. **固定 m8 有明显 cliff，但仍有感知/语义价值。** 在相邻的 1、4、7、13、19 dB 点，VAR Digital 均保持低于 legacy MSE-DeepJSCC 的 LPIPS、高于其 DINO；PSNR 则始终较低。
4. 当前 100 张是开发机制集。正式论文主表必须冻结至少 1,000 张未用于阈值或权重选择的新样本。

逐图主 CSV SHA-256：`78506b3c4cf48fe19d319b5791db3c4fae4b4892cd74b076bd7a81d749c9fd44`。
