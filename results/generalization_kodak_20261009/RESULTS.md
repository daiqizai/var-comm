# Kodak24 frozen-method results

## 中文结论

- 这批结果支持特征一致性与源图预测一致性的改善，但不支持“全面胜出”。DINOv2-L 和 ConvNeXt 源图预测一致率相对三个基线、三个 SNR 的全部 9 项源级配对区间均严格高于零。它们是代理指标，不是真实语义标签准确率。
- 4 dB 必须保留的不利结果：本文相对潜空间连续 JSCC 的 LPIPS 差值为 +0.0267 [+0.0108, +0.0409]，正值表示更差；PSNR 相对连续 JSCC、BPG 和 Swin 分别为 -1.55 [-1.83, -1.25]、-3.14 [-3.50, -2.77]、-3.15 [-3.84, -2.22] dB。
- 相对 BPG，本文三个 SNR 的 PSNR 均更低，同时 LPIPS、DINOv2-L 和源图预测一致率均更有利。相对潜空间连续 JSCC，10 dB 仍有 PSNR 损失，19 dB 四项指标的点态配对区间均有利。
- 19 dB 相对 Swin 的 PSNR 差值为 -0.84 [-1.48, +0.13] dB，区间跨零，不能据此宣称胜出、持平或等价。Swin 的 19 dB 属于训练及校准范围外工作点。
- 只有 24 张预先固定的中心裁剪图；每图三个噪声先平均，随后按来源图配对。不同方法没有共享同一次实际接收观测。不能扩展为原尺寸 Kodak、任意域泛化或预训练集完全无重叠的结论。

## Manuscript-ready English

We evaluated all 24 Kodak images using one predetermined 256 by 256 center crop per image, N=1024, SNRs of 4, 10, and 19 dB, and three fixed noise realizations per source and method. All weights and source/channel policies remained frozen, and VAR completion used the unconditional null embedding. Across all three SNRs and all three reference methods, the source-paired 95% intervals were strictly positive for DINOv2-L similarity and ConvNeXt source-prediction agreement. Relative to latent continuous JSCC, the DINOv2-L gains were +0.2330, +0.2066, +0.2372, and agreement gains were +27.78, +33.33, +37.50 percentage points, respectively. These metrics measure feature and prediction consistency with the source, not ground-truth semantic accuracy.

The gains involve a pixel-fidelity tradeoff. At 4 dB, the proposed method had worse LPIPS than latent continuous JSCC (difference +0.0267 [+0.0108, +0.0409]) and lower PSNR than all three references. Its LPIPS was lower than that of adaptive-resolution BPG + LDPC and adapted SwinJSCC-80k at every tested SNR, and lower than latent continuous JSCC at 10 and 19 dB. PSNR remained below adaptive-resolution BPG + LDPC at all three SNRs and below adapted SwinJSCC-80k at 4 and 10 dB. At 19 dB, PSNR exceeded latent continuous JSCC by +0.87 [+0.57, +1.18] dB, while the difference from adapted SwinJSCC-80k was -0.84 [-1.48, +0.13] dB; the latter interval crosses zero and establishes neither a directional difference nor equivalence.

Intervals are pointwise percentile source-bootstrap intervals from one shared set of 10,000 resamples (seed 2026100901), after averaging three noises within each source. No old result was re-bootstrapped. Swin at 19 dB is outside its training and calibration range. The 24 fixed center crops provide limited cross-collection evidence, not full-resolution Kodak evaluation or proof of non-overlap with pretrained-model training data. The intervals are not multiplicity-adjusted and do not establish universal superiority.


## Single-method means and existing 95% intervals

All numbers below are formatted directly from `analysis_v1/summary.csv`. Agreement and its interval are multiplied by 100. No scientific value or interval is recomputed.

| SNR (dB) | Method | PSNR (dB) | LPIPS | DINOv2-L cosine | Agreement (%) |
|---:|---|---:|---:|---:|---:|
| 4 | Proposed: partial-scale digital + unconditional VAR | 18.59 [17.59, 19.62] | 0.2736 [0.2486, 0.2981] | 0.6489 [0.5841, 0.7126] | 58.33 [37.50, 79.17] |
| 4 | Latent continuous JSCC | 20.14 [19.17, 21.15] | 0.2469 [0.2278, 0.2660] | 0.4159 [0.3554, 0.4791] | 30.56 [15.28, 45.83] |
| 4 | Adaptive-resolution BPG + LDPC | 21.72 [20.55, 22.95] | 0.7442 [0.6828, 0.8041] | 0.1442 [0.0916, 0.2089] | 4.17 [0.00, 12.50] |
| 4 | SwinJSCC-80k (adapted) | 21.74 [20.40, 23.09] | 0.6064 [0.5578, 0.6536] | 0.2702 [0.1978, 0.3513] | 13.89 [5.56, 25.00] |
| 10 | Proposed: partial-scale digital + unconditional VAR | 20.74 [19.60, 21.92] | 0.1815 [0.1616, 0.2014] | 0.7779 [0.7439, 0.8119] | 59.72 [40.28, 79.17] |
| 10 | Latent continuous JSCC | 21.14 [20.09, 22.25] | 0.1987 [0.1800, 0.2173] | 0.5713 [0.5209, 0.6201] | 26.39 [12.50, 41.67] |
| 10 | Adaptive-resolution BPG + LDPC | 23.33 [22.02, 24.69] | 0.5979 [0.5334, 0.6636] | 0.3652 [0.2837, 0.4489] | 16.67 [4.17, 33.33] |
| 10 | SwinJSCC-80k (adapted) | 22.79 [21.35, 24.27] | 0.5536 [0.5040, 0.6011] | 0.4184 [0.3400, 0.4988] | 19.44 [5.56, 34.72] |
| 19 | Proposed: partial-scale digital + unconditional VAR | 22.42 [21.19, 23.68] | 0.1330 [0.1174, 0.1490] | 0.8510 [0.8220, 0.8783] | 62.50 [41.67, 79.17] |
| 19 | Latent continuous JSCC | 21.55 [20.45, 22.69] | 0.1850 [0.1661, 0.2037] | 0.6137 [0.5627, 0.6624] | 25.00 [8.33, 41.67] |
| 19 | Adaptive-resolution BPG + LDPC | 24.99 [23.56, 26.46] | 0.4704 [0.4119, 0.5300] | 0.5608 [0.4791, 0.6417] | 33.33 [16.67, 54.17] |
| 19 | SwinJSCC-80k (adapted) | 23.26 [21.76, 24.79] | 0.5341 [0.4831, 0.5835] | 0.5249 [0.4459, 0.6041] | 19.44 [6.94, 34.72] |

## Source-paired differences and existing 95% intervals

Direction: proposed minus reference. Negative LPIPS favors proposed; positive PSNR, DINOv2-L, and agreement favor proposed. Agreement differences are percentage points (pp), not relative percentages. Conclusions use these paired intervals, not overlap between single-method intervals.

| SNR (dB) | Reference | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2-L | ΔAgreement (pp) |
|---:|---|---:|---:|---:|---:|
| 4 | Latent continuous JSCC | -1.55 [-1.83, -1.25] | +0.0267 [+0.0108, +0.0409] | +0.2330 [+0.1796, +0.2875] | +27.78 [+11.11, +45.83] |
| 4 | Adaptive-resolution BPG + LDPC | -3.14 [-3.50, -2.77] | -0.4706 [-0.5183, -0.4227] | +0.5047 [+0.4445, +0.5616] | +54.17 [+33.33, +75.00] |
| 4 | SwinJSCC-80k (adapted) | -3.15 [-3.84, -2.22] | -0.3328 [-0.3795, -0.2876] | +0.3787 [+0.3222, +0.4359] | +44.44 [+23.61, +65.28] |
| 10 | Latent continuous JSCC | -0.40 [-0.69, -0.07] | -0.0172 [-0.0254, -0.0097] | +0.2066 [+0.1707, +0.2439] | +33.33 [+9.72, +56.94] |
| 10 | Adaptive-resolution BPG + LDPC | -2.58 [-2.94, -2.24] | -0.4164 [-0.4691, -0.3662] | +0.4127 [+0.3335, +0.4882] | +43.06 [+20.83, +63.92] |
| 10 | SwinJSCC-80k (adapted) | -2.05 [-2.70, -1.10] | -0.3721 [-0.4128, -0.3319] | +0.3595 [+0.2959, +0.4208] | +40.28 [+18.06, +61.11] |
| 19 | Latent continuous JSCC | +0.87 [+0.57, +1.18] | -0.0520 [-0.0580, -0.0459] | +0.2372 [+0.1921, +0.2848] | +37.50 [+16.67, +58.33] |
| 19 | Adaptive-resolution BPG + LDPC | -2.57 [-2.90, -2.26] | -0.3373 [-0.3847, -0.2914] | +0.2901 [+0.2224, +0.3640] | +29.17 [+8.33, +50.00] |
| 19 | SwinJSCC-80k (adapted) | -0.84 [-1.48, +0.13] | -0.4011 [-0.4417, -0.3596] | +0.3261 [+0.2579, +0.3996] | +43.06 [+19.44, +65.28] |

## Execution and failure accounting

The four local actual completion receipts confirm 864 completed reconstruction frames and 1080 new PHY decoder calls: 432 proposed, 432 BPG, 216 Swin, and zero latent continuous JSCC. The proposed receipt records 216 audited VAR null-embedding calls, no true labels, and no training. BPG records two child exits of zero, 216 decoded frames, zero unresolved ledger attempts, no calibration, and an unchanged original root ledger. Actual scoring completed 580 unique reconstruction evaluations plus 24 source preparations; exact same-source reconstruction pixels were reused without dropping any frame. The metric completion, itself hash-bound by the analysis completion, records 3456 metric rows. The analysis records 48 summaries, 36 paired contrasts, 1152 source means, and exactly one new-data bootstrap draw array.

Execution receipts: [proposed](actual_receipts/VAR_UNCONDITIONAL.json), [latent continuous JSCC](actual_receipts/P1024.json), [BPG](actual_receipts/BPG_ADAPTIVE.json), [Swin](actual_receipts/SWIN80K.json), and [metrics](metrics_v1/completion.json). The independent review reads these top-level actual counts; it does not claim every remote waveform or reconstruction archive has been copied locally.

The independently read `status_counts.csv` retains four body-CRC-rejected frames at 10 dB for the proposed method (`BODY_CRC_REJECT_KEEP`), together with all 68 ordinary received frames at that point. All 72 frames at the other proposed points and all reference points are present. All 216 BPG frames are decoded. These status statements do not replace detailed per-frame receiver evidence.

## Read-only audit and provenance

- Verified all 48 summary and 36 paired rows against the actual analysis completion hashes; exact method/SNR/metric coverage, 24-source/three-noise/72-frame fields, finite ordered confidence intervals, and VAR-minus-reference means agree with the CSVs.
- Every DINOv2-L and agreement paired interval excludes zero in the favorable direction. The low-SNR adverse LPIPS/PSNR results and the Swin 19 dB PSNR interval crossing zero are explicitly retained.
- This writer only reads completed CSVs and formats prose/tables. It performs no model call, channel simulation, bootstrap, policy selection, or modification of original result files.
- Independent PNG inspection found complete labels, legends, intervals, negative differences, and correct percent/percentage-point units. The main plot preserves the hollow Swin 19 dB marker and dashed 10-to-19 dB segment.

- `summary.csv` SHA-256: `4c6188be738bf54e020b1aa60a329c75b8b1b46d42ab9c71ce8ec65a71ed9ee5`
- `paired.csv` SHA-256: `4d39d64cee3be75e1419d466ee59f89676064792679b65351567838d45057d55`
- `status_counts.csv` SHA-256: `decbb9db40daee995ad886ac1593601ad40d9065bc3247b065bb76ff1532278d`
- `metrics_v1/completion.json` SHA-256: `5e2b88980effa4cce45f2e5ea78ebaa5660a812d706e8512313ffc0dbac205dd`
- `actual_receipts/VAR_UNCONDITIONAL.json` SHA-256: `7d0c9f6e3a75ca0f66065c4af101b1805801c14f8caf970aa5be3074e4d11b38`
- `actual_receipts/P1024.json` SHA-256: `e89465d8a8f3eb881902d432e89b528af346e868b919c51b5e05cf6b6f354625`
- `actual_receipts/BPG_ADAPTIVE.json` SHA-256: `06390f6e65bebba441d259dff09a34e77b74690de89a499c58bdf61f8c001155`
- `actual_receipts/SWIN80K.json` SHA-256: `6e868e988a282d3ae44428e4c298bcfa6c43f005e94bb7203ce410f00481096c`

Recreate only these prose files: `python experiments/generalization_kodak_20261009/write_results.py`.
