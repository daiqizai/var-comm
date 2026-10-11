# Closed full-calibration milestones and bottleneck diagnostic

Both actual child processes exited zero. Milestones contain exactly 96,000 logical frames (91,800 new, 4,200 verified reuses); the bottleneck diagnostic contains 900 frames (300 new, 600 verified reuses). Full frame grids, source identities, failures, duplicates and input/output hashes passed. There are no rejected headers or fixed-fallback frames in these diagnostic grids; no row was filtered.

Milestones use all 1,000 original calibration sources and all six displayed SNR values at N1024. P uses three original noises; Swin uses its original one-noise calibration protocol. The first 100 calibration sources form the separate bottleneck diagnostic. These descriptive means do not have bootstrap intervals and do not enter model selection or stopping. They are not holdout results.

## Selected checkpoint minus parent (descriptive)

| Branch | SNR (dB) | Delta PSNR (dB) | Delta LPIPS | Delta DINOv2-L | Delta agreement (pp) |
|---|---:|---:|---:|---:|---:|
| P | 1 | 0.229997 | -0.016745 | 0.034835 | 3.566667 |
| P | 4 | 0.285995 | -0.017250 | 0.039824 | 3.266667 |
| P | 7 | 0.319738 | -0.017275 | 0.041270 | 5.600000 |
| P | 10 | 0.341682 | -0.017152 | 0.043434 | 4.600000 |
| P | 13 | 0.354786 | -0.017192 | 0.043830 | 5.500000 |
| P | 19 | 0.366365 | -0.017511 | 0.043461 | 5.466667 |
| Swin | 1 | 0.613092 | -0.018891 | 0.027073 | 2.500000 |
| Swin | 4 | 0.757846 | -0.026527 | 0.039777 | 4.800000 |
| Swin | 7 | 0.828494 | -0.030611 | 0.061517 | 2.500000 |
| Swin | 10 | 0.922299 | -0.035918 | 0.059169 | 4.200000 |
| Swin | 13 | 1.011944 | -0.040580 | 0.061512 | 3.300000 |
| Swin | 19 | 1.087604 | -0.046382 | 0.065417 | 2.900000 |

The selected checkpoints improve all four displayed parent-relative means at all six SNR values. This does not imply monotonic late-training improvement: P DINOv2-L and agreement decrease at some SNR values between 80k and 100k; Swin agreement also decreases at several SNR values between 160k and 200k. The full milestone curves retain these changes. No test-metric-based reselection is made.

P stops under the registered empirical plateau rule. Swin reaches its additional-update budget while still improving its selection objective; convergence is not claimed. Swin changes include fresh optimizer/sampling/RNG initialization from 80k weights and expanded 1--19 dB coverage. Its original 80k/19 dB point remains outside the old training/calibration range.

## Bottleneck interpretation

The full F-to-Dc path is a free-latent reference, not an equal-budget achievable upper bound. Both no-noise communication paths retain the real N1024 bottleneck and receiver SNR conditioning at 19 dB. Their gap to the full-latent reference suggests a substantial loss in the tested communication bottleneck; it does not establish a capacity limit for all continuous JSCC architectures. Agreement can fluctuate with noise and is not source-label classification accuracy.

The three diagnostic figure groups were exported as vector PDF, editable SVG and 600 dpi PNG. The publication uses a fixed copy under `figures/` so later automatic rendering cannot mutate published bytes. Reproduce the source figures with `python experiments/baseline-delivery-20261012/plot_evaluation.py diagnostics`; this snapshot packaging entry is once-only.

Formal common500 and existing-new100 scoring and its gated source-paired statistical successor remain separate. Fair timing is pending the exclusive resource window and timing-entry qualification.
