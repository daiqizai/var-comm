# Interim calibration-only baseline evaluation

Completed while both training branches continued on separate GPUs. These fixed checkpoint snapshots are not the final selected models. No holdout image was evaluated and the new scores do not enter model selection, learning-rate decisions or stopping.

Population: the first100 sources in the original frozen calibration order, shared by all four methods. N1024; SNR1/4/7/10/13/19dB; noises4101/4102/4103;7200 retained method frames. Average the three repetitions within each source, then average100 sources. These descriptive means have no bootstrap intervals. This small diagnostic does not replace the complete1000-source milestone diagnostics or final500-source evaluation.

P40k and P80k use the same frozen Dc, original continuous1024-complex-symbol interface and noise construction. Swin80k and Swin150k regenerate their own channel masks, normalization powers, paid metadata and receive observations. Metadata failures retain the original fixed-gray output. Old Swin80k at19dB remains outside its training/calibration range. Swin150k includes the explicitly approved fresh Adam/order/RNG initialization, expanded1–19dB coverage and microbatch4/effective16; changes cannot be attributed solely to training duration.

The four metrics are float64-MSE PSNR, LPIPS-Alex, DINOv2 ViT-L/14 cosine and ConvNeXt agreement with the source-image prediction. Agreement is not classification accuracy. Frozen metric code, weights and preprocessing are reused. Source labels are not used. The diagnostic was checked against all overlapping original calibration MSE/header outcomes and P LPIPS values.

## Fixed checkpoint means

| Method | SNR (dB) | PSNR (dB) | LPIPS | DINOv2-L | ConvNeXt agreement (%) |
|---|---:|---:|---:|---:|---:|
| P40k | 1 | 18.7875 | 0.27496 | 0.25359 | 39.00 |
| P40k | 4 | 19.6923 | 0.22766 | 0.39921 | 53.33 |
| P40k | 7 | 20.3070 | 0.19996 | 0.48106 | 58.33 |
| P40k | 10 | 20.6854 | 0.18480 | 0.52130 | 61.00 |
| P40k | 13 | 20.8958 | 0.17713 | 0.53971 | 60.33 |
| P40k | 19 | 21.0641 | 0.17200 | 0.55726 | 60.67 |
| P80k | 1 | 18.9840 | 0.26072 | 0.29420 | 47.67 |
| P80k | 4 | 19.9326 | 0.21391 | 0.43738 | 59.33 |
| P80k | 7 | 20.5730 | 0.18650 | 0.51760 | 65.33 |
| P80k | 10 | 20.9589 | 0.17181 | 0.55669 | 67.67 |
| P80k | 13 | 21.1711 | 0.16411 | 0.57544 | 66.67 |
| P80k | 19 | 21.3441 | 0.15834 | 0.58940 | 66.00 |
| Swin80k | 1 | 20.3715 | 0.60396 | 0.12901 | 10.33 |
| Swin80k | 4 | 21.3085 | 0.56341 | 0.21980 | 15.67 |
| Swin80k | 7 | 21.8962 | 0.53750 | 0.27835 | 28.00 |
| Swin80k | 10 | 22.2314 | 0.52147 | 0.31316 | 25.33 |
| Swin80k | 13 | 22.4132 | 0.51314 | 0.36067 | 30.00 |
| Swin80k | 19 | 22.6044 | 0.50587 | 0.39628 | 33.00 |
| Swin150k | 1 | 20.7876 | 0.58723 | 0.14282 | 14.00 |
| Swin150k | 4 | 21.7753 | 0.54527 | 0.24408 | 19.67 |
| Swin150k | 7 | 22.4686 | 0.51470 | 0.32825 | 27.33 |
| Swin150k | 10 | 22.9342 | 0.49414 | 0.39167 | 31.00 |
| Swin150k | 13 | 23.2315 | 0.48113 | 0.43552 | 30.67 |
| Swin150k | 19 | 23.5074 | 0.46905 | 0.48245 | 32.67 |

## Source-paired mean differences

New checkpoint minus its own parent. Positive PSNR/DINO/agreement and negative LPIPS are favorable. Agreement differences are percentage points; no relative-percent conversion.

| Comparison | SNR (dB) | Delta PSNR (dB) | Delta LPIPS | Delta DINOv2-L | Delta agreement (pp) |
|---|---:|---:|---:|---:|---:|
| P80k - P40k | 1 | +0.1965 | -0.01424 | +0.04061 | +8.67 |
| P80k - P40k | 4 | +0.2403 | -0.01375 | +0.03818 | +6.00 |
| P80k - P40k | 7 | +0.2659 | -0.01346 | +0.03654 | +7.00 |
| P80k - P40k | 10 | +0.2736 | -0.01300 | +0.03539 | +6.67 |
| P80k - P40k | 13 | +0.2753 | -0.01303 | +0.03573 | +6.33 |
| P80k - P40k | 19 | +0.2801 | -0.01366 | +0.03214 | +5.33 |
| Swin150k - Swin80k | 1 | +0.4161 | -0.01674 | +0.01380 | +3.67 |
| Swin150k - Swin80k | 4 | +0.4668 | -0.01814 | +0.02428 | +4.00 |
| Swin150k - Swin80k | 7 | +0.5724 | -0.02281 | +0.04990 | -0.67 |
| Swin150k - Swin80k | 10 | +0.7029 | -0.02733 | +0.07851 | +5.67 |
| Swin150k - Swin80k | 13 | +0.8184 | -0.03201 | +0.07485 | +0.67 |
| Swin150k - Swin80k | 19 | +0.9031 | -0.03682 | +0.08617 | -0.33 |

Full precision is retained in summary.csv, paired.csv and the four per-frame CSVs. No historical scientific result was overwritten. GPU0 performs only this finite diagnostic; P and Swin training continue on GPUs1 and2. This shared-host inference is not a unified latency benchmark.

Reproduce only into a fresh versioned output directory after inspecting active jobs. The inference entry is experiments/baseline-diagnostics-20261011/run_interim_evaluation.py; it requires the existing external assets and fixed checkpoint hashes. The reporting entry is report_interim_evaluation.py. Neither should modify the bound live training sources.
