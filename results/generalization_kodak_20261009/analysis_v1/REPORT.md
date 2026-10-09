# Kodak24 frozen-method transfer

Actual population: 24 distinct source photographs, one deterministic 256x256 center crop each; N=1024; SNRs 4, 10, and 19 dB; three fixed noises per method/source. This is not an ImageNet-label accuracy experiment and does not establish absence from all pretrained training corpora.

New statistics only: 48 summaries, 36 VAR-minus-reference paired contrasts, and 1152 source means. One 10000x24 resampling array uses fixed seed 2026100901. All 864 frames are included, including failures. Old main/supplementary results and intervals are untouched.

No model/PHY/noise or policy evaluation runs in this script. The input CSV is bound to its actual scoring completion. Ground-truth labels are unavailable; ConvNeXt 0/1 source-prediction agreement is used. DINO-L is dinov2_vitl14_cosine. PSNR averages per-image PSNR values, not PSNR of average MSE.

Swin at 19 dB remains outside training/calibration; the 10--19 dB main-curve segment is dashed because 13 dB is not a tested point. All results, negative effects, and confidence intervals remain visible. LPIPS signs are unchanged; agreement differences scale by 100 to percentage points.

Pairing is by source after three-noise means, not by received waveform. The 24-image benchmark and center crop restrict generality; all CIs are pointwise, not multiplicity-adjusted. No quality ranking chooses sources, noise realizations, or a new strategy.

| SNR | Reference | Metric | VAR - reference | 95% CI |
|---:|---|---|---:|---|
| 4 | Latent continuous JSCC | psnr_db | -1.54829 | [-1.82946, -1.25014] |
| 4 | Latent continuous JSCC | lpips_alex | 0.0266614 | [0.0108259, 0.0409372] |
| 4 | Latent continuous JSCC | dinov2_vitl14_cosine | 0.23296 | [0.179639, 0.287497] |
| 4 | Latent continuous JSCC | convnext_top1_source_prediction | 27.7778 | [11.1111, 45.8333] |
| 10 | Latent continuous JSCC | psnr_db | -0.39994 | [-0.694554, -0.0710506] |
| 10 | Latent continuous JSCC | lpips_alex | -0.017246 | [-0.0253652, -0.00971495] |
| 10 | Latent continuous JSCC | dinov2_vitl14_cosine | 0.206579 | [0.170724, 0.243862] |
| 10 | Latent continuous JSCC | convnext_top1_source_prediction | 33.3333 | [9.72222, 56.9444] |
| 19 | Latent continuous JSCC | psnr_db | 0.86825 | [0.571495, 1.18338] |
| 19 | Latent continuous JSCC | lpips_alex | -0.0520102 | [-0.058031, -0.0458719] |
| 19 | Latent continuous JSCC | dinov2_vitl14_cosine | 0.237225 | [0.192127, 0.284788] |
| 19 | Latent continuous JSCC | convnext_top1_source_prediction | 37.5 | [16.6667, 58.3333] |
| 4 | Adaptive-resolution BPG + LDPC | psnr_db | -3.13671 | [-3.50348, -2.76717] |
| 4 | Adaptive-resolution BPG + LDPC | lpips_alex | -0.47057 | [-0.5183, -0.422715] |
| 4 | Adaptive-resolution BPG + LDPC | dinov2_vitl14_cosine | 0.504669 | [0.444502, 0.56155] |
| 4 | Adaptive-resolution BPG + LDPC | convnext_top1_source_prediction | 54.1667 | [33.3333, 75] |
| 10 | Adaptive-resolution BPG + LDPC | psnr_db | -2.58437 | [-2.93661, -2.2402] |
| 10 | Adaptive-resolution BPG + LDPC | lpips_alex | -0.416412 | [-0.46908, -0.366171] |
| 10 | Adaptive-resolution BPG + LDPC | dinov2_vitl14_cosine | 0.412691 | [0.333503, 0.488164] |
| 10 | Adaptive-resolution BPG + LDPC | convnext_top1_source_prediction | 43.0556 | [20.8333, 63.9236] |
| 19 | Adaptive-resolution BPG + LDPC | psnr_db | -2.57418 | [-2.90235, -2.25752] |
| 19 | Adaptive-resolution BPG + LDPC | lpips_alex | -0.337334 | [-0.384742, -0.291388] |
| 19 | Adaptive-resolution BPG + LDPC | dinov2_vitl14_cosine | 0.290145 | [0.222356, 0.36397] |
| 19 | Adaptive-resolution BPG + LDPC | convnext_top1_source_prediction | 29.1667 | [8.33333, 50] |
| 4 | SwinJSCC-80k (adapted) | psnr_db | -3.15288 | [-3.83656, -2.22345] |
| 4 | SwinJSCC-80k (adapted) | lpips_alex | -0.332826 | [-0.37952, -0.287575] |
| 4 | SwinJSCC-80k (adapted) | dinov2_vitl14_cosine | 0.378658 | [0.322152, 0.435853] |
| 4 | SwinJSCC-80k (adapted) | convnext_top1_source_prediction | 44.4444 | [23.6111, 65.2778] |
| 10 | SwinJSCC-80k (adapted) | psnr_db | -2.0475 | [-2.69885, -1.09536] |
| 10 | SwinJSCC-80k (adapted) | lpips_alex | -0.372102 | [-0.412834, -0.331894] |
| 10 | SwinJSCC-80k (adapted) | dinov2_vitl14_cosine | 0.359475 | [0.295856, 0.420783] |
| 10 | SwinJSCC-80k (adapted) | convnext_top1_source_prediction | 40.2778 | [18.0556, 61.1111] |
| 19 | SwinJSCC-80k (adapted) | psnr_db | -0.839831 | [-1.48142, 0.129268] |
| 19 | SwinJSCC-80k (adapted) | lpips_alex | -0.401105 | [-0.441733, -0.359578] |
| 19 | SwinJSCC-80k (adapted) | dinov2_vitl14_cosine | 0.326077 | [0.257861, 0.399636] |
| 19 | SwinJSCC-80k (adapted) | convnext_top1_source_prediction | 43.0556 | [19.4444, 65.2778] |

Reproduce into a new output directory:

`python experiments/generalization_kodak_20261009/analyze.py --rows "/home/liulu/projects/VAR_COMM/results/generalization_kodak_20261009/metrics_v1/rows.csv" --completion "/home/liulu/projects/VAR_COMM/results/generalization_kodak_20261009/metrics_v1/completion.json" --out NEW_DIRECTORY`

Actual PNG visual inspection is required after export; programmatic text bounds are checked during plotting. The per-frame CSV and fixed dataset/protocol request retain the original method identities and source/noise mapping.
