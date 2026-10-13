# Frozen final-baseline Kodak24 comparison

All 24 fixed 256x256 center crops; N1024; SNR4/10/19; original noise labels2001/2002/2003. Every failure frame is included. Equal noise labels do not claim identical received observations across different waveforms.

Two new baselines contribute432 actual logical frames and1728 metric cells. The existing raw and adaptive BPG anchors contribute432 verified reused frames. There are24 new single-method intervals and24 new NeST-minus-baseline intervals; the original raw/BPG summary rows and12 NeST-minus-BPG intervals are copied verbatim. No old interval is recomputed, and no Kodak selection, training, or calibration occurs.

The original analyze.py interval function and original shared10000x24 bootstrap index array construction are reused with seed2026100901. Three noises are averaged within each source before source-level inference. LPIPS deltas keep their original sign; agreement is a source-prediction proxy.

| Method | SNR (dB) | Metric | Mean | 95% CI |
|---|---:|---|---:|---|
| NeST-Com (raw) | 4 | psnr_db | 18.5871 | [17.5875, 19.6203] |
| NeST-Com (raw) | 4 | lpips_alex | 0.273599 | [0.248577, 0.298129] |
| NeST-Com (raw) | 4 | dinov2_vitl14_cosine | 0.648899 | [0.584062, 0.712555] |
| NeST-Com (raw) | 4 | convnext_top1_source_prediction | 58.3333 | [37.5, 79.1667] |
| NeST-Com (raw) | 10 | psnr_db | 20.7449 | [19.6007, 21.9197] |
| NeST-Com (raw) | 10 | lpips_alex | 0.181486 | [0.161569, 0.201373] |
| NeST-Com (raw) | 10 | dinov2_vitl14_cosine | 0.777903 | [0.743942, 0.811931] |
| NeST-Com (raw) | 10 | convnext_top1_source_prediction | 59.7222 | [40.2778, 79.1667] |
| NeST-Com (raw) | 19 | psnr_db | 22.4156 | [21.1905, 23.6832] |
| NeST-Com (raw) | 19 | lpips_alex | 0.133019 | [0.11736, 0.148978] |
| NeST-Com (raw) | 19 | dinov2_vitl14_cosine | 0.850967 | [0.822005, 0.878342] |
| NeST-Com (raw) | 19 | convnext_top1_source_prediction | 62.5 | [41.6667, 79.1667] |
| Adaptive-resolution BPG + LDPC | 4 | psnr_db | 21.7239 | [20.5542, 22.9504] |
| Adaptive-resolution BPG + LDPC | 4 | lpips_alex | 0.744169 | [0.682786, 0.80406] |
| Adaptive-resolution BPG + LDPC | 4 | dinov2_vitl14_cosine | 0.14423 | [0.0916431, 0.208898] |
| Adaptive-resolution BPG + LDPC | 4 | convnext_top1_source_prediction | 4.16667 | [0, 12.5] |
| Adaptive-resolution BPG + LDPC | 10 | psnr_db | 23.3293 | [22.0241, 24.6893] |
| Adaptive-resolution BPG + LDPC | 10 | lpips_alex | 0.597898 | [0.533385, 0.663603] |
| Adaptive-resolution BPG + LDPC | 10 | dinov2_vitl14_cosine | 0.365212 | [0.283738, 0.44887] |
| Adaptive-resolution BPG + LDPC | 10 | convnext_top1_source_prediction | 16.6667 | [4.16667, 33.3333] |
| Adaptive-resolution BPG + LDPC | 19 | psnr_db | 24.9898 | [23.5634, 26.4589] |
| Adaptive-resolution BPG + LDPC | 19 | lpips_alex | 0.470353 | [0.411902, 0.529986] |
| Adaptive-resolution BPG + LDPC | 19 | dinov2_vitl14_cosine | 0.560822 | [0.47908, 0.64173] |
| Adaptive-resolution BPG + LDPC | 19 | convnext_top1_source_prediction | 33.3333 | [16.6667, 54.1667] |
| Latent-space JSCC (100k) | 4 | psnr_db | 20.3464 | [19.3558, 21.3939] |
| Latent-space JSCC (100k) | 4 | lpips_alex | 0.227287 | [0.209007, 0.245523] |
| Latent-space JSCC (100k) | 4 | dinov2_vitl14_cosine | 0.492561 | [0.433508, 0.55049] |
| Latent-space JSCC (100k) | 4 | convnext_top1_source_prediction | 31.9444 | [16.6319, 48.6111] |
| Latent-space JSCC (100k) | 10 | psnr_db | 21.3713 | [20.3097, 22.4797] |
| Latent-space JSCC (100k) | 10 | lpips_alex | 0.181627 | [0.164362, 0.198769] |
| Latent-space JSCC (100k) | 10 | dinov2_vitl14_cosine | 0.617493 | [0.576555, 0.658472] |
| Latent-space JSCC (100k) | 10 | convnext_top1_source_prediction | 41.6667 | [25, 59.7222] |
| Latent-space JSCC (100k) | 19 | psnr_db | 21.7853 | [20.6876, 22.9186] |
| Latent-space JSCC (100k) | 19 | lpips_alex | 0.168247 | [0.151259, 0.185075] |
| Latent-space JSCC (100k) | 19 | dinov2_vitl14_cosine | 0.653535 | [0.613601, 0.692139] |
| Latent-space JSCC (100k) | 19 | convnext_top1_source_prediction | 43.0556 | [25, 61.1111] |
| SwinJSCC-200k (adapted) | 4 | psnr_db | 22.3916 | [21.2326, 23.6014] |
| SwinJSCC-200k (adapted) | 4 | lpips_alex | 0.57374 | [0.51822, 0.628105] |
| SwinJSCC-200k (adapted) | 4 | dinov2_vitl14_cosine | 0.253978 | [0.180721, 0.335446] |
| SwinJSCC-200k (adapted) | 4 | convnext_top1_source_prediction | 13.8889 | [1.38889, 29.1667] |
| SwinJSCC-200k (adapted) | 10 | psnr_db | 23.5706 | [22.3085, 24.8699] |
| SwinJSCC-200k (adapted) | 10 | lpips_alex | 0.502158 | [0.44821, 0.556339] |
| SwinJSCC-200k (adapted) | 10 | dinov2_vitl14_cosine | 0.467942 | [0.391601, 0.546919] |
| SwinJSCC-200k (adapted) | 10 | convnext_top1_source_prediction | 19.4444 | [6.94444, 33.3333] |
| SwinJSCC-200k (adapted) | 19 | psnr_db | 24.1444 | [22.8151, 25.5166] |
| SwinJSCC-200k (adapted) | 19 | lpips_alex | 0.472186 | [0.419781, 0.525898] |
| SwinJSCC-200k (adapted) | 19 | dinov2_vitl14_cosine | 0.566463 | [0.494349, 0.639463] |
| SwinJSCC-200k (adapted) | 19 | convnext_top1_source_prediction | 31.9444 | [15.2778, 50] |

All CSV values retain the original precision; the Markdown table is for reading. The four-metric figure uses every mean and interval at true numerical SNR positions. Its editable SVG and vector PDF do not contain rasterized data curves. PNG visual review must be recorded separately.
