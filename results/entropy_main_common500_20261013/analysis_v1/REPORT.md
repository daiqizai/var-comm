# Frozen entropy partial-scale common500 extension

All 4,500 new logical frames and their failures are retained. Three registered noises are averaged within each source before source-level statistics. The frozen original common500 bootstrap (10,000 draws, seed 2026100701) computes only 12 new marginal intervals and 60 direct paired intervals. Original intervals are copied unchanged.

This is a subsequent evaluation on the already used common500 population, not a newly blind test. Pairing is by source, not a claim that distinct systems share received observations. Confidence intervals are pointwise. LPIPS retains its original sign; agreement is source-prediction consistency, not classification accuracy.

Raw and entropy coding differ in framing and failure rules, so the four matched-prior systems are not a pure two-factor ablation. Source lengths count 500 unique sources per SNR/method, not 1,500 noise realizations. Arithmetic lengths include termination. Compression saving concerns the same token sequence and does not imply equal wireless-symbol savings.

| SNR | Reference | Metric | Entropy partial minus reference | 95% CI |
|---|---|---|---:|---|
| 4 | EC_VAR_WHOLE | psnr_db | 0.61603 | [0.563878, 0.672817] |
| 4 | EC_VAR_WHOLE | lpips_alex | -0.0245828 | [-0.0262145, -0.0229994] |
| 4 | EC_VAR_WHOLE | dinov2_vitl14_cosine | 0.0462606 | [0.036038, 0.0566874] |
| 4 | EC_VAR_WHOLE | convnext_top1_source_prediction | 5.4 | [2, 8.8] |
| 10 | EC_VAR_WHOLE | psnr_db | 0.704705 | [0.639213, 0.769803] |
| 10 | EC_VAR_WHOLE | lpips_alex | -0.027358 | [-0.0297322, -0.0250038] |
| 10 | EC_VAR_WHOLE | dinov2_vitl14_cosine | 0.0549401 | [0.045318, 0.0647774] |
| 10 | EC_VAR_WHOLE | convnext_top1_source_prediction | 6.6 | [4, 9.4] |
| 19 | EC_VAR_WHOLE | psnr_db | 0.336906 | [0.305018, 0.368942] |
| 19 | EC_VAR_WHOLE | lpips_alex | -0.00532786 | [-0.00575445, -0.00490619] |
| 19 | EC_VAR_WHOLE | dinov2_vitl14_cosine | 0.00851784 | [0.00618772, 0.0108725] |
| 19 | EC_VAR_WHOLE | convnext_top1_source_prediction | -0.2 | [-2, 1.6] |
| 4 | RAW_PARTIAL | psnr_db | 0.280772 | [0.236557, 0.327004] |
| 4 | RAW_PARTIAL | lpips_alex | -0.00910909 | [-0.0103844, -0.00784891] |
| 4 | RAW_PARTIAL | dinov2_vitl14_cosine | 0.0203039 | [0.0136482, 0.0274138] |
| 4 | RAW_PARTIAL | convnext_top1_source_prediction | 3 | [1, 5.2] |
| 10 | RAW_PARTIAL | psnr_db | 0.299117 | [0.246795, 0.354985] |
| 10 | RAW_PARTIAL | lpips_alex | -0.00746418 | [-0.00889301, -0.00615277] |
| 10 | RAW_PARTIAL | dinov2_vitl14_cosine | 0.0138766 | [0.00964145, 0.0183706] |
| 10 | RAW_PARTIAL | convnext_top1_source_prediction | 2.33333 | [0.6, 4.13333] |
| 19 | RAW_PARTIAL | psnr_db | 0.594726 | [0.559388, 0.630176] |
| 19 | RAW_PARTIAL | lpips_alex | -0.010864 | [-0.0113969, -0.0103117] |
| 19 | RAW_PARTIAL | dinov2_vitl14_cosine | 0.0153107 | [0.011711, 0.0188719] |
| 19 | RAW_PARTIAL | convnext_top1_source_prediction | -0.2 | [-2.2, 1.8] |
| 4 | P100000 | psnr_db | -2.20954 | [-2.306, -2.11173] |
| 4 | P100000 | lpips_alex | 0.044469 | [0.0412684, 0.0476445] |
| 4 | P100000 | dinov2_vitl14_cosine | 0.109068 | [0.0916085, 0.126102] |
| 4 | P100000 | convnext_top1_source_prediction | 9.86667 | [6, 13.7333] |
| 10 | P100000 | psnr_db | -0.788567 | [-0.876624, -0.696584] |
| 10 | P100000 | lpips_alex | -0.00179063 | [-0.00385744, 0.000251225] |
| 10 | P100000 | dinov2_vitl14_cosine | 0.15365 | [0.140685, 0.166869] |
| 10 | P100000 | convnext_top1_source_prediction | 13.1333 | [9.6, 16.6667] |
| 19 | P100000 | psnr_db | 0.915705 | [0.821472, 1.00889] |
| 19 | P100000 | lpips_alex | -0.0402777 | [-0.0418865, -0.0386311] |
| 19 | P100000 | dinov2_vitl14_cosine | 0.192765 | [0.180257, 0.205367] |
| 19 | P100000 | convnext_top1_source_prediction | 13 | [9.33333, 16.6667] |
| 4 | Swin200000 | psnr_db | -4.22272 | [-4.33563, -4.11032] |
| 4 | Swin200000 | lpips_alex | -0.266475 | [-0.277039, -0.256542] |
| 4 | Swin200000 | dinov2_vitl14_cosine | 0.309115 | [0.290318, 0.327907] |
| 4 | Swin200000 | convnext_top1_source_prediction | 40.4 | [35.7333, 45] |
| 10 | Swin200000 | psnr_db | -3.01193 | [-3.10274, -2.92043] |
| 10 | Swin200000 | lpips_alex | -0.297192 | [-0.30739, -0.287515] |
| 10 | Swin200000 | dinov2_vitl14_cosine | 0.334408 | [0.316473, 0.352439] |
| 10 | Swin200000 | convnext_top1_source_prediction | 47 | [42.5333, 51.3333] |
| 19 | Swin200000 | psnr_db | -1.53788 | [-1.61579, -1.45963] |
| 19 | Swin200000 | lpips_alex | -0.321827 | [-0.331926, -0.312072] |
| 19 | Swin200000 | dinov2_vitl14_cosine | 0.327651 | [0.310214, 0.345577] |
| 19 | Swin200000 | convnext_top1_source_prediction | 42 | [37.4667, 46.4667] |
| 4 | BPG_ADAPTIVE | psnr_db | -3.29429 | [-3.39395, -3.19282] |
| 4 | BPG_ADAPTIVE | lpips_alex | -0.462226 | [-0.472825, -0.451817] |
| 4 | BPG_ADAPTIVE | dinov2_vitl14_cosine | 0.430163 | [0.410721, 0.449775] |
| 4 | BPG_ADAPTIVE | convnext_top1_source_prediction | 58.2 | [53.8, 62.6] |
| 10 | BPG_ADAPTIVE | psnr_db | -2.52897 | [-2.61478, -2.44231] |
| 10 | BPG_ADAPTIVE | lpips_alex | -0.417905 | [-0.429379, -0.406859] |
| 10 | BPG_ADAPTIVE | dinov2_vitl14_cosine | 0.386092 | [0.366236, 0.405797] |
| 10 | BPG_ADAPTIVE | convnext_top1_source_prediction | 55.2 | [50.6, 59.8] |
| 19 | BPG_ADAPTIVE | psnr_db | -2.06981 | [-2.15207, -1.98938] |
| 19 | BPG_ADAPTIVE | lpips_alex | -0.34018 | [-0.351697, -0.328886] |
| 19 | BPG_ADAPTIVE | dinov2_vitl14_cosine | 0.251931 | [0.234543, 0.269407] |
| 19 | BPG_ADAPTIVE | convnext_top1_source_prediction | 39.4 | [34.8, 44] |

Agreement differences above use percentage points. No failure was dropped, no policy was reselected, no old interval was recalculated, and no training, new bandwidth, Kodak, HiFi or additional SNR extension was started. Unified timing is reported by its separate registered owner; this statistical delivery does not assert that timing is complete.
