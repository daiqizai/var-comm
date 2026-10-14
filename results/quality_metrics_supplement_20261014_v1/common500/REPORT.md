# Common500 supplemental quality metrics

| SNR (dB) | Method | DISTS ↓ (95% CI) | DreamSim ↓ (95% CI) | MS-SSIM ↑ (95% CI) |
|---:|---|---:|---:|---:|
| 4 | EC_VAR_PARTIAL | 0.191632 [0.188974, 0.194317] | 0.183874 [0.176957, 0.191081] | 0.567083 [0.554471, 0.579970] |
| 4 | RAW_PARTIAL | 0.195885 [0.193427, 0.198384] | 0.192088 [0.185266, 0.199056] | 0.551168 [0.539478, 0.563232] |
| 4 | EC_VAR_WHOLE | 0.202827 [0.200332, 0.205405] | 0.204600 [0.197372, 0.211839] | 0.531136 [0.519129, 0.543528] |
| 4 | P100000 | 0.221804 [0.219321, 0.224307] | 0.231678 [0.225244, 0.238252] | 0.718020 [0.709953, 0.726177] |
| 4 | Swin200000 | 0.348841 [0.345875, 0.351849] | 0.435911 [0.426573, 0.445461] | 0.780809 [0.773630, 0.787985] |
| 4 | BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024 | 0.402213 [0.398588, 0.405770] | 0.573672 [0.563258, 0.584042] | 0.703235 [0.694186, 0.712456] |
| 10 | EC_VAR_PARTIAL | 0.150414 [0.148086, 0.152827] | 0.107680 [0.103778, 0.111793] | 0.721171 [0.711026, 0.731391] |
| 10 | RAW_PARTIAL | 0.154141 [0.151828, 0.156499] | 0.113082 [0.108970, 0.117386] | 0.711061 [0.701203, 0.720976] |
| 10 | EC_VAR_WHOLE | 0.162194 [0.159236, 0.165162] | 0.131809 [0.126030, 0.137728] | 0.671196 [0.658383, 0.684346] |
| 10 | P100000 | 0.198638 [0.196056, 0.201238] | 0.171151 [0.165866, 0.176525] | 0.771219 [0.763809, 0.778541] |
| 10 | Swin200000 | 0.322391 [0.319355, 0.325444] | 0.356391 [0.347359, 0.365576] | 0.838779 [0.832756, 0.844788] |
| 10 | BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024 | 0.340877 [0.337392, 0.344434] | 0.416548 [0.406017, 0.427352] | 0.794977 [0.787754, 0.802237] |
| 19 | EC_VAR_PARTIAL | 0.124323 [0.122044, 0.126713] | 0.078476 [0.075503, 0.081614] | 0.816139 [0.808291, 0.823910] |
| 19 | RAW_PARTIAL | 0.130591 [0.128421, 0.132833] | 0.083048 [0.080030, 0.086197] | 0.797518 [0.789567, 0.805422] |
| 19 | EC_VAR_WHOLE | 0.127645 [0.125458, 0.129898] | 0.081628 [0.078669, 0.084797] | 0.805808 [0.798092, 0.813498] |
| 19 | P100000 | 0.191096 [0.188474, 0.193757] | 0.154602 [0.149603, 0.159646] | 0.790066 [0.782998, 0.797161] |
| 19 | Swin200000 | 0.308058 [0.304904, 0.311234] | 0.317946 [0.309244, 0.326853] | 0.865552 [0.859989, 0.871012] |
| 19 | BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024 | 0.291959 [0.288188, 0.295694] | 0.296377 [0.286910, 0.306227] | 0.857186 [0.851511, 0.862778] |

All original common500 sources, N1024, SNR4/10/19 and each method's original three noise seeds are retained. There are27000 logical image pairs and81000 supplemental scalar scores. Failed/gray frames retain their original denominator. Scores reuse only exactly verified native reference/reconstruction pairs. DISTS/DreamSim are lower-is-better; MS-SSIM is higher-is-better. Official frozen implementations, weights and RGB preprocessing are the same as the delivered fixed16 supplement. The original P100k/Swin200k evaluator had not saved pixels; the user separately authorized frozen9000-frame deterministic replay, and all9000 recovered outputs exactly match the old per-frame reconstruction SHA and header status. The other four methods use sealed original native arrays. No training, new checkpoint/policy selection, new samples/SNRs, recomputation of old scores or change to the main figure occurred.

Each source first averages its original three noise scores. The registered10000 source resamples, seed2026100701, produce pointwise95% intervals. The separate paired table resamples signed per-source method differences directly, not differences of marginal intervals. All15 method pairs and all three metrics/workpoints are reported; these exploratory intervals do not constitute simultaneous significance or a universal winner. HiFi development16 remains separate at13dB. Supplemental scores cannot feed back into main policy/checkpoint selection.
