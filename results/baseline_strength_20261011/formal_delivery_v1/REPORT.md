# Frozen baseline supplement: formal evaluation and paired statistics

The selected P100000 and Swin200000 checkpoints completed 18,000 original-common500 and 1,800 existing-new100 method frames. The independent scoring and statistics/plot processes all returned actual exit zero. Every source, SNR and registered noise appears exactly once per new method; failure rows remain in the denominator. No training or checkpoint reselection followed these scores.

Common500 uses the original continuous-baseline noises 2001/2002/2003; new100 uses 9301/9302/9303. Digital common500 anchors retain their original 6201/6202/6203 noises. Pairing is by source after averaging each method over its own registered three noises, not by identical received observations. The two cohorts are never combined.

Only new single-method intervals and the registered contrasts are computed, using the original source-level 10,000-replicate bootstrap and seed 2026100701. All old intervals are copied unchanged. These are subsequent baseline comparisons on previously used evaluation cohorts, not newly blind tests. Frozen metric definitions, model identities and numerical settings are retained; cross-host bitwise equivalence is not claimed.

## Source-paired PSNR contrasts

Values below are method minus reference in dB, with direct paired 95% intervals. Full-precision CSVs contain all four metrics, including negative and zero values.

### common500

| Method | Reference | SNR (dB) | Delta PSNR (dB) | 95% interval |
|---|---|---:|---:|---|
| P100000 | P40000 | 1 | 0.224741 | [0.206502, 0.243031] |
| P100000 | P40000 | 4 | 0.271241 | [0.251502, 0.291093] |
| P100000 | P40000 | 7 | 0.317924 | [0.296395, 0.339001] |
| P100000 | P40000 | 10 | 0.346385 | [0.322860, 0.369095] |
| P100000 | P40000 | 13 | 0.363871 | [0.339007, 0.388216] |
| P100000 | P40000 | 19 | 0.379172 | [0.352203, 0.405246] |
| Swin200000 | Swin80000 | 1 | 0.301459 | [0.253910, 0.354204] |
| Swin200000 | Swin80000 | 4 | 0.377801 | [0.329281, 0.432358] |
| Swin200000 | Swin80000 | 7 | 0.432555 | [0.383569, 0.486424] |
| Swin200000 | Swin80000 | 10 | 0.487212 | [0.433741, 0.546955] |
| Swin200000 | Swin80000 | 13 | 0.537906 | [0.482197, 0.601016] |
| Swin200000 | Swin80000 | 19 | 0.598373 | [0.537994, 0.665787] |
| RAW_PARTIAL | P100000 | 1 | -2.503974 | [-2.597732, -2.408680] |
| RAW_PARTIAL | P100000 | 4 | -2.490312 | [-2.579335, -2.402259] |
| RAW_PARTIAL | P100000 | 7 | -1.830234 | [-1.912183, -1.746367] |
| RAW_PARTIAL | P100000 | 10 | -1.087685 | [-1.166763, -1.007335] |
| RAW_PARTIAL | P100000 | 13 | -0.761444 | [-0.832188, -0.685904] |
| RAW_PARTIAL | P100000 | 19 | 0.320979 | [0.248049, 0.395986] |
| RAW_PARTIAL | Swin200000 | 1 | -4.397617 | [-4.515427, -4.283002] |
| RAW_PARTIAL | Swin200000 | 4 | -4.503492 | [-4.615045, -4.392025] |
| RAW_PARTIAL | Swin200000 | 7 | -3.935949 | [-4.036428, -3.838517] |
| RAW_PARTIAL | Swin200000 | 10 | -3.311050 | [-3.405872, -3.218770] |
| RAW_PARTIAL | Swin200000 | 13 | -3.087318 | [-3.171021, -3.005621] |
| RAW_PARTIAL | Swin200000 | 19 | -2.132607 | [-2.203607, -2.062871] |

### new100

| Method | Reference | SNR (dB) | Delta PSNR (dB) | 95% interval |
|---|---|---:|---:|---|
| EC_VAR_PARTIAL | P100000 | 4 | -2.345379 | [-2.602797, -2.087003] |
| EC_VAR_PARTIAL | P100000 | 10 | -0.816204 | [-0.999779, -0.626546] |
| EC_VAR_PARTIAL | P100000 | 19 | 0.973817 | [0.785397, 1.174796] |
| EC_VAR_PARTIAL | Swin200000 | 4 | -4.345640 | [-4.643092, -4.039533] |
| EC_VAR_PARTIAL | Swin200000 | 10 | -3.006286 | [-3.242628, -2.757587] |
| EC_VAR_PARTIAL | Swin200000 | 19 | -1.449755 | [-1.633567, -1.255552] |

## Observed trade-off

On common500, raw partial is below the new latent continuous baseline in PSNR at 1–13 dB and above it at 19 dB; it is below the new adapted Swin baseline in PSNR at all six SNRs. Its DINOv2-L and source-prediction agreement means exceed both new baselines across these SNRs. LPIPS favors the raw partial method over new Swin throughout, but crosses the latent continuous baseline as SNR increases. These metric-specific trade-offs do not support a blanket claim of superiority on all four metrics.

## Reading the figures

PSNR and DINOv2-L are higher-is-better. LPIPS retains its original sign and is lower-is-better. ConvNeXt source-prediction agreement is not classification accuracy; its single-method values and intervals display percent, while paired differences display percentage points. All six common500 SNR positions use their true numerical spacing.

The old adapted Swin80k point at 19 dB remains outside its training/calibration range and is shown with an open marker and dashed final segment. The new adapted Swin200k model combines additional training with expanded 1–19 dB coverage and the explicitly authorized fresh optimizer, sampler and RNG from the 80k weights. It reached the 120k additional-update budget while still improving; convergence and official optimality are not claimed. P100k resumed its full training state and met the registered empirical plateau rule. No capacity expansion was run.

Four final figure groups are available under `../final_comparison_figures_v1/` as vector PDF, editable-text SVG and 600 dpi PNG, with exact plot rows and English captions. PNG visual review is recorded separately; generation alone is not review.

## Remaining delivery

Same-machine timing of the four digital modes and two selected baselines is still pending implementation qualification and an exclusive resource window. No old 4090D timing is mixed into an H800 speed comparison. The final training recovery states and these lightweight result increments depend on the existing original model/data/runtime recovery archives.
