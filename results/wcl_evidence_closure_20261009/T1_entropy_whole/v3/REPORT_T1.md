# T1: Whole-scale entropy coding at the published common500 sources

This is a post-hoc same-source supplement to the already reported holdout. Policies were selected only on the original calibration1000, using DINOv2-L; the new experiment does not rewrite the original registration or results.

Two pure-arithmetic whole-scale families use the frozen tokenizer, null-class VAR, and Dc. Static probabilities use only the original training20k. Both pay the registered header, length and CRC costs, use actual LDPC reception, and fall back by measured bit length to a minimum whole prefix m4. No raw/arithmetic minimum substitution is used.

N=1024; SNRs 4, 10 and 19 dB; 500 sources and three noises per source. The new entropy namespace uses t1_phy.standard_noise with seeds 6201/6202/6203 fixed before execution. Raw comparisons reuse their original receive records. Pairing is by source; equal numeric seeds do not assert equal received observations across waveforms or implementations.

## Actual failure and fallback results

| Family | SNR | Gray frames / 1500 | Fallback fraction | Decoded frames with token errors |
|---|---:|---:|---:|---:|
| Static-entropy whole-scale transmission with VAR completion | 4 | 0 | 1.000000 | 0 |
| Static-entropy whole-scale transmission with VAR completion | 10 | 0 | 0.000000 | 0 |
| Static-entropy whole-scale transmission with VAR completion | 19 | 0 | 0.996000 | 0 |
| VAR-entropy whole-scale transmission with VAR completion | 4 | 0 | 0.990000 | 0 |
| VAR-entropy whole-scale transmission with VAR completion | 10 | 0 | 1.000000 | 0 |
| VAR-entropy whole-scale transmission with VAR completion | 19 | 0 | 0.000000 | 0 |

Token-error diagnostics are evaluated after the actual output is fixed. They never select or repair a receiver output. Arithmetic parser failures, body CRC rejection and header rejection are retained separately in failure_breakdown.csv. The original raw KEEP outputs remain unchanged; CRC-DROP is a separately identified same-reception diagnostic.

The raw_crc_drop_breakdown.csv separates original header rejection, body-CRC rejection with diagnostic gray output, accepted erroneous tokens, and accepted correct tokens. Its token errors include both whole-prefix and transmitted partial tokens; accepted CRC-undetected errors retain their original output. raw_crc_drop_per_frame.csv preserves the exact receive proof and all four actual metrics.

## DINOv2-L paired comparisons

Each row is method minus reference after averaging the three noises within each source. Every preregistered pair is retained, including negative and zero differences.

| Method | Reference | SNR | Mean difference | 95% interval | Direction |
|---|---|---:|---:|---|---|
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 4 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 10 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 19 | 0.0004856 | [0.0000000, 0.0013706] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 4 | -0.0267458 | [-0.0354971, -0.0181086] | interval favors reference |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 10 | -0.0758578 | [-0.0861784, -0.0655495] | interval favors reference |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 19 | -0.0515701 | [-0.0582120, -0.0452728] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 4 | 0.0007891 | [0.0000991, 0.0017973] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 10 | 0.0347943 | [0.0286928, 0.0409932] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (KEEP) | 19 | 0.0588486 | [0.0519578, 0.0659731] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 4 | -0.0259567 | [-0.0347972, -0.0171883] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 10 | -0.0410635 | [-0.0517208, -0.0303707] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (KEEP) | 19 | 0.0067928 | [0.0037670, 0.0098363] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Static-entropy whole-scale transmission with VAR completion | 4 | 0.0007891 | [0.0000991, 0.0017973] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Static-entropy whole-scale transmission with VAR completion | 10 | 0.0347943 | [0.0286928, 0.0409932] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Static-entropy whole-scale transmission with VAR completion | 19 | 0.0583629 | [0.0514441, 0.0654414] | interval favors method |
| Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw whole-scale transmission with VAR completion (KEEP) | 4 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw whole-scale transmission with VAR completion (KEEP) | 10 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw whole-scale transmission with VAR completion (KEEP) | 19 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw partial-scale transmission with VAR completion (KEEP) | 4 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw partial-scale transmission with VAR completion (KEEP) | 10 | -0.0278648 | [-0.0347835, -0.0213620] | interval favors reference |
| Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | Raw partial-scale transmission with VAR completion (KEEP) | 19 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 4 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 10 | 0.0000000 | [0.0000000, 0.0000000] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 19 | 0.0004856 | [0.0000000, 0.0013706] | interval includes zero |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 4 | -0.0267458 | [-0.0354971, -0.0181086] | interval favors reference |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 10 | -0.0479930 | [-0.0603826, -0.0350500] | interval favors reference |
| Static-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 19 | -0.0515701 | [-0.0582120, -0.0452728] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 4 | 0.0007891 | [0.0000991, 0.0017973] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 10 | 0.0347943 | [0.0286928, 0.0409932] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw whole-scale transmission with VAR completion (CRC-DROP diagnostic) | 19 | 0.0588486 | [0.0519578, 0.0659731] | interval favors method |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 4 | -0.0259567 | [-0.0347972, -0.0171883] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 10 | -0.0131987 | [-0.0260156, -0.0002283] | interval favors reference |
| VAR-entropy whole-scale transmission with VAR completion | Raw partial-scale transmission with VAR completion (CRC-DROP diagnostic) | 19 | 0.0067928 | [0.0037670, 0.0098363] | interval favors method |

## Direct implications for the original partial-scale system

The following comparisons retain the direction entropy whole-scale minus original raw partial-scale. Positive PSNR/DINO and negative LPIPS favor entropy. Agreement differences are shown in percentage points. These statements are read directly from the paired CSV, including unfavorable evidence for the original system.

- Static-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 4 dB: psnr_db -0.3604723 [-0.3994548, -0.3239328]; interval favors reference; lpips_alex +0.0159502 [0.0147598, 0.0171570]; interval favors reference; dinov2_vitl14_cosine -0.0267458 [-0.0354971, -0.0181086]; interval favors reference; ConvNeXt agreement (percentage points) -2.4000000 [-5.4000000, 0.6000000]; interval includes zero.
- Static-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 10 dB: psnr_db -1.1761278 [-1.2378052, -1.1150384]; interval favors reference; lpips_alex +0.0404304 [0.0385591, 0.0423630]; interval favors reference; dinov2_vitl14_cosine -0.0758578 [-0.0861784, -0.0655495]; interval favors reference; ConvNeXt agreement (percentage points) -6.8666667 [-10.1333333, -3.7333333]; interval favors reference.
- Static-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 19 dB: psnr_db -1.3309120 [-1.3810098, -1.2820166]; interval favors reference; lpips_alex +0.0346597 [0.0334450, 0.0359236]; interval favors reference; dinov2_vitl14_cosine -0.0515701 [-0.0582120, -0.0452728]; interval favors reference; ConvNeXt agreement (percentage points) -1.0000000 [-3.8000000, 1.8000000]; interval includes zero.
- VAR-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 4 dB: psnr_db -0.3352585 [-0.3797265, -0.2916916]; interval favors reference; lpips_alex +0.0154737 [0.0141726, 0.0167579]; interval favors reference; dinov2_vitl14_cosine -0.0259567 [-0.0347972, -0.0171883]; interval favors reference; ConvNeXt agreement (percentage points) -2.4000000 [-5.4000000, 0.6000000]; interval includes zero.
- VAR-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 10 dB: psnr_db -0.4055881 [-0.4979969, -0.3121755]; interval favors reference; lpips_alex +0.0198938 [0.0167958, 0.0229410]; interval favors reference; dinov2_vitl14_cosine -0.0410635 [-0.0517208, -0.0303707]; interval favors reference; ConvNeXt agreement (percentage points) -4.2666667 [-7.4666667, -1.1333333]; interval favors reference.
- VAR-entropy whole-scale transmission with VAR completion versus original raw partial-scale at 19 dB: psnr_db +0.2578195 [0.2387331, 0.2774542]; interval favors method; lpips_alex -0.0055361 [-0.0059450, -0.0051331]; interval favors method; dinov2_vitl14_cosine +0.0067928 [0.0037670, 0.0098363]; interval favors method; ConvNeXt agreement (percentage points) +0.0000000 [-1.8000000, 1.8000000]; interval includes zero.

The original raw partial-scale method cannot be described as uniformly superior when a paired interval favors the entropy method. A zero agreement difference is retained as zero; it does not cancel differences in other metrics.


All four metrics and all 132 paired rows are in paired.csv. LPIPS retains its negative-is-better sign. ConvNeXt is source-prediction agreement, not classification accuracy; stored values are fractions and paired differences are absolute differences.

Intervals use the original source-level bootstrap implementation, 10,000 replicates and seed 2026100701. Existing raw intervals are copied; identical vectors reuse existing intervals. Intervals are pointwise, with no multiple-comparison adjustment. Inclusion of zero is not evidence of equivalence.

## Compression and computational cost

source_lengths.csv includes every m4–m9 prefix slot for calibration and holdout, per-SNR paid source capacity, actual fit, selected m and full fallback attempts. Actual encoded lengths include flushing. An absent short VAR calibration prefix is marked as unnecessary when m6 already fits the minimum capacity; a holdout prefix above the maximum frozen target is marked uncomputed. Neither is replaced with an estimated length. m9_sendable_summary.csv reports the actual fraction that fits each frozen MCS; missing m9 lengths never count as failures or successes.

Actual online TX/RX/e2e mean, median and p95 timings are joined for 12 of twelve entropy/raw-anchor method/SNR points, using only condition=all and 48 measured calls from fixed16 sources. Seconds are converted to milliseconds. Conditional success/failure timing is not substituted for all-frame timing.
All timing observations, including long-latency frames, are retained. An empirical mean can exceed the 95th percentile when a few calls are long; that is not a reason to trim them. Fixed16 timing is a small engineering sample and does not establish precise tail latency or universal deployment performance.
A quality advantage at a particular metric/SNR does not establish universal optimality. If entropy improves quality, that improvement must be reported; any quality–compute tradeoff requires the actual T4 measurements.

## Execution and reuse

- Actual entropy packet decoder calls: 18000.
- New holdout VAR renders: 500 plus one exact original-render replay qualification.
- Four-metric execution counts: `{"native_BPG_quality_reuses": 66, "new_quality_calls": 500, "new_reference_preparations": 0, "old_quality_reuses": 0, "old_reference_reuses": 500, "original_common500_quality_reuses": 1500}`.
- This reporting command performs zero model, channel or bootstrap calls.
