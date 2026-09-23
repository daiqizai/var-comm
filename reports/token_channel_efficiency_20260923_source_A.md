# Source representation and actual bitstreams: experiment A complete

Formal scope:1000 calibration and100 original development sources, kept separate. Each source has actual raw/arithmetic m6–m10 token roundtrips and20 D0/Dc representation references. The25000+2500 per-source rows and10000+1000 ledger rows are published with hashes. Two preliminary calibration probes are excluded from these formal counts. No new holdout, new training, model weights, pixels or actual token/bitstream arrays are included in this source-only publication.

These are noiseless source/interface references, with no finite-channel N or E. Arithmetic length is the actual written integer bitstream, including termination, not cross entropy. Byte-storage padding is reported separately and is not claimed as transmitted source bits. Selected payload length includes the predeclared raw fallback. Header, CRC/tail and FEC/rate-matching remain separate costs; source-bit reduction does not establish channel-use reduction.

## Actual source lengths

| Population | m | Raw bits | Arithmetic mean bits | P95 bits | Selected payload mean bits | Raw fallback | Source-bit reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 6 | 1092 | 896.081 | 1000.00 | 896.081 | 0.00% | 17.94% |
| calibration | 7 | 1860 | 1540.084 | 1673.00 | 1540.084 | 0.00% | 17.20% |
| calibration | 8 | 3060 | 2484.842 | 2699.00 | 2484.842 | 0.00% | 18.80% |
| calibration | 9 | 5088 | 3988.696 | 4343.05 | 3988.696 | 0.00% | 21.61% |
| calibration | 10 | 8160 | 5954.985 | 6514.20 | 5954.985 | 0.00% | 27.02% |
| development | 6 | 1092 | 898.510 | 982.05 | 898.510 | 0.00% | 17.72% |
| development | 7 | 1860 | 1540.410 | 1658.15 | 1540.410 | 0.00% | 17.18% |
| development | 8 | 3060 | 2484.720 | 2693.15 | 2484.720 | 0.00% | 18.80% |
| development | 9 | 5088 | 4000.650 | 4344.15 | 4000.650 | 0.00% | 21.37% |
| development | 10 | 8160 | 5982.650 | 6567.70 | 5982.650 | 0.00% | 26.68% |

All registered levels are retained. Source-token roundtrip succeeds for every formal source/mode in both families; this is noiseless codec correctness, not wireless reliability. Full distributions (mean,median,P90/P95,min,max) and flush/storage fields are in the CSVs.

## Decoder/representation grid

| Population | Path | PSNR dB | LPIPS | DINO |
| --- | --- | ---: | ---: | ---: |
| calibration | D0_F | 21.423189 | 0.144196 | 0.877946 |
| calibration | D0_Fq | 22.544885 | 0.103638 | 0.936114 |
| calibration | Dc_F | 26.177880 | 0.051278 | 0.957089 |
| calibration | Dc_Fq | 23.161176 | 0.096641 | 0.911709 |
| development | D0_F | 21.744214 | 0.149040 | 0.881553 |
| development | D0_Fq | 22.835888 | 0.105751 | 0.937683 |
| development | Dc_F | 26.472460 | 0.052310 | 0.957301 |
| development | Dc_Fq | 23.464584 | 0.098871 | 0.914152 |

Decoder adaptation changes the interface response: on original development, Dc(F) and Dc(Fq) differ by about3.008dB and0.04656LPIPS, while D0(Fq) has higher mean DINO than Dc(Fq). These descriptive comparisons do not isolate a decoder-independent quantization loss, establish a deployed communication winner, or supply free continuous latent information to RX.

## Prefix references: official ten-scale cumulative mapping

| Population | Path | PSNR dB | LPIPS | DINO |
| --- | --- | ---: | ---: | ---: |
| calibration | D0_prefix_m6 | 15.765350 | 0.528642 | 0.154981 |
| calibration | D0_VAR_m6 | 16.236114 | 0.296425 | 0.775421 |
| calibration | Dc_prefix_m6 | 15.971103 | 0.560457 | 0.102689 |
| calibration | Dc_VAR_m6 | 16.826769 | 0.279487 | 0.771471 |
| calibration | D0_prefix_m7 | 16.887791 | 0.412461 | 0.318305 |
| calibration | D0_VAR_m7 | 17.813170 | 0.229957 | 0.836809 |
| calibration | Dc_prefix_m7 | 17.107167 | 0.428861 | 0.244666 |
| calibration | Dc_VAR_m7 | 18.470253 | 0.214671 | 0.827911 |
| calibration | D0_prefix_m8 | 18.241265 | 0.304418 | 0.562365 |
| calibration | D0_VAR_m8 | 19.396638 | 0.173890 | 0.885613 |
| calibration | Dc_prefix_m8 | 18.591905 | 0.291822 | 0.500742 |
| calibration | Dc_VAR_m8 | 20.109259 | 0.161077 | 0.869723 |
| calibration | D0_prefix_m9 | 20.007081 | 0.199426 | 0.799103 |
| calibration | D0_VAR_m9 | 20.971554 | 0.132629 | 0.916903 |
| calibration | Dc_prefix_m9 | 20.501996 | 0.178748 | 0.760937 |
| calibration | Dc_VAR_m9 | 21.686118 | 0.122513 | 0.895751 |
| development | D0_prefix_m6 | 16.030072 | 0.531085 | 0.148544 |
| development | D0_VAR_m6 | 16.593995 | 0.295587 | 0.791689 |
| development | Dc_prefix_m6 | 16.289128 | 0.560141 | 0.089623 |
| development | Dc_VAR_m6 | 17.173915 | 0.278304 | 0.786601 |
| development | D0_prefix_m7 | 17.159342 | 0.411346 | 0.320053 |
| development | D0_VAR_m7 | 18.101699 | 0.233804 | 0.849902 |
| development | Dc_prefix_m7 | 17.471025 | 0.424828 | 0.244155 |
| development | Dc_VAR_m7 | 18.753152 | 0.218405 | 0.841467 |
| development | D0_prefix_m8 | 18.500434 | 0.305580 | 0.548868 |
| development | D0_VAR_m8 | 19.691624 | 0.177141 | 0.889730 |
| development | Dc_prefix_m8 | 18.874802 | 0.290428 | 0.499452 |
| development | Dc_VAR_m8 | 20.394318 | 0.164718 | 0.875285 |
| development | D0_prefix_m9 | 20.267994 | 0.202388 | 0.792859 |
| development | D0_VAR_m9 | 21.243728 | 0.135896 | 0.923859 |
| development | Dc_prefix_m9 | 20.748462 | 0.179666 | 0.762796 |
| development | Dc_VAR_m9 | 21.945783 | 0.125811 | 0.902628 |

m10 is the complete true quantized latent Fq; no unnecessary suffix generation is performed. These direct-prefix and VAR references are not substitutes for independently trained Prefix-RX communication models.

## Frozen common targets

| Target | Calibration reference | PSNR minimum | LPIPS maximum |
| --- | --- | ---: | ---: |
| high | Dc_Fq | 22.98349190 | 0.09431767 |
| balanced | Dc_VAR_m8 | 19.90531254 | 0.16185233 |
| coarse | Dc_prefix_m6 | 15.76817036 | 0.56598300 |

These are the registered median-based calibration targets. The exact values and calibration source-CSV SHA are in `experiments/token_channel_efficiency_20260923/quality_targets.json`. The development registration embeds the same frozen object before development loads. All targets, including unattained ones and negative resource savings, must remain in subsequent B1/B2 tables.

## Remaining scope

Only experiment A is complete here. P2048/P3060 have separate real-device optimizer/gradient/energy/resume acceptance, and budget training is ongoing. New digital quality grids, selected development/online timing,16QAM energy-constrained results, minimum tested N and merged short-prefix C remain incomplete. The old P4084/digital/mixed results keep their original scope; no new finite-channel ranking is inferred from this report.
