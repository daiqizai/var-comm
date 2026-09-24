# QPSK development and matched continuous comparison — 2026-09-25

QPSK development is complete: **39,000 real frames from 100 original development sources, 26 eligible candidates, five SNRs and three noise seeds**, plus 2,600 complete online timings and 780 warmups. The original delivery chain has moved to 16QAM calibration. No new holdout, selector or training run was introduced.

The independent CPU publisher verifies all 2,600 sealed digital cells and 300 continuous cells, completion and stage snapshots, registration/source bindings, received-preview hashes, exact frame-table identities, summaries, all timing/warmup keys, endpoints and timed quality replay. Continuous selected replay remains 12/12 PASS. The combined comparison contains 43,500 frames and 2,900 measured timings, using exactly the same 100 source/preprocessing identities and frozen decoder.

## Resource result

Within the **tested budgets N2048/3060/4084**, the calibration-selected QPSK digital families do not reduce the minimum N relative to the selected continuous models at the registered targets. This conclusion is limited to these models, training histories, source population, PHY and tested budgets.

The table below gives minimum tested N. A dash means the mean PSNR and LPIPS target was not jointly reached; it is not a missing run.

| Target | SNR dB | Continuous N | Arithmetic QPSK N | Raw QPSK N |
|---|---:|---:|---:|---:|
| High | 1 / 4 | — | — | — |
| High | 7 | 4084 | — | — |
| High | 13 / 19 | 3060 | — | — |
| Balanced | 1 | 3060 | — | — |
| Balanced | 4 | 2048 | 4084 | 4084 |
| Balanced | 7 | 2048 | 3060 | 3060 |
| Balanced | 13 / 19 | 2048 | 2048 | 3060 |
| Coarse | all five SNRs | 2048 | 2048 | 2048 |

Digital saving is defined as **1 − digital_N / continuous_N**. At the balanced target it is −99.414% for both digital families at 4 dB, −49.414% for both at 7 dB, and 0% / −49.414% for arithmetic / raw at 13 and 19 dB. Coarse-target savings are 0%. Unreached targets retain null savings. These are measured-grid minima without interpolation, not a continuous optimum or an information-theoretic bound.

Frozen targets (PSNR minimum / LPIPS maximum): high 22.983491897583008 / 0.09431766718626022; balanced 19.905312538146973 / 0.16185232996940613; coarse 15.768170356750488 / 0.56598299741745. They were frozen from calibration before this development run. Joint source/frame attainment rates are separately retained and must not be confused with population mean attainment.

Full evidence: [minimum uses](../results/token_channel_efficiency_20260923/QPSK_development_v1/minimum_uses.json), [CSV](../results/token_channel_efficiency_20260923/QPSK_development_v1/minimum_uses.csv), [savings plot](../results/token_channel_efficiency_20260923/QPSK_development_v1/minimum_N_savings.svg).

## Example quality and time at 13 dB

| Family / N | Frozen choice | Mean PSNR dB | Mean LPIPS | Mean full TX+RX ms |
|---|---|---:|---:|---:|
| Continuous / 2048 | selected 30k | 22.7503 | 0.11452 | 20.7253 |
| Continuous / 3060 | selected 37.5k | 24.4285 | 0.08087 | 20.7226 |
| Continuous / 4084 | historical selected 10k | 25.1314 | 0.06974 | 20.6619 |
| Arithmetic / 2048 | m10 cap | 20.7360 | 0.15926 | 321.4162 |
| Arithmetic / 3060 | m10 cap | 22.5092 | 0.11841 | 344.3727 |
| Arithmetic / 4084 | m10 cap | 23.4646 | 0.09887 | 359.3084 |
| Raw / 2048 | m8 | 20.3943 | 0.16472 | 129.1092 |
| Raw / 3060 | m9 | 21.9458 | 0.12581 | 130.6681 |
| Raw / 4084 | m9 | 21.9458 | 0.12581 | 130.4239 |

All quality and time values above are at 13 dB. Full TX and RX components are in [combined_timing_summary.csv](../results/token_channel_efficiency_20260923/QPSK_development_v1/combined_timing_summary.csv). Arithmetic m labels are registered prefix caps: any capacity-driven fallback and actual transmitted m remain in each frame's ledger.

The raw N3060/N4084 quality equality at 13 dB is preserved. Additional allocated uses do not imply extra decoded source information. Source bits, control/FEC costs, actual N/E, decoded mode and failure fields remain in the frame records.

All digital choices are frozen by the original 1,000-source calibration policy (SHA256 ba98f50508d5a13a97e1325162b5da8421253bf5e8273a2d964948c07e24cd84). No development choice or retraining is performed. The 26-method grid remains available even where a candidate is not selected. Four original raw capacity exclusions are retained separately.

[Quality vs N](../results/token_channel_efficiency_20260923/QPSK_development_v1/quality_vs_N.svg) and [quality vs time](../results/token_channel_efficiency_20260923/QPSK_development_v1/quality_vs_time.svg) use the same validated combined table as all differences and resource estimates. The plots connect only measured points and do not imply interpolated target attainment.

## Statistics, failures and timing boundaries

All three noise seeds are averaged within source before averaging across images. Thirty same-N/SNR digital-versus-continuous comparisons include MSE, PSNR, LPIPS and DINO differences with 10,000 paired image-bootstrap resamples in [paired.json](../results/token_channel_efficiency_20260923/QPSK_development_v1/paired.json). Distinct noise namespaces and lengths do not imply identical received waveforms.

[Resource bootstrap](../results/token_channel_efficiency_20260923/QPSK_development_v1/minimum_uses_bootstrap.json) resamples those same source images jointly across methods, retains the frozen policy, and recomputes mean-target attainment and minimum tested N. Unreached repetitions are explicitly counted. Its saving interval is **conditional on both families reaching the target**; it must be read alongside the attained/unreached counts, not as an unconditional interval. Neither bootstrap includes training-seed variability. P2048/P3060 and historical P4084 had different recorded training opportunities, so differences are not attributed to bandwidth alone.

The complete QPSK development grid retains **7,085 body-CRC failures, zero header failures and zero source-overflow erasures**. Failures remain in quality and resource calculations.

Timing uses ten fixed original source indices 0,11,...,99, five SNRs, seed2001, two repetitions and three warmups per source/method. Every run covers CPU uint8 RGB → CPU waveform (TX) and CPU observation → CPU float RGB (RX), with one channel-noise application outside RX timing. Replay error is checked <=2e-5. These compute timings include the present PHY and decoder implementation; they are not over-the-air delay or an optimized-hardware lower bound. Offline source reuse does not replace the full timed TX/RX path.

QPSK and continuous are per-frame E=2N protocols. The forthcoming fixed-constellation 16QAM results have actual variable frame energy and will be reported separately.

## Reproduction and remaining work

[Publication index](../results/token_channel_efficiency_20260923/QPSK_development_v1/index.json) binds all files by SHA256. Original digital CSV bytes are preserved as ordered parts, with a header only in the first part; concatenate in per_frame_parts/manifest.json order and verify the original hash. All source-level means, complete bit/control/FEC ledgers, original timings/warmups, failures, policy and stage receipts are included. Continuous raw records are reused by SHA reference from the earlier grid milestone. Pixels and weights stay local.

Digital table SHA256: a30bc97adc9b539f7c30c6d4752f25d0e53cd53d3f94880df4f9ca90cc8afd83.

Combined table SHA256: ae3908260db3ad12b6e5facaeca7d547c5037add438e3e2c479f087461a5085c.

CPU publisher: [tools/publish_qpsk_development.py](../tools/publish_qpsk_development.py). It refuses to overwrite an existing publication and changes no active execution dependency.

**Full study delivery remains pending**: 16QAM calibration/development, combined B1/B2 review, C first matrix/followups and seed replications, four historical common-metric/PHY/timing workers, and the final merged paired report. The original queue continues; no additional worker is started.
