# QPSK calibration and selected continuous development — 2026-09-25

Two real stages are complete and independently audited. QPSK development is now running in the existing delivery chain. This is a stage delivery; the merged study, digital savings, C experiments and historical reference GPU checks remain incomplete.

## Scope and acceptance

| Completed stage | Sources | Eligible methods | Sealed cells | Actual frames | Online timings / warmups |
|---|---:|---:|---:|---:|---:|
| QPSK calibration | 1,000 original calibration | 26 | 26,000 | 390,000 | 0 / 0 |
| Selected continuous development | 100 original development | P2048, P3060, P4084 | 300 | 4,500 | 300 / 90 |

Both use registered SNR 1/4/7/13/19 dB and three noise seeds. Four of the 30 registered QPSK candidates are explicit raw-capacity exclusions, retained in candidates.json. No frames from eligible methods are discarded on header/CRC failure or source overflow. The QPSK calibration table retains 10 header failures and 71,746 body-CRC failures (these categories can overlap); source-overflow erasures are zero. Continuous development has zero counts in these fields.

The continuous stage passed **12/12 real selected replay checks** (two registered sources, SNR 1/13 dB, three budgets). These checks actually load the selected trained heads, extending the earlier shared-execution seeded-probe acceptance. They do not certify unexecuted historical methods.

| Budget | Selected step | Selected checkpoint SHA256 |
|---|---:|---|
| P2048 | 30,000 | 78cb4c480c97c23a4577ee63716b5a8e5cdb9726d87e9cdacc9656ff38cd7751 |
| P3060 | 37,500, selected from the completed 40k run | 962a1de0b7f25ba080c04b6e2b91b3c1cb999878985509ebbb917f4b55cce794 |
| P4084 | historical 10,000 | b6bd1f0bb01bbd734eeb6a63afac1b8f7c57998085c616011f37554035434e1a |

The P2048/P3060 calibration stopping decisions remain unchanged. P3060's 40k regression is retained. Rule-based stopping is not a claim of theoretical convergence. The frozen visual model, decoder, losses, original A/B and hardware settings remain unchanged; P2048 microbatch8 and P3060 microbatch4 retain effective batch16.

## Continuous development results

Noise repeats are averaged within each source, then the 100 source values are averaged. PSNR is mean per-frame PSNR, not PSNR calculated from the population mean MSE.

| SNR dB | P2048 PSNR / LPIPS | P3060 PSNR / LPIPS | P4084 PSNR / LPIPS |
|---:|---:|---:|---:|
| 1 | 19.806 / 0.21277 | 21.198 / 0.15910 | 21.859 / 0.14056 |
| 4 | 21.198 / 0.16054 | 22.665 / 0.11811 | 23.353 / 0.10221 |
| 7 | 22.024 / 0.13406 | 23.568 / 0.09755 | 24.244 / 0.08449 |
| 13 | 22.750 / 0.11452 | 24.428 / 0.08087 | 25.131 / 0.06974 |
| 19 | 22.957 / 0.10954 | 24.688 / 0.07632 | 25.417 / 0.06578 |

Complete MSE, PSNR, LPIPS and DINO values and N/E distributions are in [summary.csv](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/summary.csv). Plots: [quality vs SNR](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/quality_snr_db.svg), [quality vs N](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/quality_N.svg).

For example, P3060 improves mean PSNR over P2048 at 1 dB by 1.392 dB (image-bootstrap 95% interval 1.275–1.512 dB). All three budget pairs at all five SNRs are in [paired differences](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/paired_budget_differences.json). Bootstrap uses 10,000 resamples of source images after averaging noise. These intervals do **not** include training-seed variability. Different N values have distinct registered noise namespaces; paired images do not imply identical received waveforms.

## Frozen quality targets and resource accounting

The pre-development targets require both PSNR and LPIPS. High: PSNR >=22.983491897583008 and LPIPS <=0.09431766718626022; balanced: >=19.905312538146973 and <=0.16185232996940613; coarse: >=15.768170356750488 and <=0.56598299741745.

| SNR dB | High: minimum tested continuous N | Balanced | Coarse |
|---:|---:|---:|---:|
| 1 | Not reached | 3060 | 2048 |
| 4 | Not reached | 2048 | 2048 |
| 7 | 4084 | 2048 | 2048 |
| 13 | 3060 | 2048 | 2048 |
| 19 | 3060 | 2048 | 2048 |

These are simultaneous **population mean** targets, not a guarantee that every source meets the target. Per-source attainment rates and intervals remain in [continuous_targets.json](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/continuous_targets.json). No interpolation is used. Continuous N is the actual complex channel-use count and energy follows the registered per-frame 2N constraint. Digital source bits are not substituted for channel uses.

**Digital minimum N and same-quality savings are still pending completed digital development.** The 1,000-source calibration table is not compared directly against the 100-source development table. Unreached targets remain explicit.

## Online compute timing

| Method | Mean TX ms | Mean RX ms | Mean TX+RX ms |
|---|---:|---:|---:|
| P2048 | 8.2235 | 12.5131 | 20.7366 |
| P3060 | 8.2252 | 12.5304 | 20.7556 |
| P4084 | 8.2263 | 12.4297 | 20.6560 |

Each method uses ten fixed original development sources (indices 0,11,...,99), five SNRs, seed2001 and two measured repetitions, with three warmups per source/method. CPU uint8 RGB to CPU waveform defines TX; CPU observation to CPU float RGB defines RX. Channel noise is outside RX. Each timed replay is checked against the corresponding quality output (maximum absolute difference <=2e-5). These are online compute timings, not over-the-air transmission latency; sub-millisecond differences are not presented as a speed ranking. Raw timings, warmups and [timing plots](../results/token_channel_efficiency_20260923/grid_milestones_v1/continuous_development/online_timing.svg) are included.

## QPSK calibration decisions

The audit recomputed every eligible method's source-averaged MSE + 0.1 LPIPS utility and verified all 30 family/N/SNR choices against the complete candidate sets. DINO is report-only. Policy [ba98f505...](../results/token_channel_efficiency_20260923/grid_milestones_v1/QPSK_calibration/policy.json) is frozen for QPSK development. Full candidate [quality vs SNR](../results/token_channel_efficiency_20260923/grid_milestones_v1/QPSK_calibration/quality_snr_db.svg) and [quality vs N](../results/token_channel_efficiency_20260923/grid_milestones_v1/QPSK_calibration/quality_N.svg) retain low-SNR failures.

## Publication and verification

The independent CPU-only publisher [tools/publish_grid_milestones.py](../tools/publish_grid_milestones.py) verified 26,300 sealed cells, exact original frame-table hashes, local received-preview hashes, source/preprocessing/SNR/noise/run coverage, source-averaged summaries, actual N/E, registration bindings (64 calibration / 67 continuous), stage receipts and selected lineage. Its engineering tests validate byte-preserving CSV publication, not model quality.

[Publication index](../results/token_channel_efficiency_20260923/grid_milestones_v1/index.json) records file SHA256 values. The complete original CSVs are split into ordered parts below the repository file-size limit. Concatenate per_frame_parts/manifest.json paths as **bytes in listed order**; only the first part has a header. The reconstructed hash must equal original_sha256. This retains every original frame and bit/control/FEC ledger field; no image pixels, weights or tensors are uploaded. Per-source means, failure counts and all receipts accompany the frame parts.

Canonical QPSK table SHA256: 22333be91c39b276d3b1777bddf164135a692082a69fd4a50d8d6b7f501ed00c.

Canonical continuous table SHA256: a9d07c4ee41ef89977fc81cb3c702f9dc0d45ae796cecc34eea25fc86f9d240c.

## Remaining delivery

The original queue continues QPSK development, 16QAM calibration/development, combined B1/B2 statistics, C first matrix and followups, then the four registered historical metric/replay/timing workers. Full selected/lineage comparisons, cross-seed uncertainty, historical PHY/metric/runtime compatibility, same-quality savings and the final merged report remain required. No new queue, holdout, learned content selector, original A/B retraining or shared hardware change is introduced by this publication.
