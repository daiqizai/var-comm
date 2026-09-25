# 16QAM development: completed grid and frozen-policy resource results

The real grid completed 100 original development sources x 29 encodable candidates x five SNRs x three registered noise seeds: 43,500 frames and 2,900 sealed cells. All 2,900 online TX/RX timing calls and 870 warmups are retained. The one raw N2048/m10 capacity constraint remains explicit in the 30-candidate ledger. There were 0 header failures, 12,754 body CRC failures, and 0 overflow erasures; failure frames remain scored.

The independent CPU publisher verifies stage/completion hashes, every sealed cell and preview, 64 context bindings, source/preprocessing/run/noise identities, actual N and paid headers/FEC, original CSV/summary, frozen policy and complete timing/replay grid. It joins the same previously qualified selected continuous 4,500 frames, 300 timings and 90 warmups into a 48,000-frame object:
`e7b3037e9248b82b45a1db7aa332d8b69055c968ec8e2dd35df43684e7d85880`.
The policy SHA remains
`e4e652bb01e620a9cfbf7a890709154f7e072cb2a8102e9477e97c8760c80755`.
Selected continuous checkpoints remain P2048 30k, P3060 37.5k and historical P4084 10k; their different training opportunities are disclosed.

## Minimum tested N

| SNR | Target | Continuous | Arithmetic 16QAM | Raw 16QAM |
| --- | --- | --- | --- | --- |
| 1/4 | high | not reached | not reached | not reached |
| 7 | high | 4084 | not reached | not reached |
| 13/19 | high | 3060 | not reached | not reached |
| 1 | balanced | 3060 | not reached | not reached |
| 4 | balanced | 2048 | not reached | not reached |
| 7 | balanced | 2048 | 3060 | 3060 |
| 13/19 | balanced | 2048 | 2048 | 2048 |
| all five | coarse | 2048 | 2048 | 2048 |

Savings are 1 - digital N / continuous N. Both digital families have -49.414% at balanced 7 dB and 0% at balanced 13/19 dB and every coarse point. Unreached cases remain null. There is no positive saving within this registered grid, frozen policy and target set; this is not a universal infeasibility statement. Relative to the separately published QPSK result, raw 16QAM balanced 13/19 dB uses N2048 instead of N3060, while balanced 4 dB is now unreached. Policies are calibration-selected; no development per-image mode selection is used.

These are actual-N comparisons under different frame-energy constraints. Continuous uses per-frame E=2N; 16QAM uses fixed constellation scaling with average-symbol energy 2 and varying per-frame E. The complete E mean/min/P05/P95/max ledger and energy figure are separate. Do not describe these as strict equal-frame-energy results or replace N with source bits.

## Example at 13 dB

| Method | N | PSNR dB | LPIPS | Full online TX+RX ms |
| --- | --- | --- | --- | --- |
| continuous | 2048 | 22.7503 | 0.114515 | 20.7253 |
| continuous | 3060 | 24.4285 | 0.080868 | 20.7226 |
| continuous | 4084 | 25.1314 | 0.069738 | 20.6619 |
| arithmetic m9 | 2048 | 21.9458 | 0.125811 | 334.0274 |
| arithmetic m10 cap | 3060 | 23.4646 | 0.098871 | 361.0772 |
| arithmetic m10 cap | 4084 | 23.4646 | 0.098871 | 362.0961 |
| raw m9 | 2048 | 21.9458 | 0.125811 | 131.2272 |
| raw m10 | 3060 | 23.4646 | 0.098871 | 29.8874 |
| raw m10 | 4084 | 23.4646 | 0.098871 | 30.1338 |

The full raw m10 path avoids suffix generation by the registered design; timing is not inferred from payload length. Actual arithmetic fallback and received fields remain in the per-frame ledger. Timing uses CPU uint8 RGB -> CPU waveform and CPU observation -> CPU float RGB, three warmups and two measured repeats for each of ten registered sources/methods across five SNRs at noise seed 2001. One channel noise application is outside RX timing. Replay error is checked <=2e-5.

## Statistics and publication

Results are in `results/token_channel_efficiency_20260923/QAM16_development_v1`: original CSV lossless parts, all source/bit/control/FEC fields, source means, 30 same-N/SNR paired comparisons, frozen targets/policy/lineage, minimum tested N, failure and energy tables, quality-N/time and energy SVGs, and SHA index. All means/differences/intervals/plots use the same complete source/population/preprocessing/SNR/noise/run object.

Bootstrap averages registered noise within each source, then resamples paired source images 10,000 times. Training-seed variation is separate and not included. Minimum-N bootstrap keeps both continuous and digital unreached counts; savings intervals are conditional on both attaining the mean target, not unconditional intervals. The frozen policy is not refit on development or bootstrap samples. Equal seed indices do not imply identical observations across waveform lengths/namespaces.

A first CPU publication attempt used calibration's image_id field for development's indexed RGB binding and stopped before creating the publication directory. The independent publisher was corrected to verify index and RGB SHA against ordered registered sources, with a regression test. Original failed log is retained locally; no GPU output, registered model/PHY or bound source changed. The publication directory is immutable and main must not be rerun. Scientific figures were regenerated from the same published tables for visual inspection; no pixels or weights are published.

This stage does not complete C, the four historical GPU workers, or final all-method paired delivery. Keep the existing queue and its exclusive GPU/thermal gates. New holdout and content selectors remain deferred.

## Completed B1/B2 aggregate review

The unchanged scheduler subsequently completed B1_B2_v1 and advanced to C_first_matrix. Independent CPU review reproduced the 87,000-frame common development object (SHA 56a17b9fbeb391bff09043246c144808e776da8ff7714f17e81fc4df4bfbfa07), all 60 paired comparisons and minimum-N decisions, and 5,800 original online timing rows. It compared every field of all 912,000 compact records (390,000 QPSK calibration, 435,000 16QAM calibration, 87,000 development) with their sealed original cells, including waveform/observation SHA, received fields, failed frames and actual bit/FEC/N/E ledgers. All 66 registered generated artifacts and report/lineage hashes matched. Six PNGs passed image integrity checks; representative quality-N and quality-time layouts were visually reviewed.

The aggregate artifacts and original scheduler report are retained unchanged; its report filename token_channel_efficiency_B1_B2_20260924.md is the registered filename, not a claim that this completion occurred on that date. The independent review receipt is results/token_channel_efficiency_20260923/B1_B2_review_v1/audit.json. The first review attempt stopped on continuous cell filename formatting; the auditor was corrected to use actual N for continuous cells, preserving its failed log. No experiment or original generated output changed. These are engineering audit fixes, not new quality runs.
