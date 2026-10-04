# N2048 complete original M1 grid

This isolated extension keeps the published M1 receiver, visual models, source
populations, seeds, accounting and selection rule. Whole m4–m8 and partial
m4–m7 produce 69 legal actions per modulation, 138 total. Every action is
calibrated on 1,000 sources × 3 seeds × 5 SNRs; no screening shortlist is used.
Development has 100 sources × 3 seeds and eight method labels per PHY/SNR.

## Active run

- Original execution evidence: `outputs/M1-N2048-FULL-GRID-20261004`.
- Qualified faster execution: `outputs/M1-N2048-FULL-GRID-20261004-FAST-R1`.
- `fast_owner.py` runs qualification, a two-source benchmark and exact row
  parity against the original execution, then complete calibration/development.
- `deliver_when_complete.py` waits for that registered run, scores the frozen
  unified metric suite, builds the report, runs repository checks, pushes
  normally, and stops. Any failure is recorded and is not automatically retried.
- Inspect `calibrate_status.json`, `delivery_status.json` and failure receipts
  before starting anything. Do not launch another owner or edit bound sources.

The execution wrapper polls an already healthy resource monitor every five
seconds. The first check, detected hot states and exceptions are not cached.
New contention or heat may first be detected up to five seconds plus one check
boundary later. Original thermal thresholds, source-safe stopping, inference,
precision and RNG are unchanged. Two complete calibration sources, 4,140 rows,
matched exactly; measured mean source-grid time fell from about 106.7 to 34.9
seconds. This is offline cached workload timing, not online receiver latency.

## Reporting

QPSK uses exact per-frame E=4096. Report 16QAM's actual fixed-constellation energy
separately. K=0 selections are whole-scale cases, not ordering gains. The oracle
pays its mask and uses the old token-mismatch ordering; it is not an image-optimal
mathematical upper bound. Final image metrics and intervals are source paired,
with the three noise repeats averaged first. Historical P2048 and N512/N1024
results are reused only with verified source/model/reference provenance.

No adapter training, old queue, new loss, new visual model, m9 or partial-m8
codec extension is included. Scientific completion is determined by the actual
run receipts, not by code presence or CPU test success.
