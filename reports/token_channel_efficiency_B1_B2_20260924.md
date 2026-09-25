# Token/channel efficiency B1/B2 actual resource results

Full source A is reported separately. This report uses 100 original
development sources, five SNRs and three noise seeds. All failures are retained.
Digital policies were frozen from1000-source calibration before development.
The selected P2048/P3060 checkpoints follow calibration-only continuation
decisions; P4084 is the explicitly retained historical10k checkpoint. Different
training opportunities are disclosed, not attributed to bandwidth alone.

Of 60 registered target/SNR/family/MCS cells, 34 have both
families meeting the mean quality target in the measured budget grid. Every
unreached target and negative saving is retained in minimum_uses.csv/json.
No interpolation or continuous-optimum claim is made. Per-frame joint attainment
is separate from mean attainment.

QPSK/continuous use per-frame E=2N.16QAM is shown separately with fixed
constellation scaling and actual varying energy; it is not a strict constant-E
comparison. Source bits are not channel uses. Prefix fallback, paid control and
CRC failures remain in the frame ledger.

Means, paired differences,10000 source-bootstrap intervals and plots derive from
one validated source/SNR/seed table. These intervals do not include training-seed
variation. Timing repeats full CPU RGB -> CPU waveform and CPU observation ->
CPU RGB execution, with one noise application outside RX timing. Offline quality
may reuse source encoding; reported timing never does.

Results: results/token_channel_efficiency_20260923/B1_B2_v1. Compact frame indices
resolve source ID/preprocessing and complete method/run context via index.json.
No weights, raw images, tokens or large tensor caches are published.

This B1/B2 milestone does not complete merged short-prefix C, its N3060
confirmation or training-seed repetitions. New holdout and selectors remain deferred.
