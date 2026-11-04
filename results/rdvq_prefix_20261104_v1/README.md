# RDVQ actual-bit prefix study — first-batch evidence

**P0 and P1 are closed. P2 matched training is active. The first batch is not
complete.** This directory is an independent research record; the NeST-Com WCL
manuscript and stopped research branches are unchanged.

- [P1 results and limitations](p1_delivery_v1/REPORT_zh.md): 47,212 native
  source-pair metric records; 14,400 real calibration radio frames; source curves
  and conditional expected controller comparisons. Unfavorable metrics remain.
- [P0 qualification](../../experiments/rdvq-prefix-20261104/reporting/P0_ACCEPTANCE_zh.md)
  and [reproduction guide](../../experiments/rdvq-prefix-20261104/reporting/REPRODUCING.md).
- [Code and packet layout](../../experiments/rdvq-prefix-20261104/README.md),
  [P1/P2 protocol](../../experiments/rdvq-prefix-20261104/P1_P2_PROTOCOL.md),
  [P2 execution](../../experiments/rdvq-prefix-20261104/P2_EXECUTION.md).
- [Canonical source identities](p1_delivery_v1/source_identities.csv),
  [raw PFS evidence](p1_delivery_v1/raw_evidence_index.json),
  [retained failures](p1_delivery_v1/retained_failures.json),
  [environment](environment_identity.json), [data-hash clarification](data_identity_clarification.json).
- [Data review](p1_delivery_v1/validation/data_review.json) and
  [actual visual review](p1_delivery_v1/validation/visual_review.json).
  The initial collector/figure metadata retain their packaging-time PENDING
  status; these later timestamped review records document the completed review.

P1 indicates a small DISTS benefit from the source-informed controller reference,
with worse PSNR means at all six points and mixed/worse perceptual metrics.
This does not motivate a complex learned selector. Controller test values are
conditional expectations using measured reliability, **not new test PHY events**.

P2 compares R1, token-prefix R2, actual-rate-calibrated R3, and estimated-only R3
at 4,000 matched updates. R2/R3/estimated-only R3 each have three training seeds.
The unique registered continuation owns training, native evaluation and final
statistics. This milestone contains its frozen source/configuration, not any
unobserved P2 efficacy result. Rolling checkpoints and all large arrays stay on
PFS. Resource caps are P0/P1/P2 = 16/64/256 GPU-device-hours; P2 reserves 190.
P3/P4 are not automatically run.

At P1 closure, charged GPU time is 0.08347030765 h for P0 and 7.00941781839 h for
P1. Actual process receipts retain inclusive parent CPU separately from leaves;
do not sum both. Repository CPU engineering checks passed 264 tests. These tests
are not real-image acceptance; model/PHY evidence is recorded separately.
