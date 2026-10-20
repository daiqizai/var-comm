# Current five-method selected examples

All 20 receiver outputs exactly match the first formal timing output hashes. All 20 are decoded outcomes; no rejected outcome was replaced. Four original crops are pixel-identical to the prior selection.

Additional work: 20 RX reconstructions, 20 PSNR and 20 LPIPS pairs. Reused: 20 actual received inputs and 4 original PNGs. No TX/source encoding, channel events, training, timing grid, bootstrap or common500 evaluation.

The original metric worker exited 1 only at its final hook-count assertion, after all 20 pair records were saved. The LPIPS implementation directly invokes its backbone forward function and bypasses those hooks. The original failure is retained; a separate CPU-only saved-output audit validates the complete records without rerunning any metric. Do not report the original scorer as exit 0 or its hook counter as measured telemetry.

Metrics are defined on exact native float outputs, matching the main evaluation; PNG materials are uniformly quantized display copies, with no crop or enhancement.
