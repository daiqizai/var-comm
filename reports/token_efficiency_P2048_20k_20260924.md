# P2048 initial 20k milestone (2026-09-24)

The actual P2048 training completed its first 20,000 updates and all nine full calibration rounds. This is a milestone, not convergence or completed development evaluation. P3060 entered the existing serialized queue automatically; no second trainer was launched.

The publisher checked the real selected checkpoint SHA, historical registration, extra microbatch8 execution identity, 61 bound files, and all 135,000 numerical calibration rows. Each round contains exactly 1000 sources x five SNRs x three calibration noise seeds; source indices and image identities are identical across rounds. CSV hashes, finite metrics, U_image construction, selection means and the selected step were independently checked. Original real GPU qualification receipts are included, not rerun or replaced by synthetic tests.

The selected checkpoint is step20,000. Registered selection utility at15k/17.5k/20k is0.03018624593348553 /0.029956301330402495 /0.02970021492034818. Consecutive relative improvements are0.76175% and0.85487%, both above the registered0.2% threshold. The delivery scheduler must evaluate this rule after both initial budgets finish and then extend by10k. Publication itself does not modify decisions or launch work.

Selection utility includes the registered latent auxiliary term. MSE+0.1LPIPS (U_image) is separately reported; it is not silently substituted for the original selection criterion. P2048 microbatch8 keeps effective batch16 and the registered precision/loss; small floating trajectory differences were already disclosed in its execution registration.

Artifacts: [numerical calibration and lineage](../results/token_channel_efficiency_20260923/budget_milestones/P2048_seed2026092304_20k/), [audit](../results/token_channel_efficiency_20260923/budget_milestones/P2048_seed2026092304_20k/audit.json), [curve](../results/token_channel_efficiency_20260923/budget_milestones/P2048_seed2026092304_20k/calibration_curve.svg).

Only numerical rows, receipts, hashes and a scientific plot are published. Checkpoints, source pixels and caches remain local. No development or new holdout was read by this publication. Remaining: extensions, real shared execution acceptance, selected development/timing, full B1/B2/C and historical reference compatibility/metrics/timing delivery. New holdout and learned selector stay deferred.
