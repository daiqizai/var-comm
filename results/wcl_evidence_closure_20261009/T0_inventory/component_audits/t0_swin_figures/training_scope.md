# Frozen Swin checkpoint and training scope

The evaluated checkpoint is **exact step 80000**, SHA-256 `8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21`. The actual training process was paused at step 81551. The fixed80k `selected_swin.json` explicitly records `selected_by=explicit_user_fixed_step`: calibration verified the user's selected checkpoint; it was **not** the best checkpoint selected across a completed converged run. The earlier generic training protocol's best-calibration rule must not override this later exact-milestone selection record.

The registered neural configuration is the author's SwinJSCC SA+RA base modules at 256x256, initialized randomly, trained on the original ImageNet 20k. C6 and C13 were sampled in balanced alternate updates; training SNR was an integer sampled uniformly from 1 through 13. Loss was clamped-RGB MSE, FP32, TF32 off, deterministic algorithms. The original calibration population has 1000 sources at SNR 1/4/7/10/13, one registered noise seed 4101, and both supported budgets/rates. The 80k verification recorded 10000 source-working-point conditions and zero header failures over those calibration conditions.

At N1024 only C6 is within the trained-rate and actual-capacity intersection. C13 requires 1664 body complex symbols before any header; C7 is accepted by the source code but lacks checkpoint training support. No new channel count, checkpoint, neural training, or adapted policy is introduced by this audit.

**19 dB remains outside both training and calibration ranges.** Keep the 19 dB point, hollow marker and extrapolation segment in external plots. Use the public name “SwinJSCC-80k (adapted)” or the equivalent explicit adapted-checkpoint label; do not call it the author's official optimal or fully converged result.

Budget truncation and `scientific_convergence_proven=false` are stated in the exact training-view completion and selection records. The local scope includes those records and the exact protocol, but a complete per-update learning curve was not located in this scoped audit. If a curve is wanted, fetch the already-existing remote training logs/calibration history; absence of a local curve is not evidence that training or calibration was not performed.

All read files, their byte counts and SHA-256 are in `provenance.json`. `adapter_validation.json` records the completed 20-source same-observation implementation check separately from paid-metadata deployment quality.
