# Phase 1: local repairs and affected-result requalification

Baseline: 9e8bee77c946fb8ee6618cf4c1ae6cf2bd10d6dc. No uncommitted work or active user training at start. Original A/B, historical outputs and selected weights retained. The user authorized phase1 followed by phase2 as two separate commits. Source is changed only in the actual root.

## Repairs and scope

See results/review_20260923_phase1/issue_status.json for N01–N08. Unified predictor selection records carry the explicit control/candidate key and checked checkpoint SHA. The existing two pipelines retain encoder [residual,Ptx] and receiver start Prx; configuration no longer claims a separate Fb_TX condition. No predictor retraining occurred.

Linear control failure masks the complete innovation, not only the received measurement. Accepted zero norm remains a valid observation and is distinguished from packet rejection. W pools every calibration source/noise row before a single division and clip. The exact original projection A is reused by SHA, norm range derives from the original20k training population, and all32 norm-control plus992 measurement uses are paid. Base and enhancement calibration seed streams are independent and explicitly recorded. All failures remain in evaluation.

Both digital and enhancement statistics share complete source/SNR/seed/preprocessing validation and aggregation. Policy means reproduce exactly; paired intervals are regenerated with the registered source-bootstrap seed. Timing uses one noise application outside RX time, CPU RGB endpoints and direct GPU-to-CPU output without the redundant digital round trip. Every method is warmed5 times and measured twice in rotating order.

Follow-up resume now checks parent/latest hashes and registered code/configuration/decoder/cache scope using existing B binding helpers. Previously unbound old training state cannot silently resume under changed code. This run did not retrain or interrupt the historical adaptation/predictor trainers, so a real interruption/resume of those old runs is NOT_RUN, not inferred from CPU guards.

## Real results

Linear source projection: PSNR20.62047, LPIPS0.166431, DINO0.869913. Residual projection:20.56769,0.169248,0.866022. Changes from v5 are small; this is an implementation repair, not evidence of a new large gain. Full development coverage is100 sources x5 SNR x3 noises x2 methods=3000 rows. W fit covers3000 frames per SNR/arm.

Existing predictor candidate versus original-residual control: PSNR+0.072412 dB (95% source-bootstrap CI[0.059143,0.087049]); LPIPS−0.00129084 (CI[−0.00158210,−0.00099747]); DINO+0.00220297 (CI[0.00098087,0.00341839]). These are small complete-pipeline gains on the existing development population. They are not a new holdout result or isolated proof about residual compressibility.

Recomputed selected calibration utilities differ slightly from the old completion. The old execution source identity remains NOT_ESTABLISHED; old5000-step training is not denied. New selected checkpoint SHA and full rows are saved. The first new evaluation attempt finished valid calibration then failed at its development entry on a float64-noise/float32-model mismatch. That entry was corrected; unchanged calibration dependencies, selected SHA and complete keys were checked before reusing those rows in the successful v2 run. Failed logs remain local; no failed partial output is presented as completed quality.

## Acceptance and delivery

62 CPU tests pass, including duplicate/mismatched pairing, rejected norm fallback, legal zero norm, coverage, selected SHA and resume/config identity counterexamples. Synthetic CPU behavior tests are separate from actual model/data/GPU evaluation. Timing has52 real consistency comparisons and1300 measured calls; any supplemental stored-image comparison is recorded separately. New per-frame results, summaries, source-paired intervals, calibration rows, source identity and output lineage are in results/review_20260923_phase1. Weights, source images, tensor caches and old outputs are excluded from Git.

No original A/B training, new method, loss change, data expansion or holdout access was performed in phase1. Historical v5 linear and old predictor completion are superseded only for the affected reevaluated scope; unrelated original results remain preserved.

Supplemental acceptance: all30 comparisons of the three neural methods against stored normal-quality reconstructions PASS; maximum pixel difference4.58956e-6 (threshold2e-5).
