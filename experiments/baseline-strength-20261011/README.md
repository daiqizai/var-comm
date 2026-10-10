# Registered baseline-strength continuation

Read PROTOCOL.md and the two JSON protocols. The user-authorized Swin optimizer reset and finite microbatch4 correction are described in results/baseline_strength_20261011/R0_inventory/R0_inventory.md. P uses the original complete40000-step state; Swin starts from80000 weights with fresh Adam/order/RNG, preserving its historical exposure. Both use effective16, original frozen mathematical functions, FP32 and no TF32.

The two existing launch_job.py owners must not be duplicated. Inspect results/baseline_strength_20261011/{P_continue,Swin_1to19}/status.json and the corresponding personal temporary jobs/*/exit.json. STOP in the branch result directory or SIGTERM to the verified worker requests a safe checkpoint. Unknown failures do not retry automatically. Source files bound by live workers must not be edited.

Training is bounded by P total100k and Swin added120k/total240k; original calibration-only objective, source sets, all cells and absolute two-window stability determine selection/stopping. The first stage reaches P80k / Swin added80k before a stability decision. Failed cases remain in denominators. No capacity experiment is authorized by default.

Each branch requires a successful R0 receipt before --mode train. Reproduction also requires the external exact-input manifest, parent checkpoints and existing frozen runtime; paths are explicit in baseline_runtime.py. Checkpoint weights and datasets must never enter Git. Every1000 updates creates a full recovery state; Swin keeps two atomic recovery slots plus full milestone states and all calibration weight candidates. CSV history is in the temporary log tree with complete calibration tables in the result tree.

This entry completes training and calibration selection. Its completion explicitly leaves milestone four-metric diagnostics, P no-noise diagnostics, subsequent common500/new100 frozen-model evaluation and same-environment timing pending; it does not silently launch those stages or claim them complete.
