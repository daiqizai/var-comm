# Baseline continuation R0 — 2026-10-11

The original P40k full checkpoint was actually loaded on CPU: model/buffers, all60 AdamW states at step40000, LR2e-4, 20k-source order/cursor, channel generator, and CPU/CUDA RNG are present. SHA matches the published parent.

The available Swin80k file contains model, step and registration identity only. No81551 full checkpoint or80k optimizer/order/RNG was found in the inspected recovery inventories; the old20000 SSH endpoint still closes the connection. The user explicitly authorized weights-initialized additional training. Fresh Adam1e-4, new registered order/channel/RNG seeds and historical80000 exposure are recorded; this is not exact continuation. No recovered optimizer LR is claimed.

All421 required original RGB and continuous-latent train/calibration cache files were restored from the local20260929 archive and verified against the original P/Swin registrations:20,000 train and1,000 calibration sources. No dataset download or holdout preview occurred.

P R0 passed on H800 GPU0. Swin microbatch16 hit the private16GiB allocator cap during the first isolated update; the failed log is retained. The finite correction uses original train_update with microbatch4/effective16 and passed on H800 GPU2. Both successful probes perform two isolated updates and replay the second after a real save/reload; all probe state is discarded. P verified frozen Dc/LPIPS hashes and absent parameter gradients, nonzero communication gradients, shape[16,1024,2] and total energy2048. Swin verified real paid N1024 metadata, zero-noise physical/direct equivalence and save/reload equality. Tolerances are stated in the source. These are same-environment engineering checks, not a claim of Torch1.12/2.11 or cross-device bitwise training equivalence.

A pre-training review corrected P's original unused noise-index draw to randint(2), as required by the original two training_base_seeds. The successful P probe and its original source bindings are immutable; a separate admission records this unexecuted-training-line correction and Swin-only changes. No formal update had run before this correction.

New code and durable text records live under modelTeam/code/liulu; model/recovery states under modelTeam/liberai-checkpoint/liulu; caches and ordinary logs under modelTeam/liberai-tmp/liulu. No PFS symlinks or shared environment modifications were made. Original restored assets are reused in place.

Both formal owners are dispatched independently with16GiB private allocator caps, two CPU affinities each and finite training/calibration budgets. Startup full calibration is not an added training update. The old monitor remains paused. Milestone four-metric diagnostics, P bottleneck diagnostics, final frozen-model evaluations and unified timing remain subsequent pending stages.
