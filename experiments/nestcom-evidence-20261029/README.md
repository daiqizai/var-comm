# Bounded NeST-Com evidence diagnostics, 2026-10-29

Scientific execution is complete. Base HEAD: `8bab318fbc8f8f115e96c9aeffeef10d3c2bb4b6`. Runtime: `/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_evidence_20261029_v1`.

Frozen protocols/results: `results/nestcom_evidence_20261029_v1/`. Registrations bind code, actual RX inputs, model hashes, numerical routes, budgets and process identities. Do not launch these registries again or overwrite successful outputs.

## Executed sequence

1. `prepare_inputs_v2.py` joins full source IDs, original indices, preprocessing identities, raw pixels and archive/native hashes. It verifies the complete original 100-source bank order. The initial schema failure made zero model calls and remains preserved.
2. `register_receivers_v2.py` and `control.py` launched E1 roles `e1_a1` through `e1_a4` and E2 roles `e2_61` through `e2_64`, each running `receiver_worker.py <registration>` once. E1 rebuilds context/KV and follows the greedy tail from actual class/prefix, with CFG=4. E2 keeps S=10 transforms, the full latent grid and the same decoder, using only received WHOLE contributions. Four fixed original-path qualification replays per experiment exactly matched native pixels.
3. `prepare_metrics.py` and `metric_worker.py <registration>` use GPU6/torch 2.11 for PSNR, LPIPS and DreamSim. `metric_worker_clip_path_v2.py <registration>` uses GPU1/torch 2.10 for CLIP. There are 485 new native pairs per metric; source features retain the original physical-device and numerical identity. The original CLIP path attempt failed before inference; the independent path revision uses an existing hash-identical relocated weight file.
4. `statistics_schema_v3.py` ran once after closure, using the original 10,000 x 100 bank for 52 mean and 44 directly paired confidence intervals. The preceding schema attempt failed before any interval calculation on the historical archive/key/slot locator. The immutable failure and independent schema adapter are retained.

Registration/control JSON files contain the exact executed `argv`. This README does not authorize a fresh launch. `test_receiver.py` and `test_statistics_schema_v3.py` are synthetic CPU engineering checks, not image-quality evidence. New PHY, calibration, training, source encoding, tokenizer calls, bootstrap draws and `figure_restore` calls are zero.

## Environment and provenance

Execution used PRO6000 GPUs on shared PFS: `var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python` with torch 2.11, and `/usr/bin/python` with torch 2.10 for CLIP. FP32 and TF32-off routes are fixed by registration. E1 used GPUs 4-7; E2 used GPUs 0-3; metrics used GPUs 6 and 1. Models, arrays and full logs remain outside Git.

The pre-execution `cache_identity.json` is not a compute ledger. Consult `actual_compute_and_reuse.json`, `execution_closure.json`, registrations and `failure_preservation.md`. All executed roles have closed, with original failures preserved. Publication documents are independent of bound scientific code. No unknown failure, new training or paused-branch recovery is claimed.
