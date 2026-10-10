# Full1000 recheck after the106-action pilot

This stage is gated by the actual complete pilot receipt and its bound ranking CSV. It is not launched automatically by the pilot and is not a test result.

For each of10 and19 dB, the fixed recheck set is the union of pilot top5, the original WHOLE winner, the original PARTIAL winner, and all seven original full-budget K=0 configurations. Candidate/wire duplicates are evaluated once. The original PARTIAL choice is a fixed reference; K>0 configurations cannot win the expanded-WHOLE selection. The rule conservatively retains all full-budget actions rather than selecting which are relevant after seeing results.

All1,000 sources from the original calibration manifest and all three original seeds4101/4102/4103 are used. The original noise and public counter functions are unchanged. The maximum union has14 actions per SNR,84,000 logical frames and168,000 header/body packet calls. The actual union and cap are written by metadata preparation before any launch. Existing original-calibration and pilot RX/images/scores are reused only on exact identities and hashes. A separate SQLite ledger owns this stage; no previous ledger is written.

The receiver, reconstruction, source-first metric reduction and identity guards match the pilot implementation. The pilot's real one-image RGB/metric qualification is referenced rather than rerun for sequential execution. New source and state checkpoints permit source-level resumption, with unresolved decode/render reservations blocking automatic repetition.

An optional two-worker GPU mode is registered before the full recheck with `--gpu-workers 2`. It uses the original M1 cohort admission code, with exact PID, process start, UID and command checks. The parent holds the common visual lock and the full-run GPU lock; two children hold separate worker locks and own even/odd source indices. No concurrent worker shares a source. The original thermal checks remain active, and an unrelated GPU process stops this owner rather than being killed. Each process retains the original six intra-op threads, two inter-op threads, deterministic FP32 flags and batch-one VAR/metric calls.

Before the first parallel production source, the two workers replay one fixed non-gray received state each from pilot sources 0 and 1. Both models must first be loaded; the two compute windows must overlap. The resulting float32 RGB hashes must be exactly equal to the already sealed pilot images and PSNR/LPIPS/DINO-L differences must satisfy the original pilot tolerances. This costs at most two extra VAR calls and two metric images, no additional PHY. Both qualification receipts are required before production starts. Reservations prohibit automatic retries of interrupted qualification. This mode is an execution option, not a claim of measured speedup; actual elapsed time and completed source counts determine throughput. GPU capacity is checked operationally by the owner, not inferred from availability of this script.

Only the full1,000-source, three-noise mean DINOv2 ViT-L/14 cosine selects the expanded whole configuration. Ties use candidate ID. Other metrics and failures are reported without selecting on them. The resulting `expanded_whole_policy.json` names a new method and preserves the original partial policy descriptor. It lists only changed SNRs as requiring future same-source holdout additions; this script never launches or reads holdout, changes old policy files, or creates new bootstrap intervals.

Prepare after the complete pilot:

```text
t2_recheck.py prepare --pilot-request <pilot request> --original-policy <original policies.json> --out <new WCL full1000 directory> --request <new request> --workers 8 --gpu-workers 2 --deadline-unix <explicit deadline>
```

Review the actual union/cap in `prepared_plan.json`, then run `cpu-worker --request ... --index 0..7` in the original CPU-only two-thread environment. After all CPU workers finish, run `t2_gpu_parallel.py owner --request ... --session full1000_01` in the original deterministic GPU environment for the registered two-worker mode. The owner records exact child identities, logs and a completion or failure receipt. A resume uses a fresh session name and reuses complete scientific source checkpoints; it does not repeat unresolved operations. For a request registered with one GPU worker instead, use `t2_recheck.py gpu-worker --request ...`.

`t2_recheck.py close --request ...` requires the complete source/action/noise grid and no unresolved packet reservations. Its completion status explicitly excludes holdout validation. The ranking CSV includes the original proxy score and whole-candidate rank, pilot rank, modulation, information/transmitted bits, head/body/padding symbols, actual PSNR/LPIPS/DINO-L and header/body rejection fractions.
