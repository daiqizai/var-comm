# T4 fresh entropy cost and common raw anchors

Status: runnable owner and actual entropy/raw endpoint implementation; **no timing execution or result is claimed**. Execution waits for frozen T1 policies and an exclusive computation window controlled by the coordinator.

The original A3 timing is preserved as historical evidence. It used the same fixed16 development sources, SNR 7/13/19, one warmup and three measured repetitions per source/SNR, FP32/B1, CPU six threads and two interop threads in its registered unified environment. The new T1 PHY qualification uses two CPU threads and one interop thread, and new T4 points are 4/10/19. Consequently new entropy costs must be accompanied by new raw whole and raw partial anchors in the same actual runtime and thread configuration; historical external-method values are not described as directly remeasured alongside entropy.

## Fixed bounded population

Original development indices in original order: `[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]`. Sources and uint8 preprocessing hashes are bound to the original A3 source manifest before execution. SNRs are `[4,10,19]`; noise seed is 6201. Raw anchors preserve the original development noise namespace, source/SNR/seed rule and frozen-point frame counters. Entropy uses T1 source/seed-defined full-frame standard normal variates with explicit SNR scaling. These are timing cases, not an identical-observation cross-waveform quality comparison. No source, noise or failed frame is replaced after observing an output.

There are four methods: frozen original raw whole, frozen original raw partial, frozen EC_STATIC_WHOLE, and frozen EC_VAR_WHOLE. Each has three warmups and three measured repetitions **per source/SNR**: 144 warmup frames plus 144 measured frames, 288 actual fresh frames per method. Total is 1,152 fresh frames, 576 measured, and an upper bound of 2,304 actual paid header/body decoder calls. Header rejection omits the corresponding body call; unused reserved slots are reported. No separate quality or bootstrap calls are authorized by timing. The new independent timing budget does not consume or extend old science ledgers.

## Exact software boundaries

TX is original CPU uint8 image tensor to complete CPU float64 1,024-by-2 I/Q frame. It freshly runs visual Encoder/VQ, all necessary source processing, all actual probability/entropy/fallback attempts, paid header, LDPC and modulation. EC_VAR_WHOLE uses fresh frozen null-class probability state for each attempted prefix. This conservative implementation does not claim a hypothetical optimized shared-probability latency. Static CDFs are a fixed pretrained component and can reside in memory; per-image token/bitstream caches cannot replace online work.

RX is the actual CPU I/Q observation to one CPU float32 RGB image. It freshly runs the paid header, actual LDPC, CRC/length/padding checks, actual received-profile source decode/canonical check, VAR and the frozen Dc. It cannot read original image, source tokens, TX profile, or TX probability tables. Header/CRC/source-parser failures use the frozen gray fallback and remain in the latency population. Incorrectly accepted legal headers are decoded according to their actual received profiles. Model/software exceptions remain execution failures and are not relabelled channel failures.

Each case has one measured outer wall-clock window with synchronized nested TX/RX windows. The deterministic channel simulation has its own measured interval; end-to-end excluding channel is the same outer interval minus that channel interval. TX and RX totals are never added and relabelled as a separately measured end-to-end quantity. Component windows are inclusive and cannot be summed into totals. Model loading, file IO, metrics, scheduling, guard polling, output hashes and audit-file writes are outside measured intervals. Existing PHY/source functions' in-memory parsing/checking remain part of their actual software endpoints.

Packet decoder slots are durably reserved before the measured frame. During measured RX, fresh callbacks and their results are recorded in memory; completion is persisted afterward. No cached ledger result can replace an actual timed decoder. An interruption leaves an unresolved reservation and blocks automatic continuation; a failed run is preserved. All four methods need the same runtime and CPU resource limits, GPU synchronization, FP32 flags, batch size, loaded-model boundaries and exclusive machine window. The coordinator must pin and record these before execution. Three warmups do not enter measurement summaries.

## Required evidence and pending integration

The actual entropy endpoint is `scripts/t4_timing_endpoint.py`, raw anchors are `scripts/t4_raw_endpoint.py`, and the complete owner is `scripts/t4_timing_owner.py`. The owner has metadata-only `prepare`, read-only `check-admission`, and explicit `run` commands. The frozen-policy JSON supplies `calibration_completion: {path, sha256}` and six `rows`, each containing `family`, `snr_db`, `candidate_id`, `target_m`, `q` and `nominal_rate`. Registration checks both families' completed 1,000-source, three-noise calibration and requires each policy to equal its recorded rank-one winner. It separately authenticates original raw policies and retains K142 at 19 dB. No quality-based reselection occurs.

Execution uses the already validated unified UM environment, FP32/B1, six CPU threads and two interop threads for all four methods. The T1 PHY constructor is called with `configure_threads=False` after the common model loader so it preserves the shared timing thread settings; its actual source/backend/catalogue/qualification bindings remain enforced. The original raw planner file SHA is verified before and after timing and returned from a narrow in-memory audit-hash lookup during the old raw PHY shell. This removes audit-file IO from measured windows without memoizing numerical operations, metadata decoding, or output reconstruction. All source bindings and model identities are verified outside the measured windows.

The owner takes the original shared visual lock and its new owner lock, then checks that GPU compute processes and other repository Python jobs are absent. It never kills another task. Existing attempts, failures and unresolved reservations block automatic retries. A completed output is reused without remeasurement. The controller must wait until T1/T2 computation has quiesced before the timing window; no detached process or future successor is launched by this owner.

The only permitted repository Python observer is the immediate parent running the SHA-bound `run_recorded_child.py` with exact child argv, `--cwd` equal to the repository and `--record-dir` equal to `<T4 out>/parent_observer`. Admission verifies same UID, parent/child PID and process-start ticks, the observer's immutable launch record, child-start/launch time consistency, stable parent identity on a second read, and `/proc/<parent>/wchan == do_wait`. No other ancestor is exempt. Run the timing command through that observer to retain the actual `child.wait()` exit receipt; the observer itself performs no scientific work.

Example remote registration, after the coordinator has the frozen policy and sets actual paths:

```sh
"$UM_PY" "$R/experiments/wcl-evidence-closure-20261009/scripts/t4_timing_owner.py" prepare \
  --root "$R" --out "$R/outputs/WCL-EVIDENCE-CLOSURE-20261009/T4_resources_cost/timing_v1" \
  --a3-request "$A3_REQUEST" --raw-cost-request "$RAW_COST_REQUEST" \
  --frozen-policy "$T1_FROZEN_POLICY" --phy-qualification "$T1_PHY_COMPLETION" \
  --static-completion "$T1_STATIC_COMPLETION" --deadline-unix "$DEADLINE_UNIX"
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$UM_PY" \
  "$R/experiments/wcl-evidence-closure-20261009/scripts/t4_timing_owner.py" check-admission --request "$T4_REQUEST"
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$UM_PY" \
  "$R/experiments/wcl-evidence-closure-20261009/scripts/t4_timing_owner.py" run --request "$T4_REQUEST"
```

`t4_timing_selfcheck.py` checks budget reservation ordering, no callback-window audit IO, cap enforcement, failed-attempt preservation, full measured-source coverage, CRC-failed KEEP retention, explicit empty conditional groups, and absence of TX/source-truth inputs to the receivers. It never calls a real or fake PHY/model and is explicitly not a scientific timing completion.

The final export must include per-call TX/RX/outer/channel timing, actual resource/energy fields, received failure state, source/fallback attempts, waveform/observation/output identities and actual decoder count. Report overall mean, median and p95 for each method/SNR and success/failure conditional sample counts. The 16-source timing population does not support universal deployment latency or precise extreme-tail claims. A completion receipt requires all 576 measured calls plus 576 warmups, exact fixed population, and zero unresolved reservations. New quality-versus-TX-cost tables must identify the frozen policy and distinguish development timing from holdout quality.
