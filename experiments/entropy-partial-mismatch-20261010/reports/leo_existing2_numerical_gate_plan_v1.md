# Shared-host numerical gate: existing calibration sources 0 and 1

Status: metadata audit complete; the GPU gate is not implemented or run. This document does not authorize a launch. Its proposed scope is deliberately smaller than the subsequent registered EC-partial source and actual-link qualification.

## Existing evidence

The inspected `fixed_existing32_seed_v1` transport manifest contains 547 files. Original calibration indices 0 and 1 are, in order, `n01440764/ILSVRC2012_val_00045866_n01440764` and `n01443537/ILSVRC2012_val_00044095_n01443537`. Their pixel/token archives, source checkpoints, complete arithmetic streams and old whole-prefix image archives were rehashed against that manifest. The JSON companion records exact original paths and SHAs.

The original H archives contain only `m6_bits` through `m9_bits`. T1 sealed assets additionally contain m4/m5 and independently recovered tokens. The source32 image witnesses contain `m7_image`, `m8_image`, `m9_image`, each float32 CHW 256x256 VAR completion. They are not Direct-decoding witnesses.

A member-name inspection of all 184 NPZ archives in this selected seed found CDF arrays only in the frozen **static** counts/CDF archive, and no Direct-named array. All 32 original H source checkpoints were inspected: some inherited records include CDF computation times, but no original VAR CDF array or hash. Therefore original VAR full-table CDF identity and original Direct float32 identity are **UNOBSERVABLE_IN_SELECTED_SEED**. Exact old arithmetic bytes do not prove equality of every unrecorded CDF entry. Missing witnesses are never replaced with a fabricated PASS.

## Minimal proposed actual calls

| Call category | Cap |
| --- | ---: |
| Model construction | 1 |
| Original Encoder/tokenizer on cached pixels | 2 |
| Fresh VAR TX passes, each producing m4–m9 | 2 |
| Independent RX of old m4–m9 streams | 12 |
| VAR whole reconstructions m7/m8/m9 | 6 |
| Direct reconstruction / quality / PHY / training | 0 |

Source TX/RX uses at most 18 + 78 = 96 prior-scale evaluations; six full VAR reconstructions use another 60, for **156 prior-scale evaluations and six Dc forwards**. The same-host TX/RX CDF trace comparison piggybacks on these calls; it does not add model calls. The only pixels read are the two original cached source arrays. No original JPEG or new confirmation100 is accessed.

Every attempt is a separately reserved **new migration-gate call**. These calls do not silently expand, consume or count as completion of the original 32-source EP budget, 457 random-PHY checks or 48 actual-image link frames. Any later exact reuse needs an explicit registration/admission review before the later stage begins. No automatic qualification rerun or budget expansion follows failure.

Acceptance requires exact original token arrays, all 12 original complete arithmetic bitstreams, independent decode and canonical reencode with 30 lookahead bits, within-new-host TX/RX integer-CDF equality, six exact old float32 VAR reconstructions, unchanged model state hashes, actual child exit0 and zero unresolved call reservations. A mismatch saves its arrays/hash/max-difference evidence and stops without relaxed tolerance.

## Why the old outer loader cannot simply be launched

* `t1_owned_source_gate.run_owned` is the actual prior entry; no `t1_source_qualify.py` was found in the inspected WCL script directory. It hardcodes 32 sources and reuses historical VAR token proofs. Those proofs alone do not qualify a different GPU.
* `t1_codec_runtime.build_frozen_native` looks up source bindings by absolute path. Its `quality_driver -> m1_native -> common/assets` chain also reads old output registrations, calibration statistics and quality models and applies the old exclusive GPU0 guard.
* `t2_pilot.gpu_build` requires equality of the full old model/quality/runtime identity, including the old hardware name and CPU settings. The new GPU and requested two CPU threads are a new qualification context, not an equal old identity.
* `SourceCodec` already accepts a new repository root for its exact integer source code and provider. However its static completion and the original H/source checkpoints contain absolute archive paths. `latent_enhancement_b.common.load_gate/load_decoder` also dereference old checkpoint and provenance paths. They need an explicit resolver at the new assembly boundary; the original JSON must remain untouched.

The proposed isolated adapter constructs a minimal native-compatible object using the original `next_scale_prior.load_models`, exact official author code, original `ContinuousDecoder` construction plus the selected checkpoint's `decoder` state, and the original `_Prior`, `IndependentProvider` and `render_received`. It may import exact SHA-bound primitives independently, but must identify itself as a new assembly implementation, not the original `Native` loader. No quality model is needed for this gate. The complete old native loader is neither monkeypatched nor represented as qualified.

The official VAR `models/*.py` is an external dependency: seven files / 58,493 bytes were located in the old archive audit and are not present in the five-file model seed. Root must stage these exact files. The current engineering NumPy venv is not a scientific runtime; the original frozen environment restoration and import/ABI check remain prerequisites.

## New shared-host contract

All inputs, environment/cache/tmp, locks, logs and outputs stay under `/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu`. Original paths map one-to-one to pinned new regular files; no `/home` compatibility symlinks or changes to old scientific records are allowed. Select only GPU UUID `GPU-bf8340be-bd73-7413-62d5-0537be7f30cb`, expose it as logical cuda:0, and run one worker with two CPU and two interop threads. Record other tenants without touching their processes.

Before importing torch, bind private cache/tmp and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; preserve FP32, deterministic algorithms, TF32 off, cuDNN benchmark off and highest matmul precision. Record exact torch/CUDA/cuDNN/GPU versions and actual thread values. A fresh resource snapshot, private nonblocking lock, fresh output, finite deadline and parent actual wait are required. OOM, loss of resource margin, requested stop or mismatch terminates only this owned task and preserves evidence. No memory-fraction setting is called a hard allocation guarantee. This shared GPU gate supplies no T4 timing claim.

The result, even if all observed checks pass, will be limited to two-source whole-stream numerical compatibility. It cannot claim old full-CDF or Direct equivalence, all1000 compatibility, arbitrary corrupt-prefix compatibility, the full EC-partial source/PHY gate, or production readiness. Later real source streams still require independent actual RX verification under the frozen protocol.
