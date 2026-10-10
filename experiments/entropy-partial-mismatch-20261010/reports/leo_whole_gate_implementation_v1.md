# Existing two-source whole compatibility gate — implementation v1

This is a separately budgeted migration gate, not production admission. Its
implementation and fake CPU tests are complete; no real model, GPU, PHY or new
source call has been run by this preparation task. The original source32 and
48-frame EP gates, scientific JSON, weights and mathematical source files remain
unchanged. The earlier metadata plan is retained unchanged. This implementation
preserves the original six PyTorch threads and two interop threads; its two-CPU
affinity is a disclosed scheduling restriction, not a change to those numbers.

## Inputs and exact scope

Use the original calibration indices 0 and 1 only, with their original 680-token
arrays and cached uint8 CHW pixels. The selected source seed supplies 19 exact
referenced file pins for this gate. Metadata and array checks actually passed
locally for both sources: six original VAR arithmetic streams per source (m4–m9),
the corresponding independently received tokens, and old float32 VAR image
witnesses at m7, m8 and m9. The historical H m6–m9 streams agree with the sealed
T1 streams. This read-only check made no model call.

The real gate, when separately started, reserves before each operation:

| Operation | Maximum |
|---|---:|
| Frozen model assembly | 1 |
| Original Encoder/tokenizer pass | 2 |
| Fresh VAR entropy TX pass, producing m4–m9 | 2 |
| Independent old-stream RX plus canonical integer re-encoding | 12 |
| VAR whole reconstruction using independently recovered tokens | 6 |
| `_Prior.logits` scale evaluations, including nested operations | 156 |
| Frozen Dc forward | 6 |

The 156 evaluations are 2 × (9 TX + 39 independent RX + 30 render). There are 32
exact output checks: two token arrays, 12 bitstreams, 12 recovered token arrays
and six RGB arrays. Dtype, shape and actual bytes must match. Each RX receives
only its bit vector and public complete-prefix m, constructs its own prior, and
advances from its own decoded tokens. The new TX CDF hash is used only as an
external equality assertion, never as an RX probability input. Rendering uses
the independently recovered tokens.

Old full VAR CDF arrays/hashes and old Direct RGB witnesses are unavailable in
this selected seed and remain `UNOBSERVABLE_IN_SELECTED_SEED`. The gate cannot
prove old CDF equality from matching bitstreams. It performs no Direct, quality,
bootstrap, PHY or training operation.

## Native source and relocated environment

The seven original author model files match the official fixed revision
`78b95394fc5896192e3a003e4b295f8ea743c48f` byte for byte. The old `dist.py` SHA was
not present in the audited old indexes. Its reviewed upstream file,
`b55322b4ae4c8c2b600b98870dc98b26435a813b72b3468a4315d7d74771e370`, is therefore an
explicit new migration dependency candidate. The original seven-file seed is
unchanged. No claim is made that the old dependency closure was recovered in full.

The gate binds the upstream audit and commit metadata, verifies all nine candidate
files, and loads that exact `dist.py` path. It never calls `dist.initialize` or
`set_gpu_id`; CUDA logical device 0 must first be shown to be the selected physical
GPU UUID. The two constructor uses of `get_device()` see `cuda`; distributed state
must remain uninitialized. No substitute package or invented stub is admitted.

The input-map builder consumes the actual frozen-runtime relocation completion.
It preserves the lexical private-venv interpreter, follows the SHA-bound original
selection and restored JSONL mapping, and separately records newly generated
`.pth` and `pyvenv.cfg` bytes. Multiple regular copies of the same original Python
binary are supported. Runtime package/source files are checked against their
original byte pins; generated files are not relabelled as historical bytes.

The child environment is rebuilt from an allowlist. It takes `LD_LIBRARY_PATH`
from the actual UM import qualification, checks every path is personal, and does
not inherit host `LD_PRELOAD`, `LD_AUDIT`, `PYTHONHOME`, `PYTHONPATH` or proxy
settings. Actual Python 3.10.12, torch 2.11.0+cu128, NumPy 2.2.6, CUDA 12.8,
cuDNN 91900, FP32, TF32 off, deterministic algorithms, highest precision and
threads 6/2 are checked. Original mathematical function/class bodies are compiled
unchanged from SHA-bound source definitions; old application loaders and global
GPU guards are not executed.

There are two read-only `/proc/self/maps` snapshots, after model assembly and at
successful close. Mapped original runtime libraries and previously observed new
host libraries must match their pins. Additional `libcuda.so*` and `libnvidia-*`
driver libraries are explicitly new-host observations, with end verification;
other unregistered native libraries stop the gate. These observations do not
claim old-host dynamic-library identity.

## Shared-resource and failure boundary

Exactly one child uses GPU UUID
`GPU-bf8340be-bd73-7413-62d5-0537be7f30cb` (physical index 5), batch one and two
permitted CPU cores. Before launch, GPU utilization must be at most 50% and free
memory at least 20 GiB. During the gate, fresh resource checks retain the 20 GiB
free-memory requirement; utilization is not a runtime stop criterion because it
includes this worker. The PyTorch allocator is capped at 16 GiB. This is not a
hard total-GPU reservation and does not cover every library allocation or other
users' processes. No other process or machine-wide setting is changed.

An owner file lock prevents duplicate gate owners. The child verifies its actual
live parent, the parent's launch receipt, PID, registered argv and interpreter.
The owner really waits for that child and only succeeds after exit 0, exact
closed counts and all input pins reverify. Each output directory is a one-attempt
claim. Time is bounded by both the absolute deadline and at most 1,800 seconds.
Busy resources, OOM, missing references, changed pins, unresolved reservations
or any first mismatch preserve failure artifacts and stop without replay,
automatic retry, threshold expansion or a successor job.

Mismatch evidence identifies tokens, bitstream, independently recovered tokens,
new-host TX/RX CDF or VAR rendering as applicable. New CDF traces are preserved
on failure. A failed cross-architecture exact gate does not prove that a future,
explicitly separate new-hardware comparison protocol is impossible. It does not
authorize such a protocol or weaken this gate's checks. No timing comparison is
performed, and this shared run cannot be used as T4 timing evidence.

## Exact staged entry points

All actual remote files stay under
`/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu`. Root owns execution. The frozen-runtime
completion must actually exist and pass its import/wait checks before the first
command can succeed. The four receipt hashes must be the actual files' hashes.

1. Run `scripts/leo_whole_input_map_v1.py` with `--source-receipt-sha256`,
   `--model-receipt-sha256`, `--candidate-receipt-sha256`,
   `--runtime-receipt-sha256`, `--deadline-unix` and a fresh `--out`. Defaults name
   the actual private receipt locations and candidate directory; path overrides
   remain confined to the personal base. This reads metadata/SHA only and emits
   `spec.json`, `runtime_projection.json`, `map_receipt.json` and exact
   `next_prepare_argv`. It does not execute that argv.
2. Execute the emitted `next_prepare_argv`. This verifies the entire bound input
   and environment map and emits the actual `request.json` path and SHA.
3. Separately run the same private engineering Python with
   `-B scripts/leo_whole_gate_v1.py run --request <actual request path>
   --request-sha256 <actual request SHA>`. The owner launches the registered
   frozen private visual interpreter and really waits. No fourth step is
   scheduled automatically.

The accompanying local checks receipt gives the exact script hashes and observed
CPU test result. Metadata/import PASS is never relabelled as numerical PASS.
