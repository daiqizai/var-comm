# R1 received-state projection and isolated reconstruction

This entry is implementation preparation. No real R1 registration, physical
frame, reconstruction, scoring call, policy selection or R2 launch is implied by
its CPU tests or software-import probe. The finite owner remains the only science
dispatcher. It must first authenticate completed R0 evidence and the reviewed
whole-chain registration.

`receiver_visual.py project` runs in the original CPU environment. It verifies
the actual successful PHY child wait, source order, frozen nominations, complete
physical/logical grid, unchanged noise keys and counters, common standard-noise
hashes across true SNRs and candidates, actual noise variance, all immutable
physical/packet reservations and completions, and the complete canonical arm
mapping. It performs no model, decoder, channel or metric calls.

The projection produces two separate JSON lists:

- `model_inputs.json` contains only opaque view/event/cache identities, actual
  received raster tokens or the fixed gray state, and the frozen model identity.
- `logical_join.json` preserves source IDs, candidate/physical IDs, SNRs, seeds,
  canonical arms and provenance for the independent CPU closure and metric stage.

The visual process reads only the first list through
`visual_input_completion.json`. It does not open the physical frame files,
logical join, source archive/checkpoints, clean tokens, source images or labels.
The score worker joins actual image rows with `logical_join.json` after rendering.
`completion.json` and `audit.json` in the projection directory remain CPU evidence.

The actual receive state is validated across the full ten-scale raster domain,
including legal wrong-but-accepted token values and long legal received IDs.
Gray states produce float32 RGB 0.5 without constructing or calling VAR. Token
states call the original SHA-bound `render_received` and `_Prior`, with class1000,
deterministic argmax, the same frozen VAE/VAR/Dc state hashes, FP32 and original
6/2-thread numerical settings. Selected original mathematical definitions are
compiled unchanged as AST nodes. No clean R0 rendering entry or source resolver
is called in the model process.

`build_model_bundle(qualified_request_pin, output_path)` is a future CPU builder
API. It verifies the existing qualification witnesses in that CPU process, then
exports only model weights, model-code paths, original software-runtime pins and
the numerical recipe. Source witnesses and the original full visual request are
not exported. The GPU process independently checks the model-only schema,
actual imported Python/Torch/NumPy file hashes and actual model state hashes.

The renderer installs a Python `open` audit gate. Research PFS reads are limited
to explicit model/software/code pins, its pure RX input, owner control metadata
and its own output directory. The gate accepts only corresponding bytecode for
pinned Python files and missing optional metadata members of already pinned
software distributions and the absent standard-library zip of the pinned
interpreter. An existing unbound archive remains forbidden. The import probe
requires an empty blocked-read ledger even when imports suppress exceptions.
It does not allow the whole runtime assets tree. This is
a runtime assertion for reviewed Python code, **not an operating-system sandbox**.

An exact cache key binds opaque source/pixel/protocol/candidate/noise/actual-RX
identity, normalized received input and model/numeric identity. The receiver rule
is excluded only to reuse byte-identical KEEP/DROP inputs from the same actual
physical event. Two logical rows remain in that case; arm aliases never create
extra rows. Different physical events cannot reuse an image merely because their
token count, gray output, source name or reconstructed pixels match.

The visual completion uses status `COMPLETE_R1_VISUAL`, `image_rows` (a JSON-list
pin), `logical_rows`, actual call ledger, resource/access evidence and the exact
model-input pin. Image rows contain opaque `view_id`, receive/model hashes,
`reconstruction={image_archive,image_sha256,actual_input_key}`, reuse provenance
and scientific failure kind. Images are float32 RGB NPZ files under the registered
personal temporary directory. Unknown failures preserve the partial image list,
unresolved model ledger and original error; there is no automatic retry.

`audit_projection(reg)` repeats only the CPU physical/projection closure.
`audit_visual(reg)` checks actual successful visual exit, every image/hash/cache
binding, gray pixels, complete logical rows, model-call order and the original
SQLite ledger. Each unique token reconstruction must have one VAR call, ten
prior-scale calls and one Dc call. Fixed gray and exact reuse contribute no model
calls. The maximum remains 46,800 registered logical outputs with no extra
qualification frames, tokenizer calls, training or bootstrap.

The shared owner handles the real six-method-timing exclusion lock, parent
identity, finite deadline and resource admission. Rendering uses the same
18 GiB free/free-plus-own gate and 16 GiB private allocator limit; these are
shared-environment quality reconstructions, not latency measurements.

Future owner-only CLI forms (not launch authorization):

```text
receiver_visual.py project --registration MASTER.json --expected-registration SHA --parent-pid PID --parent-ticks START
receiver_visual.py visual --registration VISUAL.json --expected-registration SHA --parent-pid PID --parent-ticks START
```

CPU tests: `python test_receiver_visual.py`. The separate
`probe_receiver_imports.py --attempt N` performs only explicit metadata/software
imports with CUDA hidden. Its receipts distinguish zero-model import compatibility
from actual numerical qualification or experiment completion.
