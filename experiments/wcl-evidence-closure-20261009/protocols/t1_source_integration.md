# T1 source API and execution integration

The root owner schedules all jobs. These modules do not acquire another user's GPU, install dependencies, retrain models, or modify existing scientific outputs.

## Source stage order and calls

1. `t1_static_counts.py`: completed original20k statistics, CPU only. Already-frozen core/counting files must not be hot-edited.
2. `t1_source_check.py`: completed first32 m7/m8/m9 source audit; explicitly reused old VAR independent roundtrips and actual new static CPU roundtrips. It records visual parity and short-prefix followups as pending.
3. `t1_source_population.py --count 1000`: CPU preparation, original calibration order. Reuses the first32 static m5..m9 streams, creates first32 static m4, encodes/decode-checks remaining static m4..m9, reuses all historical H VAR m6..m9 streams. It writes source_checkpoints/0000.json through0999.json and sources/0000.npz through0999.npz. Static family is ready immediately; missing VAR short streams are explicit.
4. After the current GPU owner finishes, a new owned source window calls `t1_owned_source_gate.run_owned(...)` and/or `t1_owned_shortprefixes.run_owned(...)`. No independent scheduler is provided. `boundary()` is a mandatory live-owner/STOP/thermal/deadline callback supplied by the root owner. Models can be loaded once with `t1_codec_runtime.build_frozen_native(root,h_completion,signal_handler)` and shared sequentially across these source tasks.
5. `t1_owned_shortprefixes` computes VAR m4/m5 only when the existing m6 stream exceeds the minimum927-bit source capacity. It uses one TX pass with two flushes and two independent RX contexts, checking every recovered token. Earlier verified first32 fallback records may be reused. Source images whose m6 already fits can never reach m5/m4 for the finite candidate set, so no unnecessary VAR call is made for them.
6. `t1_merge_sources.py` takes CPU source completion plus the final GPU short-prefix completion. If that final completion already includes reused first32 fallback records, supply only it; do not pass duplicate first32 and final records. The merge writes a fresh complete source layer and checks all72 family-candidate conditions per source. It does not overwrite either input stage. Every candidate must resolve to a present actual stream before PHY evaluation.

Example CPU preparation:

```text
python -B experiments/wcl-evidence-closure-20261009/scripts/t1_source_population.py --root /home/liulu/projects/VAR_COMM --calibration /home/liulu/projects/VAR_COMM/outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002/m1_calibration_registration.json --h-completion /home/liulu/projects/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/full1000_source/completion.json --static-completion <NEW>/static_train20k_v1/completion.json --source32-completion <NEW>/source32_v1/completion.json --count 1000 --out <NEW>/calibration_source1000_v1
```

The prefix preparation is experimental caching, not measured online TX/RX time. Holdout source preparation is not implemented by this calibration-only entry point and must follow policy freeze in a separately bound stage.

## Single owned GPU window

`t1_source_owner.py` is the explicitly launched CLI wrapper. Supply `--root`, `--environment-request` (the already-qualified T2 environment request), `--calibration`, `--h-completion`, `--static-completion`, `--source32-completion`, `--source-population-completion`, a fresh `--out`, and finite `--deadline-unix`. It requires `CUDA_VISIBLE_DEVICES=0`, acquires the original visual lock and a new owner GPU lock, uses the existing frozen `t2_pilot.gpu_build`, then executes source32 gate, needed calibration short prefixes, and CPU merge. No subprocess, remote connection, automatic relaunch, or background scheduler is created.

The budget is determined solely by the completed CPU population: for `n` sources whose existing VAR m6 stream exceeds927 bits, exactly `n` TX operations and `2n` independent RX operations are admitted. These total at most `14n` new source-prior scale evaluations; clean reconstruction has a separate maximum192 image renders and96 static CPU source decodes. Each operation has an immutable reservation before execution and a separate completion after execution. Failed operations leave explicit unresolved reservations. A used output directory cannot be re-entered automatically. For the actual1000-source population reported by the root owner, `n=491`, giving491 TX,982 independent RX, and6874 source-prior scale evaluations. This is a pre-run budget, not a completion claim.

`t1_reference_cache.RawCalibrationReference` reuses original calibration images only after matching source identity, complete received token state, frozen visual identity and numerical settings, and the sealed completion/checkpoint/NPZ/image hashes. It scans original row order and never selects by quality. Gate success is `T1_SOURCE32_REAL_RECOVERY_GATE_COMPLETE`; wrapper success is `T1_SOURCE_OWNER_COMPLETE`. The sealed source completion is under the wrapper output's `source1000_sealed/completion.json`.

## Per-source schema

`schema = T1_SOURCE_ASSET_V1`

- `source_index`, `source_id`, `source_tokens_sha256`, `preprocessing_id`.
- `source_assets_checkpoint = {path, sha256}` points to the original calibration source asset and target pixels.
- `archive = {path, sha256}` identifies the new stream/token NPZ.
- `tokens_key = tokens` identifies the complete source token array for **TX only**.
- `streams[family][str(m)]` identifies `bits_key`, `payload_bits`, `payload_sha256`, `received_tokens_key`, `received_tokens_sha256`, and `receiver_cache_proof`.
- `receiver_cache_proof` records independent roundtrip/canonical success, exact30 zero lookahead reads, origin, source-model identity, and upstream checked evidence.

Typical keys are `static_m7_bits`, `static_m7_received_tokens`, `var_m9_bits`, `var_m9_received_tokens`. Missing unreachable VAR m4/m5 are allowed; the candidate selection loop must stop as soon as a longer stream fits. A missing reachable stream is an asset error, never a scored channel failure.

## Receiver source decoding

```python
from t1_codec_runtime import SourceCodec, InvalidSourceStream
codec = SourceCodec(root, static_completion=static_completion, native=native)
try:
    parsed = codec.decode(family, actual_received_payload_bits, received_m)
    received_tokens = parsed['received_tokens']
except InvalidSourceStream:
    # Fixed protocol source-parse failure output; no true-source inspection.
    ...
```

The caller must already have accepted the actual paid header, selected its public profile, decoded its fixed k/n body, checked CRC, and validated charged length/padding. `decode` accepts no source ID, source tokens, source pixels, original class or TX probability table. Only `InvalidSourceStream` is a protocol parsing failure. Model/software errors are raised and must not be scored as gray channel failures.

An exact offline RX cache may be consulted with:

```python
from t1_source_asset_schema import exact_received_cache, model_id
hit = exact_received_cache(record, restricted_stream_arrays,
    family=family, m=received_m, received_bits=actual_received_payload_bits,
    expected_source_model_id=model_id(family, static_completion_sha))
```

The helper does not read the original source tokens. Provide only relevant stream and independently verified received-token arrays in `restricted_stream_arrays`. It requires family/m/model, exact observed payload length/hash and bit-for-bit equality. A miss goes through the actual decoder above. This cache is prohibited during online timing. It cannot be used to call an erroneous accepted bitstream rejected simply because it differs from the source.

## Clean recovery equivalence

The owned32 source gate reuses prior VAR token-roundtrip evidence; no new m7/m8/m9 VAR probability pass is required. It independently decodes static source streams and renders from the actual decoded tokens through the frozen common renderer/Dc. Raw reference output can be supplied through optional `reference_lookup(index,m,tokens)`; each hit must provide exact received tokens, m/K/source ID, model identity, image SHA and bound source files. The gate validates those before reusing the reference. Missing reference images are rendered once. New entropy decoded inputs must still reproduce the bound reference exactly. These checks measure equality, not a policy score.

## Current validation boundary

Eight initial CPU synthetic checks passed (including strict20k identity, complete scales, changed-shard rejection and integer codec roundtrips). Four runtime/receipt synthetic checks passed, including malformed/noncanonical stream rejection. Exact-bit cache hit plus changed-bit/model/family misses passed. A synthetic merged source passed all72 registered family-candidate conditions. These are engineering checks, not real GPU admission or scientific quality results. Actual native construction, short-prefix probabilities and image-equivalence receipts remain the root owner's scheduled real-model work.
