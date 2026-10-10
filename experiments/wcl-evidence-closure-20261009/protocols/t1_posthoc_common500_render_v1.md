# T1 fixed-policy common500 physical reception and reconstruction

This is a post-hoc supplement on the previously reported common500 population.
It is not an independent new holdout and cannot select or retune a policy.
The original main results, raw KEEP rules, ledgers and files stay immutable.

`t1_holdout_render.py` consumes the completed calibration-only six-policy freeze
and the actual source assets from `t1_holdout_sources.py`. There are two frozen
entropy families, SNRs 4/10/19 dB, 500 original ordered sources and seeds
6201/6202/6203: exactly 9,000 logical frames, with a separate 18,000 packet-call
cap. Parent `ledger-init` creates the WAL before CPU children; their constructors
are serialized, while actual paid decoding remains parallel. A reserved but
unresolved actual decoder call blocks automatic repetition.

The original 144-profile paid-header catalogue and qualified T1 physical backend
are reused. Every frame has 68 paid header and 956 paid body complex symbols;
there is no frame energy normalization or free side information. The fresh T1
noise namespace uses the fixed source ID and registered seed, shared across the
two entropy families. This does not claim equal observations with published raw
or continuous methods. The public counter is
`(six_SNR_index * 500 + original_source_index) * 3 + noise_seed - 6201`.

The actual selected prefix is checked against the frozen policy, its source's
complete actual TX-length table and the source preparation's `selected_by_snr`.
Only the actual selected independent roundtrip stream is used as TX payload.
Accepted erroneous headers are allowed to choose their actual received family,
prefix and coding parameters. Recovery consumes that actual profile and payload.
An independent RX cache is eligible only for the exact received bits, family,
prefix and frozen source-model identity. A miss uses the real source decoder.
Header, CRC, framing and canonical source rejection yield exact constant 0.5
float32 RGB; software/model errors stop the run.

After all CPU sources complete, `plan-render` seals every physical receipt and
every distinct source-decoding identity. It registers the exact number of
uncached VAR/static decoding identities, known new reconstruction states and a
conservative one-reconstruction-per-unknown-state bound. The program recomputes
this read-only plan before GPU work and refuses changes. There can be at most
9,000 source decodes and 9,000 production renders, plus one renderer qualification.
Complete exact-received-bit cache hits consume neither source-model inference
nor PHY calls. Repeated states within one source consume only one reconstruction.

Old raw images are reused only for the same original source and exact actual
receiver state. Only sealed VAR-completion float32 images are eligible, never
Direct or TX-truth reconstructions. A one-frame deterministic exact float32
replay qualifies the renderer before reuse. Both old archive SHA and RGB SHA are
checked. A local read-only audit found 2,001 eligible distinct whole/gray states
across all 500 sources; that is availability, not a claim that new entropy frames
hit those states. Each new source decode/render is reserved durably before the
operation, so an interrupted ambiguous operation cannot silently repeat.

The per-frame export retains target and actual prefix, K=0, transmitted tokens,
arithmetic length, modulation, nominal and actual code rate, k/n, symbol budgets,
known padding, actual energy/rho, all header/body/parser/source statuses and
fallback attempts. A separate diagnostic compares decoded tokens with original
source tokens only after selecting the output; it cannot alter receiver state,
acceptance or image generation. Failed and zero-information frames remain rows.

This consumer makes zero metric calls and runs no bootstrap. Its closure is
`T1_POSTHOC_COMMON500_RENDER_COMPLETE`, with exactly 9,000 rows in `per_frame.csv`,
the metadata source-request pin, frozen policies and ordered500 IDs. Every image
reference carries `image_path`, `image_key`, `image_slot`, `image_sha256` and
`image_file_sha256`; the closure seals all referenced old and new archives.
The separate frozen four-metric consumer performs scoring and statistics.

## Root-owned execution order

All paths below are explicit registered remote paths; root supplies the existing
frozen runtime. Do not start this stage until both full calibration families and
the selected source500 assets have actually completed.

```text
python t1_holdout_render.py prepare --root <R> --source-request <posthoc_source_request.json> --source-completion <sources500/completion.json> --environment-request <frozen_environment_request.json> --original-image-completion <R>/outputs/MAIN-RAW64-20261007/unified500_same_rx_parallel_images_r2/completion.json --out <new_WCL_out> --workers 8 --deadline-unix <finite_deadline>
python t1_holdout_render.py ledger-init --request <new_WCL_out>/request.json
CUDA_VISIBLE_DEVICES='' python t1_holdout_render.py cpu-worker --request <new_WCL_out>/request.json --index <0..7>
python t1_holdout_render.py plan-render --request <new_WCL_out>/request.json
CUDA_VISIBLE_DEVICES=0 python t1_holdout_render.py gpu-worker --request <new_WCL_out>/request.json
python t1_holdout_render.py close --request <new_WCL_out>/request.json
```

CPU workers may run together after the single initializer. GPU runs only after
all CPU sources and the finite plan exist, in the existing root-owned GPU window.
The new consumer does not start successors, monitors or background processes.

## Local validation

`python -m unittest discover -s experiments/wcl-evidence-closure-20261009/scripts -p test_t1_holdout_render.py -v`

Ten synthetic tests exercise actual received-header selection, true cache misses,
rejection retention, immutable selected prefixes, finite population/budget,
diagnostic isolation, float image hashes, durable interrupted source recovery,
and an 18-frame synthetic CPU→plan→GPU pass with real SQLite accounting and
resume. Synthetic callbacks are not scientific PHY or model calls.
