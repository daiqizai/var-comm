# CLIP-I supplement: frozen N1024 five-method common500

This extension evaluates already-fixed native RGB float32 reconstructions only.
The population is the original common500, retaining source indices 0–499,
SNR 4/10/19 dB and all three original noises: 22,500 logical image pairs.
NeST-Com uses the current consistency-delivery PARTIAL rows, including T1-C
m10-corrected 19 dB. P100k, Swin200k and adaptive BPG use the byte-verified
supplemental-metric native-array roster. ARPC combines the closed original100
and remaining400 receiver records. Development examples and N2048 are excluded.

The evaluator is the unchanged repository `metric_models.py` implementation:
OpenAI CLIP ViT-L/14 (224), official commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, checkpoint SHA256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
Native float tensors are resized bicubically with antialiasing, center-cropped
to224 and normalized by the frozen CLIP constants. Both operands are images.
FP32, eval/inference mode, TF32 off, deterministic operations, batch1 and6/2
threads are fixed. No caption, text encoder forward, softmax, percentage
conversion, PIL requantization or new image generation is used.

`inventory.py` verifies the complete grid, source identities, file and pixel
hashes, actual status and fallback lineage. Missing files or mismatches block
scoring. All actual gray/rejected outputs keep their original weight. The
eight frozen CLIP asset files are restored byte-for-byte from existing static
volumes031/032; no package installation, environment upgrade or new checkpoint.
Previous current-five-method metric deliveries contain no CLIP-I column.
Historical unified metrics cover a different population/method setup and do
not supply an admissible current-pixel/device feature cache. No historical
scalar is substituted by method label. New embeddings are cached by complete
registered evaluator identity and exact native pixel hash, including shared
gray images. Pair scores are cached by both source and reconstruction hashes.

`test_protocol.py` is synthetic CPU-only. After inventory/tests/asset hashes
pass, root freezes registration and code. `control.py <registration-SHA>` runs
one bounded `score.py` child, keeping a persistent actual wait, PID/start,
argv, stdout/stderr and the shared timing lock. The scoring ledger reserves
each load/embedding/pair before execution. Unknown failures stay immutable;
no automatic restart or silent replay of successful calls. Each completed
embedding is durably saved. Caps are derived from the actual closed inventory.

Statistical analysis first averages the three original noises within each
source/method/SNR, then weights the500 sources equally. It reuses the exact
existing seed2026100701 10000×500 bootstrap index bank, SHA256
`5ca12fe75af6a09e1d3c18e205d7394bf4cdca53c25f7ca21005e5de30117634`,
with zero new index draws. Report all15 marginal cells and all30 method-pair
cells (ten pairs at three SNRs), including the12 NeST-minus-baseline contrasts.
Paired intervals resample signed source differences directly, preserving
negative/cross-zero results. These are pointwise95% source-paired intervals;
historical methods do not share identical noise realizations.

This supplemental evaluation cannot change frozen models, policies, receiver
rules, model selection, the original seven scores or the primary figures.
No training, channel/decoder call, VAR/diffusion reconstruction, tokenizer,
caption, N2048 expansion or whole-paper rewrite is part of this execution.
