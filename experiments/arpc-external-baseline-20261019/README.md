# ARPC external comparison — execution record

This is a new, gated A0→A3 task under the user's 2026-10-19 instruction to run
the supplied package through final delivery. It supersedes the package's default
A0/A1-only stop, while retaining its scientific limits, qualification gates and
excluded expansions. Existing NeST-Com results, checkpoints, figures and shared
H800 timing remain unchanged. Backups remain deferred as requested.

Repository reference and observed HEAD: `3ac13a02feccbbe5f87ca2bc801c18ae07dd29a0`.
No later commit or tracked modification was present at the initial audit.
Upstream: [Joanna-0421/ARPC](https://github.com/Joanna-0421/ARPC/tree/f7480889175271a58c7e5f8e48e0b58f247ccf63),
commit `f7480889175271a58c7e5f8e48e0b58f247ccf63`, MIT source license.
The two adapted checkpoints are the exact public Drive links in that README,
not the unadapted Infinity base weights. No separate checkpoint license text is
provided in that repository; the upstream explicitly distributes them for
inference. Flan-T5-XL and BLIP checkpoint/model-card revisions are recorded in
`asset_sources.json`; their published cards declare Apache-2.0 and BSD-3-Clause.

## Current stage

A0 preparation is running. The deterministic cal100 and external_common100_v1
lists and the first eight calibration sources have been recorded before any
new caption, tokenizer, prior, reconstruction, metric or channel call.
Public model downloads and isolated dependency preparation are engineering
operations, not completed A1 experiments. A1 entry files are prepared but must
not execute until acquisition, environment, source and registration checks pass.

## Static findings

- Native 256×256 ratio-1 preset `0.06M` contains scale sides 1,2,4,6,8,12,16.
- Upstream `demo.py` defaults to 1024; it is not the executed input preset.
- Source `encoding` performs one native probability traversal, uses one
  independently terminated arithmetic trial per 16-bit token group, and chooses
  compressed bits only when shorter than 16. This rule is preserved.
- Author `trans_list`/`help_list` preserve Python segment boundaries; they are
  not independently transmittable files. The new lossless container makes those
  boundaries and the caption explicit; it does not serialize JSON/Pickle.
- The final native `decoding` function does not read its `gt_ls_Bl` argument.
  Its one scale loop is bounded to the actually transmitted prefix. All original
  full-schedule conditioning/interpolation remains intact. Passing a shortened
  global schedule would change the model and is not used.
- `decompress_cfg` overwrites only `si <= gt_leak` with received prefix tokens;
  later tokens are generated. RX supplies only decoded tokens, never source GT.
- Native reported bpp divides by 1024²; new rates use actual 256² pixels.
- A1 reference and independent RX use author CFG=3, tau=0.5, top-k=900,
  top-p=0.97, reset shared generation seed 0 per endpoint. No quality-driven seed
  or configuration search is permitted.

## Qualification limits

A1: 8 captions, 8 image tokenizations, 8 source probability traversals; endpoints
1/4/7 give 24 independent prefix roundtrips and at most 24 reference +24 RX image
generations. Wireless frames and training updates stay zero. Integer CDFs and
prefix tokens must agree exactly; final uint8 pixels must agree exactly, with
float output max absolute difference ≤1e-6. Comparisons occur only after RX exit.
Low capacity/no-fit and weak image quality are reported; they cannot be repaired
with hidden source data or omitted from denominators.

One `leo-tactile` device is used serially for native TX and RX, after a fresh
availability check. The isolated target uses that host's existing Torch 2.10.0
CUDA 13 runtime; upstream-pinned Transformers 4.51.3 and other recorded inference
dependencies are installed in a new target. FlashAttention 2.8.3.post2 is built
for the actual Blackwell architecture. No shared training environment is altered.
This environment difference and native mixed arithmetic are disclosed. The
current preparation is not a speed comparison with the historical H800 result.

A2/A3 are allowed only after actual A1 closure, visual sanity and measured cost
review. Future maxima remain 100-source clean screening, ≤84 strategies,
≤2 candidates/SNR, 18,000 calibration logical frames on original cal1000,
≤24 engineering wireless frames, and 900 evaluation logical frames on the
fixed common500 subset. Existing methods use exactly the same 100 source IDs
and their completed per-source scores; 19 dB entropy results use T1-C m10.

Unknown science failures stop the chain and retain logs, reservations and
partial outputs. No automatic replay of a completed scientific call is allowed.
