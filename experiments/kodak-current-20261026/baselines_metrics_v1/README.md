# Kodak current baseline reuse and four metrics

This lane uses the fixed Kodak24 crop population, N1024, SNR 4/10/19,
and original noise IDs 2001/2002/2003. The parent population and scope
registrations are SHA pinned. All six configurations retain 216 logical
frames, including real rejection outputs. No sample or status filtering.

`audit_reuse.py` is a CPU identity audit. It verified the final P100k and
Swin200k native arrays, original wave/observation/noise hashes, checkpoints,
source crops and prior PSNR/LPIPS records. The original BPG archives were
restored from verified local backup volumes without running the codec.
Their native scoring hashes and embedded source arrays match. The 648
logical frames reuse 504 native archives and 648 values of each old metric.
The exact historical noise namespaces and old runtime provenance remain
attached. BPG's historical output-hash serializer differs from the canonical
metric serializer; the common file SHA and both native scoring identities
are retained. No float arrays are recovered from display PNGs.

`prepare_metrics.py` accepts an explicit closed-input descriptor and
validates the full 1296-frame grid, native file/array SHA, immutable source
population, completion/actual-wait receipts and gone process generations.
It preserves all original frame metadata in the plan. Existing PSNR/LPIPS
values may propagate only across identical source and reconstruction
pixels with identical frozen metric math/assets. Conflicting scores abort.

The scorer uses the exact frozen main metric loader and projections from
the ARPC sensitivity registration. PSNR and LPIPS-Alex retain the original
formula/weights. DreamSim retains ensemble, FP32 and tensor square224
bicubic-antialias preprocessing. CLIP-I is OpenAI CLIP ViT-L/14 image-image
cosine using the frozen float224 preprocessing; no captions or softmax.
The three-metric role uses the original H800 visual environment with Torch
2.11.0+cu128; CLIP uses the original PRO native Torch 2.10.0+cu130 environment.
GPU physical identities are explicitly registered for this run. Both use
batch1, FP32, TF32 off, deterministic mode and 6/2 CPU threads; no hardware
timing comparison is claimed. Resource requirements are 8 GiB free at
startup, 2 GiB free while running and allocator cap12 GiB.

The two model roles start only after root reviews the registration and all
science input runs have closed. `owner.py` provides a unique exclusive
dispatch directory, true child.wait, PID/start-ticks, finite7200s scoring
and120s cleanup, and persistent stdout/stderr/failure records. Successful
scalar calls and feature outputs have durable per-call receipts; exact
duplicates reuse them. Failed or unresolved calls stop and require
diagnosis. There is no automatic retry, no reconstruction, channel run,
training, tokenizer, caption call or bootstrap in this lane.

After both roles actually exit0 and all processes are gone,
`merge_metrics.py` writes `frame_metrics.csv` with four full-precision
scores per logical frame and an auditable compute/reuse ledger. Source
means, the original24-source paired bootstrap and figures belong to the
parent delivery lane, not this scorer.

`test_metrics.py` runs only temporary synthetic arrays and scalar functions.
It checks population/failure denominators, hash conventions, negative zero,
completed-call recovery, limits, corrupt receipts and failure retention.
