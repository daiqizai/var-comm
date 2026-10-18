# Fixed 100-source seven-metric comparison

This independent entry is gated by actual A3 receiver closure and retains the
single A3 7200-second deadline. It does not calibrate/select a policy or checkpoint.
Actual native float32 reconstruction arrays and original uint8/255 references
form at most800 distinct source/image pairs. All900 logical frames, including
gray/parse/channel failures, remain in the denominator. No existing baseline
is re-scored. The same100 source IDs are used for every method.

The original four-metric implementation is byte pinned. Its private population
metadata and DINO usage label change to fixed evaluation; numerical functions,
class methods, weights and preprocessing remain the frozen implementations.
Seven metrics are PSNR, LPIPS-Alex, DINOv2-L cosine, ConvNeXt source-prediction
agreement, DISTS, DreamSim ensemble v0.2.0 and MS-SSIM. Supplementary loaders,
weights, source code and protocol are exactly those used for the completed
common500/T1-C supplements. No PIL requantization or changed crops are introduced.
Source-label argument0 is unused for DINO-only features; agreement compares
source/reconstruction predictions, not label accuracy.

Four-metric models and three supplementary models run sequentially on primary
GPU0, FP32, TF32 off,6/2 threads, batch1. Start free8GiB, runtime floor2GiB,
allocator12GiB; these resource limits are disclosed and are not timing parity.
Each model/reference/pair operation has a durable reservation/completion record.
Unknown failures retain state and stop. No automatic retry or repeat of successful
scientific calls is provided. All source and output identities are SHA bound.

Baselines use final P100k, Swin200k and BPG records. Entropy NeST-Com19dB uses
only T1-C m10 PARTIAL new_mean, all seven metrics;4/10dB retain frozen originals.
ARPC and NeST use6201/6202/6203; other historical baselines use2001/2002/2003.
Pairing is by source, without a claim of identical noise across all methods.

The source first averages its three noise realizations. A single formal
10000x100 NumPy RNG draw matrix, seed2026100701, is preserved with SHA.
All five descriptive means/pointwise95% percentile intervals reuse this matrix.
The21 primary NeST-minus-ARPC intervals resample directly paired source deltas.
Negative values, zeros and crossing-zero intervals remain intact. These are
pointwise intervals with no multiplicity adjustment or overall-winner claim.
New eight synthetic tests verify populations, denominator failures, exact paired
intervals, negative deltas, original math and complete T1-C19dB replacement.

Only after real score/statistics/owner/supervisor exits0 and no unresolved calls
may root close the stage, render actual final summaries and publish reviewed
source/lightweight results. New model training, old baseline experiments,
timing repeats and additional hypotheses are excluded.
