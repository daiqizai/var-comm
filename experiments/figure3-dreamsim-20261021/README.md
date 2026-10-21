# Fixed Fig.3 DreamSim supplement

This entry scores only the ten existing native float reconstructions for
Pastries95 and Shopping-cart75, N1024, 10 dB, timing repetition r3.
Method order: P100000, Swin200000, BPG, ARPC, NEST. Source images and all
reconstruction file/array hashes are registered before any model call.
Original LPIPS-Alex scores are copied verbatim from display_manifest.csv.

DreamSim reuses the frozen main evaluator and ensemble weights, FP32, batch1,
224x224 bicubic antialias floating-point resize, with no crop or quantization.
Distance is one minus cosine similarity of the ensemble image embeddings.
The two source embeddings are reused; ten reconstruction embeddings and ten
distances are permitted. No other metrics, reconstructions, channels or
bootstrap are run. GPU1 on the existing H800 machine is used; this is a quality
supplement, not a new latency comparison. Startup free>=8 GiB, running free>=2
GiB, allocator cap6 GiB, six CPU threads, two interop threads, TF32/AMP off.

Run once with the registration SHA: frozen_visual_py310/bin/python -B score.py
dispatch --registration SHA. The dispatch directory is exclusive, the parent
actually waits for the child, and each inference call is ledgered. A failure
is retained and never automatically retried. Logs/features stay outside Git.
