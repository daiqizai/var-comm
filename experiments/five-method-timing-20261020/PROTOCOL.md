# Five-method current-version N1024 latency

Status: implementation preparation; no measurement has been launched.

The authorized population is the original16 fixed development sources, preserving
their source IDs and indices. Methods are entropy-coded NeST-Com with the current
m10-corrected19dB policy, Latent JSCC100k, SwinJSCC200k, adaptive-resolution BPG+LDPC,
and ARPC+LDPC with its frozen e7-target policies and actual-length fallback.
SNRs are4,10,19dB. Batch size is1. All five methods run serially on the same
physical CPU/GPU pair. The exact admitted device, CPU affinity, environment,
precision, thread counts and shared load must be recorded before execution.

## Finite work

For each method/source/SNR, three fresh warmups precede three fresh measurements.
This gives720 warmups plus720 measurements,1440 full calls. Each method contributes
288 calls,144 formal measurements,48 formal measurements per SNR.
Timing repetitions use one registered noise realization per setting, with fresh
TX and RX computation every time; they are not independent quality-noise trials.
No old successful scientific work is replayed as a hidden qualification, and no
measured case is silently retried after failure.

TX includes input normalization, any required caption, fresh image/token encoder,
source probability traversal, independently finished endpoint streams and actual-length fallback, container
and paid metadata, CRC, FEC and modulation. RX includes actual header/body decode,
parsing, independent source decode, neural completion and image reconstruction.
No TX CDF, source pixel, unsent token or source-truth correction enters RX.
NeST-Com uses the current one-pass multi-endpoint encoder, including m10 at19dB;
each fallback stream has its own real arithmetic termination. This implementation
is recorded separately from the historical per-attempt timing implementation.
For continuous JSCC, its real paid signaling and encoder/decoder are retained.
For adaptive BPG, all frozen resolution/QP candidates required by its source rule
are counted. Native256 BPG is not substituted.

The primary software total is TX+RX, expressed in ms/image. Each GPU timing
boundary synchronizes the registered CUDA device. Model construction/loading,
evaluation networks, pure provenance checks, result persistence, and software
AWGN synthesis are outside the primary total and separately recorded.
Required codec conversions and subprocess work are inside the corresponding
endpoint time. No radio airtime or energy measurement is claimed.

## Results and limits

Retain per-case timings, receiver outcomes, exact repeated-input/output hashes,
component counts, resource samples and failures. Summaries report mean, median
and p95 across the48 formal calls per method/SNR, together with source count and
repetition count. No bootstrap or quality scoring is performed.
Shared load and any framework differences are disclosed; current measurements
are not spliced with the historical six-method H800 values.

The scientific scope excludes common500 quality evaluation, N2048, new training,
new policy/checkpoint selection and timing of additional methods.
