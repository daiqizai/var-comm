# Bounded confirmation and residual-dependence diagnostic

**COMPLETE.** Both workers exited0; 243 new physical frames and the fixed16
diagnostic are closed. See [the report](../../results/two_route_confirmation_20261028_v1/REPORT_zh.md).
The actual matched-ceiling utilization is99.07% versus94.14%. All16 residual
vectors equal their conditional prediction, and all16 source vectors are the
same vector. Training remains paused; no follow-up is automatically launched.

This is a new, independent follow-up to commit 6bff6eab. Existing training remains
paused; the old STOP and checkpoints are preserved. No shared-codeword expansion,
training, model selection, new seed, or common500 evaluation is included.

A uses the original 32 development sources, N1024, design SNR2, actual0/2/4 dB and
the same three noise realizations. Both single-stream controls share the original
ceiling m5/K27, MCS (QPSK,k637,n1912), paid29-bit body fields and profile catalogue.
Only candidate granularity differs: original quarter-scale endpoints versus all
token endpoints up to that same ceiling. Reuse exact physical events only after
checking profile, information, counter, SNR and noise identity. At most288 new
physical events; no new source traversal or image/metric inference is planned.
Independent RX/metric cache misses stop with evidence, never borrow TX truth.

B uses only the progressive arm's saved step3000 checkpoint, the already fixed
16 development sources, j256, original GPU5 and FP32/TF32-off implementation.
Compare independently decoded transmitted quantized residuals with per-position
argmax of the16-way conditional entropy probabilities p_phi(F_C). Both inputs
use the same received F_C, depth vector and synthesis/decoder weights. Existing
evaluation saved scores but not streams/pixels: a separately counted deterministic
materialization is necessary; its MSE must match the previous result exactly.
Reuse saved transmitted LPIPS. Identical quantized vectors reuse the actual native
image and score. Report scalar/bit agreement, exact-vector agreement, image
differences and source-paired PSNR/LPIPS/D differences, without new bootstrap.
Equal inputs cannot establish synthesis insensitivity; no extra intervention is
authorized here. This diagnostic does not select a final model or finish B1.

Caps: A288 physical events, B16 source materializations/independent decodes,
32 synthesis+image renders,16 additional LPIPS pairs. Each worker<=1800 seconds;
total new GPU-device time<=1800 seconds and worker-wall time<=3600 seconds.
Actual use and reuse are separate. Models/arrays/logs stay on PFS. Scientific
failures are preserved, without automatic repeat. Publishing code is not execution.
