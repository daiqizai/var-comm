# A2a fixed-cal100 clean quality

Use only the closed A2a700-state manifest and fixed cal100. This separate process
does not run ARPC, reconstruct, transmit, train, bootstrap or read test outcomes.
It scores700 clean pairs and100 fixed failure-baseline pairs. Exactly identical
float image/source/model pairs may reuse a stored score, retaining all800 rows.
The gray reference is the established float32 RGB0.5 digital failure output.

PSNR uses the original float64 MSE expression; LPIPS uses the original AlexNet
loader and pair expression; DINO-L uses the original ViT-L/14 weights and exact
224 bicubic/no-antialias preprocessing. Original SHA/AST projections and weight
hydration receipts are validated. DINO-L alone is the prescreen objective;
PSNR and LPIPS are diagnostics. ConvNeXt and unused DINO-S are not constructed.
ARPC reconstruction input is its saved native float_output, not a PNG or a
new8-bit quantization. Sources remain original uint8/255. No cropping/clipping.

Run in the original frozen_visual_py310 environment on primary H800GPU0 after
fresh admission and PCI/CUDA identity confirmation. FP32, AMP off, TF32 off,
deterministic,6/2 threads,batch1 are preserved. Startup≥8GiB free/util≤25,
allocator≤6GiB,running free≥2GiB. One model load each for LPIPS and DINO; at most
100 source-DINO,800 reconstructed-DINO,800 LPIPS pairs/1600 backbone forwards.
Finite worker3600s plus outer3900s cleanup allowance. No automatic retries.

New synthetic qualification is metadata/math only, no model imports. After
the root checks actual A2 clean parent/child exits on leo, a closed receipt is
required before registration. Execute prepare once then dispatch once with
the actual registrationSHA. Per-call reservations/completions, child PID/start,
actualwait, stdout/stderr, model/metric identity and CSV are preserved.
No automatic A2b successor. Existing matched PHY proxy and a complete
84-strategy actual-fallback screen with uncertainty gate must still be evaluated.
