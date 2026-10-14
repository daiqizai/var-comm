# Fixed development16 quality supplement

The authoritative successful registration is `dependency_recovery_registration.json`.
All 48 original pairs and 144 new scalar scores are retained: 16 final HiFi,
16 paired final Swin200k, and 16 same-source/workpoint P100k reconstructions.
This is N1024/13 dB, one original noise per source in the selected development16
display set. The means and signed, directly source-paired differences are descriptive.
No common500 confidence interval or population inference is made.

DISTS, DreamSim ensemble v0.2.0 weights, and MS-SSIM use their fixed official
implementations and the unmodified old offline metric loader. Native float32
RGB inputs are used. DreamSim square224 bicubic antialiased tensor preprocessing
retains original float values without PIL re-quantization; DISTS and MS-SSIM
use native256 RGB. Exact versions, source/weight SHA, settings and actual runtime
are supplied in registration and evidence. Official sources:
https://github.com/dingkeyan93/DISTS
https://github.com/ssundaram21/dreamsim
https://github.com/VainF/pytorch-msssim

Two diagnosed initialization failures produced zero pair scores and zero metric
forwards. The first failed before any evaluator weights loaded because Windows
serialized relocated Linux paths with backslashes. The second completed one
DISTS construction and stopped during DreamSim import on a missing MarkupSafe
dependency. Both real exit1 records and failed loader reservations are preserved.
The successful entry retains exactly the same scientific function AST, pixels,
official metric source, weights and preprocessing. Three missing dependencies
(MarkupSafe3.0.3, einops0.8.1 and sniffio1.3.1) are SHA-verified official wheels
in a private evaluation overlay; the frozen original environment is unchanged.
An actual CPU import qualification blocked checkpoint/network loading and
constructed no evaluators. Every outcome is reported and feeds no selection.

By explicit user choice, common500 waits for original P100k/Swin200k pixels.
No older-checkpoint substitutions, training, receiver/image generation,
channel replay, bootstrap, checkpoint/policy selection or main-figure changes occur.
