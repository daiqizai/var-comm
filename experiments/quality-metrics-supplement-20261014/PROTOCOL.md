# Saved-output supplementary evaluation, 2026-10-14

The user authorized three supplementary metrics on existing outputs only.
No training, channel simulation, transmission, receiver reconstruction, model
selection, main-configuration selection, or checkpoint selection is permitted.
Previous scores and main figures remain unchanged. Every beneficial, adverse,
zero, and inconclusive result is reported.

## Common500: waiting for original continuous-baseline pixels

The requested six-method grid remains original common500, N1024,
SNR4/10/19, three original noises per source: 27,000 logical pairs and
81,000 intended scalar scores. Digital arms retain6201/6202/6203;
continuous and adaptive-BPG arms retain2001/2002/2003. Paired intervals
must first average each source's own three noises, then directly bootstrap
source differences (10,000 replicates, original seed2026100701).
Marginal confidence intervals must never be subtracted.

The new P100k/Swin200k common500 program preserved old metrics and
reconstruction hashes but did not save native reconstruction arrays. No saved
new-checkpoint pixel grid was found in its output or completed formal backup.
The user explicitly chose: first finish HiFi; common500 waits for original
files. It is therefore NOT_RUN. Neither old P40k/Swin80k images nor replayed
channels substitute for the9,000 missing new-checkpoint logical frames.
Raw and BPG original arrays have been located and restored from prior verified
archives, retaining their exact member SHA; this is byte restoration, not
scientific reconstruction or scoring. Their restoration does not authorize
starting the incomplete common500 score grid.

## Final-link HiFi: complete development16 descriptive table

The original16 indices are0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95.
Score all16 final HiFi outputs and all16 existing final Swin200k and P100k
counterparts, exactly48 logical image pairs/144 scalar scores. N1024/13dB,
channel seed2001, HiFi posterior seed23, original receiver schedule and
checkpoints remain. HiFi and Swin share their exact saved observation;
P100k is same-source/same-workpoint, not same-waveform. Original native
float32 CHW RGB arrays and uint8 source archives are SHA authenticated.
No paper montage, screenshot, thumbnail or JPEG reconstruction is accepted.
Exact native pixel-pair reuse retains all original logical rows and failures.

This was a fixed development display set with one saved noise per source.
Supply every frame's scores, all16-source means, sample standard deviations,
medians and ranges, and direct source-paired descriptive differences.
No inferential CI, bootstrap or population/generalization claim is made for
this selected small set. It is not merged with common500 or19dB figures.

## Fixed metrics before scores

Reuse the immutable official offline loader, SHA
dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066.
Only the three explicitly authorized evaluators are constructed:

- DISTS official revision1267d8cb626c98706db3697422701c56a85ebf2e,
  learned alpha/beta plus torchvision VGG16 IMAGENET1K_V1. Native256 RGB
  float32 in[0,1], no crop/resize/requantization, official internal normalization.
- DreamSim official revisiondb4d16c6948314e36e2e62f0bae2ee23bc974bf3,
  ensemble v0.2.0-checkpoints, DINO-B/16+CLIP-B/16+OpenCLIP-B/16 and LoRA,
  normalize_embeds=True, use_patch_model=False. Author square224 bicubic
  antialiased tensor geometry, no crop, native float precision retained instead
  of PIL requantization; official per-backbone internal normalization.
- VainF MS-SSIM revisionb057b072dd869ae3f6b88543786f44d008315f69,
  native256 RGB/data_range1, win11/sigma1.5, five weights
  [0.0448,0.2856,0.3001,0.2363,0.1333], K=(0.01,0.03), size_average=False.

DISTS/DreamSim are lower-is-better; MS-SSIM is higher-is-better.
Do not switch branch, color space, crop, interpolation, weights, sign or
handling of negative/zero values after observing results. Record every source
file and weight SHA, actual Torch/dependency versions and preprocessing.
DreamSim's learned backbones overlap other semantic evaluators; these metrics
are not independent replications of one quality claim.

## Actual execution and delivery

One SHA-bound registration and one bounded scoring attempt; unknown failure
stops without automatic replay. Private GPU3, FP32/TF32off, deterministic
operations, batch1,6/2 threads, CPU0–5,12GiB allocator and2GiB running free
margin; startup requires8GiB free. Model-load/forward reservations and outcomes
are stored in SQLite. Only original arrays and metric implementations are read.
New large assets, caches and logs stay in liberai-tmp; only source, registrations,
score tables, reports and small closure evidence enter the sole main worktree.
The supervisor records exact argv, real PID/start generations, durable raw logs
and actual child.wait. Finalization requires actual exit0, all recorded
generations gone, no failures and complete48-row/144-score closure.
Publish normal reviewed exact paths after the repository's required checks;
backup the verified increment and sync light files locally.
