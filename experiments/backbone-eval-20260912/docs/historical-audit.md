# Historical development protocol and read-only dependency audit

Scope: this is an audit of existing files, not a new model experiment. Historical
projects/checkpoints/results were neither changed nor copied. Existing training
was only inspected with `ps`/status reads; no signal was sent. The latest user
instruction is to let the current local GPU task finish, then evaluate locally.

## Population and preprocessing

- Frozen manifest: `/home/liulu/projects/VAR-MAP-GATE0/results/vae_reconstruction/imagenet100_m6_to_m10_rate_sweep/selected_samples.json`.
- SHA256: `33d2a4f13eb97bb1d47fca25df04d9ca4b7ebf0fdcefe7d9497c5a6e4830c243` (recomputed).
- 100 unique images, 100 classes, all 100 source JPEGs exist. Total source JPEG size: 15,086,999 bytes. Manifest paths and class indices must be kept paired; old manifest field `split: test` is historical and does **not** make this reused set a fresh holdout.
- Images are under `/home/liulu/projects/VAR-MAP-GATE0/data/imagenet/val/`.
- Exact preprocessing: PIL RGB; scale = `256 / min(width,height)`; resized dimensions use Python `round(width*scale), round(height*scale)`; torchvision PIL bicubic resize with `antialias=True`; torchvision center crop 256; `pil_to_tensor().float()/127.5 - 1`. Do not silently substitute a generic torchvision `Resize(256)` with potentially different rounding.

## Existing model assets

| Asset | Exact historical path | Recorded SHA256 |
|---|---|---|
| Official VAE | `/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth` | `7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c` |
| Official VAR-d16 | `/home/liulu/projects/VAR-MAP-GATE0/checkpoints/var_d16.pth` | `4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb` |
| Fidelity decoder | `/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/outputs/train/EXP-VAR-DECODER-ONLY-M89-001/checkpoints/best.pt` | `7b79b4b96f0b13ba5f77cf3cd4038ac1f0fe216e2b516e6a3a3ba57fb20d2802` |
| Metric DINOv2 ViT-S/14 | `/home/liulu/projects/CAP-VPR/artifacts/checkpoints/dinov2/dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |

The evaluation constructor re-verifies those checkpoint hashes before use. The
741 MiB fidelity checkpoint contains optimizer/training state; load only
`payload['decoder']` strictly into `vae.decoder`. It is **not** a complete
replacement VAE checkpoint. Its epoch-3 selection was a predeclared PSNR fallback
because the perceptual eligibility gate failed, not a gate-passing fidelity gain.
All non-decoder state remains the official VAE.

- VAR source: `/home/liulu/projects/VAR-MAP-GATE0/third_party/VAR`.
- DINO source: `/home/liulu/projects/VAR-MAP-GATE0/third_party/dinov2`.
- Caution: neither vendored directory is an independent git repository; `git -C ... rev-parse HEAD` returns the parent project commit, not the upstream source commit. Frozen parent config records DINO source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, tree SHA `faff1743dcb7dfd492d1312bbbdc229f9f462161611aec4a810418353204adf5`.

## Unified metric implementation

Read-only implementation:

- Historical `scripts/evaluate_var_decoder_only.py::calculate_metrics`, SHA256 `2ff8ac3b62d87bd00dc72812c7ba7333d0147310f76aaa91c7736056d2cf109b` (recomputed).
- Historical `src/cadsd_jscc/var_decoder_tuning.py`, SHA256 `c5c24e829cc575c2d8e10ca09ad78514ea527814f2015346b626ec0ca0c91ddd` (recomputed).
- PSNR: mean RGB/spatial MSE per image in `[0,1]`, `-10*log10(mse.clamp_min(1e-12))`; average per-image dB, not dB of aggregate MSE.
- SSIM: `pytorch_msssim.ssim(...,data_range=1.0,size_average=False)` with original library defaults.
- LPIPS: `lpips.LPIPS(net='alex',verbose=False)`, inputs transformed to `[-1,1]`.
- DINO: `dinov2_vits14`; tensor bicubic interpolation to 224x224, `align_corners=False`, no new explicit antialias option; ImageNet channel mean/std; `forward_features()['x_norm_clstoken']`; cosine similarity to the source CLS feature.
- Evaluate floating-point outputs before PNG quantization. Official decode: `vae.decoder(vae.post_quant_conv(fhat.float())).clamp(-1,1)`, then map to `[0,1]`.
- Historical evaluation was fp32 without autocast. New measurements use batch 1, no TF32/autocast, and record any numerical replay differences rather than relabel historical rounded metrics as new observations.

Local lightweight package metadata (no GPU initialization): torch `2.11.0+cu128`,
torchvision `0.26.0+cu128`, pytorch-msssim `1.0.0`, lpips `0.1.4`, numpy `2.2.6`,
Pillow `12.2.0`, PyYAML `6.0.3`, timm `1.0.27`. `xformers` is absent; historical
VAR builder explicitly disables flash/fused attention. LPIPS's large AlexNet
backbone is already at `/home/liulu/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
(244,408,911 bytes); LPIPS v0.1 learned head is packaged in the installed lpips
module (6,009 bytes). No hidden network dependency is needed for these metrics.

## Old source-only completion protocol

Schedule `[1,2,3,4,5,6,8,10,13,16]`, one 4096-entry codebook, 12 raw bits/index:
full = 680 indices/8160 bits, first 8 scales = 255/3060, first 9 = 424/5088.

- Direct prefix reconstruction accumulates only transmitted residual contributions **on the final 16x16 latent grid**, with the quantizer's fixed total scale count of 10. Missing residuals remain zero. Do not reduce the total number of scales for interpolation/indexing of residual modules.
- Completion uses ground-truth ImageNet class pre-shared at zero bits under this diagnostic oracle-side-information assumption. If conveyed, add 10 fixed bits and disclose reliability separately.
- One class-conditioned branch, deterministic closed-loop argmax, **no classifier-free mixing**, no random sampling, no best-of-source-based selection. This is equivalent to ordinary CFG scale 1, not XQ upstream's extra-guidance ramp with argument 1.
- Feed true prefix through the causal VAR cache; all later scales use only generated tokens. New receiver API is stricter than the old function: it physically receives only the prefix token lists, not true suffix tokens.
- Historical full progressive accumulation equalled the ordinary full `f_hat` exactly (maximum absolute difference 0).
- These are noiseless source bit counts, not 3060 complex channel uses; no entropy coder, FEC, modulation or packet side information is present.

## Historical values for sanity replay, not new results

All metrics below are the archived means from the same 100 development images.

| Decoder / input | Raw bits | PSNR | SSIM | LPIPS | DINO |
|---|---:|---:|---:|---:|---:|
| Official full | 8160 | 22.835654879 | .638011905 | .105693840 | .938358659 |
| Official first8 direct | 3060 | 18.500846949 | .474803162 | .305652943 | .548305365 |
| Official first8 completion | 3060 | 19.680456753 | .511905314 | .177602671 | .890598145 |
| Official first9 direct | 5088 | 20.239022675 | .53685 | .20286 | .79175 |
| Official first9 completion | 5088 | 21.231569118 | .56976 | .13589 | .92317 |
| Fidelity full | 8160 | 23.828756485 | .67207 | .12099 | .93000 |

Exact per-image/full-precision sources (relative to historical diffusion directory):

- `outputs/analysis/ANALYSIS-VAR-SEMANTIC-RATE-CONTRIBUTION-IMAGENET100-001/{per_image_metrics.csv,mean_metrics.csv,completion.json,budget_ledger.json}`: 1000 rows including both decoders' full/direct/completion conditions.
- `outputs/analysis/ANALYSIS-VAR-DECODER-ONLY-IMAGENET100-001/{per_image_metrics.csv,mean_metrics.csv,completion.json}`: 600 rows for both decoders' full/first8/first9 completion conditions.
- Historical source-token completion metrics were deterministic; sampling belongs only to the new explicitly separate sensitivity analysis.

## New harness boundaries

`evaluate_backbones.py` reuses the metric functions above without modifying them.
Each model runs in its own subprocess, avoiding upstream `models` namespace
collisions. It generates exact float reconstructions first, saves own-run float
spools, deletes the codec, then loads DINO/LPIPS for metrics. Thus synchronized
encoding/decoding/completion measurements exclude metrics/I/O and metric weights
are not included in codec peak allocation. Process-local caching/reserved memory
is reported honestly, not as a model parameter-only estimate.

XQ natural points count **both** product branches: first8 = 2424 bits, first9 =
3960 bits, full = 6864 bits. They are not old 3060/5088 points. WeTok full is 8192
bits. Primary XQ completion is class-only argmax; upstream-style CFG-ramp 3.25,
top-k 750/top-p .95, seeds 0/1/2 is labelled sensitivity. No image-dependent
selection is allowed. Sampling seeds are averaged within each image before
bootstrap; the effective source population stays 100, not 300.

Smoke checks include upstream XQ full reconstruction agreement, all-true-token
completion identity, and an arbitrary withheld-suffix mutation leaving primary
prefix completion unchanged. The adapter asserts transmitted-prefix identity.
The paired XQ generator's embedded tokenizer is compared against the standalone
checkpoint; any mismatch must be prominent in interpretation, never hidden.
