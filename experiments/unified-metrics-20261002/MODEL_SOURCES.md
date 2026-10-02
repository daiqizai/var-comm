# Frozen additional evaluation models

These models evaluate final reconstructions. They do not change a communication
encoder, receiver, generator, selected policy, training loss, or calibration
criterion. The existing DINO column is **DINOv2 ViT-S/14**, and its original values
and registered preprocessing remain intact. At the user's request, a separate
**DINOv2 ViT-L/14** column is added as `dinov2_vitl14_cosine`; it does not replace
the retained `dino_cosine` S/14 values.

## Models and conventions

| Metric | Registered evaluator | Input and result |
|---|---|---|
| CLIP image similarity ↑ | OpenAI CLIP ViT-L/14, original 224-pixel checkpoint | Shorter-edge bicubic resize to 224, center crop 224, official CLIP mean/std; cosine between image embeddings |
| DISTS ↓ | Official DISTS learned alpha/beta, VGG16 ImageNet-1k V1 | Native 256×256 RGB [0,1]; official internal normalization and forward; one score per pair |
| Classification ↑ | torchvision ResNet50, IMAGENET1K_V2 | Official resize 232 / crop 224 / bilinear transform; top-1 versus true ImageNet index and versus the same classifier's original-image prediction |
| DINOv2 L/14 similarity ↑ | Meta standard `dinov2_vitl14`, no registers | Same retained S/14 tensor preprocessing: square 224×224 bicubic, align_corners=False, antialias=False, ImageNet mean/std; cosine of 1024-dimensional `x_norm_clstoken` |
| DreamSim ↓ | Official ensemble, released LoRA weights | DINO ViT-B/16 + CLIP ViT-B/16 + OpenCLIP ViT-B/16; direct bicubic resize to 224×224; official per-model normalization and normalized ensemble embedding |
| MS-SSIM ↑ | VainF/pytorch-msssim | Native RGB [0,1], data_range=1, five default scales, Gaussian 11×11 window, separate per-image results |

All new evaluators receive the same floating RGB tensors, before any extra
8-bit image export. CLIP, ResNet and DreamSim use torchvision's antialiased
**tensor** resize path. DINOv2 L/14 uses the exact retained S/14 interpolation
without antialiasing. The tensor paths are not asserted to be
bitwise equal to a PIL resize after quantization. Models run in FP32, in eval
mode, without gradients or AMP. The exact torch/torchvision and optional package
versions are recorded with each evaluation manifest.

The classifier is separate from the communication/generation models. For a
class-conditioned reconstruction (D_C), the receiver already had the true
label; its classification score is descriptive and must not support the main
semantic-preservation claim. D_U is the principal classification comparison.
Original-image classifier accuracy and per-source predictions are retained.

The evaluators are not mutually independent: DreamSim itself includes DINO and
CLIP families. DISTS uses VGG16; the retained LPIPS uses its previously registered
backbone. The valid separation claim is that these semantic/classification
evaluation backbones are not the communication encoder, VAR, or Dc.

## Official implementations and pins

Pins were read from the official repositories on 2026-10-02. Every executable
Python file in each downloaded implementation must also have a full SHA256 in
the local asset manifest; a branch name alone is rejected.

| Implementation | Pinned commit |
|---|---|
| [OpenAI CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6) | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| [Official DISTS](https://github.com/dingkeyan93/DISTS/tree/1267d8cb626c98706db3697422701c56a85ebf2e) | `1267d8cb626c98706db3697422701c56a85ebf2e` |
| [Official DreamSim](https://github.com/ssundaram21/dreamsim/tree/db4d16c6948314e36e2e62f0bae2ee23bc974bf3) | `db4d16c6948314e36e2e62f0bae2ee23bc974bf3` |
| [DINO backbone implementation used by DreamSim](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a) | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| [DINOv2 implementation shared by S/14 and L/14](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| [MS-SSIM](https://github.com/VainF/pytorch-msssim/tree/b057b072dd869ae3f6b88543786f44d008315f69) | `b057b072dd869ae3f6b88543786f44d008315f69` |

The [CLIP official loader](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py)
registers both the transform and a complete checkpoint SHA256:
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
The evaluator calls its local `build_model` with that verified archive and does
not need tokenization or text inference.

The [official DISTS implementation](https://github.com/dingkeyan93/DISTS/blob/1267d8cb626c98706db3697422701c56a85ebf2e/DISTS_pytorch/DISTS_pt.py)
normally obtains VGG16 through torchvision's pretrained loader. Our constructor
supplies a verified local VGG16 checkpoint, initializes with `load_weights=False`,
and copies the released alpha/beta tensors. Its feature extraction, pooling,
normalization, distance formula and per-image forward remain unchanged.

[Torchvision's ResNet50 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
defines V2 and its transform. We construct without automatic weights and strictly
load the local `resnet50-11ad3fa6.pth`. The source manifest additionally records the
installed torchvision version. Labels must already use the registered
ImageNet-1k 0–999 ordering; no category reordering or inferred labels are allowed.

The added L/14 model uses the same frozen official DINOv2 source tree as the
retained S/14 metric. All 157 Python files in that existing tree were checked
against the official commit above. The loader reuses `dinov2.hub.backbones` and
its existing `DinoVisionTransformer` class, and rejects an imported DINOv2 module
from another tree instead of replacing it. It constructs `dinov2_vitl14` with
`pretrained=False`, then strictly loads the separately registered official local
checkpoint. Checks require 24 blocks, patch size 14, 1024-dimensional CLS features,
and zero register tokens. No source checkout, training or download occurs here.

The [official pretrained-model table](https://github.com/facebookresearch/dinov2#pretrained-models)
labels the released ViT-L/14 checkpoint as distilled. “Standard pretrained,
without registers” is therefore the accurate model description; this study does
not perform further distillation or fine-tuning. The L/14 and S/14 metrics belong
to the same model family and are not presented as independent feature spaces.

The [DreamSim factory](https://github.com/ssundaram21/dreamsim/blob/db4d16c6948314e36e2e62f0bae2ee23bc974bf3/dreamsim/model.py)
uses its released ensemble and `normalize_embeds=True`. The official forward
is cosine distance between its resulting embeddings; caching an original
embedding therefore preserves the same formula. Its automatic DINO hub call
is narrowly redirected to the pinned local official DINO implementation and
registered backbone weights. Other hub requests and all network connects fail.
The factory still applies its original converted CLIP weights and LoRA adapters.

## Asset preparation

Assets are downloaded separately, never during metric inference. Record the full
SHA256 after downloading each official file. Estimates below are planning sizes,
not integrity checks.

| Asset | Official download | Approximate size |
|---|---|---|
| CLIP ViT-L/14 | [OpenAI archive](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | 0.9 GB |
| VGG16 V1 | [torchvision archive](https://download.pytorch.org/models/vgg16-397923af.pth) | 0.554 GB |
| DISTS alpha/beta | [pinned official file](https://raw.githubusercontent.com/dingkeyan93/DISTS/1267d8cb626c98706db3697422701c56a85ebf2e/DISTS_pytorch/weights.pt) | small |
| ResNet50 V2 | [torchvision archive](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | 0.103 GB |
| DINOv2 ViT-L/14 | [official standard no-register backbone](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | 1.2 GB |
| DreamSim ensemble | [official v0.2.0-checkpoints release](https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip) | verify archive Content-Length |
| DreamSim's DINO base | [official ViT-B/16 backbone](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) | 0.35 GB |

The DreamSim archive must contain `dino_vitb16_pretrain.pth`,
`clip_vitb16_pretrain.pth.tar`, `open_clip_vitb16_pretrain.pth.tar`, and
`ensemble_lora/adapter_config.json` plus the adapter weights. The separate DINO
base file is the official backbone used by the hub factory, not the larger
DreamSim training checkpoint containing student/projection-head data.

[DreamSim's package requirements](https://github.com/ssundaram21/dreamsim/blob/db4d16c6948314e36e2e62f0bae2ee23bc974bf3/setup.py)
list torch, torchvision, numpy, Pillow, open-clip-torch, peft, timm, scipy and
transformers without an exact compatibility matrix; its code requires peft
at least 0.2.0. PEFT 0.21.0 is an available official release and can be pinned in
the isolated evaluation environment, subject to a real loading check with the
installed transformers version. A dependency/version failure is not a DreamSim
score. The evaluator records a missing optional registration explicitly and
fails on a supplied but invalid asset.

## Reusing one source for a reconstruction batch

`MetricEvaluator.expand_reference(reference_single, prepared_single, batch_count)`
returns a repeated reference tensor and its repeated feature cache for `score`.
It first checks the original floating pixels, single-image cache shape and
evaluator identity. Repeated tensors own separate storage and do not overwrite
the original-image baseline. CLIP, ResNet50, DreamSim and DINOv2 L/14 cached
reference outputs are reused without another original-image model call. DISTS
retains its original pair forward, including reference feature computation;
no cached-feature proxy or distance-formula change is introduced.

Labels remain scoring inputs only. The original-image classifier baseline is
computed once with batch size one. CPU qualification additionally compares batch
size two with two individual score calls, requiring absolute differences at most
`2e-5` for each floating metric and exact classification agreement. This check
does not qualify a larger GPU batch or imply bitwise equality between batch
sizes.

Before formal scoring, `batch_speed.py` tests GPU batches 1, 2, 4, 8 and 16 while
both the frozen replay models and all six metric models remain loaded. It uses
one analytic RGB reference and 16 fixed synthetic reconstructions; no source or
development image enters this check. Every candidate gets one full warmup and
three timed passes over the same 16 images. All five floating metric columns
must agree with individual scoring within absolute `2e-5`; all classifier
predictions, correctness flags and conditioning flags must agree exactly.
Candidate batches that run out of CUDA memory, exceed 88% of total device
memory in peak process reservation, or fail numerical agreement are rejected.
The fastest median time per image among valid candidates is registered, with a
smaller batch winning an exact timing tie. Batch one must pass and remains the
fallback; a formal scoring failure stops the run instead of changing batch size.

`metric_batch_qualification.json` records all candidates, timings, peak memory,
absolute differences, selected batch and valid tail sizes. It binds the model
manifest, evaluator identity, qualifier/evaluator source hashes, hardware and
frozen replay inputs and numerical flags. Resume requires that same binding.
The receipt is an engineering measurement with `synthetic_images=true` and
`scientific_result=false`. Its exact bytes and SHA256 accompany the registered
results. No AMP, model formula or original reconstruction batch is changed;
the old generator's random state is preserved. A tail uses only valid powers of
two no larger than the selected batch.

## Manifest contract

Each weight entry is `{"path": "/absolute/file", "sha256": "64 hex digits"}`.
An implementation entry adds `revision` (the commit above) and `files`, mapping
all Python source paths relative to `path` to full SHA256 digests.

Top-level keys:

- `clip`: `implementation`, `weights`.
- `dists`: `implementation`, `weights` (alpha/beta), `vgg16_weights`.
- `resnet50`: `weights`.
- `dinov2_vitl14`: `implementation` (the retained S/14 source tree and its verified
  official revision), `weights` (separate L/14 checkpoint with a full SHA256).
- `dreamsim`: `implementation`, `dino_implementation`, `dino_weights`, `cache`.
  `cache` has `path` and `files` for every model/config file under its root.
- `ms_ssim`: `implementation`.

The loader has no automatic download or training step. Its contract tests use
small synthetic stubs only to test shapes, caching, label handling, preprocessing
and failure behavior. They do not establish pretrained-model quality; a real
weight qualification is required before scientific scoring.

KID/FID are deferred to a larger holdout. Three noise draws for one source remain
one source cluster for paired bootstrap; 100 sources × 3 draws are not 300
independent images.
