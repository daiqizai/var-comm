"""Frozen, offline image evaluators; no communication model or policy is changed.

Inputs are RGB BCHW floating tensors in [0, 1]. Tensor-native resize deliberately
avoids the extra 8-bit quantization introduced by converting reconstructions to
PIL. Geometry, interpolation and normalization follow each official evaluator.
All external executable source and weights must be supplied in a SHA256 manifest.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import socket
import sys
from pathlib import Path
from unittest import mock

CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MANDATORY = ("clip", "dists", "resnet50", "dinov2_vitl14")
OPTIONAL = ("dreamsim", "ms_ssim")
MODEL_IDS = {
    "clip": "OpenAI CLIP ViT-L/14 (224px), image-image cosine",
    "dists": "DISTS official learned alpha/beta; torchvision VGG16 IMAGENET1K_V1",
    "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2",
    "dinov2_vitl14": "Meta DINOv2 ViT-L/14 standard LVD-142M backbone, no registers, 1024-d x_norm_clstoken cosine",
    "dreamsim": "DreamSim ensemble v0.2.0-checkpoints: dino_vitb16,clip_vitb16,open_clip_vitb16 + LoRA",
    "ms_ssim": "VainF/pytorch-msssim MS-SSIM, RGB, data_range=1, five scales",
}
SOURCE_REVISIONS = {
    "clip": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
    "dists": "1267d8cb626c98706db3697422701c56a85ebf2e",
    "dreamsim": "db4d16c6948314e36e2e62f0bae2ee23bc974bf3",
    "ms_ssim": "b057b072dd869ae3f6b88543786f44d008315f69",
    "dino": "7c446df5b9f45747937fb0d72314eb9f7b66930a",
    "dinov2_vitl14": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(spec, *, expected_sha=None):
    """Reject missing hashes, symlink ambiguity and unverified pretrained assets."""
    path = Path(spec["path"]).resolve(strict=True)
    expected = spec.get("sha256", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"A full SHA256 is required for {path}")
    if expected_sha is not None and expected != expected_sha:
        raise ValueError(f"Manifest does not identify the registered official weights: {path}")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Asset SHA256 mismatch: {path}")
    return path


def verify_tree(spec, *, source=True, expected_revision=None):
    """Validate all Python code in an implementation and every listed cache file."""
    root = Path(spec["path"]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected a directory: {root}")
    if source and not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
        raise ValueError(f"A pinned 40-character Git commit is required: {root}")
    if expected_revision is not None and spec.get("revision") != expected_revision:
        raise ValueError(f"Implementation differs from the registered source revision: {root}")
    files = spec.get("files", {})
    if not files:
        raise ValueError(f"An explicit file digest manifest is required: {root}")
    for name, digest in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Asset path escapes the declared directory: {name}")
        verify_file({"path": str(path), "sha256": digest})
    if source:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*.py") if ".git" not in p.parts}
        missing = actual - set(files)
        if missing:
            raise ValueError(f"Unregistered executable source: {sorted(missing)[:5]}")
    return root


def package_versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scipy", "peft", "timm", "open-clip-torch", "transformers", "huggingface-hub", "accelerate", "safetensors", "dreamsim", "pytorch-msssim"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


@contextlib.contextmanager
def no_network():
    """A missing asset fails locally instead of silently downloading a new one."""
    def reject(*_args, **_kwargs):
        raise RuntimeError("Network access is disabled in the frozen metric loader")
    with mock.patch.object(socket.socket, "connect", reject), mock.patch.object(socket, "create_connection", reject):
        yield


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_dict(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


@contextlib.contextmanager
def cpu_default_checkpoint_loads():
    """Keep an official loader's unspecific checkpoint reads off the live GPU.

    Explicit positional or keyword map_location values are preserved, as are
    weights_only and every other torch.load argument. The patch is local to the
    DreamSim construction block and is restored even if model loading fails.
    """
    import torch
    original_load = torch.load
    def load(*args, **kwargs):
        if len(args) < 2 and "map_location" not in kwargs:
            kwargs = dict(kwargs, map_location="cpu")
        return original_load(*args, **kwargs)
    with mock.patch.object(torch, "load", load):
        yield


def _freeze(model, device):
    model = model.to(device=device, dtype=__import__("torch").float32)
    model.eval().requires_grad_(False)
    return model


def load_clip(spec, device):
    """Use only OpenAI's build_model; avoid its tokenizer and download wrapper."""
    import torch
    root = verify_tree(spec["implementation"], expected_revision=SOURCE_REVISIONS["clip"])
    checkpoint = verify_file(spec["weights"], expected_sha=CLIP_SHA256)
    with no_network():
        official = load_source(root / "clip" / "model.py", "_registered_openai_clip_model")
        archive = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        model = official.build_model(archive.state_dict())
        del archive
    if int(model.visual.input_resolution) != 224:
        raise ValueError("Expected the registered 224px CLIP ViT-L/14")
    return _freeze(model, device)


def load_dists(spec, device):
    """Keep official DISTS forward; supply local VGG and DISTS weights explicitly."""
    import torch
    import torchvision.models as tv_models
    root = verify_tree(spec["implementation"], expected_revision=SOURCE_REVISIONS["dists"])
    vgg_path = verify_file(spec["vgg16_weights"])
    learned_path = verify_file(spec["weights"])
    if not spec["vgg16_weights"]["sha256"].startswith("397923af"):
        raise ValueError("DISTS requires official torchvision VGG16 IMAGENET1K_V1")
    factory = tv_models.vgg16
    def local_vgg16(*_args, **_kwargs):
        model = factory(weights=None)
        model.load_state_dict(_state_dict(vgg_path), strict=True)
        return model
    with no_network(), mock.patch.object(tv_models, "vgg16", local_vgg16):
        official = load_source(root / "DISTS_pytorch" / "DISTS_pt.py", "_registered_official_dists")
        model = official.DISTS(load_weights=False)
    learned = _state_dict(learned_path)
    if set(learned) != {"alpha", "beta"}:
        raise ValueError("Unexpected DISTS learned-weight keys")
    with torch.no_grad():
        for key in ("alpha", "beta"):
            target = getattr(model, key)
            if target.shape != learned[key].shape or not torch.isfinite(learned[key]).all():
                raise ValueError(f"Invalid DISTS {key}")
            target.copy_(learned[key])
    return _freeze(model, device)


def load_resnet50(spec, device):
    from torchvision.models import resnet50
    path = verify_file(spec["weights"])
    if not spec["weights"]["sha256"].startswith("11ad3fa6"):
        raise ValueError("Expected torchvision ResNet50 IMAGENET1K_V2 weights")
    with no_network():
        model = resnet50(weights=None)
        model.load_state_dict(_state_dict(path), strict=True)
    return _freeze(model, device)


def verify_dinov2_namespace(root):
    """Reuse the frozen S/14 package; never replace an already imported class."""
    root = Path(root).resolve()
    for name, module in list(sys.modules.items()):
        if name != "dinov2" and not name.startswith("dinov2."):
            continue
        origin = getattr(module, "__file__", None)
        locations = ([origin] if origin else []) + list(getattr(module, "__path__", []))
        if any(not Path(location).resolve().is_relative_to(root) for location in locations):
            raise RuntimeError(f"DINOv2 package {name} is already imported from a different source; preserve the S/14 implementation")


def validate_dinov2_vitl14_architecture(model, official_class):
    """Reject a register variant, another backbone size or a different class."""
    if type(model) is not official_class or official_class.__name__ != "DinoVisionTransformer":
        raise ValueError("Expected the frozen official DINOv2 DinoVisionTransformer class")
    patch = model.patch_embed.patch_size
    patch = (patch, patch) if isinstance(patch, int) else tuple(patch)
    if (getattr(model, "embed_dim", None) != 1024 or tuple(model.cls_token.shape) != (1, 1, 1024)
            or patch != (14, 14) or len(model.blocks) != 24
            or getattr(model, "num_register_tokens", 0) != 0
            or getattr(model, "register_tokens", None) is not None):
        raise ValueError("Expected standard DINOv2 ViT-L/14: 1024 dimensions, patch14, 24 blocks, no registers")


def load_dinov2_vitl14(spec, device):
    root = verify_tree(spec["implementation"], expected_revision=SOURCE_REVISIONS["dinov2_vitl14"])
    checkpoint = verify_file(spec["weights"])
    verify_dinov2_namespace(root)
    sys.path.insert(0, str(root))
    try:
        with no_network():
            backbones = importlib.import_module("dinov2.hub.backbones")
            architecture = importlib.import_module("dinov2.models.vision_transformer")
            verify_dinov2_namespace(root)
            # Same official factory/class as the retained S/14 metric. No hub
            # checkout or source-module replacement is performed in this process.
            model = backbones.dinov2_vitl14(pretrained=False)
            validate_dinov2_vitl14_architecture(model, architecture.DinoVisionTransformer)
            weights = _state_dict(checkpoint)
            if "register_tokens" in weights:
                raise ValueError("Register-model weights cannot replace standard dinov2_vitl14")
            model.load_state_dict(weights, strict=True)
            del weights
    finally:
        sys.path.remove(str(root))
    return _freeze(model, device)


def load_dreamsim(spec, device):
    """Official ensemble, with the one DINO hub call replaced by local assets.

    The hub adapter builds the same official DINO ViT-B/16 and loads the official
    frozen backbone. DreamSim's released adapter and converted CLIP checkpoints
    are subsequently applied by the untouched official DreamSim factory.
    """
    import torch
    root = verify_tree(spec["implementation"], expected_revision=SOURCE_REVISIONS["dreamsim"])
    cache = verify_tree(spec["cache"], source=False)
    dino_root = verify_tree(spec["dino_implementation"], expected_revision=SOURCE_REVISIONS["dino"])
    dino_checkpoint = verify_file(spec["dino_weights"])
    required = {"dino_vitb16_pretrain.pth", "open_clip_vitb16_pretrain.pth.tar", "clip_vitb16_pretrain.pth.tar", "ensemble_lora/adapter_config.json"}
    if not required.issubset(spec["cache"]["files"]):
        raise ValueError("DreamSim cache is missing registered ensemble assets")
    if not any(name in spec["cache"]["files"] for name in ("ensemble_lora/adapter_model.bin", "ensemble_lora/adapter_model.safetensors")):
        raise ValueError("DreamSim LoRA weights are not registered")
    # Include every executable/config/weight cache file; no unregistered override.
    present = {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file() and p.suffix in (".pth", ".tar", ".bin", ".safetensors", ".json", ".py") and p.relative_to(cache).as_posix() != "_archive_identity.json"}
    if present - set(spec["cache"]["files"]):
        raise ValueError("DreamSim cache contains unregistered model/config files")
    for name, mod in list(sys.modules.items()):
        if name == "dreamsim" or name.startswith("dreamsim."):
            origin = getattr(mod, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError("A different DreamSim implementation is already imported")
    sys.path.insert(0, str(root))
    old_hub_dir = torch.hub.get_dir()
    try:
        # DINO imports a top-level utils module. Scope the official one locally,
        # so the communication repository's unrelated utils module cannot leak in.
        with no_network():
            dino_utils = load_source(dino_root / "utils.py", "_registered_dreamsim_dino_utils")
            with mock.patch.dict(sys.modules, {"utils": dino_utils}):
                official_dino = load_source(dino_root / "vision_transformer.py", "_registered_dreamsim_dino")
        def local_dino_hub(repo, model, *args, **kwargs):
            if repo not in ("facebookresearch/dino:main", "facebookresearch/dino") or model != "dino_vitb16" or args:
                raise RuntimeError(f"Unregistered torch.hub request: {repo}/{model}")
            if kwargs.get("pretrained", True) is not True:
                raise RuntimeError("DreamSim DINO must load registered pretrained weights")
            backbone = official_dino.vit_base(patch_size=16, num_classes=0)
            backbone.load_state_dict(_state_dict(dino_checkpoint), strict=True)
            return backbone
        with no_network(), mock.patch.object(torch.hub, "load", local_dino_hub), cpu_default_checkpoint_loads():
            official = importlib.import_module("dreamsim")
            model, _unused_pil_transform = official.dreamsim(pretrained=True, device=str(device), cache_dir=str(cache), normalize_embeds=True, dreamsim_type="ensemble", use_patch_model=False)
    finally:
        torch.hub.set_dir(old_hub_dir)
        sys.path.remove(str(root))
    return _freeze(model, device)


def load_ms_ssim(spec, _device):
    root = verify_tree(spec["implementation"], expected_revision=SOURCE_REVISIONS["ms_ssim"])
    with no_network():
        return load_source(root / "pytorch_msssim" / "ssim.py", "_registered_msssim").ms_ssim


def validate_images(x, name="image"):
    import torch
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3 or x.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty RGB BCHW tensor")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating values")
    if x.amin().item() < 0 or x.amax().item() > 1:
        raise ValueError(f"{name} must be in [0, 1]; no hidden clipping or rescaling")
    return x.detach().float()


def image_digest(x):
    """Bind cached original features to exact floating pixel values and shape."""
    x = validate_images(x).contiguous().cpu()
    h = hashlib.sha256(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def preprocess_clip(x):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf
    x = tf.resize(x, 224, interpolation=InterpolationMode.BICUBIC, antialias=True)
    return tf.normalize(tf.center_crop(x, [224, 224]), CLIP_MEAN, CLIP_STD)


def preprocess_resnet50(x):
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)(x)


def preprocess_dreamsim(x):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tf
    # Unlike CLIP, official DreamSim resizes directly to a square, without crop.
    return tf.resize(x, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)


def preprocess_dinov2_vitl14(x):
    """Exact old S/14 geometry and normalization, without antialias or crop."""
    import torch.nn.functional as F
    resized = F.interpolate(x.float(), size=(224, 224), mode="bicubic", align_corners=False)
    mean = resized.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = resized.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return (resized - mean) / std


def dinov2_vitl14_features(model, images):
    import torch
    features = model.forward_features(preprocess_dinov2_vitl14(images))["x_norm_clstoken"].float()
    if features.shape != (len(images), 1024) or not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
        raise ValueError("DINOv2 ViT-L/14 must return finite nonzero 1024-d normalized CLS tokens")
    return features


def _labels(labels, batch, device):
    import torch
    result = torch.as_tensor(labels, device=device)
    if result.ndim == 0:
        result = result.unsqueeze(0)
    if result.shape != (batch,) or result.is_floating_point() or result.dtype == torch.bool:
        raise ValueError("Labels must be an integer vector, one ImageNet index per image")
    if result.min().item() < 0 or result.max().item() > 999:
        raise ValueError("Labels must use the registered ImageNet-1k index ordering (0..999)")
    return result.long()


class MetricEvaluator:
    """All requested mandatory assets are required; optional failures are explicit."""

    def __init__(self, manifest, device="cuda", requested=MANDATORY + OPTIONAL):
        import torch
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.models = {}
        self.metadata = {"format_version": 2, "input": "RGB BCHW float32 [0,1]; no additional 8-bit quantization", "inference": "eval; no_grad; float32; no AMP", "package_versions": package_versions(), "metrics": {}, "metric_selection_usage": "evaluation only; never policy selection or model training", "existing_dino": "Retain registered dino_cosine=DINOv2 ViT-S/14 unchanged; dinov2_vitl14_cosine is a separate added column"}
        self.metadata["numerical_backend"] = {"device": str(self.device), "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "float32_matmul_precision": torch.get_float32_matmul_precision()}
        if not set(requested).issubset(set(MANDATORY + OPTIONAL)):
            raise ValueError("Unregistered metric requested")
        loaders = {"clip": load_clip, "dists": load_dists, "resnet50": load_resnet50, "dinov2_vitl14": load_dinov2_vitl14, "dreamsim": load_dreamsim, "ms_ssim": load_ms_ssim}
        preprocess = {
            "clip": {"resize": 224, "crop": 224, "interpolation": "bicubic", "antialias": True, "mean": CLIP_MEAN, "std": CLIP_STD, "tensor_vs_PIL": "same geometry/constants, floating tensor interpolation; no claim of bitwise PIL equivalence"},
            "dists": {"resize": None, "input_size": [256, 256], "normalization": "official forward internal ImageNet mean/std", "batch_average": False},
            "resnet50": {"resize": 232, "crop": 224, "interpolation": "bilinear", "antialias": True, "normalization": "official torchvision IMAGENET1K_V2 transforms"},
            "dinov2_vitl14": {"resize": [224, 224], "crop": None, "interpolation": "torch.nn.functional.interpolate bicubic", "align_corners": False, "antialias": False, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "embedding": "forward_features()['x_norm_clstoken']", "dimensions": 1024, "num_register_tokens": 0, "matches_retained_S14_preprocessing": True},
            "dreamsim": {"resize": [224, 224], "crop": None, "interpolation": "bicubic", "antialias": True, "normalization": "official per-backbone internal normalization and normalize_embeds=True"},
            "ms_ssim": {"resize": None, "data_range": 1.0, "size_average": False, "win_size": 11, "win_sigma": 1.5, "weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], "K": [0.01, 0.03]},
        }
        for name in requested:
            if name not in manifest:
                if name in MANDATORY:
                    raise ValueError(f"Required metric assets are missing: {name}")
                self.metadata["metrics"][name] = {"status": "UNAVAILABLE_NOT_REGISTERED", "model_id": MODEL_IDS[name]}
                continue
            # An explicitly supplied but broken optional asset is an error, never a silent skip.
            self.models[name] = loaders[name](manifest[name], self.device)
            self.metadata["metrics"][name] = {"status": "READY", "model_id": MODEL_IDS[name], "preprocessing": preprocess[name], "assets": manifest[name]}
        self.metadata["backbone_relationships"] = "Communication encoder/VAR/Dc are separate from these evaluation backbones. DreamSim includes DINO and CLIP families; evaluation metrics are not mutually independent. D_C classification is label-conditioned and descriptive only."

    def prepare_reference(self, reference):
        import torch
        x = validate_images(reference, "reference").to(self.device)
        out = {"digest": image_digest(reference), "batch": len(x), "evaluator_identity": self.identity()}
        with torch.inference_mode():
            if "clip" in self.models:
                out["clip"] = self.models["clip"].encode_image(preprocess_clip(x)).float().detach().cpu()
            if "resnet50" in self.models:
                out["resnet50"] = self.models["resnet50"](preprocess_resnet50(x)).argmax(-1).detach().cpu()
            if "dreamsim" in self.models:
                out["dreamsim"] = self.models["dreamsim"].embed(preprocess_dreamsim(x)).float().detach().cpu()
            if "dinov2_vitl14" in self.models:
                out["dinov2_vitl14"] = dinov2_vitl14_features(self.models["dinov2_vitl14"], x).detach().cpu()
        return out

    def identity(self):
        """Cached originals cannot migrate across weights or preprocessing rules."""
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def expand_reference(self, reference_single, prepared_single, batch_count):
        """Repeat one validated source cache without running a source model again.

        Returns (reference_batch, prepared_batch) for score(). Each output owns
        its repeated storage, so an accidental caller edit cannot change the
        single-source baseline. DISTS keeps its unchanged official pair forward;
        this method repeats the feature caches used by the other neural metrics.
        """
        import torch
        if not isinstance(batch_count, int) or isinstance(batch_count, bool) or batch_count < 1:
            raise ValueError("batch_count must be a positive integer")
        source = validate_images(reference_single, "reference_single")
        if tuple(source.shape) != (1, 3, 256, 256):
            raise ValueError("Repeated references require one [1,3,256,256] source")
        evaluator_identity = self.identity()
        if (prepared_single.get("digest") != image_digest(source) or prepared_single.get("batch") != 1
                or prepared_single.get("evaluator_identity") != evaluator_identity):
            raise ValueError("Reference feature cache does not match this source and evaluator")
        feature_keys = {name for name in ("clip", "resnet50", "dreamsim", "dinov2_vitl14") if name in self.models}
        for name in feature_keys:
            value = prepared_single.get(name)
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid single-source feature cache: {name}")
            if name == "resnet50":
                if value.shape != (1,) or value.dtype != torch.int64 or value.min().item() < 0 or value.max().item() > 999:
                    raise ValueError("Invalid cached original-image classifier prediction")
            elif value.ndim != 2 or value.dtype != torch.float32:
                raise ValueError(f"Invalid single-source embedding: {name}")
        with torch.inference_mode():
            repeated_source = source.repeat(batch_count, 1, 1, 1)
            repeated = {"digest": image_digest(repeated_source), "batch": batch_count,
                "evaluator_identity": evaluator_identity}
            for name in feature_keys:
                value = prepared_single[name]
                repeated[name] = value.detach().repeat((batch_count,) + (1,) * (value.ndim - 1))
        return repeated_source, repeated

    def score(self, reference, reconstruction, labels, label_conditioned=False, prepared=None):
        import torch
        import torch.nn.functional as F
        x = validate_images(reference, "reference")
        y = validate_images(reconstruction, "reconstruction")
        if x.shape != y.shape or tuple(x.shape[-2:]) != (256, 256):
            raise ValueError("Registered paired inputs must have equal [B,3,256,256] shape")
        truth = _labels(labels, len(x), self.device)
        if not isinstance(label_conditioned, bool):
            raise ValueError("Use one explicit boolean label_conditioned flag for the batch")
        if prepared is None:
            prepared = self.prepare_reference(x)
        if prepared.get("digest") != image_digest(x) or prepared.get("batch") != len(x) or prepared.get("evaluator_identity") != self.identity():
            raise ValueError("Reference feature cache does not match these exact source pixels")
        x, y = x.to(self.device), y.to(self.device)
        cols = {}
        with torch.inference_mode():
            if "clip" in self.models:
                fy = self.models["clip"].encode_image(preprocess_clip(y)).float()
                cols["clip_image_cosine"] = F.cosine_similarity(prepared["clip"].to(self.device), fy, dim=-1)
            if "dists" in self.models:
                cols["dists"] = self.models["dists"](x, y, require_grad=False, batch_average=False).reshape(-1)
            if "dreamsim" in self.models:
                fy = self.models["dreamsim"].embed(preprocess_dreamsim(y)).float()
                # Official ensemble forward is exactly 1-cosine(embed(a),embed(b)).
                cols["dreamsim"] = 1 - F.cosine_similarity(prepared["dreamsim"].to(self.device), fy, dim=-1)
            if "dinov2_vitl14" in self.models:
                fy = dinov2_vitl14_features(self.models["dinov2_vitl14"], y)
                cols["dinov2_vitl14_cosine"] = F.cosine_similarity(prepared["dinov2_vitl14"].to(self.device), fy, dim=-1)
            if "ms_ssim" in self.models:
                cols["ms_ssim"] = self.models["ms_ssim"](x, y, data_range=1.0, size_average=False, win_size=11, win_sigma=1.5, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.03))
            if "resnet50" in self.models:
                pred = self.models["resnet50"](preprocess_resnet50(y)).argmax(-1)
                source_pred = prepared["resnet50"].to(self.device)
                cols.update(resnet50_prediction=pred, resnet50_source_prediction=source_pred, resnet50_top1_label=(pred == truth), resnet50_top1_source_prediction=(pred == source_pred), resnet50_source_top1_label=(source_pred == truth))
        for name, value in cols.items():
            if value.shape != (len(x),) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid metric output: {name}")
            cols[name] = value.detach().cpu().tolist()
        return [dict({name: value[i] for name, value in cols.items()}, label_conditioned=label_conditioned) for i in range(len(x))]


def official_asset_plan():
    """Download instructions only. This module never downloads or installs assets."""
    return {
        "clip": {"repository": "https://github.com/openai/CLIP", "implementation": "clip/model.py and full Python-source digest manifest", "weights_url": f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-L-14.pt", "expected_sha256": CLIP_SHA256, "approx_download_GB": 0.9},
        "dists": {"repository": "https://github.com/dingkeyan93/DISTS", "weights_url_template": "https://raw.githubusercontent.com/dingkeyan93/DISTS/{pinned_commit}/DISTS_pytorch/weights.pt", "vgg16_url": "https://download.pytorch.org/models/vgg16-397923af.pth", "approx_download_GB": 0.554},
        "resnet50": {"implementation": "installed torchvision version recorded in metadata", "weights_url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth", "approx_download_GB": 0.103},
        "dinov2_vitl14": {"repository": "https://github.com/facebookresearch/dinov2", "implementation": "Reuse the retained S/14 frozen official source tree and verified revision", "weights_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", "approx_download_GB": 1.2, "registers": False},
        "dreamsim": {"repository": "https://github.com/ssundaram21/dreamsim", "weights_url": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip", "dino_repository": "https://github.com/facebookresearch/dino", "dino_weights_url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth", "approx_download_GB": "about 1-2 GB including separate DINO backbone; verify Content-Length", "dependencies": ["torch", "torchvision", "numpy", "open-clip-torch", "peft", "Pillow", "timm", "scipy", "transformers"]},
        "ms_ssim": {"repository": "https://github.com/VainF/pytorch-msssim", "weights": None},
    }


if __name__ == "__main__":
    print(json.dumps(official_asset_plan(), indent=2))
