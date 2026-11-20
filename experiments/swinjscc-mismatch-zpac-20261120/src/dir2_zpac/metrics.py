"""Offline SHA256-verified visual quality evaluators (PSNR, MS-SSIM, LPIPS-Alex, DISTS) on A800 PFS."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from unittest import mock
import torch
import torch.nn.functional as F


RUNTIME_ROOT = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010")
SUPPLEMENT_ROOT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1"
)
METRIC_MODELS_PY = RUNTIME_ROOT / "assets/pilot_metric_code_seed_v1/metric_models.py"
SUPPLEMENT_MANIFEST = SUPPLEMENT_ROOT / "metric_assets_posix_manifest.json"
ALEXNET_OWT_PTH = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_swinjscc_mismatch_v1/cache/torch/alexnet-owt-7be5be79.pth"
)
LPIPS_ALEX_PTH = (
    RUNTIME_ROOT / "assets/frozen_namespace_v1/external/home/liulu/.local/lib/python3.10/site-packages/lpips/weights/v0.1/alex.pth"
)


def compute_psnr_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute per-image PSNR (dB) for x, y in [0, 1] of shape (B, 3, H, W)."""
    mse = (x.float() - y.float()).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


class OfflineVisualEvaluator:
    """Loads official PSNR, MS-SSIM, LPIPS-Alex, and DISTS completely offline from verified PFS assets."""

    def __init__(self, device: torch.device | str = "cuda:0", include_lpips: bool = True):
        self.device = torch.device(device)
        self.include_lpips = include_lpips

        from pytorch_msssim import ms_ssim

        self._ms_ssim_fn = ms_ssim

        spec = importlib.util.spec_from_file_location("_dir2_metric_models", str(METRIC_MODELS_PY))
        assert spec is not None and spec.loader is not None
        mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mm)
        manifest = json.loads(SUPPLEMENT_MANIFEST.read_text(encoding="utf-8"))
        with mm.no_network():
            self.dists_model = mm.load_dists(manifest["models"]["dists"], self.device)

        self.lpips_model = None
        if include_lpips:
            import lpips
            import torchvision.models as tv_models

            orig_alexnet = tv_models.alexnet

            def offline_alexnet(*_args, **_kwargs):
                net = orig_alexnet(weights=None)
                sd = torch.load(ALEXNET_OWT_PTH, map_location="cpu", weights_only=True)
                net.load_state_dict(sd)
                return net

            with mm.no_network(), mock.patch.object(tv_models, "alexnet", offline_alexnet):
                self.lpips_model = lpips.LPIPS(
                    net="alex",
                    version="0.1",
                    pretrained=True,
                    model_path=str(LPIPS_ALEX_PTH),
                    verbose=False,
                ).to(self.device).eval()
                self.lpips_model.requires_grad_(False)

    @torch.inference_mode()
    def evaluate_batch(self, target: torch.Tensor, recon: torch.Tensor) -> dict[str, list[float]]:
        """Evaluate a batch of RGB images in [0, 1], returning per-sample metric lists."""
        x = target.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        y = recon.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        psnr = compute_psnr_batch(x, y)
        msssim = self._ms_ssim_fn(
            x, y, data_range=1.0, size_average=False, win_size=11, win_sigma=1.5, K=(0.01, 0.03)
        )
        dists_val = self.dists_model(x, y, require_grad=False, batch_average=False).reshape(-1)

        out: dict[str, list[float]] = {
            "psnr_db": [float(v) for v in psnr.cpu().tolist()],
            "ms_ssim": [float(v) for v in msssim.cpu().tolist()],
            "dists": [float(v) for v in dists_val.cpu().tolist()],
        }
        if self.lpips_model is not None:
            x_norm = x * 2.0 - 1.0
            y_norm = y * 2.0 - 1.0
            lp = self.lpips_model(x_norm, y_norm).reshape(-1)
            out["lpips_alex"] = [float(v) for v in lp.cpu().tolist()]
        for k, vals in out.items():
            assert all(math.isfinite(v) for v in vals), f"Non-finite {k}: {vals}"
        return out
