"""SwinJSCC (c_T, Gamma, c_R) split operator, baselines (B0, B1, B2, Avg-SNR), ablations, and ZPAC."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .phy_channel import ChannelOutput, awgn_forward, fading_equalized_forward, FadingTrajectoryStep


DEFAULT_VENDOR_ROOT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/vendor/SwinJSCC"
)

SNR_CENTER = 7.0
SNR_SCALE = 6.0


def load_swinjscc_backbone(
    checkpoint_path: Path | str,
    *,
    model_mode: str = "SwinJSCC_w/_SA",
    c_bottleneck: int = 96,
    model_size: str = "base",
    vendor_root: Path | str = DEFAULT_VENDOR_ROOT,
    device: torch.device | str = "cuda:0",
) -> nn.Module:
    """Instantiate official SwinJSCC backbone and load pretrained checkpoint with strict verification."""
    vroot = Path(vendor_root).resolve()
    if str(vroot) not in sys.path:
        sys.path.insert(0, str(vroot))
    from net.network import SwinJSCC

    depths_map = {
        "small": ([2, 2, 2, 2], [2, 2, 2, 2]),
        "base": ([2, 2, 6, 2], [2, 6, 2, 2]),
        "large": ([2, 2, 18, 2], [2, 18, 2, 2]),
    }
    enc_depths, dec_depths = depths_map[model_size]
    chan_num = int(c_bottleneck) if model_mode in ("SwinJSCC_w/o_SAandRA", "SwinJSCC_w/_SA") else None

    common = dict(
        model=model_mode,
        img_size=(256, 256),
        C=chan_num,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    encoder_kwargs = dict(
        common,
        patch_size=2,
        in_chans=3,
        embed_dims=[128, 192, 256, 320],
        depths=enc_depths,
        num_heads=[4, 6, 8, 10],
    )
    decoder_kwargs = dict(
        common,
        embed_dims=[320, 256, 192, 128],
        depths=dec_depths,
        num_heads=[10, 8, 6, 4],
    )
    args = SimpleNamespace(
        model=model_mode,
        channel_type="awgn",
        multiple_snr="1,4,7,10,13",
        C=str(c_bottleneck),
        distortion_metric="MSE",
    )
    config = SimpleNamespace(
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        logger=None,
        device=torch.device(device),
        pass_channel=True,
        downsample=4,
        norm=False,
        CUDA=torch.cuda.is_available(),
    )
    net = SwinJSCC(args, config)
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if all(k.startswith("native.") for k in sd.keys()):
        sd = {k[len("native."):]: v for k, v in sd.items()}

    net.encoder.update_resolution(256, 256)
    net.decoder.update_resolution(16, 16)
    net.H = net.W = 256
    net.load_state_dict(sd, strict=True)
    net.encoder.update_resolution(256, 256)
    net.decoder.update_resolution(16, 16)
    net.H = net.W = 256
    return net.to(device).eval()


def _as_batch_col(val: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(val):
        return torch.full((batch_size, 1), float(val), device=device, dtype=dtype)
    t = val.to(device=device, dtype=dtype)
    if t.ndim == 0:
        return t.reshape(1, 1).expand(batch_size, 1)
    return t.reshape(batch_size, 1)


def encode_with_ct(
    backbone: nn.Module,
    image: torch.Tensor,
    c_t_db: torch.Tensor | float,
    rate_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batch-safe SwinJSCC encoder forward conditioned on per-sample c_T (dB)."""
    enc = backbone.encoder
    model_mode = backbone.model
    B = image.shape[0]
    device = image.device
    dtype = image.dtype

    x = enc.patch_embed(image)
    for layer in enc.layers:
        x = layer(x)
    x = enc.norm(x)
    L = x.shape[1]

    if model_mode == "SwinJSCC_w/o_SAandRA":
        return enc.head_list(x), None

    if model_mode == "SwinJSCC_w/_SA":
        snr_batch = _as_batch_col(c_t_db, B, device, dtype)
        temp = None
        for i in range(enc.layer_num):
            temp = enc.sm_list[i](x.detach()) if i == 0 else enc.sm_list[i](temp)
            bm = enc.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val = enc.sigmoid(enc.sm_list[-1](temp))
        x = x * mod_val
        x = enc.head_list(x)
        return x, None

    if model_mode == "SwinJSCC_w/_SAandRA":
        assert rate_channels is not None and 1 <= int(rate_channels) <= x.shape[2]
        rate = int(rate_channels)
        snr_batch = _as_batch_col(c_t_db, B, device, dtype)
        rate_batch = _as_batch_col(float(rate), B, device, dtype)

        temp = None
        for i in range(enc.layer_num):
            temp = enc.sm_list1[i](x.detach()) if i == 0 else enc.sm_list1[i](temp)
            bm = enc.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val1 = enc.sigmoid1(enc.sm_list1[-1](temp))
        x = x * mod_val1

        temp = None
        for i in range(enc.layer_num):
            temp = enc.sm_list[i](x.detach()) if i == 0 else enc.sm_list[i](temp)
            bm = enc.bm_list[i](rate_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val = enc.sigmoid(enc.sm_list[-1](temp))
        x = x * mod_val
        mask_scores = torch.sum(mod_val, dim=1)
        _, indices = mask_scores.sort(dim=1, descending=True)
        c_indices = indices[:, :rate]
        mask = torch.zeros_like(mask_scores)
        mask.scatter_(1, c_indices, 1.0)
        mask_3d = mask.unsqueeze(1).expand(-1, L, -1)
        x = x * mask_3d
        return x, mask_3d

    raise ValueError(f"Unsupported model_mode: {model_mode}")


class DualConditionLinear(nn.Module):
    """Replaces Linear(1, M) in AdaptiveModulator with Linear(2, M) taking [c_T, c_R], zero-init on c_T."""

    def __init__(self, orig_linear: nn.Linear):
        super().__init__()
        assert orig_linear.in_features == 1
        M = orig_linear.out_features
        self.linear = nn.Linear(2, M, bias=orig_linear.bias is not None)
        with torch.no_grad():
            self.linear.weight[:, 0].zero_()
            self.linear.weight[:, 1].copy_(orig_linear.weight[:, 0])
            if orig_linear.bias is not None:
                self.linear.bias.copy_(orig_linear.bias)

    def forward(self, cond_2d: torch.Tensor) -> torch.Tensor:
        return self.linear(cond_2d)


class ZeroLockedModNetDelta(nn.Module):
    """Zero-locked asymmetric rank-1 residual delta on decoder ChannelModNet gates (for P_zpac_hier)."""

    def __init__(self, num_layers: int, hidden_dim: int, out_dim: int, rank_hidden: int = 32):
        super().__init__()
        self.num_layers = num_layers
        self.gamma_pos_raw = nn.Parameter(torch.tensor(1.2))
        self.gamma_neg_raw = nn.Parameter(torch.tensor(1.2))
        self.pos_mlps = nn.ModuleList()
        self.neg_mlps = nn.ModuleList()
        for _ in range(num_layers):
            mp = nn.Sequential(nn.Linear(2, rank_hidden), nn.GELU(), nn.Linear(rank_hidden, hidden_dim))
            mn = nn.Sequential(nn.Linear(2, rank_hidden), nn.GELU(), nn.Linear(rank_hidden, hidden_dim))
            nn.init.zeros_(mp[-1].weight)
            nn.init.zeros_(mp[-1].bias)
            nn.init.zeros_(mn[-1].weight)
            nn.init.zeros_(mn[-1].bias)
            self.pos_mlps.append(mp)
            self.neg_mlps.append(mn)
        # Direct post-head_list zero-locked channel affine delta (bypasses small random sm_list weights!)
        self.post_pos = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 2 * out_dim))
        self.post_neg = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 2 * out_dim))
        nn.init.zeros_(self.post_pos[-1].weight)
        nn.init.zeros_(self.post_pos[-1].bias)
        nn.init.zeros_(self.post_neg[-1].weight)
        nn.init.zeros_(self.post_neg[-1].bias)

    def delta_for_layer(self, layer_idx: int, cr_norm: torch.Tensor, dc_pos: torch.Tensor, dc_neg: torch.Tensor) -> torch.Tensor:
        gp = torch.tanh(F.softplus(self.gamma_pos_raw) * dc_pos)
        gn = torch.tanh(F.softplus(self.gamma_neg_raw) * dc_neg)
        inp_p = torch.cat([cr_norm, dc_pos], dim=-1)
        inp_n = torch.cat([cr_norm, dc_neg], dim=-1)
        return gp * self.pos_mlps[layer_idx](inp_p) + gn * self.neg_mlps[layer_idx](inp_n)

    def modulate_post_head(self, x: torch.Tensor, cr_norm: torch.Tensor, dc_pos: torch.Tensor, dc_neg: torch.Tensor) -> torch.Tensor:
        gp = torch.tanh(F.softplus(self.gamma_pos_raw) * dc_pos).unsqueeze(1)  # (B, 1, 1)
        gn = torch.tanh(F.softplus(self.gamma_neg_raw) * dc_neg).unsqueeze(1)
        ap, bp = self.post_pos(torch.cat([cr_norm, dc_pos], dim=-1)).chunk(2, dim=-1)
        an, bn = self.post_neg(torch.cat([cr_norm, dc_neg], dim=-1)).chunk(2, dim=-1)
        scale = gp * torch.tanh(ap).unsqueeze(1) + gn * torch.tanh(an).unsqueeze(1)
        shift = gp * bp.unsqueeze(1) + gn * bn.unsqueeze(1)
        return x * (1.0 + scale) + shift


class FiLMStage(nn.Module):
    """Single residual spatial-channel stage inside a calibration branch."""

    def __init__(self, hidden_ch: int, cond_dim: int, *, pointwise_only: bool = False, use_se: bool = False):
        super().__init__()
        num_groups = 8 if hidden_ch % 8 == 0 else 4
        self.norm1 = nn.GroupNorm(num_groups, hidden_ch)
        if pointwise_only:
            self.spatial = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1)
        else:
            self.spatial = nn.Conv2d(
                hidden_ch, hidden_ch, kernel_size=3, padding=1, padding_mode="reflect", groups=hidden_ch
            )
        self.pw = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1)
        self.norm2 = nn.GroupNorm(num_groups, hidden_ch)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_ch),
            nn.GELU(),
            nn.Linear(hidden_ch, 2 * hidden_ch),
        )
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)
        self.use_se = use_se
        if use_se:
            se_mid = max(16, hidden_ch // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_ch, se_mid, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(se_mid, hidden_ch, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.spatial(F.gelu(self.norm1(h)))
        r = self.norm2(self.pw(F.gelu(r)))
        alpha, beta = self.cond_mlp(cond).chunk(2, dim=-1)
        r = r * (1.0 + alpha[:, :, None, None]) + beta[:, :, None, None]
        if self.use_se:
            r = r * (2.0 * self.se(F.gelu(r)))
        return h + r


class FiLMResStack(nn.Module):
    """2-stage residual spatial-channel calibration network with zero-initialized output projection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
        cond_dim: int,
        *,
        num_stages: int = 2,
        pointwise_only: bool = False,
        use_se: bool = False,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        self.stages = nn.ModuleList(
            [
                FiLMStage(hidden_ch, cond_dim, pointwise_only=pointwise_only, use_se=(use_se and i == num_stages - 1))
                for i in range(num_stages)
            ]
        )
        self.proj_out = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(feat)
        for stg in self.stages:
            h = stg(h, cond)
        return self.proj_out(F.gelu(h))


class ReceiverFrontendCalibrator(nn.Module):
    """Unified receiver latent calibrator supporting ZPAC, ablations, B0, B2, P_raw, and P_pointwise."""

    def __init__(
        self,
        channels: int,
        variant: str,
        *,
        hidden_ch: int = 128,
        grid_hw: tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.channels = channels
        self.variant = variant
        self.grid_hw = grid_hw

        valid_variants = {
            "B0",
            "B2_front",
            "P_raw",
            "P_nogate",
            "P_sym",
            "P_noxi",
            "P_pointwise",
            "P_zpac",
        }
        if variant not in valid_variants:
            raise ValueError(f"Unknown frontend variant: {variant}")

        self.pointwise_only = variant == "P_pointwise"
        self.use_zero_gate = variant in {"P_zpac", "P_sym", "P_noxi", "P_pointwise"}
        self.asymmetric = variant in {"P_zpac", "P_nogate", "P_noxi", "P_pointwise"}
        self.use_xi = variant in {"P_zpac", "P_nogate", "P_sym", "P_pointwise"}

        if self.use_zero_gate:
            self.gamma_pos_raw = nn.Parameter(torch.tensor(1.2))
            self.gamma_neg_raw = nn.Parameter(torch.tensor(1.2))

        if self.asymmetric:
            # Overestimation branch R+ (with joint spatial-channel Wiener prior) and Underestimation branch R-
            in_pos = (4 * channels + 2) if self.use_xi else (3 * channels)
            in_neg = 2 * channels
            self.branch_pos = FiLMResStack(
                in_pos, channels, hidden_ch, cond_dim=2, num_stages=2, pointwise_only=self.pointwise_only, use_se=False
            )
            self.branch_neg = FiLMResStack(
                in_neg, channels, hidden_ch, cond_dim=2, num_stages=2, pointwise_only=self.pointwise_only, use_se=True
            )
            if self.use_xi:
                self.wiener_shrink_mlp = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.GELU(),
                    nn.Linear(32, channels),
                )
                nn.init.zeros_(self.wiener_shrink_mlp[-1].weight)
                nn.init.zeros_(self.wiener_shrink_mlp[-1].bias)
        else:
            # Parameter-matched 2-stage single-branch network (~1.42x width matches 2x branch parameter count!)
            sym_hidden = max(16, (int(round(hidden_ch * 1.42)) // 8) * 8)
            in_single = (4 * channels + 2) if self.use_xi else (2 * channels)
            self.branch_single = FiLMResStack(
                in_single, channels, sym_hidden, cond_dim=3, num_stages=2, pointwise_only=self.pointwise_only, use_se=True
            )

    def _decompose_subbands(self, u_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pointwise_only:
            u_low = u_2d.mean(dim=1, keepdim=True).expand_as(u_2d)
            return u_low, u_2d - u_low
        u_padded = F.pad(u_2d, (1, 1, 1, 1), mode="reflect")
        u_low = F.avg_pool2d(u_padded, kernel_size=3, stride=1)
        u_high = u_2d - u_low
        return u_low, u_high

    def _posterior_wiener_priors(
        self,
        u_2d: torch.Tensor,
        u_low: torch.Tensor,
        u_high: torch.Tensor,
        u_norm_2d: torch.Tensor,
        sigma0_sq_cr: torch.Tensor,
        active_mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scale-free spatial SNR map Xi, spatial Wiener gain W_sp, and channel Wiener prior U_wiener."""
        if active_mask_2d is not None:
            denom = active_mask_2d.sum(dim=1, keepdim=True).clamp_min(1.0)
            e_sp = (u_norm_2d * active_mask_2d).square().sum(dim=1, keepdim=True) / denom
        else:
            e_sp = u_norm_2d.square().mean(dim=1, keepdim=True)  # (B, 1, H, W)

        e_ch = u_norm_2d.square().mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        s_sq = sigma0_sq_cr[:, None, None, None].clamp_min(1e-8)

        xi = torch.log1p(e_sp / s_sq)
        w_sp = F.relu(e_sp - s_sq) / (e_sp + 1e-6)
        w_ch = F.relu(e_ch - s_sq) / (e_ch + 1e-6)

        if self.pointwise_only:
            # Pointwise ablation cannot use spatial average e_ch across HxW
            u_wiener = u_2d * w_sp
        else:
            u_wiener = u_low * w_ch + u_high * (w_sp * w_ch)
        return xi, w_sp, u_wiener

    def forward(
        self,
        u_restored: torch.Tensor,
        u_norm: torch.Tensor,
        c_t_db: torch.Tensor,
        c_r_db: torch.Tensor,
        sigma0_sq_cr: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C = u_restored.shape
        H, W = self.grid_hw
        assert L == H * W and C == self.channels

        cr_col = _as_batch_col(c_r_db, B, u_restored.device, u_restored.dtype)
        ct_col = _as_batch_col(c_t_db, B, u_restored.device, u_restored.dtype)

        cr_norm = (cr_col - SNR_CENTER) / SNR_SCALE
        ct_norm = (ct_col - SNR_CENTER) / SNR_SCALE
        dc = (ct_col - cr_col) / SNR_SCALE
        dc_pos = F.relu(dc)
        dc_neg = F.relu(-dc)

        u_2d = u_restored.transpose(1, 2).reshape(B, C, H, W)
        un_2d = u_norm.transpose(1, 2).reshape(B, C, H, W)
        mask_2d = active_mask.transpose(1, 2).reshape(B, C, H, W) if active_mask is not None else None

        u_low, u_high = self._decompose_subbands(u_2d)

        aux: dict[str, torch.Tensor] = {}
        if self.asymmetric:
            cond_p = torch.cat([cr_norm, dc_pos], dim=-1)
            cond_n = torch.cat([cr_norm, dc_neg], dim=-1)
            if self.use_xi:
                xi, w_sp, u_wiener = self._posterior_wiener_priors(u_2d, u_low, u_high, un_2d, sigma0_sq_cr, mask_2d)
                feat_p = torch.cat([u_2d, u_low, u_high, u_wiener, xi, w_sp], dim=1)
                shrink_gain = F.relu(torch.tanh(self.wiener_shrink_mlp(cond_p)))[:, :, None, None]
                wiener_delta = shrink_gain * (u_wiener - u_2d)
                aux["xi_mean"] = xi.mean(dim=(1, 2, 3)).detach()
                aux["wiener_mean"] = w_sp.mean(dim=(1, 2, 3)).detach()
            else:
                feat_p = torch.cat([u_2d, u_low, u_high], dim=1)
                wiener_delta = 0.0

            r_pos = self.branch_pos(feat_p, cond_p) + wiener_delta
            feat_n = torch.cat([u_2d, u_high], dim=1)
            r_neg = self.branch_neg(feat_n, cond_n)

            if self.use_zero_gate:
                gp = torch.tanh(F.softplus(self.gamma_pos_raw) * dc_pos)[:, :, None, None]
                gn = torch.tanh(F.softplus(self.gamma_neg_raw) * dc_neg)[:, :, None, None]
            else:
                gp = torch.ones((B, 1, 1, 1), device=u_2d.device, dtype=u_2d.dtype)
                gn = torch.ones((B, 1, 1, 1), device=u_2d.device, dtype=u_2d.dtype)

            delta_2d = gp * r_pos + gn * r_neg
            aux["delta_pos_rms"] = (gp * r_pos).square().mean(dim=(1, 2, 3)).sqrt().detach()
            aux["delta_neg_rms"] = (gn * r_neg).square().mean(dim=(1, 2, 3)).sqrt().detach()
        else:
            if self.variant == "B0":
                cond_3 = torch.cat([cr_norm, torch.zeros_like(cr_norm), torch.zeros_like(cr_norm)], dim=-1)
                feat_s = torch.cat([u_2d, u_high], dim=1)
                delta_2d = self.branch_single(feat_s, cond_3)
            elif self.variant in {"P_raw", "B2_front"}:
                cond_3 = torch.cat([cr_norm, ct_norm, torch.zeros_like(cr_norm)], dim=-1)
                feat_s = torch.cat([u_2d, u_high], dim=1)
                delta_2d = self.branch_single(feat_s, cond_3)
            elif self.variant == "P_sym":
                cond_3 = torch.cat([cr_norm, dc_pos, dc_neg], dim=-1)
                xi, w_sp, u_wiener = self._posterior_wiener_priors(u_2d, u_low, u_high, un_2d, sigma0_sq_cr, mask_2d)
                feat_s = torch.cat([u_2d, u_low, u_high, u_wiener, xi, w_sp], dim=1)
                raw_s = self.branch_single(feat_s, cond_3)
                g_abs = torch.tanh(F.softplus(self.gamma_pos_raw) * (dc_pos + dc_neg))[:, :, None, None]
                delta_2d = g_abs * raw_s
            else:
                raise RuntimeError(f"Unhandled single-branch variant {self.variant}")

        u_cal_2d = u_2d + delta_2d
        u_cal = u_cal_2d.reshape(B, C, L).transpose(1, 2).contiguous()
        if active_mask is not None:
            u_cal = u_cal * active_mask
        return u_cal, aux


class CalibratedSwinJSCC(nn.Module):
    """Complete transceiver model wrapping SwinJSCC backbone + method-specific RX calibration."""

    METHODS = (
        "frozen",
        "avg_snr",
        "B0",
        "B1",
        "B2",
        "P_raw",
        "P_nogate",
        "P_sym",
        "P_noxi",
        "P_pointwise",
        "P_zpac",
        "P_zpac_hier",
    )

    def __init__(
        self,
        backbone: nn.Module,
        method: str = "P_zpac",
        *,
        hidden_ch: int = 128,
        avg_snr_fallback_db: float = 7.0,
    ):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}', expected one of {self.METHODS}")
        self.backbone = copy.deepcopy(backbone)
        self.method = method
        self.avg_snr_fallback_db = float(avg_snr_fallback_db)

        model_mode = self.backbone.model
        if model_mode == "SwinJSCC_w/_SA":
            lat_ch = self.backbone.decoder.head_list.in_features
        elif model_mode == "SwinJSCC_w/_SAandRA":
            lat_ch = self.backbone.decoder.embed_dims[0]
        else:
            raise ValueError(f"Unsupported backbone mode for calibration: {model_mode}")
        self.latent_channels = int(lat_ch)

        self.dual_decoder_bm = method in {"B1", "B2"}
        if self.dual_decoder_bm:
            for bm in self.backbone.decoder.bm_list:
                bm.fc[0] = DualConditionLinear(bm.fc[0])

        self.hier_delta: ZeroLockedModNetDelta | None = None
        if method == "P_zpac_hier":
            self.hier_delta = ZeroLockedModNetDelta(
                num_layers=self.backbone.decoder.layer_num,
                hidden_dim=self.backbone.decoder.hidden_dim,
                out_dim=self.backbone.decoder.embed_dims[0],
                rank_hidden=32,
            )

        frontend_map = {
            "frozen": None,
            "avg_snr": None,
            "B0": "B0",
            "B1": None,
            "B2": "B2_front",
            "P_raw": "P_raw",
            "P_nogate": "P_nogate",
            "P_sym": "P_sym",
            "P_noxi": "P_noxi",
            "P_pointwise": "P_pointwise",
            "P_zpac": "P_zpac",
            "P_zpac_hier": "P_zpac",
        }
        f_var = frontend_map[method]
        self.frontend = (
            ReceiverFrontendCalibrator(self.latent_channels, f_var, hidden_ch=hidden_ch)
            if f_var is not None
            else None
        )

    def configure_trainable_parameters(
        self, *, open_decoder_bm: bool = False, open_decoder_sm: bool = False
    ) -> list[str]:
        """Freeze encoder & Swin blocks completely; enable RX calibrator (+ decoder ModNet if requested)."""
        self.backbone.requires_grad_(False)
        if self.frontend is not None:
            self.frontend.requires_grad_(True)
        if self.hier_delta is not None:
            self.hier_delta.requires_grad_(True)
        if self.dual_decoder_bm or open_decoder_bm:
            self.backbone.decoder.bm_list.requires_grad_(True)
        if open_decoder_sm:
            self.backbone.decoder.sm_list.requires_grad_(True)

        return [n for n, p in self.named_parameters() if p.requires_grad]

    def decode_latent(
        self,
        u_in: torch.Tensor,
        c_t_db: torch.Tensor | float,
        c_r_db: torch.Tensor | float,
    ) -> torch.Tensor:
        dec = self.backbone.decoder
        model_mode = self.backbone.model
        B, L, _ = u_in.shape
        device = u_in.device
        dtype = u_in.dtype

        x = dec.head_list(u_in) if model_mode == "SwinJSCC_w/_SA" else u_in

        cr_col = _as_batch_col(c_r_db, B, device, dtype)
        ct_col = _as_batch_col(c_t_db, B, device, dtype)

        if self.dual_decoder_bm:
            cond_bm = torch.cat([ct_col, cr_col], dim=-1)
        else:
            cond_bm = cr_col

        cr_norm = (cr_col - SNR_CENTER) / SNR_SCALE
        dc = (ct_col - cr_col) / SNR_SCALE
        dc_pos = F.relu(dc)
        dc_neg = F.relu(-dc)

        temp = None
        for i in range(dec.layer_num):
            temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
            bm_val = dec.bm_list[i](cond_bm)
            if self.hier_delta is not None:
                bm_val = bm_val + self.hier_delta.delta_for_layer(i, cr_norm, dc_pos, dc_neg)
            bm = bm_val.unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm

        mod_val = dec.sigmoid(dec.sm_list[-1](temp))
        x = x * mod_val
        if self.hier_delta is not None:
            x = self.hier_delta.modulate_post_head(x, cr_norm, dc_pos, dc_neg)
        for layer in dec.layers:
            x = layer(x)
        x = x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_awgn(
        self,
        image: torch.Tensor,
        c_t_db: torch.Tensor | float,
        gamma_db: torch.Tensor | float,
        c_r_db: torch.Tensor | float,
        *,
        standard_noise: torch.Tensor | None = None,
        rate_channels: int | None = None,
        return_latents: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = image.shape[0]
        device = image.device
        dtype = image.dtype

        eff_ct = self.avg_snr_fallback_db if self.method == "avg_snr" else c_t_db
        ct_vec = _as_batch_col(eff_ct, B, device, dtype).squeeze(1)
        cr_vec = _as_batch_col(c_r_db, B, device, dtype).squeeze(1)
        gm_vec = _as_batch_col(gamma_db, B, device, dtype).squeeze(1)

        with torch.no_grad():
            z_raw, active_mask = encode_with_ct(self.backbone, image, ct_vec, rate_channels=rate_channels)
            ch_out: ChannelOutput = awgn_forward(
                z_raw, gm_vec, cr_vec, standard_noise=standard_noise, active_mask=active_mask
            )

        if self.frontend is not None:
            u_cal, aux = self.frontend(
                ch_out.u_restored,
                ch_out.u_norm,
                ct_vec,
                cr_vec,
                ch_out.sigma0_sq_cr,
                active_mask=active_mask,
            )
        else:
            u_cal = ch_out.u_restored
            aux = {}

        recon = self.decode_latent(u_cal, ct_vec, cr_vec)
        out: dict[str, torch.Tensor] = {"recon": recon, **aux}
        if return_latents:
            out["z_raw"] = z_raw
            out["z_norm"] = ch_out.z_norm
            out["u_restored"] = ch_out.u_restored
            out["u_cal"] = u_cal
            if active_mask is not None:
                out["active_mask"] = active_mask
        return out

    def forward_fading(
        self,
        image: torch.Tensor,
        fading_steps: list[FadingTrajectoryStep],
        *,
        standard_noise: torch.Tensor | None = None,
        rate_channels: int | None = None,
    ) -> dict[str, torch.Tensor]:
        B = image.shape[0]
        assert len(fading_steps) == B
        device = image.device
        dtype = image.dtype

        if self.method == "avg_snr":
            ct_vec = torch.full((B,), self.avg_snr_fallback_db, device=device, dtype=dtype)
        else:
            ct_vec = torch.tensor([s.c_t_db for s in fading_steps], device=device, dtype=dtype)
        cr_vec = torch.tensor([s.c_r_db for s in fading_steps], device=device, dtype=dtype)

        with torch.no_grad():
            z_raw, active_mask = encode_with_ct(self.backbone, image, ct_vec, rate_channels=rate_channels)
            ch_out = fading_equalized_forward(
                z_raw, fading_steps, standard_noise=standard_noise, active_mask=active_mask
            )

        if self.frontend is not None:
            u_cal, aux = self.frontend(
                ch_out.u_restored,
                ch_out.u_norm,
                ct_vec,
                cr_vec,
                ch_out.sigma0_sq_cr,
                active_mask=active_mask,
            )
        else:
            u_cal = ch_out.u_restored
            aux = {}

        recon = self.decode_latent(u_cal, ct_vec, cr_vec)
        return {"recon": recon, "c_t_db": ct_vec, "c_r_db": cr_vec, "gamma_db": ch_out.gamma_db, **aux}
