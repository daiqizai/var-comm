"""V2 Multi-Scale CSI-Physics-Guided & Closed-Loop Uncertainty-Aware Transceiver Calibration (MS-ZPAC & TR-ZPAC)."""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    DualConditionLinear,
    SNR_CENTER,
    SNR_SCALE,
    _as_batch_col,
)
from .phy_v2 import (
    ChannelV2Output,
    OFDMFadingStep,
    awgn_v2_forward,
    ofdm_fading_forward,
    rayleigh_v2_forward,
)


def _small_init(layer: nn.Module, std: float = 0.01) -> None:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.normal_(layer.weight, mean=0.0, std=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SpatialFiLMResBlock(nn.Module):
    """Depthwise-separable 3x3 ConvNeXt-style residual block with bottleneck FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(num_groups=min(16, channels), num_channels=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, channels * 2),
        )
        _small_init(self.film[-1], std=0.01)
        _small_init(self.pw2, std=0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.dw(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.pw2(self.act(self.pw1(h)))
        return x + h


class IQConjugatePairBlock(nn.Module):
    """Pairs rows 0..7 (Re) and rows 8..15 (Im) of SwinJSCC's (16, 16, C) latent to cancel I/Q phase/fading coupling."""

    def __init__(self, latent_ch: int, hidden_ch: int = 128):
        super().__init__()
        in_ch = 2 * latent_ch + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=min(16, hidden_ch)),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 2 * latent_ch, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        u_restored: torch.Tensor,
        u_norm: torch.Tensor,
        alpha_ratio_iq: torch.Tensor,
        sigma_post_iq: torch.Tensor,
        phase_unc_iq: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, C = u_restored.shape
        u_sp = u_restored.reshape(B, 16, 16, C)
        u_re = u_sp[:, :8, :, :].permute(0, 3, 1, 2)
        u_im = u_sp[:, 8:, :, :].permute(0, 3, 1, 2)

        un_sp = u_norm.reshape(B, 16, 16, C)
        mag_sq = un_sp[:, :8, :, :].square() + un_sp[:, 8:, :, :].square()

        sig_sp = sigma_post_iq.reshape(B, 8, 16, C)
        alp_sp = alpha_ratio_iq.reshape(B, 8, 16, C)
        phs_sp = phase_unc_iq.reshape(B, 8, 16, C)

        if active_mask is not None:
            m_sp = active_mask.reshape(B, 16, 16, C)[:, :8, :, :]
            denom = m_sp.sum(dim=-1, keepdim=True).clamp_min(1.0)
            post_snr_map = torch.log1p(((mag_sq * m_sp) / (2.0 * sig_sp + 1e-6)).sum(dim=-1, keepdim=True) / denom)
        else:
            post_snr_map = torch.log1p((mag_sq / (2.0 * sig_sp + 1e-6)).mean(dim=-1, keepdim=True))
        log_sig_map = torch.log(sig_sp.clamp_min(1e-6)).mean(dim=-1, keepdim=True).div(5.0)
        alp_dev_map = (alp_sp - 1.0).mean(dim=-1, keepdim=True)
        phs_map = phs_sp.mean(dim=-1, keepdim=True)

        phy_4 = torch.cat([post_snr_map, log_sig_map, alp_dev_map, phs_map], dim=-1).permute(0, 3, 1, 2)
        delta = self.net(torch.cat([u_re, u_im, phy_4], dim=1))
        d_re, d_im = delta[:, :C], delta[:, C:]
        return torch.cat([d_re, d_im], dim=2).permute(0, 2, 3, 1).reshape(B, L, C)


class PhysicsFrontendV2(nn.Module):
    """CSI-sufficient I/Q-paired frequency-decoupled asymmetric bottleneck calibrator."""

    def __init__(
        self,
        latent_ch: int,
        hidden_ch: int = 144,
        *,
        mode: str = "ms_zpac",
        enable_csi_gate: bool = True,
        enable_iq_pair: bool = True,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.mode = mode
        self.enable_csi_gate = enable_csi_gate
        self.enable_iq_pair = enable_iq_pair and (mode != "p_raw_v2")

        self.gamma_pos = nn.Parameter(torch.tensor(1.5))
        self.gamma_neg = nn.Parameter(torch.tensor(1.5))
        self.gamma_csi = nn.Parameter(torch.tensor(2.0))

        cond_dim = 6

        if mode == "p_raw_v2":
            self.raw_in = nn.Conv2d(latent_ch + 3, hidden_ch, kernel_size=3, padding=1)
            self.raw_b1 = SpatialFiLMResBlock(hidden_ch, 3)
            self.raw_b2 = SpatialFiLMResBlock(hidden_ch, 3)
            self.raw_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=3, padding=1)
            nn.init.zeros_(self.raw_out.weight)
            nn.init.zeros_(self.raw_out.bias)
            return

        if self.enable_iq_pair:
            self.iq_pair = IQConjugatePairBlock(latent_ch, hidden_ch=min(128, max(96, latent_ch)))
        else:
            self.iq_pair = None

        self.wiener_temp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_ch),
        )
        _small_init(self.wiener_temp[-1], std=0.01)

        self.pos_in = nn.Conv2d(3 * latent_ch + 3, hidden_ch, kernel_size=1)
        self.pos_b1 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.pos_b2 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.pos_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.pos_out.weight)
        nn.init.zeros_(self.pos_out.bias)

        self.neg_se = nn.Sequential(
            nn.Linear(latent_ch + cond_dim, hidden_ch),
            nn.GELU(),
            nn.Linear(hidden_ch, latent_ch),
            nn.Tanh(),
        )
        _small_init(self.neg_se[-2], std=0.01)

        self.neg_in = nn.Conv2d(2 * latent_ch + 3, hidden_ch, kernel_size=1)
        self.neg_b1 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.neg_b2 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.neg_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.neg_out.weight)
        nn.init.zeros_(self.neg_out.bias)

        self.csi_in = nn.Conv2d(3 * latent_ch + 3, hidden_ch, kernel_size=1)
        self.csi_b1 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.csi_b2 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.csi_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.csi_out.weight)
        nn.init.zeros_(self.csi_out.bias)

    def build_spatial_phy_map(
        self,
        ch_out: ChannelV2Output,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, C = ch_out.u_restored.shape
        sig_full = torch.cat([ch_out.sigma_post_iq, ch_out.sigma_post_iq], dim=1).reshape(B, 16, 16, C)
        alp_full = torch.cat([ch_out.alpha_ratio_iq, ch_out.alpha_ratio_iq], dim=1).reshape(B, 16, 16, C)
        un_sp = ch_out.u_norm.reshape(B, 16, 16, C)

        if active_mask is not None:
            m_sp = active_mask.reshape(B, 16, 16, C)
            denom = m_sp.sum(dim=-1, keepdim=True).clamp_min(1.0)
            local_snr = torch.log1p(((un_sp.square() * m_sp) / (sig_full + 1e-6)).sum(dim=-1, keepdim=True) / denom).div(3.0)
        else:
            local_snr = torch.log1p((un_sp.square() / (sig_full + 1e-6)).mean(dim=-1, keepdim=True)).div(3.0)
        log_sig = torch.log(sig_full.clamp_min(1e-6)).mean(dim=-1, keepdim=True).add(3.0).div(4.0)
        alp_dev = (alp_full - 1.0).mean(dim=-1, keepdim=True)
        return torch.cat([local_snr, log_sig, alp_dev], dim=-1).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        ch_out: ChannelV2Output,
        ct_vec: torch.Tensor,
        cr_vec: torch.Tensor,
        tau_vec: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C = ch_out.u_restored.shape
        H = W = 16
        u_sp = ch_out.u_restored.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        cr_norm = ((cr_vec - SNR_CENTER) / SNR_SCALE).unsqueeze(1)
        ct_norm = ((ct_vec - SNR_CENTER) / SNR_SCALE).unsqueeze(1)

        if self.mode == "p_raw_v2":
            z0 = torch.zeros_like(cr_norm)
            c_raw = torch.cat([cr_norm, ct_norm, z0], dim=-1)
            c_map = c_raw[:, :, None, None].expand(B, 3, H, W)
            h = F.gelu(self.raw_in(torch.cat([u_sp, c_map], dim=1)))
            h = self.raw_b2(self.raw_b1(h, c_raw), c_raw)
            u_out = (u_sp + self.raw_out(h)).permute(0, 2, 3, 1).reshape(B, L, C)
            if active_mask is not None:
                u_out = u_out * active_mask
            dummy_phy = torch.zeros((B, 3, H, W), device=u_sp.device, dtype=u_sp.dtype)
            dummy_cond = torch.zeros((B, 6), device=u_sp.device, dtype=u_sp.dtype)
            dummy_g = torch.zeros((B, 1), device=u_sp.device, dtype=u_sp.dtype)
            return u_out, dummy_phy, dummy_cond, dummy_g

        dc = (ct_vec - cr_vec) / SNR_SCALE
        dc_pos = F.relu(dc).unsqueeze(1)
        dc_neg = F.relu(-dc).unsqueeze(1)
        psi_col = ch_out.psi_csi.clamp(0.0, 5.0).unsqueeze(1)
        tau_col = tau_vec.clamp(0.0, 1.0).unsqueeze(1)
        mean_log_sig = torch.log(ch_out.sigma_post_iq.mean(dim=1, keepdim=True).clamp_min(1e-6)).div(5.0)

        cond = torch.cat([cr_norm, dc_pos, dc_neg, psi_col, tau_col, mean_log_sig], dim=-1)
        phy_map = self.build_spatial_phy_map(ch_out, active_mask=active_mask)

        g_pos = torch.tanh(F.softplus(self.gamma_pos) * dc_pos)
        g_neg = torch.tanh(F.softplus(self.gamma_neg) * dc_neg)
        g_csi = torch.tanh(F.softplus(self.gamma_csi) * psi_col) if self.enable_csi_gate else torch.zeros_like(g_pos)

        u_low = F.avg_pool2d(F.pad(u_sp, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        u_high = u_sp - u_low

        sig_full = torch.cat([ch_out.sigma_post_iq, ch_out.sigma_post_iq], dim=1).reshape(B, H, W, C).permute(0, 3, 1, 2)
        un_sp = ch_out.u_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)
        temp_scale = (1.0 + F.softplus(self.wiener_temp(cond)) - math.log(2.0)).clamp_min(0.1)[:, :, None, None]
        sig_eff = sig_full * temp_scale
        w_emp = F.relu(un_sp.square() - sig_eff) / (un_sp.square() + sig_eff + 1e-6)
        u_wiener = u_low * w_emp.mean(dim=(2, 3), keepdim=True) + u_high * w_emp

        hp = F.gelu(self.pos_in(torch.cat([u_sp, u_wiener, u_high, phy_map], dim=1)))
        hp = self.pos_b2(self.pos_b1(hp, cond), cond)
        r_pos = self.pos_out(hp)

        u_gap = u_sp.mean(dim=(2, 3))
        se_gain = self.neg_se(torch.cat([u_gap, cond], dim=-1))[:, :, None, None]
        hn = F.gelu(self.neg_in(torch.cat([u_sp * (1.0 + se_gain), u_high, phy_map], dim=1)))
        hn = self.neg_b2(self.neg_b1(hn, cond), cond)
        r_neg = self.neg_out(hn)

        if self.iq_pair is not None:
            iq_delta = self.iq_pair(
                ch_out.u_restored,
                ch_out.u_norm,
                ch_out.alpha_ratio_iq,
                ch_out.sigma_post_iq,
                ch_out.phase_unc_iq,
                active_mask=active_mask,
            ).reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            iq_delta = torch.zeros_like(u_sp)

        mmse_sp = ch_out.u_mmse_restored.reshape(B, H, W, C).permute(0, 3, 1, 2)
        mmse_diff = mmse_sp - u_sp
        hc = F.gelu(self.csi_in(torch.cat([u_sp, mmse_diff, iq_delta, phy_map], dim=1)))
        hc = self.csi_b2(self.csi_b1(hc, cond), cond)
        r_csi = self.csi_out(hc) + iq_delta

        u_cal_sp = (
            u_sp
            + g_pos[:, :, None, None] * r_pos
            + g_neg[:, :, None, None] * r_neg
            + g_csi[:, :, None, None] * r_csi
        )
        u_cal = u_cal_sp.permute(0, 2, 3, 1).reshape(B, L, C).contiguous()
        g_tot = torch.clamp(g_pos + g_neg + g_csi, 0.0, 1.0)
        return u_cal, phy_map, cond, g_tot


class StageZeroLockedAdapter(nn.Module):
    """Zero-locked spatial-channel adapter injected between Swin Decoder blocks at resolution (H_s, W_s)."""

    def __init__(self, dim: int, cond_dim: int = 6):
        super().__init__()
        mid = max(64, dim // 2)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, dim * 2),
        )
        _small_init(self.film[-1], std=0.01)

        self.down = nn.Linear(dim, mid)
        self.fuse_pw = nn.Conv2d(mid + 3, mid, kernel_size=1)
        self.dw_conv = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid)
        self.pw_conv = nn.Conv2d(mid, mid, kernel_size=1)
        self.up = nn.Linear(mid, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        x: torch.Tensor,
        H_s: int,
        W_s: int,
        phy_map_16: torch.Tensor,
        cond: torch.Tensor,
        g_tot: torch.Tensor,
    ) -> torch.Tensor:
        if float(g_tot.abs().max().item()) == 0.0 and not self.training:
            return x
        B, L, _ = x.shape
        h = self.norm(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h_mid = F.gelu(self.down(h))
        h_2d = h_mid.reshape(B, H_s, W_s, -1).permute(0, 3, 1, 2).contiguous()

        p_2d = (
            phy_map_16
            if (H_s == 16 and W_s == 16)
            else F.interpolate(phy_map_16, size=(H_s, W_s), mode="bilinear", align_corners=False)
        )
        h_2d = F.gelu(self.fuse_pw(torch.cat([h_2d, p_2d], dim=1)))
        h_2d = F.gelu(self.pw_conv(self.dw_conv(h_2d)))
        h_tok = h_2d.permute(0, 2, 3, 1).reshape(B, L, -1)
        return x + g_tot.unsqueeze(1) * self.up(h_tok)


class UncertaintyAwareTXAdapter(nn.Module):
    """Zero-locked closed-loop transmitter conditioning driven by known feedback staleness tau_unc (0 extra bits)."""

    def __init__(self, embed_dim: int = 320, latent_ch: int = 96):
        super().__init__()
        self.gamma_unc = nn.Parameter(torch.tensor(2.0))
        self.shrink_raw = nn.Parameter(torch.tensor(0.0))
        self.risk_beta_raw = nn.Parameter(torch.tensor(-0.5))
        self.c_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.c_mlp[-1].weight)
        nn.init.zeros_(self.c_mlp[-1].bias)

        mid = 96
        self.norm = nn.LayerNorm(embed_dim)
        self.film = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim * 2),
        )
        _small_init(self.film[-1], std=0.01)
        self.power_head = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.GELU(),
            nn.Linear(mid, latent_ch),
        )
        nn.init.zeros_(self.power_head[-1].weight)
        nn.init.zeros_(self.power_head[-1].bias)

    def compute_c_eff(
        self,
        c_t_raw: torch.Tensor,
        tau_unc: torch.Tensor,
        c_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_clamp = c_t_raw.clamp(1.0, 13.0)
        g_unc = torch.tanh(F.softplus(self.gamma_unc) * tau_unc.clamp(0.0, 1.0))

        w_sh = torch.sigmoid(self.shrink_raw)
        beta_r = F.softplus(self.risk_beta_raw)
        prior_pull = w_sh * (c_prior - c_clamp) - beta_r * F.relu(c_clamp - c_prior)

        mlp_in = torch.stack(
            [
                (c_clamp - SNR_CENTER) / SNR_SCALE,
                (c_prior - SNR_CENTER) / SNR_SCALE,
                tau_unc.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        mlp_delta = self.c_mlp(mlp_in).squeeze(-1) * SNR_SCALE * 0.5
        c_eff = (c_clamp + g_unc * (prior_pull + mlp_delta)).clamp(1.0, 13.0)
        return c_eff, g_unc

    def forward_delta_z(
        self,
        x_stem: torch.Tensor,
        c_t_raw: torch.Tensor,
        c_eff: torch.Tensor,
        c_prior: torch.Tensor,
        tau_unc: torch.Tensor,
        g_unc: torch.Tensor,
    ) -> torch.Tensor:
        if float(g_unc.abs().max().item()) == 0.0 and not self.training:
            B, L, _ = x_stem.shape
            return torch.zeros((B, L, self.power_head[-1].out_features), device=x_stem.device, dtype=x_stem.dtype)
        cond_tx = torch.stack(
            [
                (c_t_raw.clamp(-5.0, 25.0) - SNR_CENTER) / SNR_SCALE,
                (c_eff - SNR_CENTER) / SNR_SCALE,
                (c_prior - SNR_CENTER) / SNR_SCALE,
                tau_unc.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        h = self.norm(x_stem)
        gamma, beta = self.film(cond_tx).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return g_unc[:, None, None] * self.power_head(h)


class TransceiverZPACv2(nn.Module):
    """Unified V2 wrapper supporting Frozen, Clamped, AvgSNR, B1, P-Raw, MS-ZPAC (RX-Only), and TR-ZPAC (Joint)."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "ms_zpac",
        hidden_ch: int = 144,
        avg_snr_fallback_db: float = 7.0,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.method = method.lower()
        self.avg_snr_fallback_db = float(avg_snr_fallback_db)

        for p in self.backbone.parameters():
            p.requires_grad = False

        model_mode = self.backbone.model
        self.latent_channels = (
            int(self.backbone.encoder.head_list.out_features)
            if model_mode == "SwinJSCC_w/_SA"
            else int(self.backbone.encoder.embed_dims[-1])
        )
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.ua_tx: UncertaintyAwareTXAdapter | None = None
        self.frontend: PhysicsFrontendV2 | None = None
        self.post_head_film: nn.Module | None = None
        self.bm_cond_mlp: nn.Module | None = None
        self.stage_adapters: nn.ModuleDict | None = None
        self.dual_decoder_bm = False

        if self.method in ("frozen", "frozen_clamped", "avg_snr"):
            return

        if self.method == "b1_v2":
            self._convert_decoder_bm_to_dual()
            for m in self.backbone.decoder.bm_list.parameters():
                m.requires_grad = True
            for m in self.backbone.decoder.sm_list.parameters():
                m.requires_grad = True
            return

        if self.method == "p_raw_v2":
            self.frontend = PhysicsFrontendV2(self.latent_channels, hidden_ch=hidden_ch, mode="p_raw_v2")
            return

        enable_csi = self.method != "ms_zpac_zeromis"
        enable_iq = self.method != "ms_zpac_noiq"
        enable_ms = self.method != "ms_zpac_noms"

        self.frontend = PhysicsFrontendV2(
            self.latent_channels,
            hidden_ch=hidden_ch,
            mode="ms_zpac",
            enable_csi_gate=enable_csi,
            enable_iq_pair=enable_iq,
        )

        self.bm_cond_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.bm_cond_mlp[-1].weight)
        nn.init.zeros_(self.bm_cond_mlp[-1].bias)

        self.post_head_film = nn.Sequential(
            nn.Linear(6, 96),
            nn.GELU(),
            nn.Linear(96, dec_dims[0] * 2),
        )
        nn.init.zeros_(self.post_head_film[-1].weight)
        nn.init.zeros_(self.post_head_film[-1].bias)

        if enable_ms:
            self.stage_adapters = nn.ModuleDict(
                {
                    "s0_end": StageZeroLockedAdapter(dec_dims[0], cond_dim=6),
                    "s1_mid": StageZeroLockedAdapter(dec_dims[1], cond_dim=6),
                    "s1_end": StageZeroLockedAdapter(dec_dims[1], cond_dim=6),
                    "s2_end": StageZeroLockedAdapter(dec_dims[2], cond_dim=6),
                    "s3_end": StageZeroLockedAdapter(dec_dims[3], cond_dim=6),
                }
            )

        if self.method == "tr_zpac":
            self.ua_tx = UncertaintyAwareTXAdapter(
                embed_dim=int(self.backbone.encoder.embed_dims[-1]),
                latent_ch=self.latent_channels,
            )

    def _convert_decoder_bm_to_dual(self) -> None:
        dec = self.backbone.decoder
        for mod in dec.bm_list:
            mod.fc[0] = DualConditionLinear(mod.fc[0])
        self.dual_decoder_bm = True

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_from_stem(
        self,
        x_stem: torch.Tensor,
        c_t_raw_db: torch.Tensor,
        tau_unc: torch.Tensor,
        c_prior_db: torch.Tensor,
        rate_channels: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        enc = self.backbone.encoder
        model_mode = self.backbone.model
        B, L, _ = x_stem.shape
        device = x_stem.device
        dtype = x_stem.dtype

        if self.method == "avg_snr":
            c_eff = torch.full((B,), self.avg_snr_fallback_db, device=device, dtype=dtype)
            g_unc = torch.zeros((B,), device=device, dtype=dtype)
        elif self.method == "frozen_clamped":
            c_eff = c_t_raw_db.clamp(1.0, 13.0)
            g_unc = torch.zeros((B,), device=device, dtype=dtype)
        elif self.ua_tx is not None:
            c_eff, g_unc = self.ua_tx.compute_c_eff(c_t_raw_db, tau_unc, c_prior_db)
        else:
            c_eff = c_t_raw_db
            g_unc = torch.zeros((B,), device=device, dtype=dtype)

        snr_batch = c_eff.reshape(B, 1)

        if model_mode == "SwinJSCC_w/_SA":
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                bm = enc.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = enc.sigmoid(enc.sm_list[-1](temp))
            z_out = enc.head_list(x_stem * mod_val)
            if self.ua_tx is not None:
                z_out = z_out + self.ua_tx.forward_delta_z(x_stem, c_t_raw_db, c_eff, c_prior_db, tau_unc, g_unc)
            return z_out, None, c_eff

        if model_mode == "SwinJSCC_w/_SAandRA":
            assert rate_channels is not None and 1 <= int(rate_channels) <= x_stem.shape[2]
            rate = int(rate_channels)
            rate_batch = _as_batch_col(float(rate), B, device, dtype)

            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list1[i](x_stem.detach()) if i == 0 else enc.sm_list1[i](temp)
                bm = enc.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val1 = enc.sigmoid1(enc.sm_list1[-1](temp))
            x = x_stem * mod_val1

            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x.detach()) if i == 0 else enc.sm_list[i](temp)
                bm = enc.bm_list[i](rate_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = enc.sigmoid(enc.sm_list[-1](temp))
            x = x * mod_val
            if self.ua_tx is not None:
                x = x + self.ua_tx.forward_delta_z(x_stem, c_t_raw_db, c_eff, c_prior_db, tau_unc, g_unc)

            mask_scores = torch.sum(mod_val.detach(), dim=1)
            _, indices = mask_scores.sort(dim=1, descending=True)
            c_indices = indices[:, :rate]
            mask = torch.zeros_like(mask_scores)
            mask.scatter_(1, c_indices, 1.0)
            mask_3d = mask.unsqueeze(1).expand(-1, L, -1)
            x = x * mask_3d
            return x, mask_3d, c_eff

        raise ValueError(f"Unsupported model_mode: {model_mode}")

    def decode_v2(
        self,
        u_in: torch.Tensor,
        c_t_eff_db: torch.Tensor,
        c_r_db: torch.Tensor,
        phy_map_16: torch.Tensor | None,
        cond: torch.Tensor | None,
        g_tot: torch.Tensor | None,
    ) -> torch.Tensor:
        dec = self.backbone.decoder
        model_mode = self.backbone.model
        B, L, _ = u_in.shape

        x = dec.head_list(u_in) if model_mode == "SwinJSCC_w/_SA" else u_in

        cr_col = c_r_db.reshape(B, 1)
        ct_col = c_t_eff_db.reshape(B, 1)

        if self.dual_decoder_bm:
            cond_bm = torch.cat([ct_col, cr_col], dim=-1)
        elif self.bm_cond_mlp is not None and cond is not None and g_tot is not None:
            cond_bm = cr_col + g_tot * self.bm_cond_mlp(cond) * SNR_SCALE
        else:
            cond_bm = cr_col

        temp = None
        for i in range(dec.layer_num):
            temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
            bm = dec.bm_list[i](cond_bm).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm
        mod_val = dec.sigmoid(dec.sm_list[-1](temp))
        x = x * mod_val

        if self.post_head_film is not None and cond is not None and g_tot is not None:
            gb = self.post_head_film(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            x = x * (1.0 + (g_tot * gamma).unsqueeze(1)) + (g_tot * beta).unsqueeze(1)

        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]
        for i_layer, layer in enumerate(dec.layers):
            H_s, W_s = resolutions[i_layer]
            n_blk = len(layer.blocks)
            for b_idx, blk in enumerate(layer.blocks):
                x = blk(x)
                if self.stage_adapters is not None and phy_map_16 is not None and cond is not None and g_tot is not None:
                    if i_layer == 1 and b_idx == (n_blk // 2 - 1):
                        x = self.stage_adapters["s1_mid"](x, H_s, W_s, phy_map_16, cond, g_tot)
            if self.stage_adapters is not None and phy_map_16 is not None and cond is not None and g_tot is not None:
                key = f"s{i_layer}_end"
                if key in self.stage_adapters:
                    x = self.stage_adapters[key](x, H_s, W_s, phy_map_16, cond, g_tot)
            if layer.upsample is not None:
                x = layer.upsample(x)

        return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()

    def forward_channel_v2(
        self,
        image_or_stem: torch.Tensor,
        c_t_raw_db: torch.Tensor | float,
        gamma_db: torch.Tensor | float,
        c_r_db: torch.Tensor | float,
        *,
        channel_mode: str = "awgn",
        tau_unc: torch.Tensor | float = 0.0,
        c_prior_db: torch.Tensor | float = 7.0,
        ofdm_steps: list[OFDMFadingStep] | None = None,
        standard_noise: torch.Tensor | None = None,
        h_complex_iq: torch.Tensor | None = None,
        pilot_noise_iq: torch.Tensor | None = None,
        pilot_symbols: int = 0,
        num_rbs: int = 0,
        equalizer_mode: str = "unbiased_mmse",
        rate_channels: int | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = image_or_stem.shape[0]
        device = image_or_stem.device
        dtype = image_or_stem.dtype

        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)

        if ofdm_steps is not None:
            ct_vec = torch.tensor([s.c_t_raw_db for s in ofdm_steps], device=device, dtype=dtype)
            cr_vec = torch.tensor([s.c_r_db for s in ofdm_steps], device=device, dtype=dtype)
            gm_vec = cr_vec
            tau_vec = torch.tensor([s.tau_unc for s in ofdm_steps], device=device, dtype=dtype)
            prior_vec = torch.tensor([s.gamma_avg_db for s in ofdm_steps], device=device, dtype=dtype)
        else:
            ct_vec = _as_batch_col(c_t_raw_db, B, device, dtype).squeeze(1)
            cr_vec = _as_batch_col(c_r_db, B, device, dtype).squeeze(1)
            gm_vec = _as_batch_col(gamma_db, B, device, dtype).squeeze(1)
            tau_vec = _as_batch_col(tau_unc, B, device, dtype).squeeze(1)
            prior_vec = _as_batch_col(c_prior_db, B, device, dtype).squeeze(1)

        if self.ua_tx is not None:
            z_raw, active_mask, c_eff = self.encode_from_stem(
                x_stem, ct_vec, tau_vec, prior_vec, rate_channels=rate_channels
            )
        else:
            with torch.no_grad():
                z_raw, active_mask, c_eff = self.encode_from_stem(
                    x_stem, ct_vec, tau_vec, prior_vec, rate_channels=rate_channels
                )

        if ofdm_steps is not None or channel_mode == "ofdm_fading":
            assert ofdm_steps is not None
            ch_out = ofdm_fading_forward(
                z_raw,
                ofdm_steps,
                standard_noise=standard_noise,
                equalizer_mode=equalizer_mode,
                active_mask=active_mask,
            )
        elif channel_mode == "rayleigh":
            ch_out = rayleigh_v2_forward(
                z_raw,
                gm_vec,
                cr_vec,
                standard_noise=standard_noise,
                h_complex_iq=h_complex_iq,
                pilot_noise_iq=pilot_noise_iq,
                pilot_symbols=pilot_symbols,
                num_rbs=num_rbs,
                active_mask=active_mask,
            )
        elif channel_mode == "awgn":
            ch_out = awgn_v2_forward(
                z_raw, gm_vec, cr_vec, standard_noise=standard_noise, active_mask=active_mask
            )
        else:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")

        if self.frontend is not None:
            u_cal, phy_map, cond, g_tot = self.frontend(
                ch_out, c_eff, ch_out.c_r_db, tau_vec, active_mask=active_mask
            )
        else:
            u_cal = ch_out.u_restored
            phy_map = None
            cond = None
            g_tot = None

        recon = self.decode_v2(u_cal, c_eff, ch_out.c_r_db, phy_map, cond, g_tot)
        return {
            "recon": recon,
            "c_eff_db": c_eff,
            "c_r_db": ch_out.c_r_db,
            "gamma_db": ch_out.gamma_db,
        }
