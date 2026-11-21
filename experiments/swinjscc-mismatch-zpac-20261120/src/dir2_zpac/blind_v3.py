"""V3 Completely CSI-Free / Blind / Feedback-Free / Pilot-Free Self-Probing Hierarchical JSCC (SPH-ZPAC)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SNR_CENTER, SNR_SCALE
from .models_v2 import SpatialFiLMResBlock, StageZeroLockedAdapter, _small_init
from .phy_v2 import ChannelV2Output, expand_rb_to_symbols, sample_correlated_rbs


def build_walsh3_basis(n_sym: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build 3 mutually orthonormal real Walsh basis vectors (3, n_sym) for n_sym divisible by 4."""
    assert n_sym % 4 == 0, f"n_sym={n_sym} must be divisible by 4"
    scale = 1.0 / math.sqrt(float(n_sym))
    u1 = torch.full((n_sym,), scale, device=device, dtype=dtype)
    idx = torch.arange(n_sym, device=device)
    u2 = torch.where((idx % 2) == 0, scale, -scale).to(dtype=dtype)
    u3 = torch.where(((idx // 2) % 2) == 0, scale, -scale).to(dtype=dtype)
    return torch.stack([u1, u2, u3], dim=0)  # (3, n_sym)


@dataclass
class BlindWirelessConfig:
    mode: str                     # "awgn" or "ofdm_complex" or "awgn_phase_offset"
    snr_db: torch.Tensor          # (B,) TRUE physical SNR (NEVER shown to TX or RX!)
    num_rbs: int = 32
    rho_f: float = 0.85


def physical_blind_channel_forward(
    s_tx_iq: torch.Tensor,        # (B, K, 2) unit-complex-power transmitted symbols (E[|s|^2] == 1)
    snr_db: torch.Tensor | float, # TRUE physical SNR (hidden from transceiver!)
    *,
    channel_type: str = "ofdm_complex",
    num_rbs: int = 32,
    rho_f: float = 0.85,
    h_rb_override: torch.Tensor | None = None,  # optional (B, M, 2) deterministic complex fading
    standard_noise_iq: torch.Tensor | None = None,  # optional (B, K, 2) N(0,1) per real/imag
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pass s_tx_iq over unknown physical wireless channel; return ONLY raw antenna y_rx_iq to receiver!"""
    B, K, two = s_tx_iq.shape
    assert two == 2
    device = s_tx_iq.device
    dtype = s_tx_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )
    sigma_n_sq = torch.pow(10.0, -snr_t / 10.0)  # complex noise variance
    sigma_real = (0.5 * sigma_n_sq).sqrt()[:, None]

    eps = (
        torch.randn_like(s_tx_iq)
        if standard_noise_iq is None
        else standard_noise_iq.to(device=device, dtype=dtype).reshape(B, K, 2)
    )
    n_r = eps[:, :, 0] * sigma_real
    n_i = eps[:, :, 1] * sigma_real

    s_r, s_i = s_tx_iq[:, :, 0], s_tx_iq[:, :, 1]

    if channel_type == "awgn":
        y_r = s_r + n_r
        y_i = s_i + n_i
        hr_sym = torch.ones((B, K), device=device, dtype=dtype)
        hi_sym = torch.zeros((B, K), device=device, dtype=dtype)
    elif channel_type == "ofdm_complex":
        if h_rb_override is not None:
            hr_rb = h_rb_override[:, :, 0].to(device=device, dtype=dtype)
            hi_rb = h_rb_override[:, :, 1].to(device=device, dtype=dtype)
        else:
            hr_rb, hi_rb = sample_correlated_rbs(B, num_rbs, rho_f, device=device, dtype=dtype)
        hr_sym = expand_rb_to_symbols(hr_rb, K)
        hi_sym = expand_rb_to_symbols(hi_rb, K)
        y_r = hr_sym * s_r - hi_sym * s_i + n_r
        y_i = hr_sym * s_i + hi_sym * s_r + n_i
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")

    y_rx_iq = torch.stack([y_r, y_i], dim=-1)  # (B, K, 2) — ONLY tensor visible to blind RX!
    truth_diag = {
        "true_snr_db": snr_t,
        "true_sigma_n_sq": sigma_n_sq,
        "true_hr_sym": hr_sym,
        "true_hi_sym": hi_sym,
    }
    return y_rx_iq, truth_diag


class OpenLoopHierarchicalTX(nn.Module):
    """CSI-Free Open-Loop Hierarchical Encoder Head: learns progressive channel power allocation from image x alone."""

    def __init__(self, embed_dim: int = 320, latent_ch: int = 96, base_c0_db: float = 7.0):
        super().__init__()
        self.base_c0_db = float(base_c0_db)
        mid = 128
        self.norm = nn.LayerNorm(embed_dim)
        # Content-adaptive hierarchical importance gate + residual feature shaper (zero CSI input!)
        self.se_pool = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.GELU(),
            nn.Linear(mid, latent_ch),
            nn.Tanh(),
        )
        _small_init(self.se_pool[-2], std=0.01)

        self.res_head = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.GELU(),
            nn.Linear(mid, latent_ch),
        )
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)

        # Learnable progressive unequal-error-protection spectrum across latent channels
        self.log_spectrum = nn.Parameter(torch.zeros(1, 1, latent_ch))

    def forward(self, z_base: torch.Tensor, x_stem: torch.Tensor) -> torch.Tensor:
        h = self.norm(x_stem)
        ch_gate = self.se_pool(h.mean(dim=1)).unsqueeze(1)  # (B, 1, C)
        spec_scale = torch.exp(self.log_spectrum.clamp(-1.0, 1.0))
        z_hier = z_base * spec_scale * (1.0 + 0.3 * ch_gate) + self.res_head(h)
        return z_hier


class SelfProbingSubspaceModem(nn.Module):
    """Zero-overhead Walsh-Hadamard 3-vector self-probing subspace projector & blind Wiener CSI estimator."""

    def __init__(self, num_rbs: int = 32, eta1_init: float = 0.045, eta3_init: float = 0.005):
        super().__init__()
        self.num_rbs = int(num_rbs)
        # Learnable pilot-subspace power fraction eta1 constrained safely to [0.015, 0.10]
        inv_sig = math.log((eta1_init - 0.015) / (0.10 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_sig)))
        self.eta3 = float(eta3_init)

        # Lightweight 1D complex RB frequency denoiser (refines analytical Wiener smoother across M RBs)
        # Input channels per RB: [h_tilde_r, h_tilde_i, E_rb, log_sig_n, wiener_r, wiener_i] -> 6 ch
        self.rb_refiner = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.rb_refiner[-1].weight)
        nn.init.zeros_(self.rb_refiner[-1].bias)

    @property
    def eta1(self) -> torch.Tensor:
        return 0.015 + (0.10 - 0.015) * torch.sigmoid(self.eta1_raw)

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Map (B, L, C) latent into exact unit-power self-probing complex symbols s_tx_iq (B, K, 2)."""
        B, L, C = z_raw.shape
        K = (L * C) // 2
        M = self.num_rbs
        N_b = K // M
        assert M * N_b == K
        device = z_raw.device
        dtype = z_raw.dtype

        # Frame RMS scale p_img (will be recovered blindly via u3 at RX — zero side-channel!)
        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        p_img = pwr.sqrt()  # (B,)
        z_unit = z_raw / p_img[:, None, None]

        flat = z_unit.reshape(B, 2 * K)
        s_r = flat[:, :K].reshape(B, M, N_b)
        s_i = flat[:, K:].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)  # (3, N_b)
        u1, u2, u3 = basis[0], basis[1], basis[2]

        # Remove projections onto span(u1, u2, u3) so semantic data lies strictly in orthogonal complement
        proj_r = torch.einsum("bmn,kn->bmk", s_r, basis)  # (B, M, 3)
        proj_i = torch.einsum("bmn,kn->bmk", s_i, basis)  # (B, M, 3)
        s_perp_r = s_r - torch.einsum("bmk,kn->bmn", proj_r, basis)
        s_perp_i = s_i - torch.einsum("bmk,kn->bmn", proj_i, basis)

        eta1 = self.eta1
        eta3 = self.eta3
        data_pwr_target = 1.0 - eta1 - eta3
        cur_data_pwr = (s_perp_r.square() + s_perp_i.square()).mean(dim=(1, 2)).clamp_min(1e-8)
        scale_data = (data_pwr_target / cur_data_pwr).sqrt()[:, None, None]

        s_data_r = s_perp_r * scale_data
        s_data_i = s_perp_i * scale_data

        # Constant-modulus encoding of p_img along u3 centered at empirical mean 2.82: |phi_p + j*phi_q|^2 == 1.0!
        phi_p = torch.tanh(0.35 * (p_img - 2.82)).clamp(-0.98, 0.98)  # (B,)
        phi_q = (1.0 - phi_p.square()).clamp_min(1e-6).sqrt()         # (B,)

        amp1 = (eta1 * float(N_b)).sqrt()
        amp3 = math.sqrt(eta3 * float(N_b))

        tx_r = s_data_r + amp1 * u1[None, None, :] + (amp3 * phi_p[:, None, None]) * u3[None, None, :]
        tx_i = s_data_i + (amp3 * phi_q[:, None, None]) * u3[None, None, :]

        return torch.stack([tx_r.reshape(B, K), tx_i.reshape(B, K)], dim=-1)

    def blind_demod_and_equalize(
        self,
        y_rx_iq: torch.Tensor,  # (B, K, 2) — ONLY input from antenna!
        latent_ch: int,
    ) -> tuple[ChannelV2Output, torch.Tensor]:
        """Recover (sigma_n^2, h_m, p_img, c_R) completely blindly from y_rx_iq and equalize!"""
        B, K, _ = y_rx_iq.shape
        M = self.num_rbs
        N_b = K // M
        L = (2 * K) // latent_ch
        device = y_rx_iq.device
        dtype = y_rx_iq.dtype

        y_r = y_rx_iq[:, :, 0].reshape(B, M, N_b)
        y_i = y_rx_iq[:, :, 1].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)
        u1, u2, u3 = basis[0], basis[1], basis[2]

        # 1. Exact interference-free blind noise variance from orthogonal null vector u2
        q2_r = torch.einsum("bmn,n->bm", y_r, u2)
        q2_i = torch.einsum("bmn,n->bm", y_i, u2)
        sigma_n_hat = (q2_r.square() + q2_i.square()).mean(dim=1).clamp(1e-5, 10.0)  # (B,)

        # 2. Raw spread-spectrum complex RB channel estimate from u1
        eta1 = self.eta1
        amp1 = (eta1 * float(N_b)).sqrt()
        htilde_r = torch.einsum("bmn,n->bm", y_r, u1) / amp1  # (B, M)
        htilde_i = torch.einsum("bmn,n->bm", y_i, u1) / amp1  # (B, M)
        sigma_e_raw = sigma_n_hat / (eta1 * float(N_b))       # (B,)

        # 3. Analytical 5-tap frequency-correlation Wiener smoother + Constant-Shell RB energy cue
        e_rb = (y_r.square() + y_i.square()).mean(dim=-1)  # (B, M) ~ |h_m|^2 + sigma_n^2
        # Smooth htilde across adjacent RBs (5-tap triangular kernel [1, 2, 3, 2, 1]/9 blended with LMMSE shrinkage)
        k5 = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=device, dtype=dtype).reshape(1, 1, 5) / 9.0
        ht_r_pad = F.pad(htilde_r.unsqueeze(1), (2, 2), mode="replicate")
        ht_i_pad = F.pad(htilde_i.unsqueeze(1), (2, 2), mode="replicate")
        hsmooth_r = F.conv1d(ht_r_pad, k5).squeeze(1)
        hsmooth_i = F.pad(htilde_i.unsqueeze(1), (2, 2), mode="replicate")
        hsmooth_i = F.conv1d(hsmooth_i, k5).squeeze(1)

        # Blend raw & smoothed based on estimated raw pilot SNR (keep sharp RB variations at high SNR, smooth more at low SNR)
        w_raw = (1.0 / (1.0 + 3.0 * sigma_e_raw))[:, None]
        h_wien_r = (w_raw * htilde_r + (1.0 - w_raw) * hsmooth_r) / (1.0 + 0.35 * sigma_e_raw[:, None])
        h_wien_i = (w_raw * htilde_i + (1.0 - w_raw) * hsmooth_i) / (1.0 + 0.35 * sigma_e_raw[:, None])

        log_sig_feat = torch.log(sigma_n_hat.clamp_min(1e-5))[:, None].expand(B, M).div(4.0)
        ref_in = torch.stack([htilde_r, htilde_i, e_rb, log_sig_feat, h_wien_r, h_wien_i], dim=1)  # (B, 6, M)
        delta_h = self.rb_refiner(ref_in)
        hhat_r_rb = h_wien_r + delta_h[:, 0, :]
        hhat_i_rb = h_wien_i + delta_h[:, 1, :]

        # Effective residual channel estimation variance after RB smoothing (~0.45 * raw variance)
        sigma_e_eff = 0.45 * sigma_e_raw

        # 4. Strip u1 reference from y_rx and apply Gain-Unbiased Blind MMSE Equalization
        y_clean_r = y_r - (hhat_r_rb[:, :, None] * amp1) * u1[None, None, :]
        y_clean_i = y_i - (hhat_i_rb[:, :, None] * amp1) * u1[None, None, :]

        hhat_sq_rb = hhat_r_rb.square() + hhat_i_rb.square()  # (B, M)
        hhat_abs_rb = hhat_sq_rb.clamp_min(1e-10).sqrt()
        eff_var = (sigma_n_hat + sigma_e_eff)[:, None]        # (B, 1)
        denom_rb = (hhat_sq_rb + eff_var).clamp_min(1e-6)
        alpha_rb = hhat_sq_rb / denom_rb                      # (B, M)
        alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.05)  # (B, 1)

        # Phase de-rotate + unbiased MMSE weight per RB
        cos_rb = (hhat_r_rb / hhat_abs_rb)[:, :, None]
        sin_rb = (hhat_i_rb / hhat_abs_rb)[:, :, None]
        y_derot_r = cos_rb * y_clean_r + sin_rb * y_clean_i
        y_derot_i = -sin_rb * y_clean_r + cos_rb * y_clean_i

        w_ub_rb = ((hhat_abs_rb / denom_rb) / alpha_bar)[:, :, None]
        shat_r = y_derot_r * w_ub_rb
        shat_i = y_derot_i * w_ub_rb

        # 5. Blindly decode frame scale p_img from u3 across all M RBs, then strip u3!
        amp3 = math.sqrt(self.eta3 * float(N_b))
        u3_re = torch.einsum("bmn,n->bm", shat_r, u3).mean(dim=1) / amp3
        phi_p_hat = u3_re.clamp(-0.97, 0.97)
        p_img_hat = (2.82 + torch.atanh(phi_p_hat) / 0.35).clamp(0.4, 7.0)  # (B,)

        # Strip u3 projection and rescale data back to unit pre-subspace variance
        p3_r = torch.einsum("bmn,n->bm", shat_r, u3)[:, :, None] * u3[None, None, :]
        p3_i = torch.einsum("bmn,n->bm", shat_i, u3)[:, :, None] * u3[None, None, :]
        data_scale = (1.0 / (1.0 - eta1 - self.eta3)).sqrt()
        s_data_hat_r = (shat_r - p3_r) * data_scale
        s_data_hat_i = (shat_i - p3_i) * data_scale

        u_mmse_norm = torch.cat([s_data_hat_r.reshape(B, K), s_data_hat_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        # Blind wideband effective SINR in dB
        sinr_wb = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1)).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-4.0, 22.0)

        h_abs_iq = expand_rb_to_symbols(hhat_abs_rb, K)
        alpha_ratio_iq = expand_rb_to_symbols(alpha_rb / alpha_bar, K)
        sigma_post_iq = expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K)
        phase_unc_iq = expand_rb_to_symbols((sigma_e_eff[:, None] / (hhat_sq_rb + sigma_e_eff[:, None] + 1e-6)).sqrt(), K)
        psi_csi = (hhat_abs_rb - 1.0).square().mean(dim=1) + sigma_e_eff

        ch_v2 = ChannelV2Output(
            u_restored=u_mmse_restored,
            u_mmse_restored=u_mmse_restored,
            u_norm=u_mmse_norm,
            z_norm=u_mmse_norm,
            pwr=p_img_hat.square(),
            gamma_db=cr_hat_db,
            c_r_db=cr_hat_db,
            h_abs_iq=h_abs_iq,
            alpha_ratio_iq=alpha_ratio_iq,
            sigma_post_iq=sigma_post_iq,
            phase_unc_iq=phase_unc_iq,
            psi_csi=psi_csi,
        )
        return ch_v2, cr_hat_db


class LegacyBlindNoiseEstimator(nn.Module):
    """Blind SNR & spatial noise estimator for unmodified frozen SwinJSCC latents (no u1/u2 subspace!)."""

    def __init__(self, latent_ch: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch * 2 + 2, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.snr_head = nn.Sequential(
            nn.Linear(96 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # [delta_snr_db, delta_log_p_scale]
        )
        nn.init.zeros_(self.snr_head[-1].weight)
        nn.init.zeros_(self.snr_head[-1].bias)

    def forward(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, K, _ = y_rx_iq.shape
        L = (2 * K) // latent_ch
        device = y_rx_iq.device
        dtype = y_rx_iq.dtype

        u_raw = torch.cat([y_rx_iq[:, :, 0], y_rx_iq[:, :, 1]], dim=1).reshape(B, L, latent_ch)
        u_sp = u_raw.reshape(B, 16, 16, latent_ch).permute(0, 3, 1, 2).contiguous()

        # High-pass spatial Laplacian residual for blind noise variance proxy (Donoho MAD style)
        u_blur = F.avg_pool2d(F.pad(u_sp, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        u_hp = u_sp - u_blur
        hp_var_map = (u_hp.square() * (9.0 / 8.0)).mean(dim=1, keepdim=True)
        tot_pwr_map = u_sp.square().mean(dim=1, keepdim=True)

        tot_complex_pwr = (y_rx_iq[:, :, 0].square() + y_rx_iq[:, :, 1].square()).mean(dim=1)
        excess_noise = (tot_complex_pwr - 1.0).clamp_min(1e-4)
        hp_noise = (2.0 * hp_var_map.mean(dim=(1, 2, 3))).clamp_min(1e-4)

        snr_init_db = (-10.0 * torch.log10(0.5 * excess_noise + 0.5 * hp_noise)).clamp(-4.0, 20.0)

        h_feat = self.net(torch.cat([u_sp, u_hp, torch.log(hp_var_map.clamp_min(1e-6)), torch.log(tot_pwr_map.clamp_min(1e-6))], dim=1))
        pooled = h_feat.mean(dim=(2, 3))
        stats_3 = torch.stack(
            [
                (snr_init_db - SNR_CENTER) / SNR_SCALE,
                torch.log(tot_complex_pwr.clamp_min(1e-5)),
                torch.log(hp_noise),
            ],
            dim=-1,
        )
        head_out = self.snr_head(torch.cat([pooled, stats_3], dim=-1))
        cr_hat_db = (snr_init_db + head_out[:, 0] * 4.0).clamp(-4.0, 22.0)
        p_hat = (2.82 * torch.exp(head_out[:, 1].clamp(-1.0, 1.0))).clamp(0.5, 6.0)

        sigma_post = 0.5 * torch.pow(10.0, -cr_hat_db / 10.0)
        ones_iq = torch.ones((B, K), device=device, dtype=dtype)
        zeros_iq = torch.zeros((B, K), device=device, dtype=dtype)
        est_sig_scale = (p_hat / tot_complex_pwr.clamp_min(0.5).sqrt())[:, None, None]
        u_restored = u_raw * est_sig_scale

        ch_v2 = ChannelV2Output(
            u_restored=u_restored,
            u_mmse_restored=u_restored,
            u_norm=u_raw,
            z_norm=u_raw,
            pwr=p_hat.square(),
            gamma_db=cr_hat_db,
            c_r_db=cr_hat_db,
            h_abs_iq=ones_iq,
            alpha_ratio_iq=ones_iq,
            sigma_post_iq=sigma_post[:, None].expand(B, K).contiguous(),
            phase_unc_iq=zeros_iq,
            psi_csi=torch.ones((B,), device=device, dtype=dtype) * 0.5,
        )
        return ch_v2, cr_hat_db


class BlindMultiScaleDecoderCalibrator(nn.Module):
    """Multi-scale 4-stage Swin Decoder calibrator driven SOLELY by blind receiver estimates (zero external CSI!)."""

    def __init__(self, latent_ch: int = 96, dec_dims: list[int] = [320, 256, 192, 128], hidden_ch: int = 160):
        super().__init__()
        cond_dim = 6  # [cr_hat_norm, pos_gap, neg_gap, psi_csi, mean_log_sig, alpha_std]

        # I/Q cross-half complex residual block + spatial FiLM frontend
        self.iq_conv = nn.Sequential(
            nn.Conv2d(2 * latent_ch + 3, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 2 * latent_ch, kernel_size=1),
        )
        nn.init.zeros_(self.iq_conv[-1].weight)
        nn.init.zeros_(self.iq_conv[-1].bias)

        self.front_in = nn.Conv2d(2 * latent_ch + 3, hidden_ch, kernel_size=1)
        self.front_b1 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.front_b2 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.front_b3 = SpatialFiLMResBlock(hidden_ch, cond_dim)
        self.front_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.front_out.weight)
        nn.init.zeros_(self.front_out.bias)

        self.bm_cond_mlp = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.bm_cond_mlp[-1].weight)
        nn.init.zeros_(self.bm_cond_mlp[-1].bias)

        self.post_head_film = nn.Sequential(nn.Linear(cond_dim, 96), nn.GELU(), nn.Linear(96, dec_dims[0] * 2))
        nn.init.zeros_(self.post_head_film[-1].weight)
        nn.init.zeros_(self.post_head_film[-1].bias)

        self.stage_adapters = nn.ModuleDict(
            {
                "s0_end": StageZeroLockedAdapter(dec_dims[0], cond_dim=cond_dim),
                "s1_mid": StageZeroLockedAdapter(dec_dims[1], cond_dim=cond_dim),
                "s1_end": StageZeroLockedAdapter(dec_dims[1], cond_dim=cond_dim),
                "s2_end": StageZeroLockedAdapter(dec_dims[2], cond_dim=cond_dim),
                "s3_end": StageZeroLockedAdapter(dec_dims[3], cond_dim=cond_dim),
            }
        )

    def forward_frontend(
        self,
        ch_out: ChannelV2Output,
        cr_hat_db: torch.Tensor,
        base_c0_db: float = 7.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C = ch_out.u_restored.shape
        u_sp = ch_out.u_restored.reshape(B, 16, 16, C).permute(0, 3, 1, 2).contiguous()

        # Spatial 3-channel blind reliability map
        sig_full = torch.cat([ch_out.sigma_post_iq, ch_out.sigma_post_iq], dim=1).reshape(B, 16, 16, C)
        alp_full = torch.cat([ch_out.alpha_ratio_iq, ch_out.alpha_ratio_iq], dim=1).reshape(B, 16, 16, C)
        un_sp = ch_out.u_norm.reshape(B, 16, 16, C)

        local_snr = torch.log1p((un_sp.square() / (sig_full + 1e-6)).mean(dim=-1, keepdim=True)).div(3.0)
        log_sig = torch.log(sig_full.clamp_min(1e-6)).mean(dim=-1, keepdim=True).add(3.0).div(4.0)
        alp_dev = (alp_full - 1.0).mean(dim=-1, keepdim=True)
        phy_map = torch.cat([local_snr, log_sig, alp_dev], dim=-1).permute(0, 3, 1, 2).contiguous()  # (B, 3, 16, 16)

        # I/Q paired residual on (8, 16)
        u_grid = ch_out.u_restored.reshape(B, 16, 16, C)
        u_re = u_grid[:, :8, :, :].permute(0, 3, 1, 2)
        u_im = u_grid[:, 8:, :, :].permute(0, 3, 1, 2)
        phy_8 = F.avg_pool2d(phy_map, kernel_size=(2, 1), stride=(2, 1))
        iq_d = self.iq_conv(torch.cat([u_re, u_im, phy_8], dim=1))
        iq_sp = torch.cat([iq_d[:, :C], iq_d[:, C:]], dim=2)  # (B, C, 16, 16)

        # Empirical Wiener shrinkage feature
        sig_c = sig_full.permute(0, 3, 1, 2)
        un_c = un_sp.permute(0, 3, 1, 2)
        w_emp = F.relu(un_c.square() - sig_c) / (un_c.square() + sig_c + 1e-6)
        u_wiener = u_sp * w_emp

        cr_norm = ((cr_hat_db - SNR_CENTER) / SNR_SCALE).unsqueeze(1)
        dc_rel = (base_c0_db - cr_hat_db) / SNR_SCALE
        pos_gap = F.relu(dc_rel).unsqueeze(1)
        neg_gap = F.relu(-dc_rel).unsqueeze(1)
        psi_col = ch_out.psi_csi.clamp(0.0, 5.0).unsqueeze(1)
        mean_log_sig = torch.log(ch_out.sigma_post_iq.mean(dim=1, keepdim=True).clamp_min(1e-6)).div(5.0)
        alp_std = ch_out.alpha_ratio_iq.std(dim=1, keepdim=True).clamp(0.0, 3.0)
        cond = torch.cat([cr_norm, pos_gap, neg_gap, psi_col, mean_log_sig, alp_std], dim=-1)  # (B, 6)

        h = F.gelu(self.front_in(torch.cat([u_sp + iq_sp, u_wiener, phy_map], dim=1)))
        h = self.front_b3(self.front_b2(self.front_b1(h, cond), cond), cond)
        u_cal_sp = u_sp + iq_sp + self.front_out(h)
        u_cal = u_cal_sp.permute(0, 2, 3, 1).reshape(B, L, C).contiguous()

        g_ones = torch.ones((B, 1), device=u_sp.device, dtype=u_sp.dtype)
        return u_cal, phy_map, cond, g_ones


class BlindCSIFreeTransceiver(nn.Module):
    """Unified V3 CSI-Free / Blind wrapper:
      - frozen_blind: Official SwinJSCC with c_T=7, c_R=7, zero CSI, zero pilots, zero side-info
      - blind_blackbox: IEEE TCOM 2025 style end-to-end blind TX+RX training WITHOUT self-probing subspace
      - blind_rx_zpac: 100% Frozen legacy encoder + Blind noise-estimating multi-scale RX
      - sph_zpac_nohiertx: Self-probing subspace + multi-scale RX, WITHOUT Open-Loop Hierarchical TX
      - sph_zpac: Full Self-Probing Hierarchical CSI-Free Transceiver (0 CSI, 0 feedback, 0 extra pilots!)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "sph_zpac",
        num_rbs: int = 32,
        base_c0_db: float = 7.0,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.method = method.lower()
        self.num_rbs = int(num_rbs)
        self.base_c0_db = float(base_c0_db)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx: OpenLoopHierarchicalTX | None = None
        self.subspace_modem: SelfProbingSubspaceModem | None = None
        self.legacy_estimator: LegacyBlindNoiseEstimator | None = None
        self.blind_dec: BlindMultiScaleDecoderCalibrator | None = None

        if self.method == "frozen_blind":
            return

        if self.method == "blind_blackbox":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.legacy_estimator = LegacyBlindNoiseEstimator(self.latent_channels)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
            return

        if self.method == "blind_rx_zpac":
            self.legacy_estimator = LegacyBlindNoiseEstimator(self.latent_channels)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
            return

        if self.method == "sph_zpac_nohiertx":
            self.subspace_modem = SelfProbingSubspaceModem(num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
            return

        if self.method == "sph_zpac":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.subspace_modem = SelfProbingSubspaceModem(num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
            return

        raise ValueError(f"Unknown V3 blind method: {self.method}")

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_blind_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
        """Encode x_stem into exact unit-average-power complex symbols s_tx_iq (B, K, 2) with ZERO CSI!"""
        enc = self.backbone.encoder
        B, L, _ = x_stem.shape
        K = (L * self.latent_channels) // 2
        snr_c0 = torch.full((B, 1), self.base_c0_db, device=x_stem.device, dtype=x_stem.dtype)

        with torch.no_grad():
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                bm = enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = enc.sigmoid(enc.sm_list[-1](temp))
            z_base = enc.head_list(x_stem * mod_val)

        z_tx = self.hier_tx(z_base, x_stem) if self.hier_tx is not None else z_base

        if self.subspace_modem is not None:
            return self.subspace_modem.embed_tx(z_tx)

        # Standard unit-complex-power normalization (no side-channel pwr sent!)
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-12)
        z_norm = z_tx / pwr.sqrt()[:, None, None]
        flat = z_norm.reshape(B, 2 * K)
        return torch.stack([flat[:, :K], flat[:, K:]], dim=-1)

    def decode_blind_rx(self, y_rx_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode ONLY from raw antenna symbols y_rx_iq (B, K, 2) — ZERO external SNR/CSI/pwr inputs!"""
        B, K, _ = y_rx_iq.shape
        C = self.latent_channels
        L = (2 * K) // C
        dec = self.backbone.decoder

        if self.method == "frozen_blind":
            tot_pwr = (y_rx_iq[:, :, 0].square() + y_rx_iq[:, :, 1].square()).mean(dim=1).clamp_min(0.5)
            u_in = torch.cat([y_rx_iq[:, :, 0], y_rx_iq[:, :, 1]], dim=1).reshape(B, L, C) * (2.82 / tot_pwr.sqrt()[:, None, None])
            x = dec.head_list(u_in)
            snr_c0 = torch.full((B, 1), self.base_c0_db, device=u_in.device, dtype=u_in.dtype)
            temp = None
            for i in range(dec.layer_num):
                temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
                temp = temp * dec.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
            x = x * dec.sigmoid(dec.sm_list[-1](temp))
            for layer in dec.layers:
                x = layer(x)
            recon = x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
            return recon, snr_c0.squeeze(1)

        if self.subspace_modem is not None:
            ch_v2, cr_hat_db = self.subspace_modem.blind_demod_and_equalize(y_rx_iq, C)
        else:
            assert self.legacy_estimator is not None
            ch_v2, cr_hat_db = self.legacy_estimator(y_rx_iq, C)

        assert self.blind_dec is not None
        u_cal, phy_map, cond, g_ones = self.blind_dec.forward_frontend(ch_v2, cr_hat_db, base_c0_db=self.base_c0_db)

        x = dec.head_list(u_cal)
        cond_bm = cr_hat_db.reshape(B, 1) + self.blind_dec.bm_cond_mlp(cond) * SNR_SCALE
        temp = None
        for i in range(dec.layer_num):
            temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
            temp = temp * dec.bm_list[i](cond_bm).unsqueeze(1).expand(-1, L, -1)
        x = x * dec.sigmoid(dec.sm_list[-1](temp))

        gb = self.blind_dec.post_head_film(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]
        for i_layer, layer in enumerate(dec.layers):
            H_s, W_s = resolutions[i_layer]
            n_blk = len(layer.blocks)
            for b_idx, blk in enumerate(layer.blocks):
                x = blk(x)
                if i_layer == 1 and b_idx == (n_blk // 2 - 1):
                    x = self.blind_dec.stage_adapters["s1_mid"](x, H_s, W_s, phy_map, cond, g_ones)
            key = f"s{i_layer}_end"
            if key in self.blind_dec.stage_adapters:
                x = self.blind_dec.stage_adapters[key](x, H_s, W_s, phy_map, cond, g_ones)
            if layer.upsample is not None:
                x = layer.upsample(x)

        recon = x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
        return recon, cr_hat_db

    def forward_blind(
        self,
        image_or_stem: torch.Tensor,
        true_snr_db: torch.Tensor | float,
        *,
        channel_type: str = "ofdm_complex",
        rho_f: float = 0.85,
        h_rb_override: torch.Tensor | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        s_tx_iq = self.encode_blind_tx(x_stem)
        y_rx_iq, truth_diag = physical_blind_channel_forward(
            s_tx_iq,
            true_snr_db,
            channel_type=channel_type,
            num_rbs=self.num_rbs,
            rho_f=rho_f,
            h_rb_override=h_rb_override,
            standard_noise_iq=standard_noise_iq,
        )
        recon, cr_hat_db = self.decode_blind_rx(y_rx_iq)
        return {
            "recon": recon,
            "cr_hat_db": cr_hat_db,
            "s_tx_iq": s_tx_iq,
            **truth_diag,
        }
