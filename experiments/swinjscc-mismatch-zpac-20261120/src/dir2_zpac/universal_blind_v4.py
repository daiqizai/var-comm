"""V4 Universal Blind Self-Probing, Multi-User Broadcast & Hardware-Standard Digital QAM Self-Probing JSCC.

Key V4 Advances over V3 (SPH-ZPAC):
1. U-SPH-ZPAC (Upgraded Analog Universal & Broadcast Transceiver):
   - Exact RX orthogonal subspace projection P_{span(u1,u2,u3)^perp}(y_m) that cancels 100.000% of u1 & u3
     self-probing waveforms for ANY unknown complex fading h_m without residual subtraction leakage;
   - Coherence-adaptive blind RB Wiener filter that automatically detects flat AWGN vs. slow/fast multipath
     fading from inter-RB finite differences |h_tilde_{m+1} - h_tilde_m|^2, seamlessly blending 32-RB global
     coherent averaging (+15 dB extra gain on flat channels) with 5-tap local Wiener + 1D Conv smoothing;
   - 32-RB Maximal-Ratio Combined (MRC) complex phase-angle atan2(Im Z3, Re Z3) blind recovery of per-image
     latent norm p_img, invariant to MMSE amplitude scaling;
   - Supports BOTH Official SwinJSCC_w_SA_AWGN and Official SwinJSCC_w_SA_Rayleigh backbones, and simultaneous
     1-to-U Heterogeneous Wireless Broadcast from a single transmitted packet.

2. DD-QAM-SPH (Hardware-Compliant Standard 16-QAM & 64-QAM Zero-Pilot Blind Digital Transceiver):
   - Every single transmitted symbol s_{m,n} (all K = 12,288 symbols, 0 pilot slots!) is an EXACT discrete
     square M-QAM constellation point in C_16 or C_64;
   - Bussgang dithered pre-quantization subspace bias along u1 & u3 breaks the 4-fold pi/2 QAM rotational
     ambiguity and conveys p_img blindly without leaving the standard QAM grid;
   - 2-iteration differentiable soft-PAM Decision-Directed (DD) virtual-pilot refinement turns all 384 QAM
     data symbols per RB into virtual pilots and estimates blind noise variance from QAM lattice residuals;
   - Benchmarked head-to-head against IEEE TCOM Nov. 2025 style Digital Blind Black-Box JSCC (TCOM25-QAM).
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blind_v3 import (
    BlindMultiScaleDecoderCalibrator,
    OpenLoopHierarchicalTX,
    build_walsh3_basis,
)
from .models import SNR_CENTER, SNR_SCALE
from .phy_v2 import ChannelV2Output, expand_rb_to_symbols, sample_correlated_rbs


def get_pam_levels(qam_order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return exact unit-complex-energy 1D PAM levels A_M (mean(a^2) == 0.5, so E[|s_re + j s_im|^2] == 1.0)."""
    if qam_order == 16:
        pts = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device, dtype=dtype) / math.sqrt(10.0)
    elif qam_order == 64:
        pts = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], device=device, dtype=dtype) / math.sqrt(42.0)
    else:
        raise ValueError(f"Unsupported qam_order={qam_order}, expected 16 or 64")
    return pts


def ste_pam_quantize(x: torch.Tensor, levels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    """Hard nearest-neighbor PAM quantizer in forward pass + smooth soft-assignment gradient in backward pass."""
    dist_sq = (x.unsqueeze(-1) - levels).square()  # (..., L_pam)
    hard_idx = dist_sq.argmin(dim=-1)
    q_hard = levels[hard_idx]
    if not x.requires_grad:
        return q_hard
    weights = F.softmax(-dist_sq / max(temperature, 1e-4), dim=-1)
    q_soft = (weights * levels).sum(dim=-1)
    return q_soft + (q_hard - q_soft).detach()


def physical_blind_channel_v4(
    s_tx_iq: torch.Tensor,         # (B, K, 2) transmitted complex symbols
    snr_db: torch.Tensor | float,  # TRUE physical Es/N0 in dB (strictly hidden from TX & RX!)
    *,
    channel_type: str = "ofdm_complex",
    num_rbs: int = 32,
    rho_f: float = 0.85,
    h_rb_override: torch.Tensor | None = None,
    standard_noise_iq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact physical Es/N0 wireless channel: noise scales with realized per-frame symbol energy E[|s|^2]."""
    B, K, two = s_tx_iq.shape
    assert two == 2
    device = s_tx_iq.device
    dtype = s_tx_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )
    frame_es = (s_tx_iq[:, :, 0].square() + s_tx_iq[:, :, 1].square()).mean(dim=1).clamp_min(1e-8)  # (B,)
    sigma_n_sq = frame_es * torch.pow(10.0, -snr_t / 10.0)
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

    y_rx_iq = torch.stack([y_r, y_i], dim=-1)
    return y_rx_iq, {
        "true_snr_db": snr_t,
        "frame_es": frame_es,
        "true_sigma_n_sq": sigma_n_sq,
        "true_hr_sym": hr_sym,
        "true_hi_sym": hi_sym,
    }


def coherence_adaptive_wiener_rb(
    htilde_r: torch.Tensor,   # (B, M)
    htilde_i: torch.Tensor,   # (B, M)
    sigma_e_raw: torch.Tensor,  # (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Automatically detect flat AWGN vs. slow/fast frequency-selective OFDM and smooth htilde optimally."""
    device = htilde_r.device
    dtype = htilde_r.dtype
    M = htilde_r.shape[1]

    # Inter-RB finite difference excess variance: E[|dh|^2] = 2(1 - rho_f) + 2*sigma_e_raw
    diff_sq = (htilde_r[:, 1:] - htilde_r[:, :-1]).square() + (htilde_i[:, 1:] - htilde_i[:, :-1]).square()
    delta_f = F.relu(diff_sq.mean(dim=1) - 2.0 * sigma_e_raw)  # ~0 on flat AWGN, ~0.30 on rho_f=0.85, ~0.90 on rho_f=0.55

    # 1. Global flat-channel coherent average across all M RBs (variance sigma_e_raw / M!)
    hbar_r = htilde_r.mean(dim=1, keepdim=True).expand(-1, M)
    hbar_i = htilde_i.mean(dim=1, keepdim=True).expand(-1, M)

    # 2. Local 5-tap triangular Wiener smoother across adjacent RBs (sharp at high SNR, smooth at low SNR!)
    k5 = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=device, dtype=dtype).reshape(1, 1, 5) / 9.0
    hsmooth_r = F.conv1d(F.pad(htilde_r.unsqueeze(1), (2, 2), mode="replicate"), k5).squeeze(1)
    hsmooth_i = F.conv1d(F.pad(htilde_i.unsqueeze(1), (2, 2), mode="replicate"), k5).squeeze(1)

    w_raw_local = (1.0 / (1.0 + 3.0 * sigma_e_raw))[:, None]
    h_loc_r = (w_raw_local * htilde_r + (1.0 - w_raw_local) * hsmooth_r) / (1.0 + 0.30 * sigma_e_raw[:, None])
    h_loc_i = (w_raw_local * htilde_i + (1.0 - w_raw_local) * hsmooth_i) / (1.0 + 0.30 * sigma_e_raw[:, None])

    # Exponential selectivity gate: ~1.0 on flat AWGN (delta_f=0), ~0.000000 on OFDM (delta_f >= 0.25)!
    w_flat = torch.exp(-delta_f / 0.015)[:, None]
    hw_r = w_flat * hbar_r + (1.0 - w_flat) * h_loc_r
    hw_i = w_flat * hbar_i + (1.0 - w_flat) * h_loc_i

    wf_1d = w_flat.squeeze(1)
    sigma_e_eff = wf_1d.square() * (sigma_e_raw / float(M)) + (1.0 - wf_1d).square() * (0.42 * sigma_e_raw)
    return hw_r, hw_i, sigma_e_eff, delta_f


class UpgradedSubspaceModemV4(nn.Module):
    """V4 Residual-Free Orthogonal Self-Probing Modem with MRC Phase-Angle Norm Recovery & Coherence-Adaptive Wiener."""

    def __init__(self, num_rbs: int = 32, eta1_init: float = 0.028, eta3_init: float = 0.004):
        super().__init__()
        self.num_rbs = int(num_rbs)
        inv_sig = math.log((eta1_init - 0.012) / (0.070 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_sig)))
        self.eta3 = float(eta3_init)

        # 7-channel RB refiner: [htilde_r, htilde_i, e_rb, log_sig_n, hw_r, hw_i, delta_f]
        self.rb_refiner = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.rb_refiner[-1].weight)
        nn.init.zeros_(self.rb_refiner[-1].bias)

    @property
    def eta1(self) -> torch.Tensor:
        return 0.012 + (0.070 - 0.012) * torch.sigmoid(self.eta1_raw)

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        B, L, C = z_raw.shape
        K = (L * C) // 2
        M = self.num_rbs
        N_b = K // M
        device, dtype = z_raw.device, z_raw.dtype

        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        p_img = pwr.sqrt()
        z_unit = z_raw / p_img[:, None, None]

        flat = z_unit.reshape(B, 2 * K)
        s_r = flat[:, :K].reshape(B, M, N_b)
        s_i = flat[:, K:].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)  # (3, N_b)
        u1, _, u3 = basis[0], basis[1], basis[2]

        proj_r = torch.einsum("bmn,kn->bmk", s_r, basis)
        proj_i = torch.einsum("bmn,kn->bmk", s_i, basis)
        s_perp_r = s_r - torch.einsum("bmk,kn->bmn", proj_r, basis)
        s_perp_i = s_i - torch.einsum("bmk,kn->bmn", proj_i, basis)

        eta1 = self.eta1
        data_pwr_target = 1.0 - eta1 - self.eta3
        cur_data_pwr = (s_perp_r.square() + s_perp_i.square()).mean(dim=(1, 2)).clamp_min(1e-8)
        scale_data = (data_pwr_target / cur_data_pwr).sqrt()[:, None, None]

        s_data_r = s_perp_r * scale_data
        s_data_i = s_perp_i * scale_data

        # Phase-angle encoding of p_img in (-1.25, +1.25) rad: cos(theta_p) > 0.315 > 0 always!
        theta_p = 1.25 * torch.tanh(0.40 * (p_img - 2.82))
        cos_p = torch.cos(theta_p)[:, None, None]
        sin_p = torch.sin(theta_p)[:, None, None]

        amp1 = (eta1 * float(N_b)).sqrt()
        amp3 = math.sqrt(self.eta3 * float(N_b))

        tx_r = s_data_r + amp1 * u1[None, None, :] + (amp3 * cos_p) * u3[None, None, :]
        tx_i = s_data_i + (amp3 * sin_p) * u3[None, None, :]
        return torch.stack([tx_r.reshape(B, K), tx_i.reshape(B, K)], dim=-1)

    def blind_demod_and_equalize(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, K, _ = y_rx_iq.shape
        M = self.num_rbs
        N_b = K // M
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        y_r = y_rx_iq[:, :, 0].reshape(B, M, N_b)
        y_i = y_rx_iq[:, :, 1].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)
        u1, u2, u3 = basis[0], basis[1], basis[2]

        # 1. Exact MVUE blind noise variance from u2 orthogonal null
        q2_r = torch.einsum("bmn,n->bm", y_r, u2)
        q2_i = torch.einsum("bmn,n->bm", y_i, u2)
        sigma_n_hat = (q2_r.square() + q2_i.square()).mean(dim=1).clamp(1e-5, 12.0)

        # 2. Coherence-adaptive Wiener RB channel sensing from u1
        eta1 = self.eta1
        amp1 = (eta1 * float(N_b)).sqrt()
        htilde_r = torch.einsum("bmn,n->bm", y_r, u1) / amp1
        htilde_i = torch.einsum("bmn,n->bm", y_i, u1) / amp1
        sigma_e_raw = sigma_n_hat / (eta1 * float(N_b))

        hw_r, hw_i, sigma_e_eff, delta_f = coherence_adaptive_wiener_rb(htilde_r, htilde_i, sigma_e_raw)
        e_rb = (y_r.square() + y_i.square()).mean(dim=-1)
        log_sig_feat = torch.log(sigma_n_hat.clamp_min(1e-5))[:, None].expand(B, M).div(4.0)
        df_feat = delta_f[:, None].expand(B, M)
        delta_h = self.rb_refiner(torch.stack([htilde_r, htilde_i, e_rb, log_sig_feat, hw_r, hw_i, df_feat], dim=1))
        hhat_r_rb = hw_r + delta_h[:, 0, :]
        hhat_i_rb = hw_i + delta_h[:, 1, :]

        # 3. 32-RB Maximal-Ratio Combined (MRC) Phase-Angle atan2 recovery of p_img (100% scale-invariant!)
        q3_r = torch.einsum("bmn,n->bm", y_r, u3)
        q3_i = torch.einsum("bmn,n->bm", y_i, u3)
        z3_r = (hhat_r_rb * q3_r + hhat_i_rb * q3_i).sum(dim=1)
        z3_i = (hhat_r_rb * q3_i - hhat_i_rb * q3_r).sum(dim=1)
        theta_hat = torch.atan2(z3_i, z3_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)

        # 4. EXACT Orthogonal Subspace Projection P_{span(u1,u2,u3)^perp}(y_m): ZERO residual pilot leak!
        proj_yr = torch.einsum("bmn,kn->bmk", y_r, basis)
        proj_yi = torch.einsum("bmn,kn->bmk", y_i, basis)
        y_perp_r = y_r - torch.einsum("bmk,kn->bmn", proj_yr, basis)
        y_perp_i = y_i - torch.einsum("bmk,kn->bmn", proj_yi, basis)

        # 5. Gain-Unbiased Blind MMSE Equalization on pure orthogonal semantic subspace
        data_pwr = 1.0 - eta1 - self.eta3
        hhat_sq_rb = hhat_r_rb.square() + hhat_i_rb.square()
        hhat_abs_rb = hhat_sq_rb.clamp_min(1e-10).sqrt()
        # Account for data_pwr fraction in MMSE denominator sigma_n^2 / data_pwr
        eff_var = ((sigma_n_hat / data_pwr) + sigma_e_eff)[:, None]
        denom_rb = (hhat_sq_rb + eff_var).clamp_min(1e-6)
        alpha_rb = hhat_sq_rb / denom_rb
        alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.05)

        cos_rb = (hhat_r_rb / hhat_abs_rb)[:, :, None]
        sin_rb = (hhat_i_rb / hhat_abs_rb)[:, :, None]
        y_derot_r = cos_rb * y_perp_r + sin_rb * y_perp_i
        y_derot_i = -sin_rb * y_perp_r + cos_rb * y_perp_i

        # Rescale by sqrt((N_b - 3) / (N_b * data_pwr)) so pre-subspace variance matches unit semantic variance!
        dof_comp = math.sqrt(float(N_b - 3) / float(N_b))
        data_scale = (1.0 / data_pwr).sqrt() * dof_comp
        w_ub_rb = ((hhat_abs_rb / denom_rb) / alpha_bar)[:, :, None] * data_scale

        s_hat_r = y_derot_r * w_ub_rb
        s_hat_i = y_derot_i * w_ub_rb

        u_mmse_norm = torch.cat([s_hat_r.reshape(B, K), s_hat_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        sinr_wb = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1)).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-4.0, 22.0)

        h_abs_iq = expand_rb_to_symbols(hhat_abs_rb, K)
        alpha_ratio_iq = expand_rb_to_symbols(alpha_rb / alpha_bar, K)
        sigma_post_iq = expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K)
        phase_unc_iq = expand_rb_to_symbols((sigma_e_eff[:, None] / (hhat_sq_rb + sigma_e_eff[:, None] + 1e-6)).sqrt(), K)
        psi_csi = (hhat_abs_rb - 1.0).square().mean(dim=1) + sigma_e_eff

        return (
            ChannelV2Output(
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
            ),
            cr_hat_db,
        )


class DecisionDirectedQAMModemV4(nn.Module):
    """Hardware-Standard 16-QAM & 64-QAM Zero-Pilot Blind Modem (Bussgang Subspace + 2-Step Soft-DD Virtual Pilots)."""

    def __init__(self, qam_order: int = 16, num_rbs: int = 32):
        super().__init__()
        self.qam_order = int(qam_order)
        self.num_rbs = int(num_rbs)

        # Lloyd-Max optimal pre-quantization dynamic-range scaling (prevents outer-ring clipping on heavy-tailed latents!)
        g_init = 0.78 if qam_order == 16 else 0.70
        b1_init = 0.22 if qam_order == 16 else 0.18
        kap_init = 1.00 if qam_order == 16 else 1.00
        self.q_null_floor = 0.052 if qam_order == 16 else 0.013
        self.ste_temp = 0.12 if qam_order == 16 else 0.045

        self.log_g_qam = nn.Parameter(torch.tensor(math.log(g_init)))
        inv_b1 = math.log((b1_init - 0.12) / (0.36 - b1_init))
        self.beta1_raw = nn.Parameter(torch.tensor(float(inv_b1)))
        self.beta3 = 0.09
        self.bussgang_kappa = nn.Parameter(torch.tensor(float(kap_init)))

        self.rb_refiner = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.rb_refiner[-1].weight)
        nn.init.zeros_(self.rb_refiner[-1].bias)

    @property
    def g_qam(self) -> torch.Tensor:
        return torch.exp(self.log_g_qam.clamp(-0.8, 0.5))

    @property
    def beta1(self) -> torch.Tensor:
        return 0.12 + (0.36 - 0.12) * torch.sigmoid(self.beta1_raw)

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Map z_raw to 100% EXACT discrete M-QAM constellation points in C_M (zero pilot slots!)."""
        B, L, C = z_raw.shape
        K = (L * C) // 2
        M = self.num_rbs
        N_b = K // M
        device, dtype = z_raw.device, z_raw.dtype

        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        p_img = pwr.sqrt()
        z_unit = z_raw / p_img[:, None, None]

        flat = z_unit.reshape(B, 2 * K)
        s_r = flat[:, :K].reshape(B, M, N_b)
        s_i = flat[:, K:].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)
        u1, _, u3 = basis[0], basis[1], basis[2]

        proj_r = torch.einsum("bmn,kn->bmk", s_r, basis)
        proj_i = torch.einsum("bmn,kn->bmk", s_i, basis)
        s_perp_r = s_r - torch.einsum("bmk,kn->bmn", proj_r, basis)
        s_perp_i = s_i - torch.einsum("bmk,kn->bmn", proj_i, basis)

        cur_pwr = (s_perp_r.square() + s_perp_i.square()).mean(dim=(1, 2)).clamp_min(1e-8)
        s_perp_r = s_perp_r / cur_pwr.sqrt()[:, None, None]
        s_perp_i = s_perp_i / cur_pwr.sqrt()[:, None, None]

        theta_p = 1.25 * torch.tanh(0.40 * (p_img - 2.82))
        cos_p = torch.cos(theta_p)[:, None, None]
        sin_p = torch.sin(theta_p)[:, None, None]

        rt_nb = math.sqrt(float(N_b))
        g_q = self.g_qam
        b1 = self.beta1

        pre_r = g_q * s_perp_r + (b1 * rt_nb) * u1[None, None, :] + (self.beta3 * rt_nb * cos_p) * u3[None, None, :]
        pre_i = g_q * s_perp_i + (self.beta3 * rt_nb * sin_p) * u3[None, None, :]

        levels = get_pam_levels(self.qam_order, device=device, dtype=dtype)
        q_r = ste_pam_quantize(pre_r, levels, temperature=self.ste_temp)
        q_i = ste_pam_quantize(pre_i, levels, temperature=self.ste_temp)
        return torch.stack([q_r.reshape(B, K), q_i.reshape(B, K)], dim=-1)

    def blind_demod_and_equalize(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, K, _ = y_rx_iq.shape
        M = self.num_rbs
        N_b = K // M
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        y_r = y_rx_iq[:, :, 0].reshape(B, M, N_b)
        y_i = y_rx_iq[:, :, 1].reshape(B, M, N_b)

        basis = build_walsh3_basis(N_b, device=device, dtype=dtype)
        u1, u2, u3 = basis[0], basis[1], basis[2]
        levels = get_pam_levels(self.qam_order, device=device, dtype=dtype)  # (L_pam,)

        # Stage 1: Coarse Bussgang Subspace Sensing (breaks 4-fold pi/2 QAM ambiguity unambiguously!)
        e_rb = (y_r.square() + y_i.square()).mean(dim=-1)  # ~ |h_m|^2 * E_s + sigma_n^2
        q2_r = torch.einsum("bmn,n->bm", y_r, u2)
        q2_i = torch.einsum("bmn,n->bm", y_i, u2)
        v2_total = (q2_r.square() + q2_i.square()).mean(dim=1)
        # Subtract approximate quantization null floor scaled by average received energy
        est_h_sq_mean = (e_rb.mean(dim=1) - v2_total).clamp(0.2, 2.5)
        sigma_n_cur = (v2_total - self.q_null_floor * est_h_sq_mean).clamp(1e-3, 12.0)

        eff_amp1 = (self.bussgang_kappa.clamp(0.5, 1.3) * self.beta1) * math.sqrt(float(N_b))
        ht0_r = torch.einsum("bmn,n->bm", y_r, u1) / eff_amp1
        ht0_i = torch.einsum("bmn,n->bm", y_i, u1) / eff_amp1
        sigma_e0 = (sigma_n_cur + self.q_null_floor * est_h_sq_mean) / eff_amp1.square()

        h_cur_r, h_cur_i, sigma_e_cur, delta_f = coherence_adaptive_wiener_rb(ht0_r, ht0_i, sigma_e0)
        h0_r, h0_i = h_cur_r, h_cur_i
        w_flat = torch.exp(-delta_f / 0.015)[:, None]
        k3 = torch.tensor([0.2, 0.6, 0.2], device=device, dtype=dtype).reshape(1, 1, 3)

        # Stage 2: 2-Iteration Differentiable Soft-PAM Decision-Directed Virtual-Pilot Refinement over all 384 QAM symbols!
        for _ in range(2):
            h_sq = (h_cur_r.square() + h_cur_i.square()).clamp_min(1e-6)  # (B, M)
            denom = (h_sq + 0.5 * sigma_n_cur[:, None]).clamp_min(1e-5)
            # Equalize raw y_m to QAM symbol space
            st_r = ((h_cur_r[:, :, None] * y_r + h_cur_i[:, :, None] * y_i) / denom[:, :, None])
            st_i = ((h_cur_r[:, :, None] * y_i - h_cur_i[:, :, None] * y_r) / denom[:, :, None])

            tau_1d = (0.5 * sigma_n_cur[:, None] / h_sq).clamp(0.012, 2.0)  # (B, M)
            # Factorized 1D soft-PAM posterior expectation E[a | st]
            w_r = F.softmax(-(st_r.unsqueeze(-1) - levels).square() / (2.0 * tau_1d[:, :, None, None]), dim=-1)
            w_i = F.softmax(-(st_i.unsqueeze(-1) - levels).square() / (2.0 * tau_1d[:, :, None, None]), dim=-1)
            sbar_r = (w_r * levels).sum(dim=-1)  # (B, M, N_b)
            sbar_i = (w_i * levels).sum(dim=-1)  # (B, M, N_b)

            p_sbar = (sbar_r.square() + sbar_i.square()).mean(dim=-1)  # (B, M) in [0, ~1.0]
            # Least-squares / LMMSE virtual-pilot channel estimate: sum_n conj(sbar_n) * y_n / sum_n |sbar_n|^2
            num_hr = (sbar_r * y_r + sbar_i * y_i).mean(dim=-1)
            num_hi = (sbar_r * y_i - sbar_i * y_r).mean(dim=-1)
            hdd_raw_r = num_hr / (p_sbar + sigma_n_cur[:, None] / float(N_b)).clamp_min(0.05)
            hdd_raw_i = num_hi / (p_sbar + sigma_n_cur[:, None] / float(N_b)).clamp_min(0.05)

            hdd_loc_r = F.conv1d(F.pad(hdd_raw_r.unsqueeze(1), (1, 1), mode="replicate"), k3).squeeze(1)
            hdd_loc_i = F.conv1d(F.pad(hdd_raw_i.unsqueeze(1), (1, 1), mode="replicate"), k3).squeeze(1)
            hdd_sm_r = w_flat * hdd_raw_r.mean(dim=1, keepdim=True) + (1.0 - w_flat) * hdd_loc_r
            hdd_sm_i = w_flat * hdd_raw_i.mean(dim=1, keepdim=True) + (1.0 - w_flat) * hdd_loc_i

            gamma_dd = ((p_sbar - 0.28) / 0.52).clamp(0.0, 0.85)
            h_cur_r = (1.0 - gamma_dd) * h0_r + gamma_dd * hdd_sm_r
            h_cur_i = (1.0 - gamma_dd) * h0_i + gamma_dd * hdd_sm_i

            # Blind lattice-residual noise variance!
            res_r = y_r - (h_cur_r[:, :, None] * sbar_r - h_cur_i[:, :, None] * sbar_i)
            res_i = y_i - (h_cur_r[:, :, None] * sbar_i + h_cur_i[:, :, None] * sbar_r)
            lat_noise = (res_r.square() + res_i.square()).mean(dim=(1, 2)).clamp(1e-4, 12.0)
            conf_mean = gamma_dd.mean(dim=1)
            sigma_n_cur = ((1.0 - 0.70 * conf_mean) * sigma_n_cur + (0.70 * conf_mean) * lat_noise).clamp(1e-4, 12.0)
            sigma_e_cur = (1.0 - conf_mean) * sigma_e_cur + conf_mean * (sigma_n_cur / float(N_b))

        log_sig_feat = torch.log(sigma_n_cur.clamp_min(1e-5))[:, None].expand(B, M).div(4.0)
        df_feat = delta_f[:, None].expand(B, M)
        delta_h = self.rb_refiner(torch.stack([ht0_r, ht0_i, h_cur_r, h_cur_i, e_rb, log_sig_feat, p_sbar, df_feat], dim=1))
        hhat_r_rb = h_cur_r + delta_h[:, 0, :]
        hhat_i_rb = h_cur_i + delta_h[:, 1, :]

        # Stage 3: MRC Phase-Angle atan2 recovery of p_img + Orthogonal Subspace Strip + Unbiased Blind MMSE
        q3_r = torch.einsum("bmn,n->bm", y_r, u3)
        q3_i = torch.einsum("bmn,n->bm", y_i, u3)
        z3_r = (hhat_r_rb * q3_r + hhat_i_rb * q3_i).sum(dim=1)
        z3_i = (hhat_r_rb * q3_i - hhat_i_rb * q3_r).sum(dim=1)
        theta_hat = torch.atan2(z3_i, z3_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)

        proj_yr = torch.einsum("bmn,kn->bmk", y_r, basis)
        proj_yi = torch.einsum("bmn,kn->bmk", y_i, basis)
        y_perp_r = y_r - torch.einsum("bmk,kn->bmn", proj_yr, basis)
        y_perp_i = y_i - torch.einsum("bmk,kn->bmn", proj_yi, basis)

        hhat_sq_rb = hhat_r_rb.square() + hhat_i_rb.square()
        hhat_abs_rb = hhat_sq_rb.clamp_min(1e-10).sqrt()
        # Include QAM quantization distortion floor (~0.5 * q_null_floor) in effective MMSE variance!
        eff_var = (sigma_n_cur + sigma_e_cur + 0.5 * self.q_null_floor)[:, None]
        denom_rb = (hhat_sq_rb + eff_var).clamp_min(1e-6)
        alpha_rb = hhat_sq_rb / denom_rb
        alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.05)

        cos_rb = (hhat_r_rb / hhat_abs_rb)[:, :, None]
        sin_rb = (hhat_i_rb / hhat_abs_rb)[:, :, None]
        y_derot_r = cos_rb * y_perp_r + sin_rb * y_perp_i
        y_derot_i = -sin_rb * y_perp_r + cos_rb * y_perp_i

        inv_buss = 1.0 / (self.bussgang_kappa.clamp(0.5, 1.3) * self.g_qam)
        w_ub_rb = ((hhat_abs_rb / denom_rb) / alpha_bar)[:, :, None] * inv_buss
        s_hat_r = y_derot_r * w_ub_rb
        s_hat_i = y_derot_i * w_ub_rb

        u_mmse_norm = torch.cat([s_hat_r.reshape(B, K), s_hat_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        sinr_wb = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1)).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-4.0, 22.0)

        return (
            ChannelV2Output(
                u_restored=u_mmse_restored,
                u_mmse_restored=u_mmse_restored,
                u_norm=u_mmse_norm,
                z_norm=u_mmse_norm,
                pwr=p_img_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=expand_rb_to_symbols(hhat_abs_rb, K),
                alpha_ratio_iq=expand_rb_to_symbols(alpha_rb / alpha_bar, K),
                sigma_post_iq=expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K),
                phase_unc_iq=expand_rb_to_symbols((sigma_e_cur[:, None] / (hhat_sq_rb + sigma_e_cur[:, None] + 1e-6)).sqrt(), K),
                psi_csi=(hhat_abs_rb - 1.0).square().mean(dim=1) + sigma_e_cur,
            ),
            cr_hat_db,
        )


class TCOM25DigitalBlindModem(nn.Module):
    """IEEE TCOM Nov. 2025 style End-to-End Blind Digital QAM Mapper & Moment-Assisted Blind Demodulator."""

    def __init__(self, qam_order: int = 16, num_rbs: int = 32, latent_ch: int = 96):
        super().__init__()
        self.qam_order = int(qam_order)
        self.num_rbs = int(num_rbs)
        self.ste_temp = 0.12 if qam_order == 16 else 0.045
        g_init = 0.78 if qam_order == 16 else 0.70
        self.log_g_qam = nn.Parameter(torch.tensor(math.log(g_init)))

        # Learnable constellation symmetry-breaking bias per channel (allows E2E blind training to discover implicit pilots!)
        self.sym_break = nn.Parameter(torch.zeros(1, 256, latent_ch))

        # Blind RB equalizer network from 1st/2nd/4th-order QAM constellation moments [mean_r, mean_i, e_rb, m4_r, m4_i, std_rb]
        self.rb_net = nn.Sequential(
            nn.Conv1d(6, 48, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(48, 3, kernel_size=1),  # [dh_r, dh_i, d_log_sig]
        )
        nn.init.zeros_(self.rb_net[-1].weight)
        nn.init.zeros_(self.rb_net[-1].bias)

        self.snr_scale_head = nn.Sequential(
            nn.Linear(48 + 2, 48),
            nn.GELU(),
            nn.Linear(48, 2),  # [delta_snr_db, delta_log_p]
        )
        nn.init.zeros_(self.snr_scale_head[-1].weight)
        nn.init.zeros_(self.snr_scale_head[-1].bias)
        self.feat_tap = nn.Sequential(
            nn.Conv1d(6, 48, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        B, L, C = z_raw.shape
        K = (L * C) // 2
        device, dtype = z_raw.device, z_raw.dtype
        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        z_unit = z_raw / pwr.sqrt()[:, None, None] + 0.25 * torch.tanh(self.sym_break)
        g_q = torch.exp(self.log_g_qam.clamp(-0.8, 0.5))
        flat = (z_unit * g_q).reshape(B, 2 * K)
        levels = get_pam_levels(self.qam_order, device=device, dtype=dtype)
        q_r = ste_pam_quantize(flat[:, :K], levels, temperature=self.ste_temp)
        q_i = ste_pam_quantize(flat[:, K:], levels, temperature=self.ste_temp)
        return torch.stack([q_r, q_i], dim=-1)

    def blind_demod_and_equalize(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, K, _ = y_rx_iq.shape
        M = self.num_rbs
        N_b = K // M
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        y_r = y_rx_iq[:, :, 0].reshape(B, M, N_b)
        y_i = y_rx_iq[:, :, 1].reshape(B, M, N_b)

        mean_r = y_r.mean(dim=-1)
        mean_i = y_i.mean(dim=-1)
        mag_sq = y_r.square() + y_i.square()
        e_rb = mag_sq.mean(dim=-1)
        std_rb = mag_sq.std(dim=-1)
        # 4th-order Viterbi-Viterbi constellation moment Re((y_r + j y_i)^4) & Im((y_r + j y_i)^4) normalized
        y2_r = y_r.square() - y_i.square()
        y2_i = 2.0 * y_r * y_i
        m4_r = (y2_r.square() - y2_i.square()).mean(dim=-1) / (e_rb.square() + 1e-4)
        m4_i = (2.0 * y2_r * y2_i).mean(dim=-1) / (e_rb.square() + 1e-4)

        feats = torch.stack([mean_r, mean_i, e_rb.clamp(0.0, 10.0), m4_r.clamp(-3.0, 3.0), m4_i.clamp(-3.0, 3.0), std_rb.clamp(0.0, 10.0)], dim=1)
        rb_out = self.rb_net(feats)
        # Start at identity channel (1 + j0) when zero-initialized
        hhat_r = 1.0 + rb_out[:, 0, :]
        hhat_i = rb_out[:, 1, :]

        tot_pwr = e_rb.mean(dim=1).clamp_min(0.2)
        snr_init = (-10.0 * torch.log10((tot_pwr - 1.0).clamp_min(0.02))).clamp(-4.0, 18.0)
        pooled = self.feat_tap(feats).mean(dim=-1)
        g_out = self.snr_scale_head(torch.cat([pooled, ((snr_init - SNR_CENTER) / SNR_SCALE).unsqueeze(1), torch.log(tot_pwr).unsqueeze(1)], dim=-1))
        cr_hat_db = (snr_init + 4.0 * g_out[:, 0]).clamp(-4.0, 22.0)
        p_hat = (2.82 * torch.exp(g_out[:, 1].clamp(-1.0, 1.0))).clamp(0.5, 6.0)

        sigma_n_hat = torch.pow(10.0, -cr_hat_db / 10.0)
        h_sq = (hhat_r.square() + hhat_i.square()).clamp_min(1e-6)
        h_abs = h_sq.sqrt()
        denom = (h_sq + sigma_n_hat[:, None]).clamp_min(1e-5)
        alpha_rb = h_sq / denom
        alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.05)

        cos_rb = (hhat_r / h_abs)[:, :, None]
        sin_rb = (hhat_i / h_abs)[:, :, None]
        yd_r = cos_rb * y_r + sin_rb * y_i
        yd_i = -sin_rb * y_r + cos_rb * y_i
        w_ub = ((h_abs / denom) / alpha_bar)[:, :, None] / torch.exp(self.log_g_qam.clamp(-0.8, 0.5))

        sh_r = yd_r * w_ub
        sh_i = yd_i * w_ub
        u_norm = torch.cat([sh_r.reshape(B, K), sh_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_rest = u_norm * p_hat[:, None, None]

        return (
            ChannelV2Output(
                u_restored=u_rest,
                u_mmse_restored=u_rest,
                u_norm=u_norm,
                z_norm=u_norm,
                pwr=p_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=expand_rb_to_symbols(h_abs, K),
                alpha_ratio_iq=expand_rb_to_symbols(alpha_rb / alpha_bar, K),
                sigma_post_iq=expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K),
                phase_unc_iq=torch.zeros((B, K), device=device, dtype=dtype),
                psi_csi=(h_abs - 1.0).square().mean(dim=1) + 0.2,
            ),
            cr_hat_db,
        )


class UniversalBlindTransceiverV4(nn.Module):
    """Unified V4 Wrapper supporting:
      - u_sph: Upgraded Residual-Free Analog Universal & Broadcast Self-Probing Transceiver
      - dd_qam16_sph / dd_qam64_sph: Hardware-Standard 16-QAM / 64-QAM Zero-Pilot Blind Digital Transceiver
      - tcom25_qam16 / tcom25_qam64: IEEE TCOM Nov. 2025 style Digital Blind Black-Box JSCC
      - frozen_qam16 / frozen_qam64: Unmodified SwinJSCC + Standard QAM without pilots
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "u_sph",
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
        self.modem: nn.Module | None = None
        self.blind_dec: BlindMultiScaleDecoderCalibrator | None = None
        self.qam_order: int | None = None

        if self.method == "u_sph":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = UpgradedSubspaceModemV4(num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method in ("dd_qam16_sph", "dd_qam64_sph"):
            q = 16 if "16" in self.method else 64
            self.qam_order = q
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = DecisionDirectedQAMModemV4(qam_order=q, num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method in ("tcom25_qam16", "tcom25_qam64"):
            q = 16 if "16" in self.method else 64
            self.qam_order = q
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = TCOM25DigitalBlindModem(qam_order=q, num_rbs=num_rbs, latent_ch=self.latent_channels)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method in ("frozen_qam16", "frozen_qam64"):
            self.qam_order = 16 if "16" in self.method else 64
        else:
            raise ValueError(f"Unknown V4 method: {self.method}")

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_blind_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
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
        if self.modem is not None:
            return self.modem.embed_tx(z_tx)

        # Frozen QAM baseline
        assert self.qam_order is not None
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        g0 = 0.78 if self.qam_order == 16 else 0.70
        flat = (z_tx / pwr.sqrt()[:, None, None] * g0).reshape(B, 2 * K)
        levels = get_pam_levels(self.qam_order, device=x_stem.device, dtype=x_stem.dtype)
        q_r = ste_pam_quantize(flat[:, :K], levels)
        q_i = ste_pam_quantize(flat[:, K:], levels)
        return torch.stack([q_r, q_i], dim=-1)

    def decode_blind_rx(self, y_rx_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = y_rx_iq.shape
        C = self.latent_channels
        L = (2 * K) // C
        dec = self.backbone.decoder

        if self.modem is None:
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
            return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous(), snr_c0.squeeze(1)

        ch_v2, cr_hat_db = self.modem.blind_demod_and_equalize(y_rx_iq, C)
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

        return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous(), cr_hat_db

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
        y_rx_iq, truth_diag = physical_blind_channel_v4(
            s_tx_iq,
            true_snr_db,
            channel_type=channel_type,
            num_rbs=self.num_rbs,
            rho_f=rho_f,
            h_rb_override=h_rb_override,
            standard_noise_iq=standard_noise_iq,
        )
        recon, cr_hat_db = self.decode_blind_rx(y_rx_iq)
        return {"recon": recon, "cr_hat_db": cr_hat_db, "s_tx_iq": s_tx_iq, **truth_diag}
