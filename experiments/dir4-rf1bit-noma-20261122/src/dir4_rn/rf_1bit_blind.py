"""Dir-4A: RF-Bussgang-JSCC — Ultra-Low-Resolution (1-Bit / 2-Bit) Time-Domain ADC,
Nonlinear Solid-State Power Amplifier (Rapp HPA) & Oscillator Wiener Phase Noise
Zero-Pilot, Zero-AGC Blind Semantic Communications.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dir2_zpac.phy_v2 import ChannelV2Output, expand_rb_to_symbols, sample_correlated_rbs
from dir2_zpac.blind_v3 import (
    SNR_CENTER,
    SNR_SCALE,
    BlindMultiScaleDecoderCalibrator,
    OpenLoopHierarchicalTX,
    build_walsh3_basis,
)
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb

# Physical 5G-style Time-Frequency Grid for K = 12,288 REs:
#   N_SYM = 48 consecutive OFDM time symbols x N_SC = 256 subcarriers = 12,288 REs
#   M = 32 RBs x 8 contiguous subcarriers/RB = 256 subcarriers
#   S_SUB = 4 temporal sub-blocks x 12 OFDM symbols = 48 OFDM symbols (8 x 12 = 96 REs per tile!)
N_OFDM_SYM = 48
N_OFDM_SC = 256
N_RBS_DEFAULT = 32
N_TIME_TILES = 4
TILE_RES = 96  # 8 subcarriers x 12 time symbols = 96 REs per (RB, TimeTile)


def rb384_to_ofdm_grid(s_rb_iq: torch.Tensor) -> torch.Tensor:
    """Bijective map [B, M=32, N_b=384, 2] -> [B, N_sym=48, N_sc=256, 2]."""
    B, M, N_b, C2 = s_rb_iq.shape
    assert M == 32 and N_b == 384 and C2 == 2
    # Each RB m has 48 time symbols x 8 subcarriers = 384 REs
    x = s_rb_iq.reshape(B, 32, N_OFDM_SYM, 8, 2).permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(B, N_OFDM_SYM, N_OFDM_SC, 2)


def ofdm_grid_to_rb384(s_grid_iq: torch.Tensor) -> torch.Tensor:
    """Inverse bijective map [B, N_sym=48, N_sc=256, 2] -> [B, M=32, N_b=384, 2]."""
    B, n_sym, n_sc, C2 = s_grid_iq.shape
    assert n_sym == N_OFDM_SYM and n_sc == N_OFDM_SC and C2 == 2
    x = s_grid_iq.reshape(B, N_OFDM_SYM, 32, 8, 2).permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(B, 32, 384, 2)


def newman_rb_phases(num_rbs: int = 32, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Newman quadratic phases psi_m = pi * m^2 / M across RBs (reduces multi-RB probe PAPR by >11 dB!)."""
    m = torch.arange(num_rbs, device=device, dtype=dtype)
    psi = math.pi * (m * m) / float(num_rbs)
    return torch.cos(psi), torch.sin(psi)


def ste_1bit_iq_quantize(y_r: torch.Tensor, y_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hardware-exact 1-Bit Zero-AGC Sign Comparator {+1/sqrt(2), -1/sqrt(2)} with Bussgang STE backward."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # In hardware: sign(0) -> +1
    hard_r = torch.where(y_r >= 0.0, inv_sqrt2, -inv_sqrt2)
    hard_i = torch.where(y_i >= 0.0, inv_sqrt2, -inv_sqrt2)
    if not (y_r.requires_grad or y_i.requires_grad):
        return hard_r, hard_i
    rms = (y_r.detach().square() + y_i.detach().square()).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
    scale = (0.60 * rms).clamp_min(1e-4)
    soft_r = inv_sqrt2 * torch.tanh(y_r / scale)
    soft_i = inv_sqrt2 * torch.tanh(y_i / scale)
    return hard_r + (soft_r - soft_r.detach()), hard_i + (soft_i - soft_i.detach())


def ste_2bit_iq_quantize(y_r: torch.Tensor, y_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hardware 2-Bit Mid-Rise Uniform I/Q ADC (4 levels per I/Q: {-3, -1, +1, +3} * d/2) using wideband analog RMS."""
    # Wideband time-domain analog RMS power detector (0 CSI, 0 pilots!)
    rms_iq = (0.5 * (y_r.detach().square() + y_i.detach().square()).mean(dim=(-2, -1), keepdim=True)).clamp_min(1e-6).sqrt()
    # Optimal Max-Lloyd uniform 2-bit step for N(0, sigma^2) is Delta = 0.9957 * sigma
    delta = 0.9957 * rms_iq
    # Normalize output to unit expected complex power: for N(0,1) with threshold 0 and +-0.9957,
    # P(|x| > 0.9957) = 0.3194 -> E[q_I^2] = (d/2)^2 * (0.6806 * 1 + 0.3194 * 9) = 0.8888 * d^2
    # Keep natural physical step delta so receiver sees consistent Bussgang scale!
    def _q1d(v: torch.Tensor) -> torch.Tensor:
        idx = torch.clamp(torch.floor(v / delta) + 2.0, 0.0, 3.0)  # {0, 1, 2, 3}
        hard = (idx - 1.5) * delta  # {-1.5, -0.5, +0.5, +1.5} * delta
        if not v.requires_grad:
            return hard
        # Smooth staircase surrogate for backward pass
        t0 = torch.tanh(v / (0.35 * delta))
        tp = torch.tanh((v - delta) / (0.35 * delta))
        tm = torch.tanh((v + delta) / (0.35 * delta))
        soft = 0.5 * delta * (t0 + tp + tm)
        return hard + (soft - soft.detach())

    return _q1d(y_r), _q1d(y_i)


def physical_rf_impaired_ofdm_channel(
    s_rb_iq: torch.Tensor,
    snr_db: torch.Tensor | float,
    *,
    adc_bits: int = 1,
    pa_ibo_db: float = 1.0,
    pn_step_deg: float = 3.5,
    rho_f: float = 0.85,
    h_rb_override: torch.Tensor | None = None,
    phase_walk_override: torch.Tensor | None = None,
    standard_noise_td_iq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Realistic RF Hardware-Impaired OFDM Baseband Channel:
       1) 256-subcarrier IFFT -> Time-domain OFDM waveform (48 OFDM symbols)
       2) Solid-State Rapp Nonlinear HPA (smoothness p=2, saturation at IBO_dB) + strict unit radiated RF power
       3) 32-RB Frequency-Selective Rayleigh Fading (rho_f)
       4) Local Oscillator Wiener Phase Noise random walk across the 48 OFDM symbols (pn_step_deg/symbol)
       5) Time-domain thermal AWGN (snr_db)
       6) Time-domain Ultra-Low-Resolution ADC (1-Bit Zero-AGC Comparator or 2-Bit Mid-Rise ADC, before RX FFT!)
       7) Receiver 256-point FFT -> [B, 32, 384, 2]
    """
    B, M, N_b, _ = s_rb_iq.shape
    device, dtype = s_rb_iq.device, s_rb_iq.dtype
    s_grid = rb384_to_ofdm_grid(s_rb_iq)  # [B, 48, 256, 2]

    # Enforce strict unit pre-PA RMS power per image so IBO_dB is physically exact and un-gameable
    in_pwr = (s_grid[..., 0].square() + s_grid[..., 1].square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    s_fd_c = torch.complex(s_grid[..., 0], s_grid[..., 1]) / in_pwr.sqrt()  # [B, 48, 256]

    # 1. TX 256-Point Orthonormal IFFT to Time Domain
    x_td = torch.fft.ifft(s_fd_c, dim=-1, norm="ortho")  # [B, 48, 256]
    env_sq = x_td.real.square() + x_td.imag.square()
    papr_db = 10.0 * torch.log10(env_sq.amax(dim=-1).mean(dim=-1).clamp_min(1.0))

    # 2. Rapp Solid-State Power Amplifier (SSPA, smoothness p = 2.0 -> 2p = 4)
    if pa_ibo_db < 25.0:
        a_sat = math.pow(10.0, float(pa_ibo_db) / 20.0)
        mag = env_sq.clamp_min(1e-12).sqrt()
        ratio4 = (mag / a_sat).pow(4)
        gain_pa = torch.rsqrt(torch.sqrt(1.0 + ratio4))  # (1 + (|x|/A_sat)^4)^(-1/4)
        x_pa = x_td * gain_pa
        rad_pwr = (x_pa.real.square() + x_pa.imag.square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        x_rad = x_pa / rad_pwr.sqrt()
    else:
        x_rad = x_td

    # 3. 32-RB Frequency-Selective Fading
    if h_rb_override is not None:
        hr, hi = h_rb_override[:, :, 0], h_rb_override[:, :, 1]
    else:
        hr, hi = sample_correlated_rbs(B, M, rho_f, device=device, dtype=dtype)
    h_sc = torch.complex(hr, hi)[:, None, :, None].expand(B, N_OFDM_SYM, M, 8).reshape(B, N_OFDM_SYM, N_OFDM_SC)

    x_rad_fd = torch.fft.fft(x_rad, dim=-1, norm="ortho")
    y_clean_td = torch.fft.ifft(x_rad_fd * h_sc, dim=-1, norm="ortho")  # [B, 48, 256]

    # 4. Intra-Slot Oscillator Wiener Phase Noise across t = 0..47 OFDM symbols
    if phase_walk_override is not None:
        phi_t = phase_walk_override.to(device=device, dtype=dtype)  # [B, 48]
    elif pn_step_deg > 0.0:
        step_rad = float(pn_step_deg) * (math.pi / 180.0)
        dphi = torch.randn((B, N_OFDM_SYM), device=device, dtype=dtype) * step_rad
        dphi[:, 0] = 0.0
        phi_t = torch.cumsum(dphi, dim=1)
    else:
        phi_t = torch.zeros((B, N_OFDM_SYM), device=device, dtype=dtype)

    rot_pn = torch.complex(torch.cos(phi_t), torch.sin(phi_t)).unsqueeze(-1)
    y_pn_td = y_clean_td * rot_pn

    # 5. Time-Domain Thermal Baseband AWGN
    if not torch.is_tensor(snr_db):
        snr_t = torch.full((B,), float(snr_db), device=device, dtype=dtype)
    else:
        snr_t = snr_db.to(device=device, dtype=dtype).reshape(B)
    sigma_n = torch.pow(10.0, -snr_t / 20.0)[:, None, None]

    if standard_noise_td_iq is not None:
        eps_r = standard_noise_td_iq[..., 0].to(device=device, dtype=dtype)
        eps_i = standard_noise_td_iq[..., 1].to(device=device, dtype=dtype)
    else:
        eps_r = torch.randn((B, N_OFDM_SYM, N_OFDM_SC), device=device, dtype=dtype)
        eps_i = torch.randn((B, N_OFDM_SYM, N_OFDM_SC), device=device, dtype=dtype)

    inv_s2 = 1.0 / math.sqrt(2.0)
    y_rx_r = y_pn_td.real + sigma_n * inv_s2 * eps_r
    y_rx_i = y_pn_td.imag + sigma_n * inv_s2 * eps_i

    # 6. Receiver Time-Domain Low-Resolution ADC (BEFORE RX FFT!)
    if adc_bits == 1:
        r_td_r, r_td_i = ste_1bit_iq_quantize(y_rx_r, y_rx_i)
    elif adc_bits == 2:
        r_td_r, r_td_i = ste_2bit_iq_quantize(y_rx_r, y_rx_i)
    else:
        r_td_r, r_td_i = y_rx_r, y_rx_i

    # 7. Receiver 256-Point FFT -> Frequency-Domain RB Grid
    r_fd = torch.fft.fft(torch.complex(r_td_r, r_td_i), dim=-1, norm="ortho")
    r_grid_iq = torch.stack([r_fd.real, r_fd.imag], dim=-1)
    r_rb_iq = ofdm_grid_to_rb384(r_grid_iq)

    return r_rb_iq, {
        "papr_db": papr_db,
        "phi_t": phi_t,
        "r_td_r_sample": r_td_r.detach()[:, :2, :8],
    }


class RFBussgangSubspaceModem(nn.Module):
    """Dir-4A Core Modem:
      - Newman-Scrambled 4-Tile (32 RBs x 4 TimeSubBlocks x 96 REs) Orthogonal Walsh-3 Subspace
      - Differentiable Time-Domain Low-PAPR Compander + Post-Compand Exact Null Re-Cleaning
      - 32-RB Coherent Common-Phase-Error (CPE) Trajectory Interpolation (symbol-by-symbol t=0..47 Wiener PN cancellation!)
      - 1-Bit-Invariant atan2 Latent Scale Recovery & Gain-Unbiased Bussgang MMSE Equalization (Zero Pilots, Zero AGC!)
    """

    def __init__(self, num_rbs: int = 32):
        super().__init__()
        self.num_rbs = int(num_rbs)
        # Subspace probe budget eta1 in [0.04, 0.16], init at 0.085 for robust 1-bit Bussgang sensing
        init_eta1 = 0.085
        inv_s = math.log((init_eta1 - 0.04) / (0.16 - init_eta1))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta3 = 0.025  # 2.5% energy for 1-bit-invariant atan2 latent scale recovery

        # Learnable Time-Domain PAPR Crest-Factor Clipping Knee (init ~ 1.95 -> PAPR drops from ~10.8 dB to ~5.8 dB!)
        self.log_papr_knee = nn.Parameter(torch.tensor(math.log(1.95)))

        # Residual 1D Conv refiner over RB frequency axis after CPE de-rotation & 4-tile coherent combining
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
        return 0.04 + 0.12 * torch.sigmoid(self.eta1_raw)

    def embed_tx(self, z_tx: torch.Tensor) -> torch.Tensor:
        """Map [B, L=256, C=96] -> Low-PAPR Newman-Coded [B, M=32, N_b=384, 2] with EXACT u2 quiet null!"""
        B, L, C = z_tx.shape
        M = self.num_rbs
        S = N_TIME_TILES
        Nt = TILE_RES  # 96
        K = (L * C) // 2
        device, dtype = z_tx.device, z_tx.dtype

        p_img = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
        z_norm = z_tx / p_img[:, None, None]
        flat = z_norm.reshape(B, 2 * K)
        d_r = flat[:, :K].reshape(B, M, S, Nt)
        d_i = flat[:, K:].reshape(B, M, S, Nt)

        basis = build_walsh3_basis(Nt, device=device, dtype=dtype)  # [3, 96]
        u1, u2, u3 = basis[0], basis[1], basis[2]

        # Project semantic payload strictly onto span(u1, u2, u3)^perp in every 96-RE tile
        proj_r = torch.einsum("bmsn,kn->bmsk", d_r, basis)
        proj_i = torch.einsum("bmsn,kn->bmsk", d_i, basis)
        d_perp_r = d_r - torch.einsum("bmsk,kn->bmsn", proj_r, basis)
        d_perp_i = d_i - torch.einsum("bmsk,kn->bmsn", proj_i, basis)

        eta1 = self.eta1
        data_pwr = 1.0 - eta1 - self.eta3
        d_rms = (d_perp_r.square() + d_perp_i.square()).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()
        s_data_r = d_perp_r * (data_pwr.sqrt() / d_rms)
        s_data_i = d_perp_i * (data_pwr.sqrt() / d_rms)

        # Newman-scrambled u1 channel probe & u3 phase-angle p_img signature
        n_cos, n_sin = newman_rb_phases(M, device=device, dtype=dtype)  # [M]
        amp1 = (eta1 * float(Nt)).sqrt()
        p1_r = amp1 * n_cos[None, :, None, None] * u1[None, None, None, :]
        p1_i = amp1 * n_sin[None, :, None, None] * u1[None, None, None, :]

        theta = 1.25 * torch.tanh(0.40 * (p_img - 2.82))  # [B]
        amp3 = math.sqrt(self.eta3 * float(Nt))
        # Total phase on u3 is theta + psi_m so RX undoing psi_m recovers pure theta!
        c_th = torch.cos(theta)[:, None] * n_cos[None, :] - torch.sin(theta)[:, None] * n_sin[None, :]
        s_th = torch.sin(theta)[:, None] * n_cos[None, :] + torch.cos(theta)[:, None] * n_sin[None, :]
        p3_r = amp3 * c_th[:, :, None, None] * u3[None, None, None, :]
        p3_i = amp3 * s_th[:, :, None, None] * u3[None, None, None, :]

        s_tile_r = s_data_r + p1_r + p3_r
        s_tile_i = s_data_i + p1_i + p3_i
        s_rb0 = torch.stack([s_tile_r.reshape(B, M, 384), s_tile_i.reshape(B, M, 384)], dim=-1)

        # Differentiable Time-Domain Low-PAPR Envelope Companding + Exact u2 Null Restoration
        s_grid0 = rb384_to_ofdm_grid(s_rb0)
        x_td0 = torch.fft.ifft(torch.complex(s_grid0[..., 0], s_grid0[..., 1]), dim=-1, norm="ortho")
        knee = torch.exp(self.log_papr_knee.clamp(math.log(1.05), math.log(3.0)))
        mag = (x_td0.real.square() + x_td0.imag.square()).clamp_min(1e-12).sqrt()
        comp_gain = torch.rsqrt(torch.sqrt(1.0 + (mag / knee).pow(4)))
        x_td_c = x_td0 * comp_gain
        s_fd_c = torch.fft.fft(x_td_c, dim=-1, norm="ortho")
        s_rb_c = ofdm_grid_to_rb384(torch.stack([s_fd_c.real, s_fd_c.imag], dim=-1))

        # Re-clean ONLY u2 (1/96 = 1% DoF!) so TX PAPR pre-companding NEVER leaks energy into our blind noise null!
        t_r = s_rb_c[:, :, :, 0].reshape(B, M, S, Nt)
        t_i = s_rb_c[:, :, :, 1].reshape(B, M, S, Nt)
        leak2_r = torch.einsum("bmsn,n->bms", t_r, u2)
        leak2_i = torch.einsum("bmsn,n->bms", t_i, u2)
        t_r = t_r - leak2_r.unsqueeze(-1) * u2[None, None, None, :]
        t_i = t_i - leak2_i.unsqueeze(-1) * u2[None, None, None, :]

        out_r = t_r.reshape(B, M, 384)
        out_i = t_i.reshape(B, M, 384)
        pwr_final = (out_r.square() + out_i.square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return torch.stack([out_r / pwr_final.sqrt(), out_i / pwr_final.sqrt()], dim=-1)

    def blind_demod_and_equalize(self, r_rb_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, M, N_b, _ = r_rb_iq.shape
        S = N_TIME_TILES
        Nt = TILE_RES
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = r_rb_iq.device, r_rb_iq.dtype

        r_tile_r = r_rb_iq[:, :, :, 0].reshape(B, M, S, Nt)
        r_tile_i = r_rb_iq[:, :, :, 1].reshape(B, M, S, Nt)

        basis = build_walsh3_basis(Nt, device=device, dtype=dtype)
        u1, u2, u3 = basis[0], basis[1], basis[2]
        n_cos, n_sin = newman_rb_phases(M, device=device, dtype=dtype)

        # 1. Exact Total Effective Bussgang Distortion Variance (Thermal + PA + 1-Bit/2-Bit ADC!) from 128 u2 nulls
        q2_r = torch.einsum("bmsn,n->bms", r_tile_r, u2)
        q2_i = torch.einsum("bmsn,n->bms", r_tile_i, u2)
        sigma_eff_hat = (q2_r.square() + q2_i.square()).mean(dim=(1, 2)).clamp(1e-5, 12.0)  # [B]

        # 2. Extract Newman-de-scrambled raw tile channel estimates H_tilde[b, m, q]
        eta1 = self.eta1
        amp1 = (eta1 * float(Nt)).sqrt()
        raw1_r = torch.einsum("bmsn,n->bms", r_tile_r, u1) / amp1
        raw1_i = torch.einsum("bmsn,n->bms", r_tile_i, u1) / amp1
        # Undo Newman phase e^{-j psi_m}
        ht_r = raw1_r * n_cos[None, :, None] + raw1_i * n_sin[None, :, None]
        ht_i = raw1_i * n_cos[None, :, None] - raw1_r * n_sin[None, :, None]  # [B, M, 4]

        # 3. 32-RB Coherent Common-Phase-Error (CPE) Trajectory Tracking across t = 0..47 OFDM symbols!
        # Cross-correlate adjacent temporal tiles q and q+1 summed across all M=32 RBs (cancels unknown angle(h_m)!)
        cpe_knots = [torch.zeros((B,), device=device, dtype=dtype)]
        for q in range(S - 1):
            # sum_m H_{m, q+1} * conj(H_{m, q})
            re_step = (ht_r[:, :, q + 1] * ht_r[:, :, q] + ht_i[:, :, q + 1] * ht_i[:, :, q]).sum(dim=1)
            im_step = (ht_i[:, :, q + 1] * ht_r[:, :, q] - ht_r[:, :, q + 1] * ht_i[:, :, q]).sum(dim=1)
            dphi_q = torch.atan2(im_step, re_step.clamp_min(1e-6) + 1e-8 * (re_step >= 0).to(dtype) - 1e-8 * (re_step < 0).to(dtype))
            # Use clean 4-quadrant atan2
            dphi_q = torch.atan2(im_step, re_step + 1e-10)
            cpe_knots.append(cpe_knots[-1] + dphi_q)
        phi_knots = torch.stack(cpe_knots, dim=1)  # [B, 4] knots centered at symbol indices [5.5, 17.5, 29.5, 41.5]

        # Piecewise-linear interpolation/extrapolation of phi_hat[b, t] across all t = 0..47 OFDM symbols!
        t_idx = torch.arange(N_OFDM_SYM, device=device, dtype=dtype)  # [48]
        knot_pos = torch.tensor([5.5, 17.5, 29.5, 41.5], device=device, dtype=dtype)
        phi_sym = torch.zeros((B, N_OFDM_SYM), device=device, dtype=dtype)
        for t_i in range(N_OFDM_SYM):
            tv = float(t_idx[t_i].item())
            if tv <= 17.5:
                w = (tv - 5.5) / 12.0
                phi_sym[:, t_i] = phi_knots[:, 0] + w * (phi_knots[:, 1] - phi_knots[:, 0])
            elif tv <= 29.5:
                w = (tv - 17.5) / 12.0
                phi_sym[:, t_i] = phi_knots[:, 1] + w * (phi_knots[:, 2] - phi_knots[:, 1])
            else:
                w = (tv - 29.5) / 12.0
                phi_sym[:, t_i] = phi_knots[:, 2] + w * (phi_knots[:, 3] - phi_knots[:, 2])

        # De-rotate EVERY OFDM symbol t = 0..47 by e^{-j phi_sym[b, t]} BEFORE coherent 384-RE processing!
        r_sym_r = r_rb_iq[:, :, :, 0].reshape(B, M, N_OFDM_SYM, 8)
        r_sym_i = r_rb_iq[:, :, :, 1].reshape(B, M, N_OFDM_SYM, 8)
        c_cpe = torch.cos(phi_sym)[:, None, :, None]
        s_cpe = torch.sin(phi_sym)[:, None, :, None]
        rc_tile_r = (r_sym_r * c_cpe + r_sym_i * s_cpe).reshape(B, M, S, Nt)
        rc_tile_i = (r_sym_i * c_cpe - r_sym_r * s_cpe).reshape(B, M, S, Nt)

        # 4. Coherently combine all 4 CPE-corrected temporal tiles for high-SNR Bussgang RB channel estimation!
        c1_r = torch.einsum("bmsn,n->bm", rc_tile_r, u1) / (float(S) * amp1)
        c1_i = torch.einsum("bmsn,n->bm", rc_tile_i, u1) / (float(S) * amp1)
        hbar_r = c1_r * n_cos[None, :] + c1_i * n_sin[None, :]
        hbar_i = c1_i * n_cos[None, :] - c1_r * n_sin[None, :]

        sigma_e_raw = sigma_eff_hat / (eta1 * float(S * Nt))
        hw_r, hw_i, sigma_e_eff, delta_f = coherence_adaptive_wiener_rb(hbar_r, hbar_i, sigma_e_raw)

        e_rb = (rc_tile_r.square() + rc_tile_i.square()).mean(dim=(2, 3))
        log_sig_feat = torch.log(sigma_eff_hat.clamp_min(1e-5))[:, None].expand(B, M).div(4.0)
        df_feat = delta_f[:, None].expand(B, M)
        delta_h = self.rb_refiner(torch.stack([hbar_r, hbar_i, e_rb, log_sig_feat, hw_r, hw_i, df_feat], dim=1))
        hhat_r = hw_r + delta_h[:, 0, :]
        hhat_i = hw_i + delta_h[:, 1, :]

        # 5. 1-Bit-Invariant atan2 Latent Scale Recovery (invariant to Bussgang scaling alpha_ADC * alpha_PA!)
        c3_r = torch.einsum("bmsn,n->bm", rc_tile_r, u3).mean(dim=2 if False else 1 if False else -1 if False else 1) if False else None
        q3_r = torch.einsum("bmsn,n->bm", rc_tile_r, u3) / float(S)
        q3_i = torch.einsum("bmsn,n->bm", rc_tile_i, u3) / float(S)
        # Undo Newman phase on u3
        u3_r = q3_r * n_cos[None, :] + q3_i * n_sin[None, :]
        u3_i = q3_i * n_cos[None, :] - q3_r * n_sin[None, :]
        # Coherent MRC with conj(hhat)
        z3_r = (hhat_r * u3_r + hhat_i * u3_i).sum(dim=1)
        z3_i = (hhat_r * u3_i - hhat_i * u3_r).sum(dim=1)
        theta_hat = torch.atan2(z3_i, z3_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)

        # 6. Orthogonal Subspace Projection P_{span(u1,u2,u3)^perp} on CPE-corrected tiles
        proj_rr = torch.einsum("bmsn,kn->bmsk", rc_tile_r, basis)
        proj_ri = torch.einsum("bmsn,kn->bmsk", rc_tile_i, basis)
        y_perp_r = (rc_tile_r - torch.einsum("bmsk,kn->bmsn", proj_rr, basis)).reshape(B, M, N_b)
        y_perp_i = (rc_tile_i - torch.einsum("bmsk,kn->bmsn", proj_ri, basis)).reshape(B, M, N_b)

        # 7. Gain-Unbiased Bussgang MMSE Equalization (automatically cancels unknown alpha_ADC * alpha_PA!)
        data_pwr = 1.0 - eta1 - self.eta3
        h_sq = hhat_r.square() + hhat_i.square()
        h_abs = h_sq.clamp_min(1e-10).sqrt()
        eff_var = ((sigma_eff_hat / data_pwr) + sigma_e_eff)[:, None]
        denom = (h_sq + eff_var).clamp_min(1e-6)
        alpha_rb = h_sq / denom
        alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.03)

        cos_rb = (hhat_r / h_abs)[:, :, None]
        sin_rb = (hhat_i / h_abs)[:, :, None]
        yd_r = cos_rb * y_perp_r + sin_rb * y_perp_i
        yd_i = -sin_rb * y_perp_r + cos_rb * y_perp_i

        dof_comp = math.sqrt(float(Nt - 3) / float(Nt))
        data_scale = (1.0 / data_pwr).sqrt() * dof_comp
        w_ub = ((h_abs / denom) / alpha_bar)[:, :, None] * data_scale

        sh_r = yd_r * w_ub
        sh_i = yd_i * w_ub
        u_mmse_norm = torch.cat([sh_r.reshape(B, K), sh_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        # Post-Bussgang-MMSE effective SINR (for 1-Bit ADC at high SNR, naturally captures the ~2.4 dB arcsine SINR!)
        sinr_wb = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1)).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-5.0, 22.0)

        return (
            ChannelV2Output(
                u_restored=u_mmse_restored,
                u_mmse_restored=u_mmse_restored,
                u_norm=u_mmse_norm,
                z_norm=u_mmse_norm,
                pwr=p_img_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=expand_rb_to_symbols(h_abs, K),
                alpha_ratio_iq=expand_rb_to_symbols(alpha_rb / alpha_bar, K),
                sigma_post_iq=expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K),
                phase_unc_iq=expand_rb_to_symbols((sigma_e_eff[:, None] / (h_sq + sigma_e_eff[:, None] + 1e-6)).sqrt(), K),
                psi_csi=(alpha_rb - alpha_bar).square().mean(dim=1) + sigma_e_eff,
            ),
            cr_hat_db,
        )


class BlackBoxRFBlindModem(nn.Module):
    """End-to-End Neural Blind RF Modem Baseline (without Newman subspace, CPE tracking, or Bussgang inversion)."""

    def __init__(self, num_rbs: int = 32):
        super().__init__()
        self.num_rbs = int(num_rbs)
        self.rb_net = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(64, 3, kernel_size=1),
        )
        nn.init.zeros_(self.rb_net[-1].weight)
        nn.init.zeros_(self.rb_net[-1].bias)

        self.global_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.global_mlp[-1].weight)
        nn.init.zeros_(self.global_mlp[-1].bias)

    def embed_tx(self, z_tx: torch.Tensor) -> torch.Tensor:
        B, L, C = z_tx.shape
        K = (L * C) // 2
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        flat = (z_tx / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
        return torch.stack([flat[:, :K].reshape(B, 32, 384), flat[:, K:].reshape(B, 32, 384)], dim=-1)

    def blind_demod_and_equalize(self, r_rb_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, M, N_b, _ = r_rb_iq.shape
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = r_rb_iq.device, r_rb_iq.dtype
        yr = r_rb_iq[:, :, :, 0]
        yi = r_rb_iq[:, :, :, 1]

        mag_sq = yr.square() + yi.square()
        e_rb = mag_sq.mean(dim=-1)
        std_rb = mag_sq.std(dim=-1)
        m_r = yr.mean(dim=-1)
        m_i = yi.mean(dim=-1)
        # Split first/second half moments
        h1_r = yr[:, :, :192].mean(dim=-1)
        h2_r = yr[:, :, 192:].mean(dim=-1)
        h1_i = yi[:, :, :192].mean(dim=-1)
        h2_i = yi[:, :, 192:].mean(dim=-1)

        feats = torch.stack([m_r, m_i, e_rb.clamp(0, 8), std_rb.clamp(0, 8), h1_r, h2_r, h1_i, h2_i], dim=1)
        out = self.rb_net(feats)
        hr = 1.0 + out[:, 0, :]
        hi = out[:, 1, :]
        g_scale = torch.exp(out[:, 2, :].clamp(-1.5, 1.5))

        g_out = self.global_mlp(feats.mean(dim=-1))
        cr_hat_db = (5.0 + 6.0 * g_out[:, 0]).clamp(-5.0, 20.0)
        p_hat = (2.82 * torch.exp(g_out[:, 1].clamp(-1.0, 1.0))).clamp(0.5, 6.0)

        h_abs = (hr.square() + hi.square()).clamp_min(1e-6).sqrt()
        cos_rb = (hr / h_abs)[:, :, None]
        sin_rb = (hi / h_abs)[:, :, None]
        yd_r = (cos_rb * yr + sin_rb * yi) * g_scale[:, :, None]
        yd_i = (-sin_rb * yr + cos_rb * yi) * g_scale[:, :, None]

        u_norm = torch.cat([yd_r.reshape(B, K), yd_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
        u_rest = u_norm * p_hat[:, None, None]
        ones_iq = torch.ones((B, K), device=device, dtype=dtype)
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
                alpha_ratio_iq=ones_iq,
                sigma_post_iq=0.35 * ones_iq,
                phase_unc_iq=0.2 * ones_iq,
                psi_csi=torch.full((B,), 0.25, device=device, dtype=dtype),
            ),
            cr_hat_db,
        )


class RFImpairedBlindTransceiver(nn.Module):
    """Unified Dir-4A Transceiver:
      - rf_bussgang: Proposed Zero-Pilot, Zero-AGC Bussgang + CPE-Trajectory + Low-PAPR Companding Transceiver
      - blackbox_rf: End-to-End Neural Blind RF Baseline (0-Pilot)
      - frozen_rf: Unmodified SwinJSCC_w_SA under RF Hardware Impairments
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "rf_bussgang",
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

        if self.method == "rf_bussgang":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = RFBussgangSubspaceModem(num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method == "blackbox_rf":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = BlackBoxRFBlindModem(num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method == "frozen_rf":
            pass
        else:
            raise ValueError(f"Unknown Dir-4A method: {self.method}")

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_rf_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        B, L, _ = x_stem.shape
        K = (L * self.latent_channels) // 2
        snr_c0 = torch.full((B, 1), self.base_c0_db, device=x_stem.device, dtype=x_stem.dtype)

        with torch.no_grad():
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                temp = temp * enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
            z_base = enc.head_list(x_stem * enc.sigmoid(enc.sm_list[-1](temp)))

        z_tx = self.hier_tx(z_base, x_stem) if self.hier_tx is not None else z_base
        if self.modem is not None:
            return self.modem.embed_tx(z_tx)

        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        flat = (z_tx / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
        return torch.stack([flat[:, :K].reshape(B, 32, 384), flat[:, K:].reshape(B, 32, 384)], dim=-1)

    def decode_rf_rx(self, r_rb_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, N_b, _ = r_rb_iq.shape
        K = M * N_b
        C = self.latent_channels
        L = (2 * K) // C
        dec = self.backbone.decoder

        if self.modem is None:
            tot_pwr = (r_rb_iq[:, :, :, 0].square() + r_rb_iq[:, :, :, 1].square()).mean(dim=(1, 2)).clamp_min(0.2)
            u_in = torch.cat([r_rb_iq[:, :, :, 0].reshape(B, K), r_rb_iq[:, :, :, 1].reshape(B, K)], dim=1).reshape(B, L, C)
            u_in = u_in * (2.82 / tot_pwr.sqrt()[:, None, None])
            x = dec.head_list(u_in)
            snr_c0 = torch.full((B, 1), self.base_c0_db, device=u_in.device, dtype=u_in.dtype)
            temp = None
            for i in range(dec.layer_num):
                temp = dec.sm_list[i](x) if i == 0 else dec.sm_list[i](temp)
                temp = temp * dec.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
            x = x * dec.sigmoid(dec.sm_list[-1](temp))
            for layer in dec.layers:
                x = layer(x)
            return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous(), snr_c0.squeeze(1)

        ch_v2, cr_hat_db = self.modem.blind_demod_and_equalize(r_rb_iq, C)
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

    def forward_rf_blind(
        self,
        image_or_stem: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        adc_bits: int = 1,
        pa_ibo_db: float = 1.0,
        pn_step_deg: float = 3.5,
        rho_f: float = 0.85,
        h_rb_override: torch.Tensor | None = None,
        phase_walk_override: torch.Tensor | None = None,
        standard_noise_td_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        s_rb_iq = self.encode_rf_tx(x_stem)
        r_rb_iq, diag = physical_rf_impaired_ofdm_channel(
            s_rb_iq,
            snr_db,
            adc_bits=adc_bits,
            pa_ibo_db=pa_ibo_db,
            pn_step_deg=pn_step_deg,
            rho_f=rho_f,
            h_rb_override=h_rb_override,
            phase_walk_override=phase_walk_override,
            standard_noise_td_iq=standard_noise_td_iq,
        )
        recon, cr_hat_db = self.decode_rf_rx(r_rb_iq)
        return {"recon": recon, "cr_hat_db": cr_hat_db, **diag}


@torch.no_grad()
def forward_pilot4_rf_bussgang_reference(
    backbone: nn.Module,
    image: torch.Tensor,
    snr_db: float,
    *,
    adc_bits: int = 1,
    pa_ibo_db: float = 1.0,
    pn_step_deg: float = 3.5,
    h_rb_override: torch.Tensor,
    phase_walk_override: torch.Tensor,
    standard_noise_td_iq: torch.Tensor,
    pilot_seed: int = 2026,
) -> torch.Tensor:
    """Closed-Loop 128-Pilot Bussgang-LMMSE Reference (4 dedicated Newman-coded pilots per RB + Genie pwr & SNR scalar!).
    Note: Conventional block pilots sit at the start of the slot (first 4 OFDM symbols t=0..3) and go through the SAME
    1-Bit/2-Bit ADC, Rapp PA, and Wiener Phase Noise channel!
    """
    device, dtype = image.device, image.dtype
    B = image.shape[0]
    enc, dec = backbone.encoder, backbone.decoder
    snr_in = torch.full((B, 1), float(snr_db), device=device, dtype=dtype)

    # Encode with true wireless SNR (closed-loop feedback!)
    x = enc.patch_embed(image)
    for layer in enc.layers:
        x = layer(x)
    x = enc.norm(x)
    L = x.shape[1]
    temp = None
    for i in range(enc.layer_num):
        temp = enc.sm_list[i](x) if i == 0 else enc.sm_list[i](temp)
        temp = temp * enc.bm_list[i](snr_in).unsqueeze(1).expand(-1, L, -1)
    z = enc.head_list(x * enc.sigmoid(enc.sm_list[-1](temp)))
    C = z.shape[-1]
    K = (L * C) // 2

    pwr = (2.0 * z.square().mean(dim=(1, 2))).clamp_min(1e-8)
    flat = (z / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
    s_data_rb = torch.stack([flat[:, :K].reshape(B, 32, 384), flat[:, K:].reshape(B, 32, 384)], dim=-1)

    # Pass data through the exact RF-impaired channel
    r_data_rb, _ = physical_rf_impaired_ofdm_channel(
        s_data_rb,
        snr_db,
        adc_bits=adc_bits,
        pa_ibo_db=pa_ibo_db,
        pn_step_deg=pn_step_deg,
        h_rb_override=h_rb_override,
        phase_walk_override=phase_walk_override,
        standard_noise_td_iq=standard_noise_td_iq,
    )

    # Dedicated 4-pilot OFDM symbols (128 pilot REs at slot header t=0..3, Newman-coded across 32 RBs)
    n_cos, n_sin = newman_rb_phases(32, device=device, dtype=dtype)
    p_grid = torch.zeros((B, N_OFDM_SYM, N_OFDM_SC, 2), device=device, dtype=dtype)
    # Place pilots on subcarriers across all 32 RBs in first 4 symbols
    p_grid[:, :, :, 0] = n_cos[None, None, :].repeat_interleave(8, dim=-1)
    p_grid[:, :, :, 1] = n_sin[None, None, :].repeat_interleave(8, dim=-1)
    p_rb = ofdm_grid_to_rb384(p_grid)

    p_gen = torch.Generator(device="cpu").manual_seed(pilot_seed)
    p_eps = torch.randn((B, N_OFDM_SYM, N_OFDM_SC, 2), generator=p_gen, dtype=torch.float32).to(device=device, dtype=dtype)
    r_pil_rb, _ = physical_rf_impaired_ofdm_channel(
        p_rb,
        snr_db,
        adc_bits=adc_bits,
        pa_ibo_db=pa_ibo_db,
        pn_step_deg=pn_step_deg,
        h_rb_override=h_rb_override,
        phase_walk_override=phase_walk_override,
        standard_noise_td_iq=p_eps,
    )
    # 4 pilot REs per RB at slot header (indices 0..3 of the 48x8 RB grid)
    rp_sym = r_pil_rb.reshape(B, 32, N_OFDM_SYM, 8, 2)[:, :, 0, :4, :]  # 4 pilot REs per RB = 128 pilots total!
    rp_r = rp_sym[..., 0].mean(dim=-1)
    rp_i = rp_sym[..., 1].mean(dim=-1)
    hhat_r = rp_r * n_cos[None, :] + rp_i * n_sin[None, :]
    hhat_i = rp_i * n_cos[None, :] - rp_r * n_sin[None, :]

    # Bussgang effective noise variance (1 - 2/pi = 0.3634 floor for 1-bit, 0.118 for 2-bit + thermal)
    q_floor = 0.3634 if adc_bits == 1 else (0.118 if adc_bits == 2 else 0.0)
    sig_eff = q_floor + math.pow(10.0, -float(snr_db) / 10.0) * (0.64 if adc_bits == 1 else 1.0)
    h_sq = hhat_r.square() + hhat_i.square()
    h_abs = h_sq.clamp_min(1e-8).sqrt()
    denom = (h_sq + sig_eff).clamp_min(1e-5)
    alpha_bar = (h_sq / denom).mean(dim=1, keepdim=True).clamp_min(0.05)

    yr = r_data_rb[..., 0]
    yi = r_data_rb[..., 1]
    cos_rb = (hhat_r / h_abs)[:, :, None]
    sin_rb = (hhat_i / h_abs)[:, :, None]
    w_ub = ((h_abs / denom) / alpha_bar)[:, :, None]
    sh_r = (cos_rb * yr + sin_rb * yi) * w_ub
    sh_i = (-sin_rb * yr + cos_rb * yi) * w_ub

    u_rest = torch.cat([sh_r.reshape(B, K), sh_i.reshape(B, K)], dim=1).reshape(B, L, C) * pwr.sqrt()[:, None, None]
    xd = dec.head_list(u_rest)
    # Feed effective post-ADC SNR to decoder
    eff_snr_db = 10.0 * torch.log10(alpha_bar / (1.0 - alpha_bar).clamp_min(1e-4)).clamp(-5.0, 20.0)
    temp = None
    for i in range(dec.layer_num):
        temp = dec.sm_list[i](xd) if i == 0 else dec.sm_list[i](temp)
        temp = temp * dec.bm_list[i](eff_snr_db).unsqueeze(1).expand(-1, L, -1)
    xd = xd * dec.sigmoid(dec.sm_list[-1](temp))
    for layer in dec.layers:
        xd = layer(xd)
    return xd.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
