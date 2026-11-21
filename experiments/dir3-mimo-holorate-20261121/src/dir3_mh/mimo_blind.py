"""Direction 3A: Zero-Pilot Spatial-Multiplexed MIMO-OFDM Blind Semantic Communications (OS-MIMO-JSCC).

Solves both U(N_t) unitary stream-mixing ambiguity and Kronecker spatial correlation without TX CSI or dedicated pilots:
  - Compresses channel uses by N_t (T = 6,144 slots for 2x2 MIMO [CBR=1/32]; T = 3,072 slots for 4x4 MIMO [CBR=1/64]);
  - Embeds (N_t + 2) orthonormal Walsh space-frequency basis vectors per RB (w_1..w_{N_t} per antenna, w_null, w_norm);
  - Recovers full N_r x N_t complex MIMO matrix H_m and exact multi-antenna noise variance sigma_n^2 blindly;
  - Strips 100.000% of all N_t antenna probes via exact right-projection Y_m P_{W^perp} and equalizes via gain-unbiased
    matrix LMMSE + Open-Loop Hadamard antenna eigen-spreading + per-stream post-MMSE error covariance conditioning.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared backbone & multi-scale building blocks from dir2_zpac
DIR2_SRC = Path(__file__).resolve().parents[3] / "swinjscc-mismatch-zpac-20261120/src"
if str(DIR2_SRC) not in sys.path:
    sys.path.insert(0, str(DIR2_SRC))

from dir2_zpac.blind_v3 import BlindMultiScaleDecoderCalibrator, OpenLoopHierarchicalTX
from dir2_zpac.models import SNR_CENTER, SNR_SCALE, encode_with_ct
from dir2_zpac.phy_v2 import ChannelV2Output
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb


def build_hadamard_pow2(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct exact normalized Sylvester-Hadamard matrix H_n / sqrt(n) of power-of-2 order."""
    assert order >= 1 and (order & (order - 1)) == 0, f"order={order} must be power of 2"
    h = torch.ones((1, 1), device=device, dtype=dtype)
    inv_rt2 = 1.0 / math.sqrt(2.0)
    while h.shape[0] < order:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0) * inv_rt2
    return h


def build_walsh_k_basis(k_vecs: int, n_sym: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build k_vecs (<= 8) mutually orthonormal real Walsh basis vectors (k_vecs, n_sym) for n_sym % 8 == 0."""
    assert 1 <= k_vecs <= 8 and n_sym % 8 == 0, f"Invalid k_vecs={k_vecs}, n_sym={n_sym}"
    h8 = build_hadamard_pow2(8, device=device, dtype=dtype)  # (8, 8), rows orthonormal
    rep = n_sym // 8
    # Tile each 8-chip Walsh pattern across rep blocks and normalize by 1/sqrt(rep)
    full = h8.unsqueeze(-1).expand(8, 8, rep).reshape(8, n_sym) / math.sqrt(float(rep))
    return full[:k_vecs].contiguous()


def complex_to_real_block(hr: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Ring isomorphism Phi(H_r + j H_i) = [[H_r, -H_i], [H_i, H_r]]: (..., Nr, Nt) -> (..., 2*Nr, 2*Nt)."""
    top = torch.cat([hr, -hi], dim=-1)
    bot = torch.cat([hi, hr], dim=-1)
    return torch.cat([top, bot], dim=-2)


def sample_kronecker_mimo_rbs(
    batch_size: int,
    num_rbs: int,
    n_rx: int,
    n_tx: int,
    *,
    rho_f: float = 0.85,
    rho_ant: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample frequency-selective & spatially-correlated Kronecker MIMO H_m in C^{B x M x Nr x Nt}."""
    keep_f = math.sqrt(max(0.0, 1.0 - rho_f * rho_f))
    inv_rt2 = math.sqrt(0.5)

    w = torch.randn((batch_size, num_rbs, n_rx, n_tx, 2), generator=generator, device="cpu", dtype=torch.float32) * inv_rt2
    hr_list = [w[:, 0, :, :, 0]]
    hi_list = [w[:, 0, :, :, 1]]
    for m in range(1, num_rbs):
        hr_list.append(rho_f * hr_list[-1] + keep_f * w[:, m, :, :, 0])
        hi_list.append(rho_f * hi_list[-1] + keep_f * w[:, m, :, :, 1])
    hr = torch.stack(hr_list, dim=1).to(device=device, dtype=dtype)  # (B, M, Nr, Nt)
    hi = torch.stack(hi_list, dim=1).to(device=device, dtype=dtype)

    if rho_ant > 1e-4:
        idx_r = torch.arange(n_rx, device=device, dtype=dtype)
        idx_t = torch.arange(n_tx, device=device, dtype=dtype)
        r_rx = torch.pow(float(rho_ant), (idx_r[:, None] - idx_r[None, :]).abs())
        r_tx = torch.pow(float(rho_ant), (idx_t[:, None] - idx_t[None, :]).abs())
        l_rx = torch.linalg.cholesky(r_rx)
        l_tx = torch.linalg.cholesky(r_tx)
        hr = torch.einsum("ij,bmjk,lk->bmil", l_rx, hr, l_tx)
        hi = torch.einsum("ij,bmjk,lk->bmil", l_rx, hi, l_tx)

    return hr, hi


def physical_mimo_channel_forward(
    x_tx_iq: torch.Tensor,         # (B, Nt, T_mimo, 2) where E[sum_t |x_t|^2] == 1.0 per slot!
    snr_db: torch.Tensor | float,  # Per-RX-antenna average SNR in dB (hidden from TX & RX!)
    *,
    num_rbs: int = 32,
    n_rx: int | None = None,
    rho_f: float = 0.85,
    rho_ant: float = 0.0,
    h_mimo_override: tuple[torch.Tensor, torch.Tensor] | None = None,  # (hr, hi) each (B, M, Nr, Nt)
    standard_noise_iq: torch.Tensor | None = None,  # (B, Nr, T_mimo, 2)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Propagate x_tx_iq over unknown Nr x Nt MIMO-OFDM channel; return ONLY raw antenna Y_rx_iq!"""
    B, n_tx, t_mimo, two = x_tx_iq.shape
    assert two == 2 and t_mimo % num_rbs == 0
    if n_rx is None:
        n_rx = n_tx
    n_b = t_mimo // num_rbs
    device, dtype = x_tx_iq.device, x_tx_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )
    # Enforce exact Es/N0 relative to realized total multi-antenna transmit power per slot
    slot_pwr = (x_tx_iq[:, :, :, 0].square() + x_tx_iq[:, :, :, 1].square()).sum(dim=1).mean(dim=1).clamp_min(1e-8)
    sigma_n_sq = slot_pwr * torch.pow(10.0, -snr_t / 10.0)
    sigma_real = (0.5 * sigma_n_sq).sqrt()[:, None, None]

    if h_mimo_override is not None:
        hr_rb = h_mimo_override[0].to(device=device, dtype=dtype)
        hi_rb = h_mimo_override[1].to(device=device, dtype=dtype)
    else:
        hr_rb, hi_rb = sample_kronecker_mimo_rbs(B, num_rbs, n_rx, n_tx, rho_f=rho_f, rho_ant=rho_ant, device=device, dtype=dtype)

    xr = x_tx_iq[:, :, :, 0].reshape(B, n_tx, num_rbs, n_b).permute(0, 2, 1, 3)  # (B, M, Nt, Nb)
    xi = x_tx_iq[:, :, :, 1].reshape(B, n_tx, num_rbs, n_b).permute(0, 2, 1, 3)

    # Complex matrix product Y_m = H_m @ X_m
    yr = torch.einsum("bmrt,bmtn->bmrn", hr_rb, xr) - torch.einsum("bmrt,bmtn->bmrn", hi_rb, xi)
    yi = torch.einsum("bmrt,bmtn->bmrn", hr_rb, xi) + torch.einsum("bmrt,bmtn->bmrn", hi_rb, xr)

    yr_flat = yr.permute(0, 2, 1, 3).reshape(B, n_rx, t_mimo)
    yi_flat = yi.permute(0, 2, 1, 3).reshape(B, n_rx, t_mimo)

    eps = (
        torch.randn((B, n_rx, t_mimo, 2), device=device, dtype=dtype)
        if standard_noise_iq is None
        else standard_noise_iq.to(device=device, dtype=dtype).reshape(B, n_rx, t_mimo, 2)
    )
    y_rx_iq = torch.stack([yr_flat + eps[:, :, :, 0] * sigma_real, yi_flat + eps[:, :, :, 1] * sigma_real], dim=-1)
    return y_rx_iq, {
        "true_snr_db": snr_t,
        "true_sigma_n_sq": sigma_n_sq,
        "hr_rb": hr_rb,
        "hi_rb": hi_rb,
    }


class OrthogonalSpaceFreqMIMOModem(nn.Module):
    """OS-MIMO-JSCC Subspace Modem: Zero-Pilot Nr x Nt Matrix Estimation + Exact Right-Nulling + Matrix LMMSE."""

    def __init__(self, n_ant: int = 2, num_rbs: int = 32):
        super().__init__()
        assert n_ant in (2, 4)
        self.n_ant = int(n_ant)
        self.num_rbs = int(num_rbs)

        eta1_init = 0.055 if n_ant == 2 else 0.105
        eta_max = 0.12 if n_ant == 2 else 0.22
        self.eta_min = 0.02
        self.eta_max = eta_max
        inv_s = math.log((eta1_init - self.eta_min) / (eta_max - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta3 = 0.006

        # Per-(r, t) link RB frequency denoiser shared across all Nr*Nt antenna pairs!
        self.link_refiner = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.link_refiner[-1].weight)
        nn.init.zeros_(self.link_refiner[-1].bias)

    @property
    def eta1(self) -> torch.Tensor:
        return self.eta_min + (self.eta_max - self.eta_min) * torch.sigmoid(self.eta1_raw)

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Encode (B, L=256, C=96) into (B, Nt, T_mimo, 2) with exact sum_t E[|x_t|^2] == 1.000000!"""
        B, L, C = z_raw.shape
        K = (L * C) // 2
        nt = self.n_ant
        M = self.num_rbs
        t_mimo = K // nt
        n_b = t_mimo // M
        device, dtype = z_raw.device, z_raw.dtype

        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        p_img = pwr.sqrt()
        z_unit = z_raw / p_img[:, None, None]

        flat = z_unit.reshape(B, 2 * K)
        sr = flat[:, :K].reshape(B, nt, M, n_b).permute(0, 2, 1, 3)  # (B, M, Nt, Nb)
        si = flat[:, K:].reshape(B, nt, M, n_b).permute(0, 2, 1, 3)

        # Open-Loop Hadamard Unitary Eigen-Spreading across the Nt antennas (prevents single-stream deep fade wipeout!)
        h_ant = build_hadamard_pow2(nt, device=device, dtype=dtype)  # (Nt, Nt)
        sr_sp = torch.einsum("ts,bmsn->bmtn", h_ant, sr)
        si_sp = torch.einsum("ts,bmsn->bmtn", h_ant, si)

        # Project along Nb onto orthogonal complement of (Nt + 2) Walsh vectors: [w_1..w_Nt, w_null, w_norm]
        basis = build_walsh_k_basis(nt + 2, n_b, device=device, dtype=dtype)  # (Nt+2, Nb)
        proj_r = torch.einsum("bmtn,kn->bmtk", sr_sp, basis)
        proj_i = torch.einsum("bmtn,kn->bmtk", si_sp, basis)
        s_perp_r = sr_sp - torch.einsum("bmtk,kn->bmtn", proj_r, basis)
        s_perp_i = si_sp - torch.einsum("bmtk,kn->bmtn", proj_i, basis)

        eta1 = self.eta1
        data_pwr_target = 1.0 - eta1 - self.eta3
        # Sum over Nt antennas, mean over (M, Nb) slots
        cur_slot_pwr = (s_perp_r.square() + s_perp_i.square()).sum(dim=2).mean(dim=(1, 2)).clamp_min(1e-8)
        scale_d = (data_pwr_target / cur_slot_pwr).sqrt()[:, None, None, None]
        sd_r = s_perp_r * scale_d
        sd_i = s_perp_i * scale_d

        # Antenna-specific orthogonal Walsh probes w_1..w_Nt (zero cross-antenna interference!)
        w_ant = basis[:nt]      # (Nt, Nb)
        w_norm = basis[nt + 1]  # (Nb,)

        amp1_per_ant = (eta1 * float(n_b) / float(nt)).sqrt()
        amp3_per_ant = math.sqrt(self.eta3 * float(n_b) / float(nt))
        theta_p = 1.25 * torch.tanh(0.40 * (p_img - 2.82))
        cos_p = torch.cos(theta_p)[:, None, None, None]
        sin_p = torch.sin(theta_p)[:, None, None, None]

        tx_r = sd_r + amp1_per_ant * w_ant[None, None, :, :] + (amp3_per_ant * cos_p) * w_norm[None, None, None, :]
        tx_i = sd_i + (amp3_per_ant * sin_p) * w_norm[None, None, None, :]

        x_r = tx_r.permute(0, 2, 1, 3).reshape(B, nt, t_mimo)
        x_i = tx_i.permute(0, 2, 1, 3).reshape(B, nt, t_mimo)
        return torch.stack([x_r, x_i], dim=-1)

    def blind_demod_and_equalize(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, nr, t_mimo, _ = y_rx_iq.shape
        nt = self.n_ant
        M = self.num_rbs
        n_b = t_mimo // M
        K = nt * t_mimo
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        yr = y_rx_iq[:, :, :, 0].reshape(B, nr, M, n_b).permute(0, 2, 1, 3)  # (B, M, Nr, Nb)
        yi = y_rx_iq[:, :, :, 1].reshape(B, nr, M, n_b).permute(0, 2, 1, 3)

        basis = build_walsh_k_basis(nt + 2, n_b, device=device, dtype=dtype)
        w_ant = basis[:nt]      # (Nt, Nb)
        w_null = basis[nt]      # (Nb,)
        w_norm = basis[nt + 1]  # (Nb,)

        # 1. Exact Multi-Antenna Blind Noise Variance from w_null (32 * Nr complex DoF!)
        qn_r = torch.einsum("bmrn,n->bmr", yr, w_null)
        qn_i = torch.einsum("bmrn,n->bmr", yi, w_null)
        sigma_n_hat = (qn_r.square() + qn_i.square()).mean(dim=(1, 2)).clamp(1e-5, 15.0)  # (B,)

        # 2. Full Nr x Nt Complex Matrix Sensing from w_1..w_Nt
        eta1 = self.eta1
        amp1_per_ant = (eta1 * float(n_b) / float(nt)).sqrt()
        ht_r = torch.einsum("bmrn,tn->bmrt", yr, w_ant) / amp1_per_ant  # (B, M, Nr, Nt)
        ht_i = torch.einsum("bmrn,tn->bmrt", yi, w_ant) / amp1_per_ant
        sigma_e_raw = sigma_n_hat / (eta1 * float(n_b) / float(nt))     # (B,) per entry of H_m

        # Smooth each (r, t) link across M=32 RBs in parallel batch (B * Nr * Nt, M)
        B_links = B * nr * nt
        ht_r_flat = ht_r.permute(0, 2, 3, 1).reshape(B_links, M)
        ht_i_flat = ht_i.permute(0, 2, 3, 1).reshape(B_links, M)
        sig_e_rep = sigma_e_raw[:, None, None].expand(B, nr, nt).reshape(B_links)

        hw_r_f, hw_i_f, sig_e_eff_f, _ = coherence_adaptive_wiener_rb(ht_r_flat, ht_i_flat, sig_e_rep)
        e_rx_rb = (yr.square() + yi.square()).mean(dim=(2, 3))[:, :, None, None].expand(B, M, nr, nt).permute(0, 2, 3, 1).reshape(B_links, M)
        log_sig_f = torch.log(sigma_n_hat.clamp_min(1e-5))[:, None, None, None].expand(B, nr, nt, M).reshape(B_links, M).div(4.0)
        ref_out = self.link_refiner(torch.stack([ht_r_flat, ht_i_flat, e_rx_rb, log_sig_f, hw_r_f, hw_i_f], dim=1))

        hhat_r = (hw_r_f + ref_out[:, 0, :]).reshape(B, nr, nt, M).permute(0, 3, 1, 2).contiguous()  # (B, M, Nr, Nt)
        hhat_i = (hw_i_f + ref_out[:, 1, :]).reshape(B, nr, nt, M).permute(0, 3, 1, 2).contiguous()
        sigma_e_eff = sig_e_eff_f.reshape(B, nr, nt).mean(dim=(1, 2))  # (B,)

        # 3. 32-RB x Nr-Antenna MRC Phase-Angle Norm Recovery from w_norm
        q3_r = torch.einsum("bmrn,n->bmr", yr, w_norm)  # (B, M, Nr)
        q3_i = torch.einsum("bmrn,n->bmr", yi, w_norm)
        # Effective composite channel vector for all-ones antenna transmission: h_sum = H_m @ 1_{Nt}
        hsum_r = hhat_r.sum(dim=-1)  # (B, M, Nr)
        hsum_i = hhat_i.sum(dim=-1)
        z3_r = (hsum_r * q3_r + hsum_i * q3_i).sum(dim=(1, 2))
        z3_i = (hsum_r * q3_i - hsum_i * q3_r).sum(dim=(1, 2))
        theta_hat = torch.atan2(z3_i, z3_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)

        # 4. Exact Right-Subspace Nulling Y_perp = Y_m (I - W^T W): 100.000% zero probe leak for ANY H_m!
        proj_yr = torch.einsum("bmrn,kn->bmrk", yr, basis)
        proj_yi = torch.einsum("bmrn,kn->bmrk", yi, basis)
        yp_r = yr - torch.einsum("bmrk,kn->bmrn", proj_yr, basis)
        yp_i = yi - torch.einsum("bmrk,kn->bmrn", proj_yi, basis)

        # 5. Exact Real-Isomorphic Symmetric Positive-Definite Matrix LMMSE Equalization!
        data_pwr_per_ant = (1.0 - eta1 - self.eta3) / float(nt)
        lam = ((sigma_n_hat + float(nt) * sigma_e_eff) / data_pwr_per_ant).clamp_min(1e-5)  # (B,)

        phi_h = complex_to_real_block(hhat_r, hhat_i)  # (B, M, 2*Nr, 2*Nt)
        gram = torch.matmul(phi_h.transpose(-1, -2), phi_h)  # (B, M, 2*Nt, 2*Nt)
        eye_2nt = torch.eye(2 * nt, device=device, dtype=dtype)[None, None, :, :]
        reg_gram = gram + lam[:, None, None, None] * eye_2nt
        g_real = torch.linalg.solve(reg_gram, phi_h.transpose(-1, -2))  # (B, M, 2*Nt, 2*Nr)

        # Effective gain matrix A = G @ H: first Nt diagonal entries equal Re([G H]_{t,t}) = alpha_{m,t} in (0, 1)!
        a_mat = torch.matmul(g_real, phi_h)  # (B, M, 2*Nt, 2*Nt)
        alpha_mt = torch.diagonal(a_mat, dim1=-2, dim2=-1)[:, :, :nt].clamp(1e-4, 0.9999)  # (B, M, Nt)
        alpha_bar = alpha_mt.mean(dim=(1, 2)).clamp_min(0.05)  # (B,)

        y_stack = torch.cat([yp_r, yp_i], dim=2)  # (B, M, 2*Nr, Nb)
        s_est_stack = torch.matmul(g_real, y_stack)  # (B, M, 2*Nt, Nb)
        se_r = s_est_stack[:, :, :nt, :]
        se_i = s_est_stack[:, :, nt:, :]

        # Gain-unbiased rescaling + inverse Hadamard antenna de-spreading!
        dof_comp = math.sqrt(float(n_b - (nt + 2)) / float(n_b))
        scale_stream = (1.0 / (alpha_bar[:, None, None, None] * data_pwr_per_ant.sqrt())) * dof_comp
        se_r = se_r * scale_stream
        se_i = se_i * scale_stream

        h_ant = build_hadamard_pow2(nt, device=device, dtype=dtype)
        sd_r = torch.einsum("st,bmtn->bmsn", h_ant, se_r)  # H_ant^T == H_ant
        sd_i = torch.einsum("st,bmtn->bmsn", h_ant, se_i)

        sh_r_flat = sd_r.permute(0, 2, 1, 3).reshape(B, K)
        sh_i_flat = sd_i.permute(0, 2, 1, 3).reshape(B, K)
        u_mmse_norm = torch.cat([sh_r_flat, sh_i_flat], dim=1).reshape(B, L, latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        # Per-symbol MIMO post-equalization noise & stream-conditioning maps
        sinr_wb = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-4.0, 24.0)

        # Average post-Hadamard stream reliability per RB
        alpha_rb_mean = alpha_mt.mean(dim=-1)  # (B, M)
        alpha_rb_min = alpha_mt.min(dim=-1).values  # (B, M) captures worst singular mode!
        h_frob = (hhat_r.square() + hhat_i.square()).sum(dim=(-2, -1)).div(float(nr * nt)).sqrt()  # (B, M)

        def expand_mimo_rb(rb_t: torch.Tensor) -> torch.Tensor:
            # Map (B, M) -> (B, Nt, M, Nb) -> (B, K) matching sh_r_flat symbol layout!
            return rb_t[:, None, :, None].expand(B, nt, M, n_b).reshape(B, K)

        sigma_post_rb = 0.5 * ((1.0 - alpha_rb_mean) / alpha_bar[:, None]).clamp_min(1e-6)
        cond_spread = (alpha_rb_mean - alpha_rb_min).mean(dim=1)  # MIMO ill-conditioning indicator!

        return (
            ChannelV2Output(
                u_restored=u_mmse_restored,
                u_mmse_restored=u_mmse_restored,
                u_norm=u_mmse_norm,
                z_norm=u_mmse_norm,
                pwr=p_img_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=expand_mimo_rb(h_frob),
                alpha_ratio_iq=expand_mimo_rb(alpha_rb_min / alpha_bar[:, None]),
                sigma_post_iq=expand_mimo_rb(sigma_post_rb),
                phase_unc_iq=expand_mimo_rb((1.0 - alpha_rb_min).clamp(0.0, 1.0)),
                psi_csi=cond_spread * 3.0 + sigma_e_eff,
            ),
            cr_hat_db,
        )


class BlackBoxBlindMIMOModem(nn.Module):
    """End-to-End Black-Box Blind MIMO Baseline (Blind-MIMOSC / TCOM'25 style without orthogonal space-frequency probes)."""

    def __init__(self, n_ant: int = 2, num_rbs: int = 32):
        super().__init__()
        self.n_ant = int(n_ant)
        self.num_rbs = int(num_rbs)
        # Learnable per-antenna symmetry-breaking signature + neural RB MIMO unmixing Conv1D
        self.ant_bias = nn.Parameter(torch.randn(1, n_ant, 1) * 0.05)
        in_ch = 2 * n_ant + n_ant * n_ant * 2
        self.unmix_net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(64, 2 * n_ant * n_ant + 2, kernel_size=1),  # [G_r, G_i (Nt x Nr), delta_snr, delta_log_p]
        )
        nn.init.zeros_(self.unmix_net[-1].weight)
        nn.init.zeros_(self.unmix_net[-1].bias)

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        B, L, C = z_raw.shape
        K = (L * C) // 2
        nt = self.n_ant
        t_mimo = K // nt
        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        z_unit = z_raw / pwr.sqrt()[:, None, None]
        flat = z_unit.reshape(B, 2 * K)
        xr = flat[:, :K].reshape(B, nt, t_mimo) + torch.tanh(self.ant_bias) * 0.20
        xi = flat[:, K:].reshape(B, nt, t_mimo)
        slot_p = (xr.square() + xi.square()).sum(dim=1).mean(dim=1).clamp_min(1e-8)
        sc = (1.0 / slot_p.sqrt())[:, None, None]
        return torch.stack([xr * sc, xi * sc], dim=-1)

    def blind_demod_and_equalize(self, y_rx_iq: torch.Tensor, latent_ch: int) -> tuple[ChannelV2Output, torch.Tensor]:
        B, nr, t_mimo, _ = y_rx_iq.shape
        nt = self.n_ant
        M = self.num_rbs
        n_b = t_mimo // M
        K = nt * t_mimo
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        yr = y_rx_iq[:, :, :, 0].reshape(B, nr, M, n_b).permute(0, 2, 1, 3)  # (B, M, Nr, Nb)
        yi = y_rx_iq[:, :, :, 1].reshape(B, nr, M, n_b).permute(0, 2, 1, 3)

        mean_yr = yr.mean(dim=-1)  # (B, M, Nr)
        mean_yi = yi.mean(dim=-1)
        # Empirical spatial covariance Y_m Y_m^H / Nb
        cov_r = (torch.einsum("bmrn,bmsn->bmrs", yr, yr) + torch.einsum("bmrn,bmsn->bmrs", yi, yi)) / float(n_b)
        cov_i = (torch.einsum("bmrn,bmsn->bmrs", yi, yr) - torch.einsum("bmrn,bmsn->bmrs", yr, yi)) / float(n_b)

        feat_rb = torch.cat(
            [
                mean_yr,
                mean_yi,
                cov_r.reshape(B, M, nr * nr).clamp(-10.0, 10.0),
                cov_i.reshape(B, M, nr * nr).clamp(-10.0, 10.0),
            ],
            dim=-1,
        ).permute(0, 2, 1)  # (B, in_ch, M)

        out = self.unmix_net(feat_rb)  # (B, 2*Nt*Nr + 2, M)
        eye_g = torch.eye(nt, nr, device=device, dtype=dtype)[None, None, :, :]
        gr = eye_g + out[:, : nt * nr, :].permute(0, 2, 1).reshape(B, M, nt, nr)
        gi = out[:, nt * nr : 2 * nt * nr, :].permute(0, 2, 1).reshape(B, M, nt, nr)

        sr_hat = torch.einsum("bmtr,bmrn->bmtn", gr, yr) - torch.einsum("bmtr,bmrn->bmtn", gi, yi)
        si_hat = torch.einsum("bmtr,bmrn->bmtn", gr, yi) + torch.einsum("bmtr,bmrn->bmtn", gi, xr := yr)
        # Wait: fix imaginary cross term cleanly: gi*yr + gr*yi!
        si_hat = torch.einsum("bmtr,bmrn->bmtn", gr, yi) + torch.einsum("bmtr,bmrn->bmtn", gi, yr)

        tot_p = (yr.square() + yi.square()).mean(dim=(1, 2, 3)).clamp_min(0.2)
        snr_init = (-10.0 * torch.log10((tot_p - 1.0).clamp_min(0.02))).clamp(-4.0, 20.0)
        cr_hat_db = (snr_init + 4.0 * out[:, -2, :].mean(dim=1)).clamp(-4.0, 22.0)
        p_hat = (2.82 * torch.exp(out[:, -1, :].mean(dim=1).clamp(-1.0, 1.0))).clamp(0.5, 6.0)

        u_norm = torch.cat(
            [
                sr_hat.permute(0, 2, 1, 3).reshape(B, K) * math.sqrt(float(nt)),
                si_hat.permute(0, 2, 1, 3).reshape(B, K) * math.sqrt(float(nt)),
            ],
            dim=1,
        ).reshape(B, L, latent_ch)
        u_rest = u_norm * p_hat[:, None, None]
        sig_p = 0.5 * torch.pow(10.0, -cr_hat_db / 10.0)
        ones_k = torch.ones((B, K), device=device, dtype=dtype)

        return (
            ChannelV2Output(
                u_restored=u_rest,
                u_mmse_restored=u_rest,
                u_norm=u_norm,
                z_norm=u_norm,
                pwr=p_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=ones_k,
                alpha_ratio_iq=ones_k,
                sigma_post_iq=sig_p[:, None].expand(B, K).contiguous(),
                phase_unc_iq=torch.zeros_like(ones_k),
                psi_csi=ones_k[:, 0] * 0.5,
            ),
            cr_hat_db,
        )


class BlindMIMOTransceiver(nn.Module):
    """Unified Dir-3A Wrapper for 2x2 & 4x4 Spatial-Multiplexed MIMO-OFDM JSCC."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "os_mimo",  # "os_mimo", "blackbox_mimo", "frozen_blind_mimo"
        n_ant: int = 2,
        num_rbs: int | None = None,
        base_c0_db: float = 7.0,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.method = method.lower()
        self.n_ant = int(n_ant)
        # Physical 5G NR PRB allocation: 384 REs/PRB -> 16 PRBs for 2x2 MIMO (6,144 REs), 8 PRBs for 4x4 MIMO (3,072 REs)!
        self.num_rbs = int(num_rbs) if num_rbs is not None else (32 // int(n_ant))
        num_rbs = self.num_rbs
        self.base_c0_db = float(base_c0_db)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx: OpenLoopHierarchicalTX | None = None
        self.modem: nn.Module | None = None
        self.blind_dec: BlindMultiScaleDecoderCalibrator | None = None

        if self.method == "os_mimo":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = OrthogonalSpaceFreqMIMOModem(n_ant=n_ant, num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method == "blackbox_mimo":
            self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            self.modem = BlackBoxBlindMIMOModem(n_ant=n_ant, num_rbs=num_rbs)
            self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)
        elif self.method == "frozen_blind_mimo":
            pass
        else:
            raise ValueError(f"Unknown MIMO method: {self.method}")

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_mimo_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        B, L, _ = x_stem.shape
        K = (L * self.latent_channels) // 2
        nt = self.n_ant
        t_mimo = K // nt
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

        # Frozen blind MIMO: split unit-total-power symbols across Nt antennas
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        z_u = z_tx / (pwr.sqrt()[:, None, None] * math.sqrt(float(nt)))
        flat = z_u.reshape(B, 2 * K)
        return torch.stack([flat[:, :K].reshape(B, nt, t_mimo), flat[:, K:].reshape(B, nt, t_mimo)], dim=-1)

    def decode_mimo_rx(self, y_rx_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, nr, t_mimo, _ = y_rx_iq.shape
        C = self.latent_channels
        K = self.n_ant * t_mimo
        L = (2 * K) // C
        dec = self.backbone.decoder

        if self.modem is None:
            tot_p = (y_rx_iq[:, :, :, 0].square() + y_rx_iq[:, :, :, 1].square()).mean(dim=(1, 2)).clamp_min(0.3)
            u_in = torch.cat(
                [y_rx_iq[:, : self.n_ant, :, 0].reshape(B, K), y_rx_iq[:, : self.n_ant, :, 1].reshape(B, K)],
                dim=1,
            ).reshape(B, L, C) * (2.82 / tot_p.sqrt()[:, None, None])
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

    def forward_mimo_blind(
        self,
        image_or_stem: torch.Tensor,
        true_snr_db: torch.Tensor | float,
        *,
        rho_f: float = 0.85,
        rho_ant: float = 0.0,
        h_mimo_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        x_tx_iq = self.encode_mimo_tx(x_stem)
        y_rx_iq, diag = physical_mimo_channel_forward(
            x_tx_iq,
            true_snr_db,
            num_rbs=self.num_rbs,
            n_rx=self.n_ant,
            rho_f=rho_f,
            rho_ant=rho_ant,
            h_mimo_override=h_mimo_override,
            standard_noise_iq=standard_noise_iq,
        )
        recon, cr_hat_db = self.decode_mimo_rx(y_rx_iq)
        return {"recon": recon, "cr_hat_db": cr_hat_db, "x_tx_iq": x_tx_iq, **diag}


@torch.inference_mode()
def forward_pilot_or_genie_svd_mimo_reference(
    backbone: nn.Module,
    xb: torch.Tensor,
    n_ant: int,
    snr_db: float,
    hr_rb: torch.Tensor,  # (B, M, Nr, Nt)
    hi_rb: torch.Tensor,
    eps_iq: torch.Tensor,  # (B, Nr, T_mimo, 2)
    *,
    mode: str = "pilot4_lmmse",  # "pilot4_lmmse" (4*Nt extra orthogonal pilot slots/RB!) or "genie_svd"
    pilot_seed: int = 777,
) -> torch.Tensor:
    """Reference Pilot-Aided Orthogonal MIMO LMMSE (256/512 extra pilot slots!) & Perfect-CSI SVD MIMO Oracle."""
    B = xb.shape[0]
    device, dtype = xb.device, xb.dtype
    M = hr_rb.shape[1]

    ct_use = max(1.0, min(13.0, snr_db)) if mode == "genie_svd" else 7.0
    z_raw, _ = encode_with_ct(backbone, xb, ct_use)
    L, C = z_raw.shape[1], z_raw.shape[2]
    K = (L * C) // 2
    t_mimo = K // n_ant
    n_b = t_mimo // M

    pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
    p_img = pwr.sqrt()
    z_unit = z_raw / (p_img[:, None, None] * math.sqrt(float(n_ant)))
    flat = z_unit.reshape(B, 2 * K)
    sr = flat[:, :K].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3)  # (B, M, Nt, Nb)
    si = flat[:, K:].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3)

    sigma_n_sq = float(10.0 ** (-snr_db / 10.0))

    if mode == "genie_svd":
        # Exact complex SVD H_m = U_m diag(sigma) V_m^H: precode with V_m at TX, combine with U_m^H at RX!
        h_cplx = torch.complex(hr_rb, hi_rb)  # (B, M, Nr, Nt)
        u_mat, s_vals, vh_mat = torch.linalg.svd(h_cplx, full_matrices=False)
        v_mat = vh_mat.conj().transpose(-1, -2)
        s_cplx = torch.complex(sr, si)  # (B, M, Nt, Nb)
        x_cplx = torch.matmul(v_mat, s_cplx)  # unit total power preserved identically!
        x_tx = torch.stack(
            [
                x_cplx.real.permute(0, 2, 1, 3).reshape(B, n_ant, t_mimo),
                x_cplx.imag.permute(0, 2, 1, 3).reshape(B, n_ant, t_mimo),
            ],
            dim=-1,
        )
        y_rx, _ = physical_mimo_channel_forward(
            x_tx, snr_db, num_rbs=M, n_rx=n_ant, h_mimo_override=(hr_rb, hi_rb), standard_noise_iq=eps_iq
        )
        y_cplx = torch.complex(
            y_rx[:, :, :, 0].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3),
            y_rx[:, :, :, 1].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3),
        )
        # Parallel eigenchannel unbiased MMSE on U_m^H Y_m = diag(s_vals) S_m + tilde_N_m!
        yt = torch.matmul(u_mat.conj().transpose(-1, -2), y_cplx)
        lam = sigma_n_sq * float(n_ant)
        alpha_eig = s_vals.square() / (s_vals.square() + lam)
        alpha_bar = alpha_eig.mean(dim=(1, 2), keepdim=True).clamp_min(0.05)
        w_eig = ((s_vals / (s_vals.square() + lam)) / alpha_bar).unsqueeze(-1)
        sh_cplx = yt * w_eig
        sh_r = sh_cplx.real.permute(0, 2, 1, 3).reshape(B, K) * math.sqrt(float(n_ant))
        sh_i = sh_cplx.imag.permute(0, 2, 1, 3).reshape(B, K) * math.sqrt(float(n_ant))
        cr_db = ct_use
    else:
        # Conventional 4*Nt orthogonal pilot time-slots per RB (4 pilots per TX antenna with total slot power 1.0)
        x_tx = torch.stack(
            [sr.permute(0, 2, 1, 3).reshape(B, n_ant, t_mimo), si.permute(0, 2, 1, 3).reshape(B, n_ant, t_mimo)],
            dim=-1,
        )
        y_rx, _ = physical_mimo_channel_forward(
            x_tx, snr_db, num_rbs=M, n_rx=n_ant, h_mimo_override=(hr_rb, hi_rb), standard_noise_iq=eps_iq
        )
        np_per_ant = 4.0
        gen = torch.Generator(device="cpu").manual_seed(pilot_seed)
        vp = torch.randn((B, M, n_ant, n_ant, 2), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        sig_p = math.sqrt(sigma_n_sq / (2.0 * np_per_ant))
        lw = 1.0 / (1.0 + sigma_n_sq / np_per_ant)
        hhat_r = lw * (hr_rb + vp[:, :, :, :, 0] * sig_p)
        hhat_i = lw * (hi_rb + vp[:, :, :, :, 1] * sig_p)
        sig_e_sq = (sigma_n_sq / np_per_ant) / (1.0 + sigma_n_sq / np_per_ant)

        phi_h = complex_to_real_block(hhat_r, hhat_i)
        lam = float(n_ant) * (sigma_n_sq + float(n_ant) * sig_e_sq)
        eye_2nt = torch.eye(2 * n_ant, device=device, dtype=dtype)[None, None, :, :]
        g_real = torch.linalg.solve(torch.matmul(phi_h.transpose(-1, -2), phi_h) + lam * eye_2nt, phi_h.transpose(-1, -2))
        a_mat = torch.matmul(g_real, phi_h)
        alpha_mt = torch.diagonal(a_mat, dim1=-2, dim2=-1)[:, :, :n_ant].clamp(1e-4, 0.9999)
        alpha_bar = alpha_mt.mean(dim=(1, 2)).clamp_min(0.05)

        yr = y_rx[:, :, :, 0].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3)
        yi = y_rx[:, :, :, 1].reshape(B, n_ant, M, n_b).permute(0, 2, 1, 3)
        se_st = torch.matmul(g_real, torch.cat([yr, yi], dim=2))
        sc = math.sqrt(float(n_ant)) / alpha_bar[:, None, None, None]
        sh_r = (se_st[:, :, :n_ant, :] * sc).permute(0, 2, 1, 3).reshape(B, K)
        sh_i = (se_st[:, :, n_ant:, :] * sc).permute(0, 2, 1, 3).reshape(B, K)
        cr_db = 7.0

    u_in = torch.cat([sh_r, sh_i], dim=1).reshape(B, L, C) * p_img[:, None, None]
    dec = backbone.decoder
    x = dec.head_list(u_in)
    snr_in = torch.full((B, 1), float(cr_db), device=device, dtype=dtype)
    temp = None
    for i in range(dec.layer_num):
        temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
        temp = temp * dec.bm_list[i](snr_in).unsqueeze(1).expand(-1, L, -1)
    x = x * dec.sigmoid(dec.sm_list[-1](temp))
    for layer in dec.layers:
        x = layer(x)
    return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
