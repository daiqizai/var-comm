"""Direction 3B: Holographic-Dispersed Rateless, Burst-Erasure & Partial-Band Jamming-Resilient Blind JSCC (HoloRate-JSCC).

Solves three simultaneous wireless resource & hostile-channel challenges with zero prior CSI and zero packet-loss signaling:
  1. Anytime-Stopping Rateless Transmission (8 / 16 / 24 / 32 RBs = CBR 1/64, 1/32, 3/64, 1/16 from ONE single model);
  2. Arbitrary Contiguous Burst Erasures (up to 50% consecutive RBs dropped) via 256-point Unitary Walsh-Hadamard
     Spatial Dispersion + 3-Channel-per-RB Full-Image Interleaving (every single RB spans all 16x16 spatial patches!);
  3. Hostile Partial-Band Jamming (+15 ~ +18 dB jammer on 25%~38% of RBs) via 4-Vector Per-RB Orthogonal Null
     Outlier-Robust Trimmed Background Sensing & Surgical Per-RB MMSE Jamming Excision.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

DIR2_SRC = Path(__file__).resolve().parents[3] / "swinjscc-mismatch-zpac-20261120/src"
if str(DIR2_SRC) not in sys.path:
    sys.path.insert(0, str(DIR2_SRC))

from dir2_zpac.blind_v3 import BlindMultiScaleDecoderCalibrator, OpenLoopHierarchicalTX
from dir2_zpac.models import SNR_CENTER, SNR_SCALE
from dir2_zpac.phy_v2 import ChannelV2Output, sample_correlated_rbs
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb

from .mimo_blind import build_hadamard_pow2, build_walsh_k_basis


def build_channel_sign_dither(length: int, ch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic +/-1 orthogonal sign pattern (1, length, ch) so Hadamard coefficients have uniform RB energy."""
    gen = torch.Generator(device="cpu").manual_seed(20261122)
    bits = torch.randint(0, 2, (1, length, ch), generator=gen, dtype=torch.int64)
    return (bits.to(dtype=dtype) * 2.0 - 1.0).to(device=device)


def physical_hostile_ofdm_channel(
    s_tx_rb_iq: torch.Tensor,      # (B, M=32, N_b=384, 2) unit average power per RB
    snr_db: torch.Tensor | float,  # Background thermal SNR in dB (hidden!)
    *,
    rho_f: float = 0.85,
    alive_rb_mask: torch.Tensor | None = None,   # (B, M) in {0.0, 1.0}: 0.0 = erased/truncated RB!
    jam_rb_mask: torch.Tensor | None = None,     # (B, M) in {0.0, 1.0}: 1.0 = jammed RB!
    jam_inr_db: float = 15.0,                    # Jammer-to-signal ratio in dB on jammed RBs (+15 dB = 31.6x power!)
    h_rb_override: torch.Tensor | None = None,   # (B, M, 2)
    standard_noise_iq: torch.Tensor | None = None,  # (B, M, N_b, 2)
    jam_noise_iq: torch.Tensor | None = None,    # (B, M, N_b, 2)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Propagate 32 RBs through unknown complex OFDM fading + arbitrary RB erasures/truncation + partial-band jamming!"""
    B, M, N_b, two = s_tx_rb_iq.shape
    assert two == 2
    device, dtype = s_tx_rb_iq.device, s_tx_rb_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )
    rb_pwr = (s_tx_rb_iq[:, :, :, 0].square() + s_tx_rb_iq[:, :, :, 1].square()).mean(dim=(1, 2)).clamp_min(1e-8)
    sigma_bg_sq = rb_pwr * torch.pow(10.0, -snr_t / 10.0)
    sigma_bg_r = (0.5 * sigma_bg_sq).sqrt()[:, None, None]

    if h_rb_override is not None:
        hr = h_rb_override[:, :, 0].to(device=device, dtype=dtype)
        hi = h_rb_override[:, :, 1].to(device=device, dtype=dtype)
    else:
        hr, hi = sample_correlated_rbs(B, M, rho_f, device=device, dtype=dtype)

    sr = s_tx_rb_iq[:, :, :, 0]
    si = s_tx_rb_iq[:, :, :, 1]
    yr = hr[:, :, None] * sr - hi[:, :, None] * si
    yi = hr[:, :, None] * si + hi[:, :, None] * sr

    eps = (
        torch.randn_like(s_tx_rb_iq)
        if standard_noise_iq is None
        else standard_noise_iq.to(device=device, dtype=dtype).reshape(B, M, N_b, 2)
    )
    yr = yr + eps[:, :, :, 0] * sigma_bg_r
    yi = yi + eps[:, :, :, 1] * sigma_bg_r

    if jam_rb_mask is not None:
        jmask = jam_rb_mask.to(device=device, dtype=dtype).reshape(B, M, 1)
        sigma_jam_sq = rb_pwr * (10.0 ** (float(jam_inr_db) / 10.0))
        sigma_jam_r = (0.5 * sigma_jam_sq).sqrt()[:, None, None]
        jeps = (
            torch.randn_like(s_tx_rb_iq)
            if jam_noise_iq is None
            else jam_noise_iq.to(device=device, dtype=dtype).reshape(B, M, N_b, 2)
        )
        yr = yr + jmask * jeps[:, :, :, 0] * sigma_jam_r
        yi = yi + jmask * jeps[:, :, :, 1] * sigma_jam_r

    if alive_rb_mask is not None:
        amask = alive_rb_mask.to(device=device, dtype=dtype).reshape(B, M, 1)
        yr = yr * amask
        yi = yi * amask

    return torch.stack([yr, yi], dim=-1), {
        "true_snr_db": snr_t,
        "true_sigma_bg_sq": sigma_bg_sq,
        "hr_rb": hr,
        "hi_rb": hi,
    }


class HoloChannelCompletionBlock(nn.Module):
    """Restores erased/truncated/jammed semantic channel-triplets from surviving channels + reliability vector r_ch!"""

    def __init__(self, latent_ch: int = 96, hidden_ch: int = 160):
        super().__init__()
        # Input per spatial token: [u_restored (96), u_restored * r_ch (96), r_ch (96)] -> 288 -> 96
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch * 3, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, latent_ch, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_restored: torch.Tensor, r_ch: torch.Tensor) -> torch.Tensor:
        """u_restored: (B, 256, 96), r_ch: (B, 96) in [0, 1] per-channel SINR reliability."""
        B, L, C = u_restored.shape
        u_sp = u_restored.reshape(B, 16, 16, C).permute(0, 3, 1, 2).contiguous()
        r_map = r_ch[:, :, None, None].expand(B, C, 16, 16)
        delta = self.net(torch.cat([u_sp, u_sp * r_map, r_map], dim=1))
        # Weight completion more strongly on degraded/erased channels while allowing refinement everywhere
        return (u_sp + delta).permute(0, 2, 3, 1).reshape(B, L, C).contiguous()


class HoloBlindAntiJamModem(nn.Module):
    """6-Walsh-Vector Self-Probing Modem (u1 probe, u_norm, + 4 orthogonal null probes u_n1..u_n4 for surgical jamming/erasure sensing)."""

    def __init__(self, num_rbs: int = 32, use_holographic_dispersion: bool = True):
        super().__init__()
        self.num_rbs = int(num_rbs)
        self.use_holo = bool(use_holographic_dispersion)

        eta1_init = 0.036
        inv_s = math.log((eta1_init - 0.015) / (0.080 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta3 = 0.006  # Slightly stronger norm anchor so even 8 surviving RBs recover p_img cleanly!

    @property
    def eta1(self) -> torch.Tensor:
        return 0.015 + (0.080 - 0.015) * torch.sigmoid(self.eta1_raw)

    def _forward_holo(self, z_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (B, L=256, C=96) -> (B, M=32, N_b=384) real & imag."""
        B, L, C = z_unit.shape
        M = self.num_rbs
        N_b = (L * C) // (2 * M)
        device, dtype = z_unit.device, z_unit.dtype

        if not self.use_holo:
            K = (L * C) // 2
            flat = z_unit.reshape(B, 2 * K)
            return flat[:, :K].reshape(B, M, N_b), flat[:, K:].reshape(B, M, N_b)

        # 256-point exact orthogonal Walsh-Hadamard spatial dispersion + deterministic sign dither
        dither = build_channel_sign_dither(L, C, device=device, dtype=dtype)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)  # (256, 256)
        z_disp = torch.matmul(h256, z_unit * dither)  # (B, 256, 96)

        # Group into M=32 progressive channel-triplets: each RB holds ALL 256 spatial frequencies of 3 channels!
        # (B, 256, 32, 3) -> (B, 32, 3 * 256 = 768) -> split 384 Re + 384 Im!
        rb_768 = z_disp.reshape(B, L, M, C // M).permute(0, 2, 3, 1).reshape(B, M, 2 * N_b)
        return rb_768[:, :, :N_b], rb_768[:, :, N_b:]

    def _inverse_holo(self, sr: torch.Tensor, si: torch.Tensor, latent_ch: int) -> torch.Tensor:
        """Exact isometric inverse of _forward_holo: (B, M=32, N_b=384) -> (B, L=256, C=96)."""
        B, M, N_b = sr.shape
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = sr.device, sr.dtype

        if not self.use_holo:
            return torch.cat([sr.reshape(B, K), si.reshape(B, K)], dim=1).reshape(B, L, latent_ch)

        rb_768 = torch.cat([sr, si], dim=-1)  # (B, 32, 768)
        z_disp = rb_768.reshape(B, M, latent_ch // M, L).permute(0, 3, 1, 2).reshape(B, L, latent_ch)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)
        dither = build_channel_sign_dither(L, latent_ch, device=device, dtype=dtype)
        return torch.matmul(h256, z_disp) * dither

    def embed_tx(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Encode z_raw (B, 256, 96) -> s_tx_rb_iq (B, M=32, N_b=384, 2) with exact unit power PER RB!"""
        B, L, C = z_raw.shape
        M = self.num_rbs
        N_b = (L * C) // (2 * M)
        device, dtype = z_raw.device, z_raw.dtype

        pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
        p_img = pwr.sqrt()
        z_unit = z_raw / p_img[:, None, None]

        sr, si = self._forward_holo(z_unit)  # (B, M, N_b)

        # 6 orthonormal Walsh vectors: w0=u1 (probe), w1=u_norm, w2..w5 = 4 local noise/jamming null vectors!
        basis = build_walsh_k_basis(6, N_b, device=device, dtype=dtype)  # (6, N_b)
        u1 = basis[0]
        u_norm = basis[1]

        proj_r = torch.einsum("bmn,kn->bmk", sr, basis)
        proj_i = torch.einsum("bmn,kn->bmk", si, basis)
        sp_r = sr - torch.einsum("bmk,kn->bmn", proj_r, basis)
        sp_i = si - torch.einsum("bmk,kn->bmn", proj_i, basis)

        eta1 = self.eta1
        data_pwr_target = 1.0 - eta1 - self.eta3
        cur_p = (sp_r.square() + sp_i.square()).mean(dim=(1, 2)).clamp_min(1e-8)
        sc = (data_pwr_target / cur_p).sqrt()[:, None, None]
        sd_r = sp_r * sc
        sd_i = sp_i * sc

        theta_p = 1.25 * torch.tanh(0.40 * (p_img - 2.82))
        cos_p = torch.cos(theta_p)[:, None, None]
        sin_p = torch.sin(theta_p)[:, None, None]

        amp1 = (eta1 * float(N_b)).sqrt()
        amp3 = math.sqrt(self.eta3 * float(N_b))

        tx_r = sd_r + amp1 * u1[None, None, :] + (amp3 * cos_p) * u_norm[None, None, :]
        tx_i = sd_i + (amp3 * sin_p) * u_norm[None, None, :]
        return torch.stack([tx_r, tx_i], dim=-1)  # (B, M, N_b, 2)

    def blind_demod_and_equalize(
        self,
        y_rx_rb_iq: torch.Tensor,  # (B, M, N_b, 2)
        latent_ch: int,
    ) -> tuple[ChannelV2Output, torch.Tensor, torch.Tensor]:
        B, M, N_b, _ = y_rx_rb_iq.shape
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = y_rx_rb_iq.device, y_rx_rb_iq.dtype

        yr = y_rx_rb_iq[:, :, :, 0]
        yi = y_rx_rb_iq[:, :, :, 1]

        # 1. Blind Packet-Erasure / Rateless Truncation Detection (E_m < 1e-5 => erased!)
        e_rb = (yr.square() + yi.square()).mean(dim=-1)  # (B, M)
        alive = (e_rb > 1e-5).to(dtype=dtype)            # (B, M) in {0.0, 1.0}
        alive_cnt = alive.sum(dim=1).clamp_min(1.0)      # (B,)

        basis = build_walsh_k_basis(6, N_b, device=device, dtype=dtype)
        u1 = basis[0]
        u_norm = basis[1]
        u_nulls = basis[2:6]  # (4, N_b) -> 8 real DoF per RB!

        # 2. Per-RB Local Interference-Plus-Noise Probe nu_m from 4 Orthogonal Null Vectors!
        qn_r = torch.einsum("bmn,kn->bmk", yr, u_nulls)  # (B, M, 4)
        qn_i = torch.einsum("bmn,kn->bmk", yi, u_nulls)
        nu_rb = (qn_r.square() + qn_i.square()).mean(dim=-1)  # (B, M)

        # Outlier-robust background thermal noise: lower-half trimmed mean across ALIVE RBs (immune to <=45% jammed RBs!)
        nu_for_sort = torch.where(alive > 0.5, nu_rb, torch.full_like(nu_rb, 1e9))
        nu_sorted, _ = torch.sort(nu_for_sort, dim=1)
        idx_grid = torch.arange(M, device=device, dtype=dtype).unsqueeze(0)
        keep_n = (0.5 * alive_cnt).floor().clamp_min(1.0).unsqueeze(1)  # lower 50% of alive RBs!
        trim_mask = (idx_grid < keep_n).to(dtype=dtype)
        # Bias correction factor for lower-50% truncated exponential/chi2_8 mean: 1 / 0.655 = 1.527
        sigma_bg_hat = ((nu_sorted * trim_mask).sum(dim=1) / keep_n.squeeze(1) * 1.527).clamp(1e-5, 15.0)  # (B,)

        # Surgical Jamming Detection: if nu_rb > 2.8 * sigma_bg_hat, use nu_rb directly to excise jammed RB!
        is_jammed = (nu_rb > 2.8 * sigma_bg_hat[:, None]) & (alive > 0.5)
        sigma_rb_hat = torch.where(is_jammed, nu_rb.clamp_max(200.0), sigma_bg_hat[:, None])  # (B, M)

        # 3. Per-RB Blind Channel Estimation on u1 (suppressing erased & jammed RBs during neighbour smoothing!)
        eta1 = self.eta1
        amp1 = (eta1 * float(N_b)).sqrt()
        ht_r = (torch.einsum("bmn,n->bm", yr, u1) / amp1) * alive
        ht_i = (torch.einsum("bmn,n->bm", yi, u1) / amp1) * alive
        sig_e_rb = sigma_rb_hat / (eta1 * float(N_b))  # (B, M)

        # Reliability-weighted 3-tap RB smoother so erased/jammed RBs borrow phase/gain cleanly without polluting clean RBs!
        rel_w = alive / (1.0 + 4.0 * sig_e_rb)  # (B, M)
        k3 = torch.tensor([0.25, 0.50, 0.25], device=device, dtype=dtype).reshape(1, 1, 3)
        num_r = F.conv1d(F.pad((ht_r * rel_w).unsqueeze(1), (1, 1), mode="replicate"), k3).squeeze(1)
        num_i = F.conv1d(F.pad((ht_i * rel_w).unsqueeze(1), (1, 1), mode="replicate"), k3).squeeze(1)
        den_w = F.conv1d(F.pad(rel_w.unsqueeze(1), (1, 1), mode="replicate"), k3).squeeze(1).clamp_min(1e-5)
        hsm_r = num_r / den_w
        hsm_i = num_i / den_w

        w_self = (alive / (1.0 + 2.5 * sig_e_rb)).clamp(0.0, 0.95)
        hhat_r = ((w_self * ht_r + (1.0 - w_self) * hsm_r) / (1.0 + 0.30 * sig_e_rb)) * alive
        hhat_i = ((w_self * ht_i + (1.0 - w_self) * hsm_i) / (1.0 + 0.30 * sig_e_rb)) * alive

        # 4. Interference-Weighted MRC Phase-Angle Norm Recovery on u_norm (ignores erased & jammed RBs!)
        q3_r = torch.einsum("bmn,n->bm", yr, u_norm)
        q3_i = torch.einsum("bmn,n->bm", yi, u_norm)
        mrc_w = alive / (sigma_rb_hat + 0.05)
        z3_r = (mrc_w * (hhat_r * q3_r + hhat_i * q3_i)).sum(dim=1)
        z3_i = (mrc_w * (hhat_r * q3_i - hhat_i * q3_r)).sum(dim=1)
        theta_hat = torch.atan2(z3_i, z3_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)

        # 5. Exact Orthogonal Projection P_{W^perp}(y_m) + Surgical Per-RB MMSE Equalization!
        proj_yr = torch.einsum("bmn,kn->bmk", yr, basis)
        proj_yi = torch.einsum("bmn,kn->bmk", yi, basis)
        yp_r = yr - torch.einsum("bmk,kn->bmn", proj_yr, basis)
        yp_i = yi - torch.einsum("bmk,kn->bmn", proj_yi, basis)

        data_pwr = 1.0 - eta1 - self.eta3
        h_sq = hhat_r.square() + hhat_i.square()
        h_abs = h_sq.clamp_min(1e-10).sqrt()
        eff_var = ((sigma_rb_hat / data_pwr) + 0.5 * sig_e_rb).clamp_min(1e-5)
        denom = h_sq + eff_var
        alpha_rb = (h_sq / denom) * alive  # 0.0 on erased RBs, ~0.02 on +15dB jammed RBs, ~0.92 on clean 10dB RBs!

        # Normalize per-RB MMSE gain ONLY on surviving clean RBs (do NOT amplify erased/jammed RBs to 1!)
        # Soft-shrinkage MMSE: w_m = h_abs / (h_sq + eff_var) * (1 / alpha_clean_mean)
        clean_mask = (alive > 0.5) & (~is_jammed)
        clean_cnt = clean_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
        alpha_clean_bar = ((alpha_rb * clean_mask.to(dtype=dtype)).sum(dim=1) / clean_cnt).clamp_min(0.10)  # (B,)

        cos_rb = (hhat_r / h_abs)[:, :, None]
        sin_rb = (hhat_i / h_abs)[:, :, None]
        yd_r = (cos_rb * yp_r + sin_rb * yp_i) * alive[:, :, None]
        yd_i = (-sin_rb * yp_r + cos_rb * yp_i) * alive[:, :, None]

        dof_comp = math.sqrt(float(N_b - 6) / float(N_b))
        # Notice: on clean RBs, E[w_m * h_abs] ≈ 1.0; on jammed/erased RBs, w_m -> 0 (surgical excision!)
        w_rb = ((h_abs / denom) * alive / alpha_clean_bar[:, None])[:, :, None] * (dof_comp / data_pwr.sqrt())
        sh_r = yd_r * w_rb
        sh_i = yd_i * w_rb

        # Inverse Holographic Dispersion back to (B, L=256, C=96)!
        u_mmse_norm = self._inverse_holo(sh_r, sh_i, latent_ch)
        # Note: if use_holo is True and only fraction rho_alive of RBs survived cleanly, each RB maps to its OWN 3 channels!
        # Wait: check _forward_holo! RB m holds ALL 256 spatial tokens of channels [3m, 3m+1, 3m+2]!
        # So surviving RBs restore their 3 channels at 100.0% full amplitude, while erased/jammed RBs put ~0 in their 3 channels!
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        # Per-channel reliability r_ch in [0, 1] (shape B, 96)
        if self.use_holo:
            r_ch = alpha_rb.unsqueeze(-1).expand(B, M, latent_ch // M).reshape(B, latent_ch)
        else:
            r_ch = alpha_rb.mean(dim=1, keepdim=True).expand(B, latent_ch)

        eff_global_alpha = alpha_rb.mean(dim=1).clamp(0.02, 0.999)
        sinr_eff = eff_global_alpha / (1.0 - eff_global_alpha).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_eff)).clamp(-4.0, 22.0)

        # Map RB-level uncertainty back to spatial-channel grid via _inverse_holo layout
        sig_rb_post = (0.5 * (1.0 - alpha_rb).clamp_min(1e-4)).unsqueeze(-1).expand(B, M, N_b)
        if self.use_holo:
            # Constant across all 256 patches, proportional to overall missing/noisy energy
            sig_iq = sig_rb_post.mean(dim=1, keepdim=True).expand(B, M, N_b).reshape(B, K)
            alp_iq = (alpha_rb / alpha_clean_bar[:, None]).mean(dim=1, keepdim=True).unsqueeze(-1).expand(B, M, N_b).reshape(B, K)
        else:
            sig_iq = sig_rb_post.reshape(B, K)
            alp_iq = (alpha_rb / alpha_clean_bar[:, None]).unsqueeze(-1).expand(B, M, N_b).reshape(B, K)

        ones_k = torch.ones((B, K), device=device, dtype=dtype)
        loss_frac = 1.0 - (clean_cnt / float(M))

        return (
            ChannelV2Output(
                u_restored=u_mmse_restored,
                u_mmse_restored=u_mmse_restored,
                u_norm=u_mmse_norm,
                z_norm=u_mmse_norm,
                pwr=p_img_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=ones_k,
                alpha_ratio_iq=alp_iq,
                sigma_post_iq=sig_iq,
                phase_unc_iq=ones_k * loss_frac[:, None],
                psi_csi=loss_frac * 3.0,
            ),
            cr_hat_db,
            r_ch,
        )


class HoloRateBlindTransceiver(nn.Module):
    """Dir-3B Wrapper:
      - holorate_blind: Full 256-Point Holographic Spatial-Channel Dispersion + 4-Null Anti-Jamming + Channel Completion
      - raster_blind_resilient: Controlled Ablation WITHOUT Holographic Spatial-Channel Interleaving (standard raster RB layout)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "holorate_blind",
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

        use_holo = self.method == "holorate_blind"
        if self.method not in ("holorate_blind", "raster_blind_resilient"):
            raise ValueError(f"Unknown Dir-3B method: {self.method}")

        self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
        self.modem = HoloBlindAntiJamModem(num_rbs=num_rbs, use_holographic_dispersion=use_holo)
        self.ch_comp = HoloChannelCompletionBlock(self.latent_channels, hidden_ch=hidden_ch)
        self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_holo_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        B, L, _ = x_stem.shape
        snr_c0 = torch.full((B, 1), self.base_c0_db, device=x_stem.device, dtype=x_stem.dtype)
        with torch.no_grad():
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                bm = enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = enc.sigmoid(enc.sm_list[-1](temp))
            z_base = enc.head_list(x_stem * mod_val)
        z_tx = self.hier_tx(z_base, x_stem)
        return self.modem.embed_tx(z_tx)  # (B, M=32, N_b=384, 2)

    def decode_holo_rx(self, y_rx_rb_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, N_b, _ = y_rx_rb_iq.shape
        C = self.latent_channels
        L = (2 * M * N_b) // C
        dec = self.backbone.decoder

        ch_v2, cr_hat_db, r_ch = self.modem.blind_demod_and_equalize(y_rx_rb_iq, C)
        # Complete missing/jammed channel-triplets using spatial context + reliability mask r_ch!
        u_completed = self.ch_comp(ch_v2.u_restored, r_ch)
        ch_v2_comp = ChannelV2Output(
            u_restored=u_completed,
            u_mmse_restored=u_completed,
            u_norm=ch_v2.u_norm,
            z_norm=ch_v2.z_norm,
            pwr=ch_v2.pwr,
            gamma_db=ch_v2.gamma_db,
            c_r_db=ch_v2.c_r_db,
            h_abs_iq=ch_v2.h_abs_iq,
            alpha_ratio_iq=ch_v2.alpha_ratio_iq,
            sigma_post_iq=ch_v2.sigma_post_iq,
            phase_unc_iq=ch_v2.phase_unc_iq,
            psi_csi=ch_v2.psi_csi,
        )

        u_cal, phy_map, cond, g_ones = self.blind_dec.forward_frontend(ch_v2_comp, cr_hat_db, base_c0_db=self.base_c0_db)
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

    def forward_hostile_blind(
        self,
        image_or_stem: torch.Tensor,
        true_snr_db: torch.Tensor | float,
        *,
        rho_f: float = 0.85,
        alive_rb_mask: torch.Tensor | None = None,
        jam_rb_mask: torch.Tensor | None = None,
        jam_inr_db: float = 15.0,
        h_rb_override: torch.Tensor | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        jam_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        s_tx_rb = self.encode_holo_tx(x_stem)
        y_rx_rb, diag = physical_hostile_ofdm_channel(
            s_tx_rb,
            true_snr_db,
            rho_f=rho_f,
            alive_rb_mask=alive_rb_mask,
            jam_rb_mask=jam_rb_mask,
            jam_inr_db=jam_inr_db,
            h_rb_override=h_rb_override,
            standard_noise_iq=standard_noise_iq,
            jam_noise_iq=jam_noise_iq,
        )
        recon, cr_hat_db = self.decode_holo_rx(y_rx_rb)
        return {"recon": recon, "cr_hat_db": cr_hat_db, "s_tx_rb": s_tx_rb, **diag}
