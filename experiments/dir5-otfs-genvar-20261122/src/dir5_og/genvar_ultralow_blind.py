"""Dir-5B: GenHolo-VAR-JSCC — Ultra-Low-Bandwidth (CBR = 1/128 [K=1,536] & CBR = 1/256 [K=768])
Generative Perceptual Blind Semantic Communications via Frozen VAR VQ-VAE + Holographic Subspace Modem.

Core Innovations:
  1. Stride-8 256-Point Walsh-Hadamard Holographic Dispersion across M = 8 Narrowband Resource Blocks
     (N_b = 192 REs/RB at CBR=1/128; N_b = 96 REs/RB at CBR=1/256):
     Spreads every spatial patch of the 16x16 VAR latent map uniformly across all 8 RBs while embedding a
     4-Walsh-vector zero-pilot Newman-chirped blind subspace (u1 channel probe, u2 MRC norm anchor, u3-u4
     orthogonal null sensors for automatic noise & deep-fade/erasure detection).
  2. SoftVQManifoldProjector (SNR-Adaptive Continuous + Soft-Codebook Attention + 10-Scale Hierarchical VAR VQ Snapping):
     Projects severely compressed (6 or 12 ch/patch) and noisy/half-erased blind wireless features directly back
     onto the frozen VAR VQ-VAE's 4,096-codeword 10-scale natural-image manifold without modifying a single
     parameter of the pretrained VAR VQ-VAE backbone.
  3. Complete Factorial Comparison against Oracle Genie-CSI Discriminative SwinJSCC (Swin_Genie_CSI),
     0-Pilot Holographic Discriminative SwinJSCC (Swin_Holo_Blind), and 0-Pilot Neural Generative VAR Baseline
     (BlackBox_VAR_Blind) at identical K = 1,536 (CBR=1/128) and K = 768 (CBR=1/256) complex symbol budgets.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from dir2_zpac.blind_v3 import (
    SNR_CENTER,
    SNR_SCALE,
    BlindMultiScaleDecoderCalibrator,
    OpenLoopHierarchicalTX,
)
from dir2_zpac.models import encode_with_ct
from dir2_zpac.phy_v2 import ChannelV2Output, expand_rb_to_symbols, sample_correlated_rbs
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb
from dir3_mh.mimo_blind import build_hadamard_pow2, build_walsh_k_basis

VAR_UPSTREAM_ROOT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets/var_upstream_candidate_v1"
)
DEFAULT_VAR_VQVAE_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028/"
    "var_comm_runtime_20261010/assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/vae_ch160v4096z32.pth"
)

N_UL_RBS = 8  # 8 frequency RBs: N_b = 192 at CBR=1/128 (K=1536), N_b = 96 at CBR=1/256 (K=768)


def cbr_geometry(cbr_denom: int) -> tuple[int, int, int]:
    """Return (K_complex_symbols, N_b_per_rb, C_bot_real_per_token) for 256x256x3 images over M=8 RBs."""
    if cbr_denom == 128:
        return 1536, 192, 12
    if cbr_denom == 256:
        return 768, 96, 6
    raise ValueError(f"Unsupported cbr_denom={cbr_denom}, expected 128 or 256")


def load_frozen_var_vqvae(
    ckpt_path: Path | str = DEFAULT_VAR_VQVAE_CKPT,
    *,
    device: torch.device | str = "cuda:0",
) -> nn.Module:
    """Load official frozen VAR VQ-VAE (V=4096, Cvae=32, 10-scale residual VQ) strictly from liulu PFS."""
    vroot = str(VAR_UPSTREAM_ROOT.resolve())
    if vroot not in sys.path:
        sys.path.insert(0, vroot)
    from models.vqvae import VQVAE

    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=False,
        share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vae.load_state_dict(sd, strict=True)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def build_ultralow_sign_dither(length: int, ch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(20261125 + ch * 97)
    bits = torch.randint(0, 2, (1, length, ch), generator=gen, dtype=torch.int64)
    return (bits.to(dtype=dtype) * 2.0 - 1.0).to(device=device)


def newman_8rb_phases(num_rbs: int = N_UL_RBS, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.arange(num_rbs, device=device, dtype=dtype)
    psi = math.pi * (m * m) / float(num_rbs)
    return torch.cos(psi), torch.sin(psi)


def physical_ultralow_ofdm_channel(
    s_rb_iq: torch.Tensor,          # [B, M=8, N_b, 2]
    snr_db: torch.Tensor | float,   # Hidden physical Es/N0 in dB
    *,
    rho_f: float = 0.85,
    alive_rb_mask: torch.Tensor | None = None,   # [B, M=8] in {0.0, 1.0} (for deep-fade / half-band erasure!)
    h_rb_override: torch.Tensor | None = None,   # [B, M=8, 2]
    standard_noise_iq: torch.Tensor | None = None,  # [B, M=8, N_b, 2]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Physical Narrowband 8-RB Frequency-Selective Rayleigh OFDM Channel with strict unit TX frame power."""
    B, M, N_b, two = s_rb_iq.shape
    assert M == N_UL_RBS and two == 2
    device, dtype = s_rb_iq.device, s_rb_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )

    # Enforce strict unit average complex symbol power per transmitted frame
    in_pwr = (s_rb_iq[..., 0].square() + s_rb_iq[..., 1].square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    s_unit = s_rb_iq / in_pwr.sqrt().unsqueeze(-1)

    if h_rb_override is not None:
        hr = h_rb_override[:, :, 0].to(device=device, dtype=dtype)
        hi = h_rb_override[:, :, 1].to(device=device, dtype=dtype)
    else:
        hr, hi = sample_correlated_rbs(B, M, rho_f, device=device, dtype=dtype)

    sr, si = s_unit[..., 0], s_unit[..., 1]
    yr = hr[:, :, None] * sr - hi[:, :, None] * si
    yi = hr[:, :, None] * si + hi[:, :, None] * sr

    sigma_n_sq = torch.pow(10.0, -snr_t / 10.0)
    sigma_r = (0.5 * sigma_n_sq).sqrt()[:, None, None]
    eps = (
        torch.randn_like(s_unit)
        if standard_noise_iq is None
        else standard_noise_iq.to(device=device, dtype=dtype).reshape(B, M, N_b, 2)
    )
    yr = yr + eps[..., 0] * sigma_r
    yi = yi + eps[..., 1] * sigma_r

    if alive_rb_mask is not None:
        amask = alive_rb_mask.to(device=device, dtype=dtype).reshape(B, M, 1)
        yr = yr * amask
        yi = yi * amask
    else:
        amask = torch.ones((B, M, 1), device=device, dtype=dtype)

    return torch.stack([yr, yi], dim=-1), {
        "true_snr_db": snr_t,
        "true_sigma_n_sq": sigma_n_sq,
        "hr_rb": hr,
        "hi_rb": hi,
        "alive_rb_mask": amask.squeeze(-1),
    }


class ConvNeXt2DBlock(nn.Module):
    """Residual 2D Spatial-Channel ConvNeXt Block with GroupNorm & SiLU on 16x16 feature maps."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        groups = 8 if dim % 8 == 0 else 4
        self.dw = nn.Conv2d(
            dim, dim, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="reflect", groups=dim
        )
        self.norm = nn.GroupNorm(groups, dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.pw2 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw2(F.silu(self.pw1(h)))
        return x + self.gamma * h


class UltraLowHoloSubspaceModem(nn.Module):
    """Zero-Pilot 8-RB Holographic Subspace Modem for Ultra-Low CBR = 1/128 (N_b=192) & CBR = 1/256 (N_b=96):
      - 256-point Orthonormal Walsh-Hadamard Spatial Dispersion + Stride-8 RB Interleaving
      - 4-Walsh-Vector Per-RB Orthogonal Decomposition:
          w0: Newman-chirped blind channel probe (eta1 ~ 0.055)
          w1: 8-RB MRC scale-invariant latent norm phase-angle anchor (eta2 = 0.012)
          w2, w3: 2-dim orthogonal null subspace for exact blind noise variance & erased-RB detection
      - Exact RX orthogonal subspace projection P_{span(w0..w3)^perp} (0.000% self-interference!)
      - Coherence-adaptive 8-RB Wiener smoothing + automatic dead/erased RB masking.
    """

    def __init__(self, cbr_denom: int = 128, *, use_holo: bool = True):
        super().__init__()
        self.cbr_denom = int(cbr_denom)
        self.k_sym, self.n_b, self.c_bot = cbr_geometry(self.cbr_denom)
        self.num_rbs = N_UL_RBS
        self.use_holo = bool(use_holo)

        eta1_init = 0.055 if self.cbr_denom == 128 else 0.070
        inv_s = math.log((eta1_init - 0.025) / (0.140 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta2 = 0.012

        self.rb_refiner = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.rb_refiner[-1].weight)
        nn.init.zeros_(self.rb_refiner[-1].bias)

    @property
    def eta1(self) -> torch.Tensor:
        return 0.025 + (0.140 - 0.025) * torch.sigmoid(self.eta1_raw)

    def _forward_holo(self, z_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [B, L=256, C_bot] -> (sr, si) each [B, M=8, N_b]."""
        B, L, C = z_unit.shape
        assert L == 256 and C == self.c_bot
        M, N_b = self.num_rbs, self.n_b
        device, dtype = z_unit.device, z_unit.dtype

        if not self.use_holo:
            flat = z_unit.reshape(B, 2 * M * N_b)
            K = M * N_b
            return flat[:, :K].reshape(B, M, N_b), flat[:, K:].reshape(B, M, N_b)

        dither = build_ultralow_sign_dither(L, C, device=device, dtype=dtype)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)  # [256, 256]
        z_disp = torch.matmul(h256, z_unit * dither)  # [B, 256, C_bot]

        # Stride-8 sequency interleaving: reshape [B, 32, 8, C_bot] -> permute [B, 8, 32, C_bot] -> [B, 8, 2*N_b]
        # Every RB m in {0..7} receives 32 uniformly strided global Hadamard frequencies across ALL C_bot channels!
        rb_2nb = z_disp.reshape(B, L // M, M, C).permute(0, 2, 1, 3).reshape(B, M, 2 * N_b)
        return rb_2nb[:, :, :N_b], rb_2nb[:, :, N_b:]

    def _inverse_holo(self, sr: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
        """Exact isometric inverse of _forward_holo: [B, M=8, N_b] -> [B, L=256, C_bot]."""
        B, M, N_b = sr.shape
        L, C = 256, self.c_bot
        device, dtype = sr.device, sr.dtype

        if not self.use_holo:
            K = M * N_b
            return torch.cat([sr.reshape(B, K), si.reshape(B, K)], dim=1).reshape(B, L, C)

        rb_2nb = torch.cat([sr, si], dim=-1)  # [B, 8, 2*N_b]
        z_disp = rb_2nb.reshape(B, M, L // M, C).permute(0, 2, 1, 3).reshape(B, L, C)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)
        dither = build_ultralow_sign_dither(L, C, device=device, dtype=dtype)
        return torch.matmul(h256, z_disp) * dither

    def embed_tx(self, z_bot: torch.Tensor) -> torch.Tensor:
        """Embed [B, 256, C_bot] -> [B, M=8, N_b, 2] unit-power zero-pilot RB symbols."""
        B, L, C = z_bot.shape
        assert L == 256 and C == self.c_bot
        M, N_b = self.num_rbs, self.n_b
        device, dtype = z_bot.device, z_bot.dtype

        p_img = (2.0 * z_bot.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()  # [B]
        z_unit = z_bot / p_img[:, None, None]  # E[z_unit^2] == 0.5

        sr, si = self._forward_holo(z_unit)  # [B, 8, N_b]
        basis = build_walsh_k_basis(4, N_b, device=device, dtype=dtype)  # [4, N_b]
        w0, w1 = basis[0], basis[1]

        # Project payload strictly orthogonal to span(w0, w1, w2, w3)
        proj_r = torch.einsum("bmn,kn->bmk", sr, basis)
        proj_i = torch.einsum("bmn,kn->bmk", si, basis)
        sr_perp = sr - torch.einsum("bmk,kn->bmn", proj_r, basis)
        si_perp = si - torch.einsum("bmk,kn->bmn", proj_i, basis)

        eta1 = self.eta1
        data_pwr = 1.0 - eta1 - self.eta2
        dof_gain = math.sqrt(float(N_b) / float(N_b - 4))
        d_scale = data_pwr.sqrt() * dof_gain

        ncos, nsin = newman_8rb_phases(M, device=device, dtype=dtype)  # [8]
        amp1 = (eta1 * float(N_b)).sqrt()
        p0_r = amp1 * ncos[None, :, None] * w0[None, None, :]
        p0_i = amp1 * nsin[None, :, None] * w0[None, None, :]

        # Encode p_img around center 1.0 (or natural bottleneck RMS) via atan2 phase angle on w1
        theta_p = 1.20 * torch.tanh(0.45 * (torch.log(p_img.clamp_min(1e-3))))  # [B] in (-1.2, +1.2)
        cp, sp = torch.cos(theta_p)[:, None], torch.sin(theta_p)[:, None]  # [B, 1]
        # Multiply exp(j*psi_m) * exp(j*theta_p)
        rot_r = ncos[None, :] * cp - nsin[None, :] * sp
        rot_i = nsin[None, :] * cp + ncos[None, :] * sp
        amp2 = math.sqrt(self.eta2 * float(N_b))
        p1_r = amp2 * rot_r[:, :, None] * w1[None, None, :]
        p1_i = amp2 * rot_i[:, :, None] * w1[None, None, :]

        tx_r = d_scale * sr_perp + p0_r + p1_r
        tx_i = d_scale * si_perp + p0_i + p1_i
        return torch.stack([tx_r, tx_i], dim=-1)

    def blind_demod_and_equalize(
        self,
        y_rb_iq: torch.Tensor,  # [B, M=8, N_b, 2]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return (z_bot_restored [B, 256, C_bot], cr_hat_db [B], aux_dict)."""
        B, M, N_b, _ = y_rb_iq.shape
        assert M == self.num_rbs and N_b == self.n_b
        device, dtype = y_rb_iq.device, y_rb_iq.dtype

        yr, yi = y_rb_iq[..., 0], y_rb_iq[..., 1]
        basis = build_walsh_k_basis(4, N_b, device=device, dtype=dtype)
        w0, w1, w_null = basis[0], basis[1], basis[2:4]  # w_null: [2, N_b]

        # 1. Detect erased/dead RBs blindly from per-RB total received energy!
        rb_energy = (yr.square() + yi.square()).mean(dim=-1)  # [B, 8]
        peak_rb = rb_energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        alive_hat = (rb_energy > (1e-4 * peak_rb)).to(dtype)  # [B, 8]: 0.0 on erased RBs, 1.0 on active RBs!
        alive_count = alive_hat.sum(dim=-1).clamp_min(1.0)    # [B]
        alive_frac = alive_count / float(M)                   # [B] in (0, 1]

        # 2. Blind thermal noise variance from 2-dim orthogonal null subspace w_null on alive RBs
        null_r = torch.einsum("bmn,kn->bmk", yr, w_null)
        null_i = torch.einsum("bmn,kn->bmk", yi, w_null)
        null_pwr_rb = (null_r.square() + null_i.square()).mean(dim=-1)  # [B, 8]
        sigma_n_hat = ((null_pwr_rb * alive_hat).sum(dim=-1) / alive_count).clamp_min(1e-5)  # [B]

        # 3. Raw zero-pilot channel projection on w0 + undo Newman 8-RB quadratic chirp
        eta1 = self.eta1
        amp1 = (eta1 * float(N_b)).sqrt()
        q0_r = torch.einsum("bmn,n->bm", yr, w0) / amp1
        q0_i = torch.einsum("bmn,n->bm", yi, w0) / amp1
        ncos, nsin = newman_8rb_phases(M, device=device, dtype=dtype)
        htilde_r = (q0_r * ncos[None, :] + q0_i * nsin[None, :]) * alive_hat
        htilde_i = (q0_i * ncos[None, :] - q0_r * nsin[None, :]) * alive_hat

        sigma_e_raw = sigma_n_hat / (eta1 * float(N_b))
        hw_r, hw_i, sigma_e_eff, delta_f = coherence_adaptive_wiener_rb(htilde_r, htilde_i, sigma_e_raw)
        log_sig = torch.log(sigma_n_hat)[:, None].expand(B, M).div(4.0)
        df_feat = delta_f[:, None].expand(B, M)
        delta_h = self.rb_refiner(torch.stack([htilde_r, htilde_i, rb_energy, log_sig, hw_r, hw_i, df_feat], dim=1))
        hhat_r = (hw_r + delta_h[:, 0, :]) * alive_hat
        hhat_i = (hw_i + delta_h[:, 1, :]) * alive_hat

        # 4. Alive-RB Maximal-Ratio Combined (MRC) phase-angle recovery of p_img from w1
        q1_r = torch.einsum("bmn,n->bm", yr, w1)
        q1_i = torch.einsum("bmn,n->bm", yi, w1)
        u1_r = (q1_r * ncos[None, :] + q1_i * nsin[None, :]) * alive_hat
        u1_i = (q1_i * ncos[None, :] - q1_r * nsin[None, :]) * alive_hat
        z1_r = (hhat_r * u1_r + hhat_i * u1_i).sum(dim=1)
        z1_i = (hhat_r * u1_i - hhat_i * u1_r).sum(dim=1)
        theta_hat = torch.atan2(z1_i, z1_r.clamp_min(1e-5)).clamp(-1.16, 1.16)
        p_img_hat = torch.exp(torch.atanh(theta_hat / 1.20) / 0.45).clamp(0.08, 12.0)  # [B]

        # 5. Exact Orthogonal Subspace Projection P_{span(w0..w3)^perp}(y_m)
        proj_yr = torch.einsum("bmn,kn->bmk", yr, basis)
        proj_yi = torch.einsum("bmn,kn->bmk", yi, basis)
        yp_r = (yr - torch.einsum("bmk,kn->bmn", proj_yr, basis)) * alive_hat[:, :, None]
        yp_i = (yi - torch.einsum("bmk,kn->bmn", proj_yi, basis)) * alive_hat[:, :, None]

        # 6. Gain-Unbiased Per-RB MMSE Equalization + Alive-Fraction Energy Compensation for Inverse Hadamard
        data_pwr = 1.0 - eta1 - self.eta2
        h_sq = hhat_r.square() + hhat_i.square()
        h_abs = h_sq.clamp_min(1e-10).sqrt()
        eff_var = ((sigma_n_hat / data_pwr) + sigma_e_eff)[:, None]
        denom = (h_sq + eff_var).clamp_min(1e-6)
        alpha_rb = (h_sq / denom) * alive_hat
        # Normalize by mean alpha across ALL M RBs so surviving RBs reconstruct unbiased global Hadamard energy!
        alpha_bar_all = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.025)
        alpha_bar_alive = (alpha_rb.sum(dim=1) / alive_count).clamp(0.025, 0.999)

        cos_rb = (hhat_r / h_abs)[:, :, None]
        sin_rb = (hhat_i / h_abs)[:, :, None]
        yd_r = cos_rb * yp_r + sin_rb * yp_i
        yd_i = -sin_rb * yp_r + cos_rb * yp_i

        dof_comp = math.sqrt(float(N_b - 4) / float(N_b))
        d_scale = (1.0 / data_pwr).sqrt() * dof_comp
        w_ub = ((h_abs / denom) / alpha_bar_all)[:, :, None] * d_scale * alive_hat[:, :, None]

        sh_r = yd_r * w_ub
        sh_i = yd_i * w_ub

        z_unit_hat = self._inverse_holo(sh_r, sh_i)  # [B, 256, C_bot]
        z_bot_restored = z_unit_hat * p_img_hat[:, None, None]

        sinr_eff = (alpha_bar_alive * alive_frac).clamp_min(1e-4) / (1.0 - alpha_bar_alive * alive_frac + 1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_eff)).clamp(-6.0, 22.0)

        return (
            z_bot_restored,
            cr_hat_db,
            {
                "z_unit_hat": z_unit_hat,
                "p_img_hat": p_img_hat,
                "alive_frac": alive_frac,
                "alive_hat": alive_hat,
                "alpha_rb": alpha_rb,
                "alpha_bar_all": alpha_bar_all.squeeze(1),
                "h_abs_rb": h_abs,
                "sigma_e_eff": sigma_e_eff,
            },
        )


class BlackBoxUltraLowModem(nn.Module):
    """Pure Neural 0-Pilot Narrowband 8-RB Modem Baseline (no H_256 holographic dispersion, no Walsh subspace)."""

    def __init__(self, cbr_denom: int = 128):
        super().__init__()
        self.cbr_denom = int(cbr_denom)
        self.k_sym, self.n_b, self.c_bot = cbr_geometry(self.cbr_denom)
        self.num_rbs = N_UL_RBS

        self.rb_net = nn.Sequential(
            nn.Conv1d(5, 48, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(48, 3, kernel_size=1),
        )
        nn.init.zeros_(self.rb_net[-1].weight)
        nn.init.zeros_(self.rb_net[-1].bias)

        self.snr_mlp = nn.Sequential(
            nn.Linear(5, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )
        nn.init.zeros_(self.snr_mlp[-1].weight)
        nn.init.zeros_(self.snr_mlp[-1].bias)

    def embed_tx(self, z_bot: torch.Tensor) -> torch.Tensor:
        B, L, C = z_bot.shape
        M, N_b = self.num_rbs, self.n_b
        K = M * N_b
        p_img = (2.0 * z_bot.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
        flat = (z_bot / p_img[:, None, None]).reshape(B, 2 * K)
        return torch.stack([flat[:, :K].reshape(B, M, N_b), flat[:, K:].reshape(B, M, N_b)], dim=-1)

    def blind_demod_and_equalize(
        self,
        y_rb_iq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, M, N_b, _ = y_rb_iq.shape
        K = M * N_b
        yr, yi = y_rb_iq[..., 0], y_rb_iq[..., 1]
        rb_e = (yr.square() + yi.square()).mean(dim=-1)  # [B, 8]
        tot_e = rb_e.mean(dim=-1).clamp_min(1e-5)
        peak_e = rb_e.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        alive_hat = (rb_e > (1e-4 * peak_e)).to(yr.dtype)
        alive_frac = alive_hat.mean(dim=-1).clamp_min(0.125)

        mean_r = yr.mean(dim=-1)
        mean_i = yi.mean(dim=-1)
        std_r = yr.std(dim=-1)
        std_i = yi.std(dim=-1)
        rb_feat = torch.stack([rb_e, mean_r, mean_i, std_r, std_i], dim=1)  # [B, 5, 8]
        ctrl = self.rb_net(rb_feat)
        gain = (1.0 + 0.5 * torch.tanh(ctrl[:, 0:1, :])).transpose(1, 2) / tot_e.sqrt()[:, None, None]
        phi = math.pi * torch.tanh(ctrl[:, 1:2, :]).transpose(1, 2)
        cp, sp = torch.cos(phi), torch.sin(phi)
        eq_r = gain * (cp * yr + sp * yi)
        eq_i = gain * (-sp * yr + cp * yi)

        g_stats = torch.stack([
            torch.log(tot_e),
            rb_e.std(dim=-1),
            alive_frac,
            std_r.mean(dim=-1),
            std_i.mean(dim=-1),
        ], dim=-1)
        pred = self.snr_mlp(g_stats)
        cr_hat_db = (SNR_CENTER + pred[:, 0] * SNR_SCALE).clamp(-6.0, 22.0)
        p_img_hat = torch.exp(0.6 * torch.tanh(pred[:, 1])).clamp(0.1, 8.0)

        z_unit_hat = torch.cat([eq_r.reshape(B, K), eq_i.reshape(B, K)], dim=1).reshape(B, 256, self.c_bot)
        z_restored = z_unit_hat * p_img_hat[:, None, None]
        return (
            z_restored,
            cr_hat_db,
            {
                "z_unit_hat": z_unit_hat,
                "p_img_hat": p_img_hat,
                "alive_frac": alive_frac,
                "alive_hat": alive_hat,
            },
        )


class SoftVQManifoldProjector(nn.Module):
    """Projects continuous reconstructed latent [B, 32, 16, 16] onto the frozen VAR VQ-VAE perceptual manifold:
      1) Differentiable Cosine Soft-Attention Lookup over VAR's frozen 4,096 x 32 codebook E
      2) Straight-Through 10-Scale Hierarchical Residual VQ Snapping via frozen vae.quantize
      3) Blind SNR & Alive-Fraction Gated Convex Manifold Fusion + 2-Stage ConvNeXt Polisher.
    """

    def __init__(self, z_ch: int = 32, hidden_ch: int = 160):
        super().__init__()
        self.z_ch = int(z_ch)
        self.log_temp = nn.Parameter(torch.tensor(math.log(18.0)))

        # Gate MLP takes [cr_norm, alive_frac] -> (w_vq_10scale, w_soft_codebook) per channel
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * z_ch),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        # Initialize bias to gentle positive manifold pull (sigmoid(-0.4) * 0.85 ~ 0.34)
        nn.init.constant_(self.gate_mlp[-1].bias, -0.4)

        self.post_refine = nn.Sequential(
            nn.Conv2d(z_ch * 2 + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, z_ch, kernel_size=1),
        )
        nn.init.zeros_(self.post_refine[-1].weight)
        nn.init.zeros_(self.post_refine[-1].bias)

    def forward(
        self,
        f_cont: torch.Tensor,       # [B, 32, 16, 16]
        vae: nn.Module,             # Frozen VAR VQVAE
        cr_hat_db: torch.Tensor,    # [B]
        alive_frac: torch.Tensor,   # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_cont.shape
        assert C == self.z_ch and H == 16 and W == 16

        # 1. Soft Cosine Codebook Attention onto frozen [4096, 32] codebook
        cb_weight = vae.quantize.embedding.weight.detach()  # [4096, 32]
        cb_norm = F.normalize(cb_weight, dim=-1)            # [4096, 32]
        q_flat = f_cont.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 256, 32]
        q_mag = q_flat.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        q_norm = q_flat / q_mag

        temp = self.log_temp.exp().clamp(2.0, 60.0)
        attn = F.softmax(temp * torch.matmul(q_norm, cb_norm.t()), dim=-1)  # [B, 256, 4096]
        cb_proj = torch.matmul(attn, cb_weight)  # [B, 256, 32]
        cb_proj_mag = cb_proj.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        f_soft_cb = (cb_proj * (q_mag / cb_proj_mag)).permute(0, 2, 1).reshape(B, C, H, W)

        # 2. Frozen 10-Scale Multi-Resolution VAR VQ Snapping with Straight-Through Estimator
        with torch.no_grad():
            f_vq_hard = vae.quantize.f_to_idxBl_or_fhat(f_cont.detach(), to_fhat=True)[-1]
        f_vq_ste = f_cont + (f_vq_hard - f_cont).detach()

        # 3. Blind SNR + Alive-RB Reliability Gated Manifold Fusion
        cr_norm = (cr_hat_db - SNR_CENTER) / SNR_SCALE
        cond = torch.stack([cr_norm, alive_frac], dim=-1)  # [B, 2]
        gates = 0.65 * torch.sigmoid(self.gate_mlp(cond))  # [B, 64]
        w_vq, w_cb = gates.chunk(2, dim=-1)
        w_vq_4d = w_vq[:, :, None, None]
        w_cb_4d = w_cb[:, :, None, None]

        f_manifold = f_cont + w_vq_4d * (f_vq_ste - f_cont) + w_cb_4d * (f_soft_cb - f_cont)

        cr_map = cr_norm[:, None, None, None].expand(B, 1, H, W)
        af_map = alive_frac[:, None, None, None].expand(B, 1, H, W)
        delta = self.post_refine(torch.cat([f_manifold, f_cont, cr_map, af_map], dim=1))
        f_out = f_manifold + delta
        return f_out, f_cont


class GenHoloVARTransceiver(nn.Module):
    """Unified Generative VAR VQ-VAE Ultra-Low-CBR Transceiver:
      - genholo_var:  Proposed Holographic 8-RB Zero-Pilot Subspace + SoftVQManifoldProjector
      - blackbox_var: Pure Neural Blind 8-RB Modem + Continuous ConvNeXt Decompressor (same frozen VAR VQVAE & loss!)
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        cbr_denom: int = 128,
        method: str = "genholo_var",
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.vae = vae  # Already frozen & shared/eval
        for p in self.vae.parameters():
            p.requires_grad = False

        self.cbr_denom = int(cbr_denom)
        self.method = method.lower()
        self.k_sym, self.n_b, self.c_bot = cbr_geometry(self.cbr_denom)
        self.z_ch = 32

        # TX & RX Architecture:
        #   - genholo_var uses Multi-Scale Pyramid Bottleneck matched to VAR's coarse-to-fine latent geometry:
        #       * CBR=1/256 (1,536 floats = 768 syms): 8x8 x 24-ch semantic base (preserves 75% of 32-D VQ channel space!)
        #       * CBR=1/128 (3,072 floats = 1,536 syms): 8x8 x 32-ch full VQ base (2,048) + 16x16 x 4-ch edge residual (1,024)
        #   - blackbox_var uses standard 16x16 x C_bot single-scale neural bottleneck
        if self.method == "genholo_var":
            self.pyr_ch = 32 if self.cbr_denom == 128 else 24
            self.fine_ch = 4 if self.cbr_denom == 128 else 0
            self.tx_pyr_down = nn.Sequential(
                nn.Conv2d(self.z_ch, hidden_ch, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                nn.SiLU(),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                nn.Conv2d(hidden_ch, self.pyr_ch, kernel_size=1),
            )
            self.tx_fine = (
                nn.Sequential(
                    nn.Conv2d(self.z_ch, hidden_ch // 2, kernel_size=1),
                    nn.SiLU(),
                    ConvNeXt2DBlock(hidden_ch // 2, dilation=1),
                    nn.Conv2d(hidden_ch // 2, self.fine_ch, kernel_size=1),
                )
                if self.fine_ch > 0
                else None
            )
            self.modem: nn.Module = UltraLowHoloSubspaceModem(self.cbr_denom, use_holo=True)
            self.rx_pyr_up = nn.Sequential(
                nn.Conv2d(self.pyr_ch + 2, hidden_ch, kernel_size=1),
                nn.SiLU(),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                nn.Conv2d(hidden_ch, hidden_ch * 4, kernel_size=1),
                nn.PixelShuffle(2),  # [B, hidden_ch, 16, 16] learned sub-pixel upsampling!
                nn.SiLU(),
            )
            rx_merge_in = hidden_ch + self.fine_ch + 2
            self.rx_decompressor = nn.Sequential(
                nn.Conv2d(rx_merge_in, hidden_ch, kernel_size=1),
                nn.SiLU(),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=2),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                nn.Conv2d(hidden_ch, self.z_ch, kernel_size=1),
            )
            self.vq_proj: SoftVQManifoldProjector | None = SoftVQManifoldProjector(self.z_ch, hidden_ch=hidden_ch)
        elif self.method == "blackbox_var":
            self.pyr_ch = 0
            self.fine_ch = self.c_bot
            self.tx_compressor = nn.Sequential(
                nn.Conv2d(self.z_ch, hidden_ch, kernel_size=1),
                nn.SiLU(),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                nn.Conv2d(hidden_ch, self.c_bot, kernel_size=1),
            )
            self.modem = BlackBoxUltraLowModem(self.cbr_denom)
            self.rx_decompressor = nn.Sequential(
                nn.Conv2d(self.c_bot + 2, hidden_ch, kernel_size=1),
                nn.SiLU(),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=2),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                ConvNeXt2DBlock(hidden_ch, dilation=1),
                nn.Conv2d(hidden_ch, self.z_ch, kernel_size=1),
            )
            self.vq_proj = None
        else:
            raise ValueError(f"Unknown GenHoloVAR method: {self.method}")

    @torch.no_grad()
    def extract_var_latents(self, image_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen (f_enc, f_vq) each [B, 32, 16, 16] from RGB image in [0, 1]."""
        self.vae.eval()
        x_m11 = image_01 * 2.0 - 1.0
        f_enc = self.vae.quant_conv(self.vae.encoder(x_m11))
        f_vq = self.vae.quantize.f_to_idxBl_or_fhat(f_enc, to_fhat=True)[-1]
        return f_enc, f_vq

    def decode_fhat_to_01(self, f_out: torch.Tensor) -> torch.Tensor:
        """Differentiable decode through frozen VAR VQ-VAE post_quant_conv + decoder -> RGB in [0, 1]."""
        rec_m11 = self.vae.decoder(self.vae.post_quant_conv(f_out))
        return (rec_m11.clamp(-1.0, 1.0) + 1.0) * 0.5

    def forward_from_fenc(
        self,
        f_enc: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        rho_f: float = 0.85,
        alive_rb_mask: torch.Tensor | None = None,
        h_rb_override: torch.Tensor | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        decode_pixels: bool = True,
    ) -> dict[str, torch.Tensor]:
        B = f_enc.shape[0]
        if self.method == "genholo_var":
            # 8x8 multi-scale base: [B, pyr_ch, 8, 8] -> [B, 4, pyr_ch//4, 64] -> [B, 256, pyr_ch//4]
            z_pyr = self.tx_pyr_down(f_enc)  # [B, 32 or 24, 8, 8]
            c_pyr_tok = self.pyr_ch // 4     # 8 (for CBR=1/128) or 6 (for CBR=1/256)
            z_pyr_tok = z_pyr.reshape(B, 4, c_pyr_tok, 64).permute(0, 1, 3, 2).reshape(B, 256, c_pyr_tok)
            if self.tx_fine is not None:
                z_fine_tok = self.tx_fine(f_enc).permute(0, 2, 3, 1).reshape(B, 256, self.fine_ch)
                z_bot = torch.cat([z_pyr_tok, z_fine_tok], dim=-1).contiguous()  # [B, 256, 12]
            else:
                z_bot = z_pyr_tok.contiguous()  # [B, 256, 6] (exact 768 complex symbols!)
        else:
            z_2d = self.tx_compressor(f_enc)  # [B, C_bot, 16, 16]
            z_bot = z_2d.permute(0, 2, 3, 1).reshape(B, 256, self.c_bot).contiguous()

        s_rb_iq = self.modem.embed_tx(z_bot)  # [B, 8, N_b, 2]
        y_rb_iq, phy = physical_ultralow_ofdm_channel(
            s_rb_iq,
            snr_db,
            rho_f=rho_f,
            alive_rb_mask=alive_rb_mask,
            h_rb_override=h_rb_override,
            standard_noise_iq=standard_noise_iq,
        )
        z_hat, cr_hat_db, aux = self.modem.blind_demod_and_equalize(y_rb_iq)

        if self.method == "genholo_var":
            c_pyr_tok = self.pyr_ch // 4
            zp_hat = z_hat[:, :, :c_pyr_tok].reshape(B, 4, 64, c_pyr_tok).permute(0, 1, 3, 2).reshape(B, self.pyr_ch, 8, 8)
            cr_8 = ((cr_hat_db - SNR_CENTER) / SNR_SCALE)[:, None, None, None].expand(B, 1, 8, 8)
            af_8 = aux["alive_frac"][:, None, None, None].expand(B, 1, 8, 8)
            up_16 = self.rx_pyr_up(torch.cat([zp_hat, cr_8, af_8], dim=1))
            cr_16 = ((cr_hat_db - SNR_CENTER) / SNR_SCALE)[:, None, None, None].expand(B, 1, 16, 16)
            af_16 = aux["alive_frac"][:, None, None, None].expand(B, 1, 16, 16)
            if self.fine_ch > 0:
                zf_hat = z_hat[:, :, c_pyr_tok:].reshape(B, 16, 16, self.fine_ch).permute(0, 3, 1, 2).contiguous()
                f_cont = self.rx_decompressor(torch.cat([up_16, zf_hat, cr_16, af_16], dim=1))
            else:
                f_cont = self.rx_decompressor(torch.cat([up_16, cr_16, af_16], dim=1))
        else:
            z_hat_2d = z_hat.reshape(B, 16, 16, self.c_bot).permute(0, 3, 1, 2).contiguous()
            cr_norm = ((cr_hat_db - SNR_CENTER) / SNR_SCALE)[:, None, None, None].expand(B, 1, 16, 16)
            af_map = aux["alive_frac"][:, None, None, None].expand(B, 1, 16, 16)
            f_cont = self.rx_decompressor(torch.cat([z_hat_2d, cr_norm, af_map], dim=1))

        if self.vq_proj is not None:
            f_out, f_pre_vq = self.vq_proj(f_cont, self.vae, cr_hat_db, aux["alive_frac"])
        else:
            f_out, f_pre_vq = f_cont, f_cont

        out: dict[str, torch.Tensor] = {
            "f_out": f_out,
            "f_pre_vq": f_pre_vq,
            "c_r_hat_db": cr_hat_db,
            "alive_frac": aux["alive_frac"],
            "s_rb_iq": s_rb_iq,
        }
        if decode_pixels:
            out["recon"] = self.decode_fhat_to_01(f_out)
        return out

    def forward_ultralow(
        self,
        image_01: torch.Tensor,
        snr_db: torch.Tensor | float,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        f_enc, _ = self.extract_var_latents(image_01)
        return self.forward_from_fenc(f_enc, snr_db, **kwargs)


class SwinUltraLowTransceiver(nn.Module):
    """Discriminative SwinJSCC Ultra-Low-CBR Counterparts at IDENTICAL K = 1,536 (CBR=1/128) & K = 768 (CBR=1/256):
      - swin_genie_csi:  Oracle Exact Genie-CSI + SNR-Adaptive TX/RX + 100% Payload DoF + H_256 Dispersion
                         (Establishes the Theoretical Discriminative Pixel-MSE Perceptual Ceiling!)
      - swin_holo_blind: Zero-Pilot 8-RB Holographic Subspace Modem (Exact Backbone/Objective Counterpart to GenHolo-VAR!)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        cbr_denom: int = 128,
        method: str = "swin_holo_blind",
        base_c0_db: float = 7.0,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.cbr_denom = int(cbr_denom)
        self.method = method.lower()
        self.base_c0_db = float(base_c0_db)
        self.k_sym, self.n_b, self.c_bot = cbr_geometry(self.cbr_denom)
        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx = (
            OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            if self.method == "swin_holo_blind"
            else None
        )
        self.tx_compressor = nn.Sequential(
            nn.Conv2d(self.latent_channels, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.c_bot, kernel_size=1),
        )
        self.rx_decompressor = nn.Sequential(
            nn.Conv2d(self.c_bot + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=2),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.latent_channels, kernel_size=1),
        )

        if self.method == "swin_holo_blind":
            self.modem: UltraLowHoloSubspaceModem | None = UltraLowHoloSubspaceModem(self.cbr_denom, use_holo=True)
        elif self.method == "swin_genie_csi":
            self.modem = None
        else:
            raise ValueError(f"Unknown SwinUltraLow method: {self.method}")

        self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def _encode_96(self, image_or_stem: torch.Tensor, snr_db: torch.Tensor | float, input_is_stem: bool) -> torch.Tensor:
        enc = self.backbone.encoder
        if not input_is_stem and self.method == "swin_genie_csi":
            z_base, _ = encode_with_ct(self.backbone, image_or_stem, snr_db)
            return z_base

        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        B, L, _ = x_stem.shape
        if self.method == "swin_genie_csi":
            snr_col = (
                torch.full((B, 1), float(snr_db), device=x_stem.device, dtype=x_stem.dtype)
                if not torch.is_tensor(snr_db)
                else snr_db.to(device=x_stem.device, dtype=x_stem.dtype).reshape(B, 1)
            )
        else:
            snr_col = torch.full((B, 1), self.base_c0_db, device=x_stem.device, dtype=x_stem.dtype)

        with torch.no_grad():
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                temp = temp * enc.bm_list[i](snr_col).unsqueeze(1).expand(-1, L, -1)
            z_base = enc.head_list(x_stem * enc.sigmoid(enc.sm_list[-1](temp)))
        return self.hier_tx(z_base, x_stem) if self.hier_tx is not None else z_base

    def forward_ultralow(
        self,
        image_or_stem: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        rho_f: float = 0.85,
        alive_rb_mask: torch.Tensor | None = None,
        h_rb_override: torch.Tensor | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        z_96 = self._encode_96(image_or_stem, snr_db, input_is_stem=input_is_stem)
        B, L, C = z_96.shape
        M, N_b = N_UL_RBS, self.n_b
        K_96 = (L * C) // 2
        device, dtype = z_96.device, z_96.dtype

        z_96_2d = z_96.reshape(B, 16, 16, C).permute(0, 3, 1, 2).contiguous()
        z_bot = self.tx_compressor(z_96_2d).permute(0, 2, 3, 1).reshape(B, 256, self.c_bot).contiguous()

        if self.method == "swin_holo_blind":
            assert self.modem is not None
            s_rb_iq = self.modem.embed_tx(z_bot)
            y_rb_iq, phy = physical_ultralow_ofdm_channel(
                s_rb_iq,
                snr_db,
                rho_f=rho_f,
                alive_rb_mask=alive_rb_mask,
                h_rb_override=h_rb_override,
                standard_noise_iq=standard_noise_iq,
            )
            z_hat, cr_hat_db, aux = self.modem.blind_demod_and_equalize(y_rb_iq)
            alive_frac = aux["alive_frac"]
            h_abs_k = expand_rb_to_symbols(aux["h_abs_rb"], K_96)
            alpha_ratio_k = expand_rb_to_symbols(aux["alpha_rb"] / aux["alpha_bar_all"][:, None], K_96)
            sigma_post_k = expand_rb_to_symbols(
                0.5 * ((1.0 - aux["alpha_rb"]) / aux["alpha_bar_all"][:, None]).clamp_min(1e-6), K_96
            )
            psi_csi = aux["sigma_e_eff"]
        else:
            # Oracle Genie-CSI: 100% payload DoF + H_256 dispersion + Exact True Channel & True Noise Variance!
            p_img = (2.0 * z_bot.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
            z_unit = z_bot / p_img[:, None, None]
            dither = build_ultralow_sign_dither(256, self.c_bot, device=device, dtype=dtype)
            h256 = build_hadamard_pow2(256, device=device, dtype=dtype)
            z_disp = torch.matmul(h256, z_unit * dither)
            rb_2nb = z_disp.reshape(B, 256 // M, M, self.c_bot).permute(0, 2, 1, 3).reshape(B, M, 2 * N_b)
            s_rb_iq = torch.stack([rb_2nb[:, :, :N_b], rb_2nb[:, :, N_b:]], dim=-1)

            y_rb_iq, phy = physical_ultralow_ofdm_channel(
                s_rb_iq,
                snr_db,
                rho_f=rho_f,
                alive_rb_mask=alive_rb_mask,
                h_rb_override=h_rb_override,
                standard_noise_iq=standard_noise_iq,
            )
            hr, hi = phy["hr_rb"], phy["hi_rb"]
            amask = phy["alive_rb_mask"]  # [B, 8]
            alive_frac = amask.mean(dim=-1).clamp_min(0.125)
            h_sq = (hr.square() + hi.square()) * amask
            denom = (h_sq + phy["true_sigma_n_sq"][:, None]).clamp_min(1e-6)
            alpha_rb = (h_sq / denom) * amask
            alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.025)

            yr, yi = y_rb_iq[..., 0], y_rb_iq[..., 1]
            sr_hat = ((hr[:, :, None] * yr + hi[:, :, None] * yi) / denom[:, :, None]) / alpha_bar[:, :, None] * amask[:, :, None]
            si_hat = ((hr[:, :, None] * yi - hi[:, :, None] * yr) / denom[:, :, None]) / alpha_bar[:, :, None] * amask[:, :, None]

            rb_hat = torch.cat([sr_hat, si_hat], dim=-1).reshape(B, M, 256 // M, self.c_bot).permute(0, 2, 1, 3).reshape(B, 256, self.c_bot)
            z_unit_hat = torch.matmul(h256, rb_hat) * dither
            z_hat = z_unit_hat * p_img[:, None, None]  # Exact true p_img!

            sinr_g = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1) + 1e-4)
            cr_hat_db = (10.0 * torch.log10(sinr_g)).clamp(-6.0, 24.0)
            h_abs_k = expand_rb_to_symbols(h_sq.sqrt(), K_96)
            alpha_ratio_k = expand_rb_to_symbols(alpha_rb / alpha_bar, K_96)
            sigma_post_k = expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K_96)
            psi_csi = torch.zeros((B,), device=device, dtype=dtype)

        z_hat_2d = z_hat.reshape(B, 16, 16, self.c_bot).permute(0, 3, 1, 2).contiguous()
        cr_norm = ((cr_hat_db - SNR_CENTER) / SNR_SCALE)[:, None, None, None].expand(B, 1, 16, 16)
        af_map = alive_frac[:, None, None, None].expand(B, 1, 16, 16)
        u_96 = self.rx_decompressor(torch.cat([z_hat_2d, cr_norm, af_map], dim=1)).permute(0, 2, 3, 1).reshape(B, 256, C)

        p_u = (2.0 * u_96.square().mean(dim=(1, 2))).clamp_min(1e-6).sqrt()
        u_norm = u_96 / p_u[:, None, None]
        ch_v2 = ChannelV2Output(
            u_restored=u_96,
            u_mmse_restored=u_96,
            u_norm=u_norm,
            z_norm=u_norm,
            pwr=p_u.square(),
            gamma_db=cr_hat_db,
            c_r_db=cr_hat_db,
            h_abs_iq=h_abs_k,
            alpha_ratio_iq=alpha_ratio_k,
            sigma_post_iq=sigma_post_k,
            phase_unc_iq=torch.zeros_like(h_abs_k),
            psi_csi=psi_csi,
        )

        dec = self.backbone.decoder
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
        return {
            "recon": recon,
            "c_r_hat_db": cr_hat_db,
            "alive_frac": alive_frac,
            "s_rb_iq": s_rb_iq,
        }
