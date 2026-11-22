"""Dir-5A: Symp-OTFS-JSCC — High-Mobility Doubly-Selective Delay-Doppler Zero-Pilot Blind Semantic Communications.

Addresses high-velocity UAV / High-Speed Rail / LEO Satellite Feeder doubly-selective wireless channels
(multipath delay spread + rapid Jakes continuous Doppler shifts + physical fast-time Inter-Carrier Interference [ICI])
with ZERO pilot symbols, ZERO Delay-Doppler guard zones, and ZERO prior CSI.

Core Innovations:
  1. Unitary Symplectic Payload Spreading (24 x 32 ISFFT/SFFT across 768 micro-coherence tiles):
     Converts rapid Jakes time-selective deep fades and frequency-selective multipath nulls into uniform
     2D Delay-Doppler diversity gain across every semantic token.
  2. Exact 14/16 Hadamard Micro-Tile Subspace Decomposition (2 time symbols x 8 subcarriers = 16 REs/tile):
     Satisfies the 2D Shannon-Nyquist Delay-Doppler sampling limit up to k_d = f_d / Delta_f = 0.22 (>1,000 km/h!)
     with zero pilot slots and zero DD guard-band overhead.
  3. Non-Causal Delay-Null Noise Floor Sensing + Sparse Delay-Doppler Empirical-Bayes Wiener Denoising +
     Exact Complex Boxcar-Deconvolved Fourier Upsampling (24x32 -> 48x256 per-RE CSI):
     Estimates thermal + ICI noise floor from CP-forbidden delay bins m in [9..24], denoises sparse physical
     scatterers in the 2D Delay-Doppler domain, deconvolves the 2x8 tile integration transfer function
     B_T[l] * B_F[m] (preserving exact sub-tile centroid phase!), and reconstructs all 12,288 instantaneous
     complex channel gains H_hat[n, k] for per-RE Decision-Free Probe Cancellation + Gain-Unbiased MMSE Equalization.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dir2_zpac.blind_v3 import (
    SNR_CENTER,
    SNR_SCALE,
    BlindMultiScaleDecoderCalibrator,
    OpenLoopHierarchicalTX,
)
from dir2_zpac.phy_v2 import ChannelV2Output

# Physical Multicarrier / OTFS Frame Geometry (K = 12,288 complex REs = CBR 1/32 for 256x256 RGB):
N_SYM = 48          # Time blocks / OFDM symbols per frame
N_SC = 256          # Subcarriers per block
N_CP = 16           # Cyclic prefix length in fast-time samples
N_T_TILES = 24      # 24 temporal micro-tiles x 2 symbols = 48 symbols (Nyquist-safe up to k_d = 0.22!)
N_F_TILES = 32      # 32 frequency RBs x 8 subcarriers = 256 subcarriers
T_SUB = 2           # Symbols per micro-tile
F_SUB = 8           # Subcarriers per micro-tile
TILE_RES = 16       # 2 x 8 = 16 REs per micro-tile (768 micro-tiles total)
PAYLOAD_MODES = 14  # 14/16 = 87.5% pure semantic DoF (256 tokens x 84 real ch = 21,504 real = 10,752 complex)
BOTTLENECK_CH = 84  # 256 * 84 = 2 * (24 * 32 * 14)

DEFAULT_TAUS = (0, 1, 3, 5, 8, 12)


def build_hadamard16(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Exact 16x16 orthonormal Sylvester-Hadamard matrix (H^T H = I_16).
    Column 0 is constant +1/4; Column 1 alternates +1/4, -1/4 along consecutive subcarriers.
    """
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype) / math.sqrt(2.0)
    h4 = torch.kron(h, h)
    h8 = torch.kron(h4, h)
    h16 = torch.kron(h, h8)  # kron([2 time], [8 freq]) so col 1 alternates along fast subcarrier axis!
    return h16


def newman_2d_phases(
    nt: int = N_T_TILES,
    nf: int = N_F_TILES,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """2D quadratic Newman chirp phases psi[t_b, f_b] = pi * (t_b^2 / nt + f_b^2 / nf) for low-PAPR probing."""
    tb = torch.arange(nt, device=device, dtype=dtype)[:, None]
    fb = torch.arange(nf, device=device, dtype=dtype)[None, :]
    psi = math.pi * ((tb * tb) / float(nt) + (fb * fb) / float(nf))
    return torch.complex(torch.cos(psi), torch.sin(psi))  # [24, 32]


def isfft_2d(x_dd: torch.Tensor) -> torch.Tensor:
    """Unitary 2D Inverse Symplectic FFT (Delay-Doppler -> Time-Frequency) on dims (1, 2).
    ISFFT: IFFT along Doppler/time axis (dim=1), FFT along Delay/freq axis (dim=2).
    """
    return torch.fft.fft(torch.fft.ifft(x_dd, dim=1, norm="ortho"), dim=2, norm="ortho")


def sfft_2d(x_tf: torch.Tensor) -> torch.Tensor:
    """Unitary 2D Symplectic FFT (Time-Frequency -> Delay-Doppler), exact inverse of isfft_2d."""
    return torch.fft.ifft(torch.fft.fft(x_tf, dim=1, norm="ortho"), dim=2, norm="ortho")


def sample_doubly_selective_multipath(
    batch_size: int,
    kd_max: float | torch.Tensor,
    *,
    taus: tuple[int, ...] = DEFAULT_TAUS,
    tau_decay: float = 4.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample P = 6 Rayleigh scatterer complex gains g [B, P, 2] and continuous Jakes Doppler shifts nu [B, P].
    Normalized unit expected total channel power: sum_p E[|g_p|^2] == 1.0.
    kd_max = f_{d,max} / Delta_f (normalized maximum Doppler shift in subcarrier spacings).
    """
    P = len(taus)
    tau_t = torch.tensor(taus, device="cpu", dtype=torch.float32)
    pdp = torch.exp(-tau_t / float(tau_decay))
    pdp = pdp / pdp.sum()  # [P], sum == 1.0

    std_p = (0.5 * pdp).sqrt()[None, :, None]  # [1, P, 1]
    g_ri = torch.randn((batch_size, P, 2), generator=generator, dtype=torch.float32) * std_p
    theta = torch.rand((batch_size, P), generator=generator, dtype=torch.float32) * (2.0 * math.pi)

    if torch.is_tensor(kd_max):
        kd_t = kd_max.detach().cpu().to(torch.float32).reshape(batch_size, 1)
    else:
        kd_t = torch.full((batch_size, 1), float(kd_max), dtype=torch.float32)
    nu_norm = kd_t * torch.cos(theta)  # Jakes U-shaped Doppler spectrum in [-kd_max, +kd_max]!

    return g_ri.to(device=device, dtype=dtype), nu_norm.to(device=device, dtype=dtype)


def physical_doubly_selective_waveform_channel(
    x_tf_iq: torch.Tensor,         # [B, N_SYM=48, N_SC=256, 2] Time-Frequency transmit grid
    snr_db: torch.Tensor | float,  # Physical Es/N0 in dB (strictly hidden from TX and blind RX!)
    *,
    kd_max: float | torch.Tensor = 0.10,
    taus: tuple[int, ...] = DEFAULT_TAUS,
    multipath_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_td_iq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact Sample-Level Fast-Time Doubly-Selective Multipath + Intra-Symbol Doppler ICI Baseband Channel:
      1. Enforce strict unit pre-channel frame power E[|X_TF|^2] == 1.0 per image
      2. 256-point orthonormal IFFT -> fast-time CP-OFDM / CP-OTFS blocks x_td[b, n, q]
      3. Sample-by-sample continuous Jakes Doppler phase rotation exp(j 2pi nu_p (n*(N_F+N_CP)/N_F + q/N_F))
         on each delayed multipath ray x_td[b, n, (q - tau_p) mod N_F] (creates physical intra-symbol ICI!)
      4. Add sample-level complex thermal AWGN w_td ~ CN(0, 10^{-SNR_dB/10})
      5. Receiver 256-point orthonormal FFT -> Y_TF[b, n, k]
    """
    B, n_sym, n_sc, two = x_tf_iq.shape
    assert n_sym == N_SYM and n_sc == N_SC and two == 2
    device, dtype = x_tf_iq.device, x_tf_iq.dtype

    snr_t = (
        torch.full((B,), float(snr_db), device=device, dtype=dtype)
        if not torch.is_tensor(snr_db)
        else snr_db.to(device=device, dtype=dtype).reshape(B)
    )

    # Strict unit average transmit power per frame (un-gameable physical power budget!)
    in_pwr = (x_tf_iq[..., 0].square() + x_tf_iq[..., 1].square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    x_tf_c = torch.complex(x_tf_iq[..., 0], x_tf_iq[..., 1]) / in_pwr.sqrt()  # [B, 48, 256]

    # TX Heisenberg 256-point IFFT to fast-time baseband waveform
    x_td = torch.fft.ifft(x_tf_c, dim=-1, norm="ortho")  # [B, 48, 256]
    env_sq = x_td.real.square() + x_td.imag.square()
    papr_db = 10.0 * torch.log10(env_sq.amax(dim=-1).mean(dim=-1).clamp_min(1.0))

    if multipath_override is not None:
        g_ri = multipath_override[0].to(device=device, dtype=dtype)
        nu_norm = multipath_override[1].to(device=device, dtype=dtype)
    else:
        g_ri, nu_norm = sample_doubly_selective_multipath(B, kd_max, taus=taus, device=device, dtype=dtype)

    g_c = torch.complex(g_ri[:, :, 0], g_ri[:, :, 1])  # [B, P]
    P = len(taus)

    # Exact fast-time continuous clock t_norm[n, q] in units of T_u = 1 / Delta_f
    n_idx = torch.arange(N_SYM, device=device, dtype=dtype)[:, None]  # [48, 1]
    q_idx = torch.arange(N_SC, device=device, dtype=dtype)[None, :]   # [1, 256]
    sym_stride = float(N_SC + N_CP) / float(N_SC)
    t_norm = n_idx * sym_stride + (q_idx / float(N_SC))  # [48, 256]

    # Stack circularly delayed fast-time signals for all P rays (exact under CP since max(taus) < N_CP = 16)
    x_delayed = torch.stack([torch.roll(x_td, shifts=int(tau), dims=-1) for tau in taus], dim=1)  # [B, P, 48, 256]
    phase_rot = 2.0 * math.pi * nu_norm[:, :, None, None] * t_norm[None, None, :, :]
    doppler_phasor = torch.complex(torch.cos(phase_rot), torch.sin(phase_rot))
    y_clean_td = (g_c[:, :, None, None] * doppler_phasor * x_delayed).sum(dim=1)  # [B, 48, 256]

    # Add sample-level thermal AWGN in fast time
    sigma_n_sq = torch.pow(10.0, -snr_t / 10.0)
    sigma_r = (0.5 * sigma_n_sq).sqrt()[:, None, None]
    eps = (
        torch.randn((B, N_SYM, N_SC, 2), device=device, dtype=dtype)
        if standard_noise_td_iq is None
        else standard_noise_td_iq.to(device=device, dtype=dtype).reshape(B, N_SYM, N_SC, 2)
    )
    w_td = torch.complex(eps[..., 0] * sigma_r, eps[..., 1] * sigma_r)
    y_td = y_clean_td + w_td

    # Receiver Wigner 256-point FFT to Time-Frequency observation grid
    y_tf_c = torch.fft.fft(y_td, dim=-1, norm="ortho")  # [B, 48, 256]
    y_tf_iq = torch.stack([y_tf_c.real, y_tf_c.imag], dim=-1)

    # Symbol-center instantaneous diagonal TF response H_diag[b, n, k] (for diagnostics & NMSE verification)
    k_idx = torch.arange(N_SC, device=device, dtype=dtype)[None, None, None, :]
    t_sym_center = (n_idx * sym_stride + 0.5)[None, None, :, :]  # [1, 1, 48, 1]
    tau_vec = torch.tensor(taus, device=device, dtype=dtype)[None, :, None, None]
    ph_tf = 2.0 * math.pi * (nu_norm[:, :, None, None] * t_sym_center - tau_vec * k_idx / float(N_SC))
    h_true_tf = (g_c[:, :, None, None] * torch.complex(torch.cos(ph_tf), torch.sin(ph_tf))).sum(dim=1)  # [B, 48, 256]

    return y_tf_iq, {
        "true_snr_db": snr_t,
        "true_sigma_n_sq": sigma_n_sq,
        "papr_db": papr_db,
        "h_true_tf": h_true_tf,
        "g_ri": g_ri,
        "nu_norm": nu_norm,
    }


def blind_dd_wiener_fourier_upsample(
    h_tile_raw: torch.Tensor,  # [B, 24, 32] complex raw blind micro-tile channel estimates
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D Delay-Doppler Empirical-Bayes Wiener Denoising + Complex Boxcar-Deconvolved Fourier Upsampling:
      - Estimates effective thermal + ICI + residual probe noise variance sigma_dd_sq directly from
        CP-forbidden non-causal delay bins m in [9..24] (384 pure-noise samples per frame, 0 extra DoF!)
      - Applies causal-delay-weighted Empirical-Bayes MMSE shrinkage in the 24x32 Delay-Doppler plane
      - Deconvolves the exact 2x8 tile averaging transfer function B_T[l] * B_F[m] (preserving sub-tile phase!)
      - Zero-pads (24, 32) -> (48, 256) in DD domain to synthesize all 12,288 instantaneous gains H_hat[b, n, k].
    Returns:
      h_hat_tf: [B, 48, 256] complex per-RE channel estimate
      sigma_tile_post: [B] post-Wiener residual channel estimation MSE per RE
      sigma_probe_raw: [B] raw per-tile probe noise variance (proportional to sigma_n_sq + sigma_ici_sq)
    """
    B, nt, nf = h_tile_raw.shape
    assert nt == N_T_TILES and nf == N_F_TILES
    device = h_tile_raw.device
    fdtype = h_tile_raw.real.dtype

    # Transform 24x32 TF micro-tile grid to 24x32 Delay-Doppler Fourier plane
    h_dd = torch.fft.ifft(torch.fft.fft(h_tile_raw, dim=1, norm="ortho"), dim=2, norm="ortho")
    pwr_dd = h_dd.real.square() + h_dd.imag.square()  # [B, 24, 32]

    # CP-forbidden delay bins m in [9..24] contain ZERO physical multipath energy (tau < 16 -> m_rb < 2.0)!
    sigma_probe_raw = pwr_dd[:, :, 9:25].mean(dim=(1, 2)).clamp_min(1e-6)  # [B]
    nv = sigma_probe_raw[:, None, None]

    m32 = torch.arange(nf, device=device)
    causal_prior = ((m32 <= 6) | (m32 >= 28)).to(fdtype)[None, None, :]  # [1, 1, 32]
    shrink = ((pwr_dd - 1.25 * nv).clamp_min(0.0) / (pwr_dd + 1e-8)) * (0.04 + 0.96 * causal_prior)
    h_dd_den = h_dd * shrink

    # Post-Wiener estimation error variance: fraction of active DD bins * nv
    active_frac = shrink.mean(dim=(1, 2)).clamp(0.02, 0.35)
    sigma_tile_post = sigma_probe_raw * active_frac

    # Exact complex boxcar transfer function B_T[l] * B_F[m] for 2x8 tile averaging
    l_signed = torch.fft.fftfreq(nt, d=1.0 / float(nt), device=device, dtype=fdtype)  # [24]
    m_signed = torch.fft.fftfreq(nf, d=1.0 / float(nf), device=device, dtype=fdtype)  # [32]
    r_idx = torch.arange(T_SUB, device=device, dtype=fdtype)[:, None]
    c_idx = torch.arange(F_SUB, device=device, dtype=fdtype)[:, None]
    ph_t = 2.0 * math.pi * r_idx * l_signed[None, :] / float(N_SYM)
    ph_f = -2.0 * math.pi * c_idx * m_signed[None, :] / float(N_SC)
    bt = torch.complex(torch.cos(ph_t), torch.sin(ph_t)).mean(dim=0)  # [24]
    bf = torch.complex(torch.cos(ph_f), torch.sin(ph_f)).mean(dim=0)  # [32]
    b2d = bt[:, None] * bf[None, :]  # [24, 32]

    h_dd_deconv = h_dd_den * torch.conj(b2d)[None, :, :] / (b2d.abs().square()[None, :, :] + 0.015)

    # Fourier zero-pad (24, 32) -> (48, 256) preserving signed positive/negative frequencies
    h_dd_pad = torch.zeros((B, N_SYM, N_SC), dtype=h_dd.dtype, device=device)
    t_half = nt // 2  # 12
    f_half = nf // 2  # 16
    for ts_src, ts_dst in [(slice(0, t_half), slice(0, t_half)), (slice(t_half, nt), slice(N_SYM - t_half, N_SYM))]:
        for fs_src, fs_dst in [(slice(0, f_half), slice(0, f_half)), (slice(f_half, nf), slice(N_SC - f_half, N_SC))]:
            h_dd_pad[:, ts_dst, fs_dst] = h_dd_deconv[:, ts_src, fs_src]

    scale_up = math.sqrt(float(N_SYM * N_SC) / float(nt * nf))  # sqrt(16) = 4.0
    h_hat_tf = torch.fft.fft(torch.fft.ifft(h_dd_pad, dim=1, norm="ortho"), dim=2, norm="ortho") * scale_up
    return h_hat_tf, sigma_tile_post, sigma_probe_raw


class SympOTFSSubspaceModem(nn.Module):
    """Unified 2D Micro-Tile Blind Subspace Modem for:
      - use_otfs=True  (Symp_OTFS_JSCC: Unitary 24x32 ISFFT/SFFT Delay-Doppler Spreading + 2D DD-Wiener Blind CSI)
      - use_otfs=False (Symp_OFDM_2DSubspace: Controlled Waveform Ablation keeping identical 2D DD-Wiener Blind CSI without ISFFT/SFFT)
    """

    def __init__(self, latent_ch: int = 96, *, use_otfs: bool = True, hidden_ch: int = 160):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.use_otfs = bool(use_otfs)

        # Bottleneck 96 -> 84 real channels (14/16 exact bijective subspace, 0 rank-projection loss!)
        self.tx_proj = nn.Linear(self.latent_ch, BOTTLENECK_CH)
        self.rx_proj = nn.Linear(BOTTLENECK_CH, self.latent_ch)
        with torch.no_grad():
            nn.init.zeros_(self.tx_proj.weight)
            nn.init.zeros_(self.tx_proj.bias)
            nn.init.zeros_(self.rx_proj.weight)
            nn.init.zeros_(self.rx_proj.bias)
            for i in range(BOTTLENECK_CH):
                self.tx_proj.weight[i, i] = 1.0
                self.rx_proj.weight[i, i] = 1.0

        # Learnable probe power fraction eta1 in [0.04, 0.18], initialized at 0.10
        eta1_init = 0.10
        inv_s = math.log((eta1_init - 0.04) / (0.18 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta2 = 0.010  # 1.0% 768-tile MRC phase-angle anchor for scale-invariant p_img recovery

        # 2D Delay-Doppler / Token Residual Refiner (compensates residual high-Doppler ICI & expands 84 -> 96 cleanly)
        self.dd_refiner = nn.Sequential(
            nn.Conv2d(BOTTLENECK_CH + 4, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, padding_mode="reflect", groups=8),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, self.latent_ch, kernel_size=1),
        )
        nn.init.zeros_(self.dd_refiner[-1].weight)
        nn.init.zeros_(self.dd_refiner[-1].bias)

    @property
    def eta1(self) -> torch.Tensor:
        return 0.04 + (0.18 - 0.04) * torch.sigmoid(self.eta1_raw)

    def embed_tx(self, z_tx: torch.Tensor) -> torch.Tensor:
        """Map [B, 256, 96] -> [B, N_SYM=48, N_SC=256, 2] unit-power TF transmit grid (0 pilots, 0 guard zones)."""
        B, L, C = z_tx.shape
        assert L == 256 and C == self.latent_ch
        device, dtype = z_tx.device, z_tx.dtype

        p_img = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()  # [B]
        z_norm = z_tx / p_img[:, None, None]

        # Compress 96 -> 84 real channels and normalize to exact unit complex symbol variance
        z_84 = self.tx_proj(z_norm)  # [B, 256, 84]
        pwr_84 = (2.0 * z_84.square().mean(dim=(1, 2), keepdim=True)).clamp_min(1e-8)
        z_84_unit = z_84 / pwr_84.sqrt()  # E[re^2 + im^2] == 1.0

        # Reshape [B, 256, 84] -> [B, 24, 32, 14] complex semantic grid
        flat = z_84_unit.reshape(B, 2 * N_T_TILES * N_F_TILES * PAYLOAD_MODES)
        half = N_T_TILES * N_F_TILES * PAYLOAD_MODES  # 10,752
        c_sem = torch.complex(
            flat[:, :half].reshape(B, N_T_TILES, N_F_TILES, PAYLOAD_MODES),
            flat[:, half:].reshape(B, N_T_TILES, N_F_TILES, PAYLOAD_MODES),
        )

        # Apply Unitary 2D ISFFT (DD -> TF) for Symp-OTFS; skip for Controlled OFDM Ablation
        c_tf = isfft_2d(c_sem) if self.use_otfs else c_sem  # [B, 24, 32, 14]

        h16 = build_hadamard16(device, dtype)  # [16, 16]
        w1 = h16[:, 0]      # [16] constant 1/4 channel probe
        w2 = h16[:, 1]      # [16] alternating subcarrier p_img phase-angle anchor
        u_perp = h16[:, 2:] # [16, 14] orthonormal payload subspace

        eta1 = self.eta1
        data_pwr = 1.0 - eta1 - self.eta2
        # Adjust payload power by 16/14 so average per-RE power across the 16-RE tile equals data_pwr!
        payload_scale = (data_pwr * (float(TILE_RES) / float(PAYLOAD_MODES))).sqrt()
        x_data_tile = payload_scale * torch.complex(
            torch.einsum("btfm,rm->btfr", c_tf.real, u_perp),
            torch.einsum("btfm,rm->btfr", c_tf.imag, u_perp),
        )  # [B, 24, 32, 16]

        # 2D Newman-chirped probe on w1 + scale-invariant p_img phase anchor on w2
        psi_c = newman_2d_phases(N_T_TILES, N_F_TILES, device=device, dtype=dtype)[None, :, :, None]  # [1, 24, 32, 1]
        amp1 = (eta1 * float(TILE_RES)).sqrt()
        p_w1 = amp1 * psi_c * w1[None, None, None, :]

        theta_p = 1.25 * torch.tanh(0.40 * (p_img - 2.82))  # [B] in (-1.25, +1.25) rad
        phasor_p = torch.complex(torch.cos(theta_p), torch.sin(theta_p))[:, None, None, None]
        amp2 = math.sqrt(self.eta2 * float(TILE_RES))
        p_w2 = amp2 * psi_c * phasor_p * w2[None, None, None, :]

        x_tile = x_data_tile + p_w1 + p_w2  # [B, 24, 32, 16]
        # Reshape [B, 24, 32, 2, 8] -> [B, 48, 256]
        x_tf_c = x_tile.reshape(B, N_T_TILES, N_F_TILES, T_SUB, F_SUB).permute(0, 1, 3, 2, 4).reshape(B, N_SYM, N_SC)
        return torch.stack([x_tf_c.real, x_tf_c.imag], dim=-1)

    def blind_demod_and_equalize(
        self,
        y_tf_iq: torch.Tensor,  # [B, 48, 256, 2]
    ) -> tuple[ChannelV2Output, torch.Tensor, dict[str, torch.Tensor]]:
        B, n_sym, n_sc, _ = y_tf_iq.shape
        assert n_sym == N_SYM and n_sc == N_SC
        device, dtype = y_tf_iq.device, y_tf_iq.dtype
        K = N_SYM * N_SC

        y_tf_c = torch.complex(y_tf_iq[..., 0], y_tf_iq[..., 1])  # [B, 48, 256]
        y_tile = y_tf_c.reshape(B, N_T_TILES, T_SUB, N_F_TILES, F_SUB).permute(0, 1, 3, 2, 4).reshape(
            B, N_T_TILES, N_F_TILES, TILE_RES
        )

        h16 = build_hadamard16(device, dtype)
        w1 = h16[:, 0]
        w2 = h16[:, 1]
        u_perp = h16[:, 2:]

        eta1 = self.eta1
        amp1 = (eta1 * float(TILE_RES)).sqrt()
        psi_c = newman_2d_phases(N_T_TILES, N_F_TILES, device=device, dtype=dtype)[None, :, :]  # [1, 24, 32]

        # 1. Raw Zero-Pilot w1 Subspace Projection + Undo 2D Newman Chirp
        proj_w1 = torch.complex(
            torch.einsum("btfr,r->btf", y_tile.real, w1),
            torch.einsum("btfr,r->btf", y_tile.imag, w1),
        )
        h_tile_raw = (proj_w1 / amp1) * torch.conj(psi_c)  # [B, 24, 32]

        # 2. Sparse 2D Delay-Doppler Wiener Denoising + Boxcar-Deconvolved 48x256 Per-RE Fourier Upsampling
        h_hat_tf, sigma_h_post, sigma_probe_raw = blind_dd_wiener_fourier_upsample(h_tile_raw)  # [B, 48, 256]
        # Convert probe noise variance back to per-RE noise+ICI variance: sigma_probe_raw = sigma_eff_re / (16 * eta1)
        sigma_re_eff = (sigma_probe_raw * (float(TILE_RES) * eta1)).clamp_min(1e-5)  # [B]

        # 3. 768-Tile Maximal-Ratio Combined (MRC) Scale-Invariant Recovery of p_img from w2
        h_tile_center = h_hat_tf.reshape(B, N_T_TILES, T_SUB, N_F_TILES, F_SUB).mean(dim=(2, 4))  # [B, 24, 32]
        proj_w2 = torch.complex(
            torch.einsum("btfr,r->btf", y_tile.real, w2),
            torch.einsum("btfr,r->btf", y_tile.imag, w2),
        ) * torch.conj(psi_c)
        z2_mrc = (torch.conj(h_tile_center) * proj_w2).sum(dim=(1, 2))  # [B]
        theta_hat = torch.atan2(z2_mrc.imag, z2_mrc.real.clamp_min(1e-5)).clamp(-1.21, 1.21)
        p_img_hat = (2.82 + torch.atanh(theta_hat / 1.25) / 0.40).clamp(0.4, 7.0)  # [B]

        # 4. Per-RE Decision-Free Probe Cancellation (cancels even intra-tile Doppler leakage of w1 & w2 into U_perp!)
        phasor_p_hat = torch.complex(torch.cos(theta_hat), torch.sin(theta_hat))[:, None, None, None]
        amp2 = math.sqrt(self.eta2 * float(TILE_RES))
        probe_tile = (
            amp1 * psi_c[:, :, :, None] * w1[None, None, None, :]
            + amp2 * psi_c[:, :, :, None] * phasor_p_hat * w2[None, None, None, :]
        )
        probe_tf = probe_tile.reshape(B, N_T_TILES, N_F_TILES, T_SUB, F_SUB).permute(0, 1, 3, 2, 4).reshape(B, N_SYM, N_SC)
        y_data_tf = y_tf_c - h_hat_tf * probe_tf

        # 5. Per-RE Gain-Unbiased MMSE Equalization across all 48 x 256 REs
        data_pwr = 1.0 - eta1 - self.eta2
        h_sq = h_hat_tf.real.square() + h_hat_tf.imag.square()  # [B, 48, 256]
        eff_var = ((sigma_re_eff / data_pwr) + sigma_h_post)[:, None, None]
        denom = (h_sq + eff_var).clamp_min(1e-6)
        alpha_re = h_sq / denom
        alpha_bar = alpha_re.mean(dim=(1, 2), keepdim=True).clamp_min(0.04)  # [B, 1, 1]

        payload_scale = (data_pwr * (float(TILE_RES) / float(PAYLOAD_MODES))).sqrt()
        x_eq_tf = (torch.conj(h_hat_tf) * y_data_tf / denom) / (alpha_bar * payload_scale)  # [B, 48, 256]

        # 6. Project onto 14-Dim Orthonormal Subspace U_perp & Invert Symplectic Transform (SFFT: TF -> DD)
        x_eq_tile = x_eq_tf.reshape(B, N_T_TILES, T_SUB, N_F_TILES, F_SUB).permute(0, 1, 3, 2, 4).reshape(
            B, N_T_TILES, N_F_TILES, TILE_RES
        )
        c_hat_tf = torch.complex(
            torch.einsum("btfr,rm->btfm", x_eq_tile.real, u_perp),
            torch.einsum("btfr,rm->btfm", x_eq_tile.imag, u_perp),
        )  # [B, 24, 32, 14]
        c_hat_sem = sfft_2d(c_hat_tf) if self.use_otfs else c_hat_tf

        half = N_T_TILES * N_F_TILES * PAYLOAD_MODES
        z_84_hat = torch.cat(
            [c_hat_sem.real.reshape(B, half), c_hat_sem.imag.reshape(B, half)], dim=1
        ).reshape(B, 256, BOTTLENECK_CH)

        # Effective post-MMSE wideband SINR (for SFFT, every DD symbol experiences exact mean SINR across all 12,288 REs!)
        abar_1d = alpha_bar.reshape(B)
        sinr_wb = abar_1d.clamp_min(1e-4) / (1.0 - abar_1d).clamp_min(1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_wb)).clamp(-5.0, 24.0)

        # 2D Token Residual Refiner + Linear Projection 84 -> 96
        z_96_lin = self.rx_proj(z_84_hat)  # [B, 256, 96]
        z_84_2d = z_84_hat.reshape(B, 16, 16, BOTTLENECK_CH).permute(0, 3, 1, 2).contiguous()

        # Pool 48x256 channel quality maps to 16x16 spatial conditioning maps
        h_abs_2d = F.adaptive_avg_pool2d(h_sq.sqrt().unsqueeze(1), (16, 16))
        alpha_var_2d = F.adaptive_avg_pool2d((alpha_re / alpha_bar).unsqueeze(1), (16, 16))
        cr_map = ((cr_hat_db - SNR_CENTER) / SNR_SCALE)[:, None, None, None].expand(B, 1, 16, 16)
        sig_map = torch.log1p(sigma_re_eff)[:, None, None, None].expand(B, 1, 16, 16)

        delta_96 = self.dd_refiner(torch.cat([z_84_2d, h_abs_2d, alpha_var_2d, cr_map, sig_map], dim=1))
        u_mmse_norm = z_96_lin + delta_96.permute(0, 2, 3, 1).reshape(B, 256, self.latent_ch)
        u_mmse_restored = u_mmse_norm * p_img_hat[:, None, None]

        # Build ChannelV2Output for BlindMultiScaleDecoderCalibrator
        if self.use_otfs:
            # After unitary SFFT, post-equalization noise & gain are isotropically uniform across all K positions!
            h_abs_iq = h_sq.sqrt().mean(dim=(1, 2), keepdim=True).expand(B, N_SYM, N_SC).reshape(B, K)
            alpha_ratio_iq = torch.ones((B, K), device=device, dtype=dtype)
            sigma_post_iq = (0.5 * ((1.0 - abar_1d) / abar_1d).clamp_min(1e-6))[:, None].expand(B, K)
            phase_unc_iq = (sigma_h_post / (h_sq.mean(dim=(1, 2)) + sigma_h_post + 1e-6)).sqrt()[:, None].expand(B, K)
        else:
            h_abs_iq = h_sq.sqrt().reshape(B, K)
            alpha_ratio_iq = (alpha_re / alpha_bar).reshape(B, K)
            sigma_post_iq = (0.5 * ((1.0 - alpha_re) / alpha_bar).clamp_min(1e-6)).reshape(B, K)
            phase_unc_iq = (sigma_h_post[:, None, None] / (h_sq + sigma_h_post[:, None, None] + 1e-6)).sqrt().reshape(B, K)

        psi_csi = (alpha_re - alpha_bar).square().mean(dim=(1, 2)) + sigma_h_post
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
            {
                "h_hat_tf": h_hat_tf,
                "sigma_re_eff": sigma_re_eff,
                "p_img_hat": p_img_hat,
            },
        )


class BlackBoxOTFSBlindModem(nn.Module):
    """End-to-End Neural Blind OTFS Modem Baseline:
    Uses exact unitary 48x256 ISFFT/SFFT OTFS waveform + 2D Conv neural equalizer in TF & DD domains,
    WITHOUT structured Newman-Hadamard micro-tile subspace probing or Sparse DD-Wiener Fourier equalization.
    """

    def __init__(self, latent_ch: int = 96):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.tf_net = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(48, 2, kernel_size=1),
        )
        nn.init.zeros_(self.tf_net[-1].weight)
        nn.init.zeros_(self.tf_net[-1].bias)

        self.snr_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.snr_mlp[-1].weight)
        nn.init.zeros_(self.snr_mlp[-1].bias)

    def embed_tx(self, z_tx: torch.Tensor) -> torch.Tensor:
        B, L, C = z_tx.shape
        K = (L * C) // 2
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        flat = (z_tx / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
        x_dd = torch.complex(flat[:, :K].reshape(B, N_SYM, N_SC), flat[:, K:].reshape(B, N_SYM, N_SC))
        x_tf = isfft_2d(x_dd)
        return torch.stack([x_tf.real, x_tf.imag], dim=-1)

    def blind_demod_and_equalize(
        self,
        y_tf_iq: torch.Tensor,
    ) -> tuple[ChannelV2Output, torch.Tensor, dict[str, torch.Tensor]]:
        B, _, _, _ = y_tf_iq.shape
        device, dtype = y_tf_iq.device, y_tf_iq.dtype
        K = N_SYM * N_SC

        yr, yi = y_tf_iq[..., 0], y_tf_iq[..., 1]
        mag = (yr.square() + yi.square()).clamp_min(1e-8).sqrt()
        tot_pwr = (yr.square() + yi.square()).mean(dim=(1, 2)).clamp_min(1e-4)

        inp_2d = torch.stack([yr, yi, mag], dim=1)  # [B, 3, 48, 256]
        delta_tf = self.tf_net(inp_2d)
        eq_tf = torch.complex(
            (yr + delta_tf[:, 0]) / tot_pwr.sqrt()[:, None, None],
            (yi + delta_tf[:, 1]) / tot_pwr.sqrt()[:, None, None],
        )
        eq_dd = sfft_2d(eq_tf)

        # Global blind statistics for neural SNR/scale MLP
        t_diff = (yr[:, 1:] - yr[:, :-1]).square().mean(dim=(1, 2)) + (yi[:, 1:] - yi[:, :-1]).square().mean(dim=(1, 2))
        f_diff = (yr[:, :, 1:] - yr[:, :, :-1]).square().mean(dim=(1, 2)) + (yi[:, :, 1:] - yi[:, :, :-1]).square().mean(dim=(1, 2))
        dd_mag = eq_dd.abs()
        stats = torch.stack([
            torch.log(tot_pwr),
            mag.std(dim=(1, 2)),
            torch.log(t_diff.clamp_min(1e-5)),
            torch.log(f_diff.clamp_min(1e-5)),
            dd_mag.mean(dim=(1, 2)),
            dd_mag.std(dim=(1, 2)),
        ], dim=-1)
        pred = self.snr_mlp(stats)
        cr_hat_db = (SNR_CENTER + pred[:, 0] * SNR_SCALE).clamp(-4.0, 22.0)
        p_img_hat = (2.82 * torch.exp(0.25 * torch.tanh(pred[:, 1]))).clamp(0.5, 6.5)

        u_norm = torch.cat([eq_dd.real.reshape(B, K), eq_dd.imag.reshape(B, K)], dim=1).reshape(B, 256, self.latent_ch)
        u_restored = u_norm * p_img_hat[:, None, None]
        ones_k = torch.ones((B, K), device=device, dtype=dtype)
        return (
            ChannelV2Output(
                u_restored=u_restored,
                u_mmse_restored=u_restored,
                u_norm=u_norm,
                z_norm=u_norm,
                pwr=p_img_hat.square(),
                gamma_db=cr_hat_db,
                c_r_db=cr_hat_db,
                h_abs_iq=mag.reshape(B, K),
                alpha_ratio_iq=ones_k,
                sigma_post_iq=0.20 * ones_k,
                phase_unc_iq=0.35 * ones_k,
                psi_csi=torch.full((B,), 0.25, device=device, dtype=dtype),
            ),
            cr_hat_db,
            {"p_img_hat": p_img_hat},
        )


class SympOTFSBlindTransceiver(nn.Module):
    """Unified Dir-5A High-Mobility Transceiver:
      - symp_otfs:     Proposed Symp-OTFS-JSCC (Unitary ISFFT/SFFT + 24x32 Subspace DD-Wiener Fourier Blind CSI)
      - symp_ofdm_2d:  Controlled Waveform Ablation (Identical 24x32 Subspace DD-Wiener Blind CSI over OFDM without ISFFT/SFFT)
      - blackbox_otfs: End-to-End Neural Blind OTFS Baseline (Unitary ISFFT/SFFT + Neural Blind Receiver, 0 Pilots)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "symp_otfs",
        base_c0_db: float = 7.0,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.method = method.lower()
        self.base_c0_db = float(base_c0_db)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
        if self.method == "symp_otfs":
            self.modem: nn.Module = SympOTFSSubspaceModem(self.latent_channels, use_otfs=True, hidden_ch=hidden_ch)
        elif self.method == "symp_ofdm_2d":
            self.modem = SympOTFSSubspaceModem(self.latent_channels, use_otfs=False, hidden_ch=hidden_ch)
        elif self.method == "blackbox_otfs":
            self.modem = BlackBoxOTFSBlindModem(self.latent_channels)
        else:
            raise ValueError(f"Unknown Dir-5A method: {self.method}")

        self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_otfs_tx(self, x_stem: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        B, L, _ = x_stem.shape
        snr_c0 = torch.full((B, 1), self.base_c0_db, device=x_stem.device, dtype=x_stem.dtype)
        with torch.no_grad():
            temp = None
            for i in range(enc.layer_num):
                temp = enc.sm_list[i](x_stem.detach()) if i == 0 else enc.sm_list[i](temp)
                temp = temp * enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
            z_base = enc.head_list(x_stem * enc.sigmoid(enc.sm_list[-1](temp)))
        z_tx = self.hier_tx(z_base, x_stem)
        return self.modem.embed_tx(z_tx)

    def decode_otfs_rx(self, y_tf_iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B = y_tf_iq.shape[0]
        L = 256
        dec = self.backbone.decoder

        ch_v2, cr_hat_db, aux = self.modem.blind_demod_and_equalize(y_tf_iq)
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
        return recon, cr_hat_db, aux

    def forward_doubly_selective(
        self,
        image_or_stem: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        kd_max: float | torch.Tensor = 0.10,
        multipath_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        standard_noise_td_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        x_tf_iq = self.encode_otfs_tx(x_stem)
        y_tf_iq, phy = physical_doubly_selective_waveform_channel(
            x_tf_iq,
            snr_db,
            kd_max=kd_max,
            multipath_override=multipath_override,
            standard_noise_td_iq=standard_noise_td_iq,
        )
        recon, cr_hat_db, aux = self.decode_otfs_rx(y_tf_iq)
        out = {
            "recon": recon,
            "c_r_hat_db": cr_hat_db,
            "papr_db": phy["papr_db"],
            "h_true_tf": phy["h_true_tf"],
        }
        out.update(aux)
        return out


@torch.inference_mode()
def forward_frozen_ofdm_doubly_selective(
    backbone: nn.Module,
    image: torch.Tensor,
    snr_db: float,
    *,
    kd_max: float = 0.10,
    base_c0_db: float = 7.0,
    multipath_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_td_iq: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Unmodified Frozen SwinJSCC over Doubly-Selective OFDM (0 pilots, 0 CSI)."""
    B = image.shape[0]
    device, dtype = image.device, image.dtype
    enc, dec = backbone.encoder, backbone.decoder
    L = 256
    C = int(enc.head_list.out_features)
    K = (L * C) // 2

    x = enc.patch_embed(image)
    for layer in enc.layers:
        x = layer(x)
    x = enc.norm(x)
    snr_c0 = torch.full((B, 1), float(base_c0_db), device=device, dtype=dtype)
    temp = None
    for i in range(enc.layer_num):
        temp = enc.sm_list[i](x) if i == 0 else enc.sm_list[i](temp)
        temp = temp * enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
    z = enc.head_list(x * enc.sigmoid(enc.sm_list[-1](temp)))

    pwr = (2.0 * z.square().mean(dim=(1, 2))).clamp_min(1e-8)
    flat = (z / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
    x_tf_iq = torch.stack([flat[:, :K].reshape(B, N_SYM, N_SC), flat[:, K:].reshape(B, N_SYM, N_SC)], dim=-1)

    y_tf_iq, phy = physical_doubly_selective_waveform_channel(
        x_tf_iq,
        snr_db,
        kd_max=kd_max,
        multipath_override=multipath_override,
        standard_noise_td_iq=standard_noise_td_iq,
    )
    rx_pwr = (y_tf_iq[..., 0].square() + y_tf_iq[..., 1].square()).mean(dim=(1, 2)).clamp_min(0.2)
    u_in = torch.cat([y_tf_iq[..., 0].reshape(B, K), y_tf_iq[..., 1].reshape(B, K)], dim=1).reshape(B, L, C)
    u_in = u_in * (2.82 / rx_pwr.sqrt()[:, None, None])

    xd = dec.head_list(u_in)
    temp = None
    for i in range(dec.layer_num):
        temp = dec.sm_list[i](xd) if i == 0 else dec.sm_list[i](temp)
        temp = temp * dec.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
    xd = xd * dec.sigmoid(dec.sm_list[-1](temp))
    for layer in dec.layers:
        xd = layer(xd)
    recon = xd.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
    return {"recon": recon, "papr_db": phy["papr_db"]}


@torch.inference_mode()
def forward_v4_usph_ofdm_doubly_selective(
    v4_model: nn.Module,
    image_or_stem: torch.Tensor,
    snr_db: float,
    *,
    kd_max: float = 0.10,
    multipath_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_td_iq: torch.Tensor | None = None,
    input_is_stem: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate V4 Universal Static-RB Blind Transceiver (U-SPH) zero-shot over Doubly-Selective OFDM!
    Demonstrates static 48-symbol RB Walsh integration washing out under rapid Jakes Doppler rotation.
    """
    x_stem = image_or_stem if input_is_stem else v4_model.extract_encoder_stem(image_or_stem)
    B = x_stem.shape[0]
    s_tx_iq = v4_model.encode_blind_tx(x_stem)  # [B, 12288, 2] ordered as [B, M=32, N_b=384, 2]
    # Map 32 RBs x (48 sym x 8 sc) -> [B, 48, 256, 2] physical OFDM grid
    x_tf_iq = s_tx_iq.reshape(B, 32, N_SYM, 8, 2).permute(0, 2, 1, 3, 4).reshape(B, N_SYM, N_SC, 2).contiguous()

    y_tf_iq, phy = physical_doubly_selective_waveform_channel(
        x_tf_iq,
        snr_db,
        kd_max=kd_max,
        multipath_override=multipath_override,
        standard_noise_td_iq=standard_noise_td_iq,
    )
    y_rx_flat = y_tf_iq.reshape(B, N_SYM, 32, 8, 2).permute(0, 2, 1, 3, 4).reshape(B, N_SYM * N_SC, 2).contiguous()
    recon, cr_hat = v4_model.decode_blind_rx(y_rx_flat)
    return {"recon": recon, "c_r_hat_db": cr_hat, "papr_db": phy["papr_db"]}


@torch.inference_mode()
def forward_pilot_4sym_interp_ofdm_doubly_selective(
    backbone: nn.Module,
    image: torch.Tensor,
    snr_db: float,
    *,
    kd_max: float = 0.10,
    pilot_syms: tuple[int, ...] = (4, 18, 30, 44),
    multipath_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_td_iq: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Generous 5G High-Mobility 4-Symbol Comb/Slot DMRS Baseline (4 x 256 = 1,024 Pilot REs = 8.33% Overhead):
    Grants 4 full OFDM pilot symbols at n in {4, 18, 30, 44} passed through the EXACT same physical doubly-selective
    channel + frequency Wiener smoothing + piecewise-linear complex TF interpolation across time + LMMSE equalization,
    WITHOUT even penalizing/puncturing the 12,288 semantic payload REs!
    Shows how temporal Nyquist aliasing (at k_d >= 0.08) + fast-time ICI break conventional 5G DMRS interpolation.
    """
    B = image.shape[0]
    device, dtype = image.device, image.dtype
    enc, dec = backbone.encoder, backbone.decoder
    L = 256
    C = int(enc.head_list.out_features)
    K = (L * C) // 2

    # Encode with true target SNR (giving DMRS baseline full SNR-adaptive advantage!)
    x = enc.patch_embed(image)
    for layer in enc.layers:
        x = layer(x)
    x = enc.norm(x)
    snr_col = torch.full((B, 1), float(snr_db), device=device, dtype=dtype)
    temp = None
    for i in range(enc.layer_num):
        temp = enc.sm_list[i](x) if i == 0 else enc.sm_list[i](temp)
        temp = temp * enc.bm_list[i](snr_col).unsqueeze(1).expand(-1, L, -1)
    z = enc.head_list(x * enc.sigmoid(enc.sm_list[-1](temp)))

    p_img = (2.0 * z.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
    flat = (z / p_img[:, None, None]).reshape(B, 2 * K)
    x_tf_iq = torch.stack([flat[:, :K].reshape(B, N_SYM, N_SC), flat[:, K:].reshape(B, N_SYM, N_SC)], dim=-1)

    y_tf_iq, phy = physical_doubly_selective_waveform_channel(
        x_tf_iq,
        snr_db,
        kd_max=kd_max,
        multipath_override=multipath_override,
        standard_noise_td_iq=standard_noise_td_iq,
    )

    # Simulate unit-power QPSK DMRS pilot symbols on n in pilot_syms through the EXACT SAME physical channel
    p_grid_c = torch.ones((B, N_SYM, N_SC), dtype=torch.complex64, device=device)
    # Newman chirp across subcarriers so pilot PAPR is low and realistic
    k_idx = torch.arange(N_SC, device=device, dtype=dtype)
    psi_f = math.pi * (k_idx * k_idx) / float(N_SC)
    p_sc = torch.complex(torch.cos(psi_f), torch.sin(psi_f))[None, None, :].expand(B, N_SYM, N_SC)
    p_grid_iq = torch.stack([p_sc.real, p_sc.imag], dim=-1)

    gen_p = torch.Generator(device="cpu").manual_seed(77123)
    eps_p = torch.randn((B, N_SYM, N_SC, 2), generator=gen_p, dtype=torch.float32).to(device=device, dtype=dtype)
    y_pilot_iq, _ = physical_doubly_selective_waveform_channel(
        p_grid_iq,
        snr_db,
        kd_max=kd_max,
        multipath_override=(phy["g_ri"], phy["nu_norm"]),
        standard_noise_td_iq=eps_p,
    )
    y_pilot_c = torch.complex(y_pilot_iq[..., 0], y_pilot_iq[..., 1])
    h_ls_all = y_pilot_c * torch.conj(p_sc)  # [B, 48, 256]

    # Frequency-domain causal delay truncation LMMSE on the 4 DMRS pilot symbols
    p_idx = list(pilot_syms)
    h_pilots = h_ls_all[:, p_idx, :]  # [B, 4, 256]
    h_pilots_delay = torch.fft.ifft(h_pilots, dim=-1, norm="ortho")
    m256 = torch.arange(N_SC, device=device)
    d_mask = ((m256 <= 20) | (m256 >= 248)).to(dtype)[None, None, :]
    sigma_n_sq = math.pow(10.0, -float(snr_db) / 10.0)
    pwr_d = h_pilots_delay.real.square() + h_pilots_delay.imag.square()
    w_d = ((pwr_d - sigma_n_sq).clamp_min(0.0) / (pwr_d + 1e-8)) * d_mask
    h_pilots_den = torch.fft.fft(h_pilots_delay * w_d, dim=-1, norm="ortho")  # [B, 4, 256]

    # Piecewise-linear complex interpolation across all n = 0..47 OFDM symbols
    h_interp = torch.zeros((B, N_SYM, N_SC), dtype=h_pilots_den.dtype, device=device)
    for n in range(N_SYM):
        if n <= p_idx[0]:
            h_interp[:, n, :] = h_pilots_den[:, 0, :]
        elif n >= p_idx[-1]:
            h_interp[:, n, :] = h_pilots_den[:, -1, :]
        else:
            for seg in range(len(p_idx) - 1):
                n0, n1 = p_idx[seg], p_idx[seg + 1]
                if n0 <= n <= n1:
                    w = float(n - n0) / float(n1 - n0)
                    h_interp[:, n, :] = (1.0 - w) * h_pilots_den[:, seg, :] + w * h_pilots_den[:, seg + 1, :]
                    break

    # LMMSE equalize payload TF grid
    y_tf_c = torch.complex(y_tf_iq[..., 0], y_tf_iq[..., 1])
    h_sq = h_interp.real.square() + h_interp.imag.square()
    denom = (h_sq + sigma_n_sq + 0.02).clamp_min(1e-6)
    alpha_bar = (h_sq / denom).mean(dim=(1, 2), keepdim=True).clamp_min(0.05)
    s_hat_c = (torch.conj(h_interp) * y_tf_c / denom) / alpha_bar

    u_hat = torch.cat([s_hat_c.real.reshape(B, K), s_hat_c.imag.reshape(B, K)], dim=1).reshape(B, L, C)
    u_hat = u_hat * p_img[:, None, None]

    xd = dec.head_list(u_hat)
    temp = None
    for i in range(dec.layer_num):
        temp = dec.sm_list[i](xd) if i == 0 else dec.sm_list[i](temp)
        temp = temp * dec.bm_list[i](snr_col).unsqueeze(1).expand(-1, L, -1)
    xd = xd * dec.sigmoid(dec.sm_list[-1](temp))
    for layer in dec.layers:
        xd = layer(xd)
    recon = xd.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()
    return {"recon": recon, "papr_db": phy["papr_db"], "h_hat_tf": h_interp, "h_true_tf": phy["h_true_tf"]}
