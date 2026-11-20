"""V2 Wireless Physical Layer: AWGN, Native/Coherent Pilot Rayleigh, and 5G/OFDM Multi-RB Block Fading."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class ChannelV2Output:
    u_restored: torch.Tensor       # (B, L, C) primary decoder input
    u_mmse_restored: torch.Tensor  # (B, L, C) gain-unbiased pilot-MMSE reference latent
    u_norm: torch.Tensor           # (B, L, C) unit-power scaled RX latent before sqrt(pwr)
    z_norm: torch.Tensor           # (B, L, C) noiseless TX unit-complex-power latent
    pwr: torch.Tensor              # (B,) TX frame power scalar
    gamma_db: torch.Tensor         # (B,) true frame-effective physical SNR (dB)
    c_r_db: torch.Tensor           # (B,) RX wideband estimated effective SNR (dB)
    h_abs_iq: torch.Tensor         # (B, K) pilot-estimated channel magnitude |h_hat| per complex symbol
    alpha_ratio_iq: torch.Tensor   # (B, K) relative LMMSE reliability alpha_k / mean(alpha) per complex symbol
    sigma_post_iq: torch.Tensor    # (B, K) post-equalization noise variance per real coord
    phase_unc_iq: torch.Tensor     # (B, K) pilot phase uncertainty proxy per complex symbol
    psi_csi: torch.Tensor          # (B,) strictly 0.0 on matched ideal AWGN; >0 under fading or pilot error


def _compute_tx_norm(
    z_raw: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if active_mask is not None:
        z_in = z_raw * active_mask
        denom = active_mask.sum(dim=(1, 2)).clamp_min(1.0)
        pwr = 2.0 * z_in.square().sum(dim=(1, 2)) / denom
    else:
        z_in = z_raw
        pwr = 2.0 * z_raw.square().mean(dim=(1, 2))
    pwr_safe = pwr.clamp_min(1e-12)
    sqrt_pwr = pwr_safe.sqrt()[:, None, None]
    z_norm = z_in / sqrt_pwr
    return z_norm, pwr_safe, sqrt_pwr


def awgn_v2_forward(
    z_raw: torch.Tensor,
    gamma_db: torch.Tensor | float,
    c_r_db: torch.Tensor | float | None = None,
    *,
    standard_noise: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> ChannelV2Output:
    """Exact complex AWGN channel matching upstream SwinJSCC with V2 sufficient statistics."""
    B, L, C = z_raw.shape
    K = (L * C) // 2
    device = z_raw.device
    dtype = z_raw.dtype

    gamma_t = (
        torch.full((B,), float(gamma_db), device=device, dtype=dtype)
        if not torch.is_tensor(gamma_db)
        else gamma_db.to(device=device, dtype=dtype).reshape(B)
    )
    cr_t = (
        gamma_t.clone()
        if c_r_db is None
        else (
            torch.full((B,), float(c_r_db), device=device, dtype=dtype)
            if not torch.is_tensor(c_r_db)
            else c_r_db.to(device=device, dtype=dtype).reshape(B)
        )
    )

    z_norm, pwr_safe, sqrt_pwr = _compute_tx_norm(z_raw, active_mask)
    sigma_real = torch.pow(10.0, -gamma_t / 10.0).mul(0.5).sqrt()[:, None, None]
    eps = torch.randn_like(z_norm) if standard_noise is None else standard_noise.to(device=device, dtype=dtype)

    u_norm = z_norm + eps * sigma_real
    if active_mask is not None:
        u_norm = u_norm * active_mask
    u_restored = u_norm * sqrt_pwr

    sigma0_sq_cr = 0.5 * torch.pow(10.0, -cr_t / 10.0)
    ones_iq = torch.ones((B, K), device=device, dtype=dtype)
    zeros_iq = torch.zeros((B, K), device=device, dtype=dtype)
    sigma_post_iq = sigma0_sq_cr[:, None].expand(B, K).contiguous()
    psi_csi = torch.zeros((B,), device=device, dtype=dtype)

    return ChannelV2Output(
        u_restored=u_restored,
        u_mmse_restored=u_restored,
        u_norm=u_norm,
        z_norm=z_norm,
        pwr=pwr_safe,
        gamma_db=gamma_t,
        c_r_db=cr_t,
        h_abs_iq=ones_iq,
        alpha_ratio_iq=ones_iq,
        sigma_post_iq=sigma_post_iq,
        phase_unc_iq=zeros_iq,
        psi_csi=psi_csi,
    )


def rayleigh_v2_forward(
    z_raw: torch.Tensor,
    gamma_db: torch.Tensor | float,
    c_r_db: torch.Tensor | float | None = None,
    *,
    standard_noise: torch.Tensor | None = None,
    h_complex_iq: torch.Tensor | None = None,
    pilot_noise_iq: torch.Tensor | None = None,
    pilot_symbols: int = 0,
    num_rbs: int = 0,
    rb_rho_f: float = 0.85,
    active_mask: torch.Tensor | None = None,
) -> ChannelV2Output:
    """Native & Pilot-Aided Rayleigh channel compatible with official SwinJSCC_w_SA_Rayleigh.

    When pilot_symbols == 0 and num_rbs == 0:
      Exact i.i.d. Rayleigh amplitude y = |h| * s + n matching upstream channel.py rayleigh_noise_layer,
      while exposing receiver CSI |h| in h_abs_iq and unbiased MMSE reference in u_mmse_restored!
    When num_rbs > 0 or pilot_symbols > 0:
      Correlated RB complex Rayleigh fading y = h * s + n with pilot LMMSE estimation h_hat and MRC phase de-rotation.
    """
    B, L, C = z_raw.shape
    K = (L * C) // 2
    device = z_raw.device
    dtype = z_raw.dtype

    gamma_t = (
        torch.full((B,), float(gamma_db), device=device, dtype=dtype)
        if not torch.is_tensor(gamma_db)
        else gamma_db.to(device=device, dtype=dtype).reshape(B)
    )
    cr_t = (
        gamma_t.clone()
        if c_r_db is None
        else (
            torch.full((B,), float(c_r_db), device=device, dtype=dtype)
            if not torch.is_tensor(c_r_db)
            else c_r_db.to(device=device, dtype=dtype).reshape(B)
        )
    )

    z_norm, pwr_safe, sqrt_pwr = _compute_tx_norm(z_raw, active_mask)
    flat = z_norm.reshape(B, 2 * K)
    s_real, s_imag = flat[:, :K], flat[:, K:]

    eps = (
        torch.randn_like(flat)
        if standard_noise is None
        else standard_noise.to(device=device, dtype=dtype).reshape(B, 2 * K)
    )
    n_r_std, n_i_std = eps[:, :K], eps[:, K:]

    sigma_n_sq = torch.pow(10.0, -gamma_t / 10.0)  # complex noise variance per symbol
    sigma_cr_sq = torch.pow(10.0, -cr_t / 10.0)
    sigma_real = (0.5 * sigma_n_sq).sqrt()[:, None]

    if h_complex_iq is None:
        if num_rbs > 0:
            hr_rb, hi_rb = sample_correlated_rbs(B, num_rbs, rb_rho_f, device=device, dtype=dtype)
            hr = expand_rb_to_symbols(hr_rb, K)
            hi = expand_rb_to_symbols(hi_rb, K)
        else:
            hr = torch.randn((B, K), device=device, dtype=dtype) * math.sqrt(0.5)
            hi = torch.randn((B, K), device=device, dtype=dtype) * math.sqrt(0.5)
    else:
        hr = h_complex_iq[..., 0].to(device=device, dtype=dtype)
        hi = h_complex_iq[..., 1].to(device=device, dtype=dtype)

    if pilot_symbols > 0:
        np_f = float(pilot_symbols)
        sigma_p_real = (sigma_n_sq / (2.0 * np_f)).sqrt()[:, None]
        if pilot_noise_iq is None:
            vp_r = torch.randn((B, K), device=device, dtype=dtype) * sigma_p_real
            vp_i = torch.randn((B, K), device=device, dtype=dtype) * sigma_p_real
        else:
            vp_r = pilot_noise_iq[..., 0].to(device=device, dtype=dtype) * sigma_p_real
            vp_i = pilot_noise_iq[..., 1].to(device=device, dtype=dtype) * sigma_p_real
        lmmse_scale = (1.0 / (1.0 + sigma_cr_sq / np_f))[:, None]
        hhat_r = lmmse_scale * (hr + vp_r)
        hhat_i = lmmse_scale * (hi + vp_i)
        sigma_e_sq = (sigma_cr_sq / np_f) / (1.0 + sigma_cr_sq / np_f)  # (B,)
    else:
        hhat_r, hhat_i = hr, hi
        sigma_e_sq = torch.zeros((B,), device=device, dtype=dtype)

    # Complex transmission y = h * s + n
    y_r = hr * s_real - hi * s_imag + sigma_real * n_r_std
    y_i = hr * s_imag + hi * s_real + sigma_real * n_i_std

    # Coherent MRC phase de-rotation using h_hat^* / |h_hat|:
    # When pilot_symbols == 0 (perfect phase), y_derot = |h| * s + rotated_awgn,
    # which is 100% identically distributed to upstream SwinJSCC rayleigh_noise_layer!
    hhat_abs = (hhat_r.square() + hhat_i.square()).clamp_min(1e-10).sqrt()
    cos_p = hhat_r / hhat_abs
    sin_p = hhat_i / hhat_abs
    y_derot_r = cos_p * y_r + sin_p * y_i
    y_derot_i = -sin_p * y_r + cos_p * y_i

    u_norm = torch.cat([y_derot_r, y_derot_i], dim=1).reshape(B, L, C)
    if active_mask is not None:
        u_norm = u_norm * active_mask
    u_restored = u_norm * sqrt_pwr

    # Gain-unbiased pilot-MMSE reference latent
    eff_var = (sigma_cr_sq + sigma_e_sq)[:, None]  # (B, 1)
    denom_mmse = (hhat_abs.square() + eff_var).clamp_min(1e-6)
    alpha_k = hhat_abs.square() / denom_mmse  # (B, K)
    alpha_mean = alpha_k.mean(dim=1, keepdim=True).clamp_min(0.05)
    w_mmse_abs = (hhat_abs / denom_mmse) / alpha_mean
    mmse_r = y_derot_r * w_mmse_abs
    mmse_i = y_derot_i * w_mmse_abs
    u_mmse_norm = torch.cat([mmse_r, mmse_i], dim=1).reshape(B, L, C)
    if active_mask is not None:
        u_mmse_norm = u_mmse_norm * active_mask
    u_mmse_restored = u_mmse_norm * sqrt_pwr

    alpha_ratio_iq = alpha_k / alpha_mean
    # Post-MMSE per-real-coord noise variance
    sigma_post_iq = 0.5 * (eff_var / denom_mmse).clamp_min(1e-6)
    phase_unc_iq = (sigma_e_sq[:, None] / (hhat_abs.square() + sigma_e_sq[:, None] + 1e-6)).sqrt()
    psi_csi = (hhat_abs - 1.0).square().mean(dim=1) + sigma_e_sq

    return ChannelV2Output(
        u_restored=u_restored,
        u_mmse_restored=u_mmse_restored,
        u_norm=u_norm,
        z_norm=z_norm,
        pwr=pwr_safe,
        gamma_db=gamma_t,
        c_r_db=cr_t,
        h_abs_iq=hhat_abs,
        alpha_ratio_iq=alpha_ratio_iq,
        sigma_post_iq=sigma_post_iq,
        phase_unc_iq=phase_unc_iq,
        psi_csi=psi_csi,
    )


def sample_correlated_rbs(
    batch_size: int,
    num_rbs: int,
    rho_f: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample CN(0, 1) frequency-correlated Rayleigh RBs via first-order Markov recursion across RBs."""
    w = torch.randn((batch_size, num_rbs, 2), generator=generator, dtype=dtype, device=device) * math.sqrt(0.5)
    if rho_f <= 0.0 or num_rbs == 1:
        return w[:, :, 0], w[:, :, 1]
    keep = math.sqrt(max(0.0, 1.0 - rho_f * rho_f))
    hr_list = [w[:, 0, 0]]
    hi_list = [w[:, 0, 1]]
    for m in range(1, num_rbs):
        hr_list.append(rho_f * hr_list[-1] + keep * w[:, m, 0])
        hi_list.append(rho_f * hi_list[-1] + keep * w[:, m, 1])
    return torch.stack(hr_list, dim=1), torch.stack(hi_list, dim=1)


def expand_rb_to_symbols(rb_tensor: torch.Tensor, num_complex_symbols: int) -> torch.Tensor:
    """Map M contiguous OFDM Resource Blocks onto K complex symbols."""
    B, M = rb_tensor.shape
    reps = (num_complex_symbols + M - 1) // M
    return rb_tensor.repeat_interleave(reps, dim=1)[:, :num_complex_symbols].contiguous()


@dataclass
class OFDMFadingStep:
    c_t_raw_db: float             # delayed quantized wideband CQI reported to TX
    c_r_db: float                 # instantaneous wideband post-MMSE effective SINR at RX
    gamma_avg_db: float           # long-term average SNR
    tau_unc: float                # known temporal feedback innovation uncertainty 1 - rho_c^(2*d)
    hr_rb: np.ndarray             # (M,) true RB real part
    hi_rb: np.ndarray             # (M,) true RB imag part
    hhat_r_rb: np.ndarray         # (M,) pilot-LMMSE estimated RB real part
    hhat_i_rb: np.ndarray         # (M,) pilot-LMMSE estimated RB imag part
    sigma_n_sq: float
    sigma_e_sq: float


def simulate_ofdm_ar1_chain(
    num_steps: int,
    avg_snr_db: float,
    rho_c: float,
    delay_blocks: int,
    seed: int,
    *,
    num_rbs: int = 32,
    rho_f: float = 0.85,
    pilot_symbols: int = 4,
    warmup: int = 32,
) -> list[OFDMFadingStep]:
    """Simulate 3GPP-style time-frequency correlated M-RB OFDM Rayleigh block fading with delayed 5-bit CQI."""
    rng = np.random.default_rng(seed)
    total = warmup + num_steps
    sigma_n_sq = float(10.0 ** (-avg_snr_db / 10.0))
    np_f = float(pilot_symbols)
    sigma_e_sq = float((sigma_n_sq / np_f) / (1.0 + sigma_n_sq / np_f))
    lmmse_scale = float(1.0 / (1.0 + sigma_n_sq / np_f))
    eff_var = sigma_n_sq + sigma_e_sq

    keep_t = math.sqrt(max(0.0, 1.0 - rho_c * rho_c))
    keep_f = math.sqrt(max(0.0, 1.0 - rho_f * rho_f))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def draw_freq_corr_innovation() -> tuple[np.ndarray, np.ndarray]:
        n_raw = rng.standard_normal((num_rbs, 2)) * inv_sqrt2
        wr = np.empty(num_rbs, dtype=np.float32)
        wi = np.empty(num_rbs, dtype=np.float32)
        wr[0], wi[0] = n_raw[0, 0], n_raw[0, 1]
        for m in range(1, num_rbs):
            wr[m] = rho_f * wr[m - 1] + keep_f * n_raw[m, 0]
            wi[m] = rho_f * wi[m - 1] + keep_f * n_raw[m, 1]
        return wr, wi

    def quantize_5bit(val_db: float, lo: float = -5.0, step: float = 0.8) -> float:
        hi = lo + 31.0 * step
        clamped = max(lo, min(hi, val_db))
        idx = max(0, min(31, round((clamped - lo) / step)))
        return float(lo + idx * step)

    hr_hist: list[np.ndarray] = []
    hi_hist: list[np.ndarray] = []
    hhat_r_hist: list[np.ndarray] = []
    hhat_i_hist: list[np.ndarray] = []
    cr_hist: list[float] = []

    hr_cur, hi_cur = draw_freq_corr_innovation()
    for t in range(total):
        if t > 0:
            wr, wi = draw_freq_corr_innovation()
            hr_cur = (rho_c * hr_cur + keep_t * wr).astype(np.float32)
            hi_cur = (rho_c * hi_cur + keep_t * wi).astype(np.float32)

        vp = rng.standard_normal((num_rbs, 2)).astype(np.float32) * math.sqrt(sigma_n_sq / (2.0 * np_f))
        hhat_r = (lmmse_scale * (hr_cur + vp[:, 0])).astype(np.float32)
        hhat_i = (lmmse_scale * (hi_cur + vp[:, 1])).astype(np.float32)

        # Analytical post-MMSE unbiased wideband SINR: mean(alpha) / (1 - mean(alpha))
        hhat_sq = hhat_r * hhat_r + hhat_i * hhat_i
        alpha_bar = float(np.mean(hhat_sq / np.maximum(hhat_sq + eff_var, 1e-8)))
        sinr_wb = max(1e-4, alpha_bar) / max(1e-4, 1.0 - alpha_bar)
        cr_db = float(np.clip(10.0 * math.log10(sinr_wb), -5.0, 24.0))

        hr_hist.append(hr_cur.copy())
        hi_hist.append(hi_cur.copy())
        hhat_r_hist.append(hhat_r)
        hhat_i_hist.append(hhat_i)
        cr_hist.append(cr_db)

    tau_unc = 0.0 if delay_blocks == 0 else float(1.0 - (rho_c ** (2 * delay_blocks)))
    out: list[OFDMFadingStep] = []
    for k in range(warmup, total):
        ct_raw = quantize_5bit(cr_hist[k - delay_blocks]) if delay_blocks > 0 else cr_hist[k]
        out.append(
            OFDMFadingStep(
                c_t_raw_db=ct_raw,
                c_r_db=cr_hist[k],
                gamma_avg_db=avg_snr_db,
                tau_unc=tau_unc,
                hr_rb=hr_hist[k],
                hi_rb=hi_hist[k],
                hhat_r_rb=hhat_r_hist[k],
                hhat_i_rb=hhat_i_hist[k],
                sigma_n_sq=sigma_n_sq,
                sigma_e_sq=sigma_e_sq,
            )
        )
    return out


def ofdm_fading_forward(
    z_raw: torch.Tensor,
    steps: list[OFDMFadingStep],
    *,
    standard_noise: torch.Tensor | None = None,
    equalizer_mode: str = "unbiased_mmse",
    active_mask: torch.Tensor | None = None,
) -> ChannelV2Output:
    """Transmit z_raw over M-RB OFDM fading and apply fair Gain-Unbiased Pilot-MMSE (or native MRC for Rayleigh ckpt)."""
    B, L, C = z_raw.shape
    K = (L * C) // 2
    assert len(steps) == B
    device = z_raw.device
    dtype = z_raw.dtype

    z_norm, pwr_safe, sqrt_pwr = _compute_tx_norm(z_raw, active_mask)
    flat = z_norm.reshape(B, 2 * K)
    s_real, s_imag = flat[:, :K], flat[:, K:]

    eps = (
        torch.randn_like(flat)
        if standard_noise is None
        else standard_noise.to(device=device, dtype=dtype).reshape(B, 2 * K)
    )
    n_r_std, n_i_std = eps[:, :K], eps[:, K:]

    hr_rb = torch.from_numpy(np.stack([st.hr_rb for st in steps], axis=0)).to(device=device, dtype=dtype)
    hi_rb = torch.from_numpy(np.stack([st.hi_rb for st in steps], axis=0)).to(device=device, dtype=dtype)
    hhat_r_rb = torch.from_numpy(np.stack([st.hhat_r_rb for st in steps], axis=0)).to(device=device, dtype=dtype)
    hhat_i_rb = torch.from_numpy(np.stack([st.hhat_i_rb for st in steps], axis=0)).to(device=device, dtype=dtype)

    hr = expand_rb_to_symbols(hr_rb, K)
    hi = expand_rb_to_symbols(hi_rb, K)
    hhat_r = expand_rb_to_symbols(hhat_r_rb, K)
    hhat_i = expand_rb_to_symbols(hhat_i_rb, K)

    sigma_n_sq = torch.tensor([st.sigma_n_sq for st in steps], device=device, dtype=dtype)
    sigma_e_sq = torch.tensor([st.sigma_e_sq for st in steps], device=device, dtype=dtype)
    sigma_real = (0.5 * sigma_n_sq).sqrt()[:, None]

    # Physical OFDM channel: y = h * s + n
    y_r = hr * s_real - hi * s_imag + sigma_real * n_r_std
    y_i = hr * s_imag + hi * s_real + sigma_real * n_i_std

    # Phase de-rotation + Gain-Unbiased Pilot-MMSE
    hhat_abs = (hhat_r.square() + hhat_i.square()).clamp_min(1e-10).sqrt()
    cos_p = hhat_r / hhat_abs
    sin_p = hhat_i / hhat_abs
    y_derot_r = cos_p * y_r + sin_p * y_i
    y_derot_i = -sin_p * y_r + cos_p * y_i

    eff_var = (sigma_n_sq + sigma_e_sq)[:, None]
    denom_mmse = (hhat_abs.square() + eff_var).clamp_min(1e-6)
    alpha_k = hhat_abs.square() / denom_mmse
    alpha_mean = alpha_k.mean(dim=1, keepdim=True).clamp_min(0.05)

    w_ub = (hhat_abs / denom_mmse) / alpha_mean
    mmse_r = y_derot_r * w_ub
    mmse_i = y_derot_i * w_ub
    u_mmse_norm = torch.cat([mmse_r, mmse_i], dim=1).reshape(B, L, C)
    if active_mask is not None:
        u_mmse_norm = u_mmse_norm * active_mask
    u_mmse_restored = u_mmse_norm * sqrt_pwr

    if equalizer_mode == "mrc_derot":
        # Native amplitude distribution |h|*s + n for SwinJSCC_w_SA_Rayleigh
        u_norm = torch.cat([y_derot_r, y_derot_i], dim=1).reshape(B, L, C)
        if active_mask is not None:
            u_norm = u_norm * active_mask
        u_restored = u_norm * sqrt_pwr
    else:
        u_norm = u_mmse_norm
        u_restored = u_mmse_restored

    cr_t = torch.tensor([st.c_r_db for st in steps], device=device, dtype=dtype)
    true_power = (hr.square() + hi.square()).mean(dim=1)
    gamma_t = 10.0 * torch.log10((true_power / sigma_n_sq.clamp_min(1e-8)).clamp_min(1e-5))

    alpha_ratio_iq = alpha_k / alpha_mean
    # Exact local post-unbiased-MMSE noise variance per real dim: 0.5 * alpha_k*(1 - alpha_k)/alpha_mean^2 + bias proxy
    sigma_post_iq = 0.5 * ((1.0 - alpha_k) / alpha_mean).clamp_min(1e-6)
    phase_unc_iq = (sigma_e_sq[:, None] / (hhat_abs.square() + sigma_e_sq[:, None] + 1e-6)).sqrt()
    psi_csi = (hhat_abs - 1.0).square().mean(dim=1) + sigma_e_sq

    return ChannelV2Output(
        u_restored=u_restored,
        u_mmse_restored=u_mmse_restored,
        u_norm=u_norm,
        z_norm=z_norm,
        pwr=pwr_safe,
        gamma_db=gamma_t,
        c_r_db=cr_t,
        h_abs_iq=hhat_abs,
        alpha_ratio_iq=alpha_ratio_iq,
        sigma_post_iq=sigma_post_iq,
        phase_unc_iq=phase_unc_iq,
        psi_csi=psi_csi,
    )
