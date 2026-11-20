"""Per-sample physical wireless channels (AWGN & AR(1) Correlated Rayleigh Fading) with paired noise."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class ChannelOutput:
    u_restored: torch.Tensor      # (B, L, C) restored latent tensor at RX input
    u_norm: torch.Tensor          # (B, L, C) unit-complex-power normalized RX latent (mean signal coord sq = 0.5)
    z_norm: torch.Tensor          # (B, L, C) noiseless TX unit-complex-power latent
    pwr: torch.Tensor             # (B,) per-image TX power scalar (2 * mean(z_raw^2))
    gamma_db: torch.Tensor        # (B,) instantaneous physical SNR in dB
    c_r_db: torch.Tensor          # (B,) receiver-estimated effective SNR in dB
    sigma0_sq_cr: torch.Tensor    # (B,) expected per-real-coord noise variance in u_norm under c_R


def make_paired_noise(
    shape: tuple[int, ...],
    source_indices: list[int] | torch.Tensor,
    realization_seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic standard normal noise paired strictly by (source_index, realization_seed)."""
    B = len(source_indices)
    assert shape[0] == B
    rows = []
    for b in range(B):
        s_idx = int(source_indices[b])
        gen = torch.Generator(device="cpu")
        # Distinct 63-bit hash seed per (realization_seed, source_index)
        combined = ((int(realization_seed) * 1000003) ^ (s_idx * 9176 + 0x5DEECE66D)) & ((1 << 62) - 1)
        gen.manual_seed(combined)
        rows.append(torch.randn(shape[1:], generator=gen, dtype=dtype))
    return torch.stack(rows, dim=0).to(device=device)


def awgn_forward(
    z_raw: torch.Tensor,
    gamma_db: torch.Tensor | float,
    c_r_db: torch.Tensor | float | None = None,
    *,
    standard_noise: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> ChannelOutput:
    """Transmit z_raw (B, L, C) over per-sample complex AWGN at physical SNR gamma_db.

    Matches upstream SwinJSCC Channel.forward exactly for B=1 when active_mask is None:
      pwr = 2 * mean(z_raw^2)
      z_norm = z_raw / sqrt(pwr)
      first half + j * second half receives CN(0, 10^(-gamma/10)) <=> each real coord N(0, 1/(2*10^(gamma/10)))
      u_restored = (z_norm + noise) * sqrt(pwr)
    Note: since additive Gaussian noise is identically distributed on real and imaginary halves,
    pairing first/second halves and unpacking back is elementwise identical to adding N(0, sigma_real^2) directly
    when given the matching unpacked standard_noise tensor!
    """
    B, L, C = z_raw.shape
    device = z_raw.device
    dtype = z_raw.dtype

    if not torch.is_tensor(gamma_db):
        gamma_t = torch.full((B,), float(gamma_db), device=device, dtype=dtype)
    else:
        gamma_t = gamma_db.to(device=device, dtype=dtype).reshape(B)

    if c_r_db is None:
        cr_t = gamma_t.clone()
    elif not torch.is_tensor(c_r_db):
        cr_t = torch.full((B,), float(c_r_db), device=device, dtype=dtype)
    else:
        cr_t = c_r_db.to(device=device, dtype=dtype).reshape(B)

    if active_mask is not None:
        z_masked = z_raw * active_mask
        denom = active_mask.sum(dim=(1, 2)).clamp_min(1.0)
        pwr = 2.0 * z_masked.square().sum(dim=(1, 2)) / denom
        z_in = z_masked
    else:
        pwr = 2.0 * z_raw.square().mean(dim=(1, 2))
        z_in = z_raw

    pwr_safe = pwr.clamp_min(1e-12)
    sqrt_pwr = pwr_safe.sqrt()[:, None, None]
    z_norm = z_in / sqrt_pwr

    sigma_real = torch.pow(10.0, -gamma_t / 10.0).mul_(0.5).sqrt()[:, None, None]
    if standard_noise is None:
        eps = torch.randn_like(z_norm)
    else:
        assert standard_noise.shape == z_norm.shape
        eps = standard_noise.to(device=device, dtype=dtype)

    noise = eps * sigma_real
    # Straight-through additive noise (exact derivative wrt z_norm is 1, matching upstream channel.py)
    u_norm = z_norm + noise
    if active_mask is not None:
        u_norm = u_norm * active_mask
    u_restored = u_norm * sqrt_pwr

    sigma0_sq_cr = 0.5 * torch.pow(10.0, -cr_t / 10.0)
    return ChannelOutput(
        u_restored=u_restored,
        u_norm=u_norm,
        z_norm=z_norm,
        pwr=pwr_safe,
        gamma_db=gamma_t,
        c_r_db=cr_t,
        sigma0_sq_cr=sigma0_sq_cr,
    )


@dataclass
class FadingTrajectoryStep:
    c_t_db: float
    gamma_true_db: float
    c_r_db: float
    h_true: complex
    h_hat: complex
    sigma_n_sq: float
    sigma_e_sq: float


def simulate_ar1_fading_chain(
    num_steps: int,
    avg_snr_db: float,
    rho_c: float,
    delay_blocks: int,
    seed: int,
    *,
    pilot_symbols: int = 4,
    quantize_step_db: float = 0.75,
    quantize_min_db: float = -3.0,
    quantize_max_db: float = 21.0,
) -> list[FadingTrajectoryStep]:
    """Simulate a Gauss-Markov AR(1) flat Rayleigh block-fading trajectory with pilot LMMSE & feedback delay d.

    h_t = rho_c * h_{t-1} + sqrt(1 - rho_c^2) * w_t,  h_0, w_t ~ CN(0, 1)
    Pilot observation: y_p = h_t + (1/sqrt(N_p)) * v_p,  v_p ~ CN(0, sigma_n^2)
    LMMSE estimate: h_hat_t = (N_p / (N_p + sigma_n^2)) * y_p
    Estimation error variance: sigma_e^2 = sigma_n^2 / (N_p + sigma_n^2), with h_t = h_hat_t + e_t, e_t perp h_hat_t.
    Effective post-MMSE SNR known at RX:
        c_{R,t} = 10 * log10( |h_hat_t|^2 / (sigma_e^2 + sigma_n^2) )
    Delayed quantized feedback used at TX:
        c_{T,t} = Q_5bit( c_{R, t - d} )  (for d=0, instantaneous quantized feedback).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    sigma_n_sq = 10.0 ** (-float(avg_snr_db) / 10.0)
    np_f = float(pilot_symbols)
    lmmse_scale = np_f / (np_f + sigma_n_sq)
    sigma_e_sq = sigma_n_sq / (np_f + sigma_n_sq)

    warmup = max(8, delay_blocks + 4)
    total = warmup + num_steps

    normals = torch.randn((total, 4), generator=gen, dtype=torch.float64)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    h_list: list[complex] = []
    h_hat_list: list[complex] = []
    cr_list: list[float] = []
    gamma_list: list[float] = []

    h_prev = complex(float(normals[0, 0]) * inv_sqrt2, float(normals[0, 1]) * inv_sqrt2)
    keep_scale = math.sqrt(max(0.0, 1.0 - rho_c * rho_c))

    for t in range(total):
        if t == 0:
            h_cur = h_prev
        else:
            w_t = complex(float(normals[t, 0]) * inv_sqrt2, float(normals[t, 1]) * inv_sqrt2)
            h_cur = rho_c * h_prev + keep_scale * w_t
            h_prev = h_cur

        vp = complex(float(normals[t, 2]), float(normals[t, 3])) * math.sqrt(sigma_n_sq / (2.0 * np_f))
        y_p = h_cur + vp
        h_hat = lmmse_scale * y_p

        gamma_lin = max(1e-5, (abs(h_cur) ** 2) / sigma_n_sq)
        cr_lin = max(1e-5, (abs(h_hat) ** 2) / (sigma_e_sq + sigma_n_sq))
        gamma_db = float(max(-8.0, min(26.0, 10.0 * math.log10(gamma_lin))))
        cr_db = float(max(-5.0, min(25.0, 10.0 * math.log10(cr_lin))))

        h_list.append(h_cur)
        h_hat_list.append(h_hat)
        gamma_list.append(gamma_db)
        cr_list.append(cr_db)

    def quantize_5bit(val_db: float) -> float:
        clamped = max(quantize_min_db, min(quantize_max_db, val_db))
        idx = round((clamped - quantize_min_db) / quantize_step_db)
        idx = max(0, min(31, idx))
        return float(quantize_min_db + idx * quantize_step_db)

    steps: list[FadingTrajectoryStep] = []
    for k in range(warmup, total):
        cr_delayed = cr_list[k - delay_blocks]
        ct_db = quantize_5bit(cr_delayed)
        steps.append(
            FadingTrajectoryStep(
                c_t_db=ct_db,
                gamma_true_db=gamma_list[k],
                c_r_db=cr_list[k],
                h_true=h_list[k],
                h_hat=h_hat_list[k],
                sigma_n_sq=sigma_n_sq,
                sigma_e_sq=sigma_e_sq,
            )
        )
    return steps


def fading_equalized_forward(
    z_raw: torch.Tensor,
    step_info: list[FadingTrajectoryStep],
    *,
    standard_noise: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> ChannelOutput:
    """Apply flat Rayleigh block fading y = h * s + n followed by pilot-LMMSE equalization at RX."""
    B, L, C = z_raw.shape
    assert len(step_info) == B
    device = z_raw.device
    dtype = z_raw.dtype

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

    flat = z_norm.reshape(B, -1)
    half = flat.shape[1] // 2
    s_real, s_imag = flat[:, :half], flat[:, half:]

    if standard_noise is None:
        eps = torch.randn_like(flat)
    else:
        assert standard_noise.shape == z_norm.shape
        eps = standard_noise.to(device=device, dtype=dtype).reshape(B, -1)
    n_real_std, n_imag_std = eps[:, :half], eps[:, half:]

    hr = torch.tensor([st.h_true.real for st in step_info], device=device, dtype=dtype)[:, None]
    hi = torch.tensor([st.h_true.imag for st in step_info], device=device, dtype=dtype)[:, None]
    hhat_r = torch.tensor([st.h_hat.real for st in step_info], device=device, dtype=dtype)[:, None]
    hhat_i = torch.tensor([st.h_hat.imag for st in step_info], device=device, dtype=dtype)[:, None]
    sigma_n_real = torch.tensor([math.sqrt(st.sigma_n_sq * 0.5) for st in step_info], device=device, dtype=dtype)[:, None]
    eff_noise_var = torch.tensor([st.sigma_e_sq + st.sigma_n_sq for st in step_info], device=device, dtype=dtype)[:, None]

    # y = h * s + n
    y_real = hr * s_real - hi * s_imag + sigma_n_real * n_real_std
    y_imag = hr * s_imag + hi * s_real + sigma_n_real * n_imag_std

    # MMSE linear equalizer: w = h_hat^* / (|h_hat|^2 + sigma_e^2 + sigma_n^2)
    # Wait: unbiased / post-MMSE gain normalization: divide by |h_hat|^2 + sigma_e^2 + sigma_n^2 (MMSE) or |h_hat|^2 + 1e-6:
    # Note that SwinJSCC expects signal scale ~ 1.0 (unbiased signal amplitude), while LMMSE scales signal by |h_hat|^2/(|h_hat|^2 + sigma_eff^2).
    # Unbiased MMSE (W-ZF regularized): h_hat^* / (|h_hat|^2 + sigma_eff^2) preserves minimum MSE directly!
    denom_eq = (hhat_r.square() + hhat_i.square() + eff_noise_var).clamp_min(1e-6)
    shat_real = (hhat_r * y_real + hhat_i * y_imag) / denom_eq
    shat_imag = (hhat_r * y_imag - hhat_i * y_real) / denom_eq

    u_norm = torch.cat([shat_real, shat_imag], dim=1).reshape(B, L, C)
    if active_mask is not None:
        u_norm = u_norm * active_mask
    u_restored = u_norm * sqrt_pwr

    gamma_t = torch.tensor([st.gamma_true_db for st in step_info], device=device, dtype=dtype)
    cr_t = torch.tensor([st.c_r_db for st in step_info], device=device, dtype=dtype)
    sigma0_sq_cr = 0.5 * torch.pow(10.0, -cr_t / 10.0)

    return ChannelOutput(
        u_restored=u_restored,
        u_norm=u_norm,
        z_norm=z_norm,
        pwr=pwr_safe,
        gamma_db=gamma_t,
        c_r_db=cr_t,
        sigma0_sq_cr=sigma0_sq_cr,
    )
