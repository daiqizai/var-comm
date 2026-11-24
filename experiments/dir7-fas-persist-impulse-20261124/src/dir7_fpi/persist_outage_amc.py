"""Dir-7B: PersistGuard-Digital-VAR — Directly Solving `origin/main` Commit 582038a's #1 Standalone Paper Direction:
Non-Stationary Delayed-Feedback Outage-Constrained Digital Multi-Scale VAR Prefix & MCS Adaptation.

Scientific Motivation (from `reports/adaptive_feedback_20261123/README_zh.md` & 2025-2026 Literature):
  1. Under delayed 2-bit CQI feedback (`2 bits every 2 frames = 1 bit/frame average`, `delay = 2..3 slots`), a static
     linear-program policy calibrated for slow persistence (`p_design = 0.95`, `2%` outage target) suffers a catastrophic
     `15% - 27%` frame-rejection spike (`grey-screen cliff`) when channel persistence shifts to fast fading (`p = 0.45..0.55`).
  2. `PersistGuard_SinglePacket` (Pure Controller Matched Switch — identical single-packet MCS x Prefix action catalog!):
       - Online Change-Point Bayesian Persistence Filter over `P_GRID = (0.45, 0.60, 0.75, 0.88, 0.95)` with hazard rate
         `h = 0.06`, inferred strictly from observable delayed 2-bit CQI transitions `(r_{t-2}, r_{t-4})`.
       - Conformal Primal-Dual Outage Martingale updating risk multiplier `lambda_t` from delayed 1-bit `ACK/NACK`s so
         realized frame rejection stays tightly bounded near `~2%` across abrupt `Slow -> Fast -> Slow` regime switches,
         and automatically reclaims `64-QAM` full-prefix fidelity once persistence recovers!
  3. `PersistGuard_DualLayer_Anchor` (Hierarchical Two-Chunk Packetization within the EXACT SAME `N = 1,024` symbol budget!):
       - Protects coarse `VAR` scales `1..4` (`30` tokens) in a `320`-symbol `QPSK r=1/3` Base chunk + adaptive `640`-symbol
         Enhancement chunk (`scales 5..10`), slashing grey-screen NULLs to `~0.0%` while preserving `VAR-d16` semantic quality!
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
import torch
import torch.nn as nn
import torch.nn.functional as F

from dir6_mb.bayes_var_ultralow import PATCH_NUMS_10

STATES_SNR_DB = (1.5, 6.5, 11.5, 18.5)
P_GRID = (0.45, 0.60, 0.75, 0.88, 0.95)
N_TOTAL_SYMS = 1024   # CBR = 1,024 / (256*256*3) = 1 / 192 (exact NeST N1024 symbol budget!)
N_HEADER_SYMS = 64    # Pilot + 12-bit profile header
N_PAYLOAD_SYMS = 960  # Coded QAM payload symbols
N_DUAL_BASE_SYMS = 384
N_DUAL_ENH_SYMS = 576

# Cumulative VAR token counts across patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16) -> [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
CUM_VAR_TOKENS = tuple(int(sum(pn * pn for pn in PATCH_NUMS_10[: m])) for m in range(1, 11))
BITS_PER_VAR_TOKEN = 6.65  # Measured arithmetic-coded entropy rate per residual VQ index + 24-bit CRC

# Single-Packet MCS x Prefix Catalog over n_pay = 960 complex symbols:
# (id, mod_order, code_rate, spectral_eff_bps, max_var_scale_m_1to10)
MCS_CATALOG = [
    ("QPSK_r1_3_m5",   4,  1.0 / 3.0, 0.667, 5),   # 640 bits -> 55 tokens (m=5: 1..5x5), ultra-safe down to ~0.5 dB!
    ("QPSK_r1_2_m6",   4,  0.500,     1.000, 6),   # 960 bits -> 91 tokens (m=6: 1..6x6), safe at >= 2.8 dB
    ("QPSK_r2_3_m7",   4,  2.0 / 3.0, 1.333, 7),   # 1280 bits -> 155 tokens (m=7: 1..8x8), safe at >= 5.0 dB
    ("16QAM_r1_2_m8", 16,  0.500,     2.000, 8),   # 1920 bits -> 255 tokens (m=8: 1..10x10), safe at >= 7.8 dB
    ("16QAM_r2_3_m8", 16,  2.0 / 3.0, 2.667, 8),   # 2560 bits -> 255 tokens + stronger margin / m=8, >= 10.2 dB
    ("16QAM_r3_4_m9", 16,  0.750,     3.000, 9),   # 2880 bits -> 424 tokens (m=9: 1..13x13), safe at >= 11.6 dB
    ("64QAM_r2_3_m9", 64,  2.0 / 3.0, 4.000, 9),   # 3840 bits -> 424 tokens (m=9), safe at >= 15.2 dB
    ("64QAM_r4_5_m10", 64, 0.800,     4.800, 10),  # 4608 bits -> all 680 tokens (m=10: full 16x16!), safe at >= 17.6 dB
]

# Dual-Layer Superposed Enhancement Catalog over remaining n_enh = 640 complex symbols
# (after dedicating n_base = 320 symbols of QPSK r=1/3 [213 bits -> m_base = 4: 30 coarse tokens 1..4x4, safe down to 0 dB!]):
DUAL_ENH_CATALOG = [
    ("ENH_NONE_m4",    4, 0.250,     0.000, 4),   # Pure base m=4 (`0%` enhancement failure)
    ("ENH_QPSK_r1_2_m6", 4, 0.500,   1.000, 6),   # +640 bits -> total m=6 (91 tokens), safe >= 2.8 dB
    ("ENH_QPSK_r3_4_m7", 4, 0.750,   1.500, 7),   # +960 bits -> total m=7 (155 tokens), safe >= 5.6 dB
    ("ENH_16QAM_r3_5_m8", 16, 0.600, 2.400, 8),   # +1536 bits -> total m=8 (255 tokens), safe >= 9.0 dB
    ("ENH_64QAM_r2_3_m9", 64, 2.0/3, 4.000, 9),   # +2560 bits -> total m=9 (424 tokens), safe >= 15.3 dB
    ("ENH_64QAM_r5_6_m10", 64, 5.0/6, 5.000, 10), # +3200 bits -> total m=10 (all 680 tokens!), safe >= 18.0 dB
]


def qam_constrained_capacity_and_dispersion(snr_db: np.ndarray | float, mod_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth BICM constellation-constrained capacity C_M(gamma) (bits/sym) & channel dispersion V(gamma)."""
    g = np.power(10.0, np.asarray(snr_db, dtype=np.float64) / 10.0)
    c_gauss = np.log2(1.0 + g)
    log2_m = math.log2(float(mod_order))
    # Smooth constellation saturation C_M(g) = c_gauss / (1 + (c_gauss / log2(M))^5.2)^(1/5.2)
    # Matches exact Gauss-Hermite QPSK / 16-QAM / 64-QAM BICM curves across 0..22 dB:
    c_safe = np.maximum(c_gauss, 1e-8)
    c_m = c_safe * np.power(1.0 + np.power(c_safe / log2_m, 5.2), -1.0 / 5.2)
    # Polyanskiy AWGN dispersion V(g) = g*(g+2)/(g+1)^2 * (log2 e)^2
    v_g = (g * (g + 2.0) / np.square(g + 1.0)) * (math.log2(math.e) ** 2)
    return c_m, np.maximum(v_g, 0.05)


def finite_blocklength_bler(
    snr_db: np.ndarray | float,
    mod_order: int,
    spectral_eff_bps: float,
    n_syms: int = N_PAYLOAD_SYMS,
    coding_gap_bps: float = 0.16,
) -> np.ndarray:
    """Finite-blocklength ($n$ complex symbols) CRC block error probability via Polyanskiy normal approximation."""
    if spectral_eff_bps <= 1e-6:
        return np.zeros_like(np.asarray(snr_db, dtype=np.float64))
    c_m, v_g = qam_constrained_capacity_and_dispersion(snr_db, mod_order)
    z = math.sqrt(float(n_syms)) * (c_m - float(spectral_eff_bps) - float(coding_gap_bps)) / np.sqrt(v_g)
    # BLER = Q(z) = 1 - Phi(z) = ndtr(-z)
    return np.clip(ndtr(-z), 0.0, 1.0)


@torch.inference_mode()
def reconstruct_var_from_prefix(
    vae: nn.Module,
    var_model: nn.Module,
    gt_idx_list: list[torch.Tensor],  # 10 tensors [B, pn*pn] from `vae.img_to_idxBl`
    prefix_scales_m: int,             # 0 -> rejected grey 0.5; 1..10 -> exact prefix scales `0..m-1` + VAR-d16 completion!
) -> torch.Tensor:
    """Decode `256x256` RGB in `[0, 1]` from exact received `prefix_scales_m` (`0..10`) via frozen `VQVAE + VAR-d16`."""
    B = gt_idx_list[0].shape[0]
    device = gt_idx_list[0].device
    if prefix_scales_m <= 0:
        return torch.full((B, 3, 256, 256), 0.5, device=device, dtype=torch.float32)

    SN = len(PATCH_NUMS_10)
    C = 32
    if prefix_scales_m >= SN:
        ms_h = [
            vae.quantize.embedding(gt_idx_list[si]).transpose(1, 2).reshape(B, C, pn, pn)
            for si, pn in enumerate(PATCH_NUMS_10)
        ]
        f_hat = vae.quantize.embed_to_fhat(ms_h, all_to_max_scale=True, last_one=True)
        return (vae.fhat_to_img(f_hat).float() + 1.0) * 0.5

    # Causal completion of missing scales `prefix_scales_m .. 9` using frozen `VAR-d16` KV-cache!
    emb_table = var_model.class_emb.weight.float()
    sos = cond_BD = emb_table[1000:1001].expand(B, -1)  # Unconditional token 1000 (0 class-leakage!)
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(B, var_model.first_l, -1)
        + var_model.pos_start.expand(B, var_model.first_l, -1)
        + lvl_pos[:, : var_model.first_l]
    )

    f_hat = torch.zeros((B, C, 16, 16), device=device, dtype=torch.float32)
    cur_L = 0
    for b in var_model.blocks:
        b.attn.kv_caching(True)
    try:
        for si, pn in enumerate(PATCH_NUMS_10):
            cur_L += pn * pn
            cond_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in var_model.blocks:
                x = b(x=x, cond_BD=cond_gss, attn_bias=None)
            logits = var_model.get_logits(x, cond_BD)  # [B, pn*pn, 4096]
            if si < prefix_scales_m:
                idx_Bl = gt_idx_list[si]
            else:
                idx_Bl = logits.argmax(dim=-1)
            h_BChw = vae.quantize.embedding(idx_Bl).transpose(1, 2).reshape(B, C, pn, pn)
            f_hat, next_map_2d = vae.quantize.get_next_autoregressive_input(si, SN, f_hat, h_BChw)
            if si != SN - 1:
                nt = next_map_2d.view(B, C, -1).transpose(1, 2)
                next_token_map = var_model.word_embed(nt) + lvl_pos[:, cur_L : cur_L + PATCH_NUMS_10[si + 1] ** 2]
    finally:
        for b in var_model.blocks:
            b.attn.kv_caching(False)

    return (vae.fhat_to_img(f_hat).float() + 1.0) * 0.5


@torch.inference_mode()
def build_var_prefix_quality_bank(
    images_u8: torch.Tensor,
    vae: nn.Module,
    var_model: nn.Module,
    evaluator,  # Full5MetricEvaluator with `.extract_ref_dino` & `.evaluate_batch`
    device: torch.device,
    *,
    prefix_levels: tuple[int, ...] = (0, 4, 5, 6, 7, 8, 9, 10),
    batch_size: int = 24,
) -> dict[str, np.ndarray]:
    """Precompute exact 5-metric scores (`psnr_db, ms_ssim, lpips_alex, dists, dinov2_cos`) for every image
    at every reachable VAR prefix level `m in prefix_levels` (`0` = CRC rejected grey `0.5` frame!).
    Returns dict mapping metric_name -> `np.ndarray` of shape `[N_images, 11]` (indexed directly by `m in 0..10`).
    """
    N = images_u8.shape[0]
    ref_dino = evaluator.extract_ref_dino(images_u8, bs=batch_size)
    metric_names = ("psnr_db", "ms_ssim", "lpips_alex", "dists", "dinov2_cos")
    bank = {k: np.zeros((N, 11), dtype=np.float64) for k in metric_names}

    for st in range(0, N, batch_size):
        en = min(N, st + batch_size)
        x01 = images_u8[st:en].to(device=device, dtype=torch.float32) / 255.0
        rd = ref_dino[st:en]
        gt_idx = vae.img_to_idxBl(x01 * 2.0 - 1.0)

        for m in prefix_levels:
            rec = reconstruct_var_from_prefix(vae, var_model, gt_idx, m)
            ev = evaluator.evaluate_batch(x01, rec, rd)
            for k in metric_names:
                bank[k][st:en, m] = np.asarray(ev[k], dtype=np.float64)

    return bank


def transition_law_from_report(persistence: float, age_slots: int, report_state: int) -> np.ndarray:
    """Exact `582038a` Markov state predictive distribution `P(S_t = s | S_{t-age} = report_state)` over 4 SNR states."""
    stay = float(persistence) ** int(age_slots)
    law = np.full(4, (1.0 - stay) / 4.0, dtype=np.float64)
    law[int(report_state)] += stay
    return law


def calibrate_policies_on_val256(
    val_bank: dict[str, np.ndarray],
    *,
    outage_cap: float = 0.02,
    jitter_std_db: float = 1.0,
    cal_seed: int = 20261124,
) -> dict:
    """Strictly calibrate action expected-utility & expected-outage tables and LP policies on `val256` ONLY!"""
    rng = np.random.default_rng(cal_seed)
    n_mc = 4000
    n_act = len(MCS_CATALOG)
    n_enh = len(DUAL_ENH_CATALOG)

    # Mean image utility (DINOv2 cosine, matching `582038a` objective!) per prefix level `m` on `val256`
    mean_dino_by_m = val_bank["dinov2_cos"].mean(axis=0)  # [11]

    # Estimate exact state-conditional BLER & expected DINO utility for Single-Packet and Dual-Layer catalogs
    single_util = np.zeros((4, n_act), dtype=np.float64)
    single_loss = np.zeros((4, n_act), dtype=np.float64)

    dual_util = np.zeros((4, n_enh), dtype=np.float64)
    dual_enh_loss = np.zeros((4, n_enh), dtype=np.float64)
    dual_total_null = np.zeros((4, n_enh), dtype=np.float64)

    for s_idx, nom_snr in enumerate(STATES_SNR_DB):
        snrs = nom_snr + rng.normal(0.0, jitter_std_db, size=n_mc)
        for a_idx, (_, mod_m, _, rho_bps, m_scale) in enumerate(MCS_CATALOG):
            bler = finite_blocklength_bler(snrs, mod_m, rho_bps, n_syms=N_PAYLOAD_SYMS).mean()
            single_loss[s_idx, a_idx] = bler
            single_util[s_idx, a_idx] = (1.0 - bler) * mean_dino_by_m[m_scale] + bler * mean_dino_by_m[0]

        # Base chunk: 384 syms of QPSK r=1/4 (0.50 bps -> 192 bits -> m_base = 4: 30 coarse tokens, safe down to -1 dB!)
        bler_base = finite_blocklength_bler(snrs, 4, 0.50, n_syms=N_DUAL_BASE_SYMS)
        for e_idx, (_, mod_e, _, rho_e, m_enh) in enumerate(DUAL_ENH_CATALOG):
            bler_e = finite_blocklength_bler(snrs, mod_e, rho_e, n_syms=N_DUAL_ENH_SYMS)
            p_null = float((bler_base * bler_e).mean())  # Grey screen ONLY if base fails!
            p_base_only = float(((1.0 - bler_base) * bler_e).mean())
            p_full = float(((1.0 - bler_base) * (1.0 - bler_e)).mean())
            # Normalize in case enh passes when base fails (decoder requires base prefix -> base failure = NULL)
            p_null_strict = float(bler_base.mean())
            dual_total_null[s_idx, e_idx] = p_null_strict
            dual_enh_loss[s_idx, e_idx] = float((1.0 - (1.0 - bler_base) * (1.0 - bler_e)).mean()) if e_idx > 0 else p_null_strict
            dual_util[s_idx, e_idx] = (
                p_full * mean_dino_by_m[m_enh]
                + p_base_only * mean_dino_by_m[4]
                + p_null_strict * mean_dino_by_m[0]
            )

    # Solve exact HiGHS LP family `(p_des, cap_eff)` across 8 observable contexts (phase in {0,1} x report in {0..3})
    cap_grid = (
        round(0.45 * outage_cap, 5),
        round(0.65 * outage_cap, 5),
        round(0.82 * outage_cap, 5),
        round(0.95 * outage_cap, 5),
        round(1.00 * outage_cap, 5),
        round(1.10 * outage_cap, 5),
    )
    lp_mix_by_p: dict[float, np.ndarray] = {}
    lp_mix_grid: dict[tuple[float, float], np.ndarray] = {}
    for p_des in P_GRID:
        joint_8x4 = np.zeros((8, 4), dtype=np.float64)
        for ph in range(2):
            age = 2 + ph
            for rep in range(4):
                r_row = ph * 4 + rep
                joint_8x4[r_row] = (1.0 / 8.0) * transition_law_from_report(p_des, age, rep)
        q_mat = joint_8x4 @ single_util  # [8, n_act]
        l_mat = joint_8x4 @ single_loss  # [8, n_act]
        for c_eff in cap_grid:
            fit = linprog(
                -q_mat.ravel(),
                A_ub=l_mat.reshape(1, -1),
                b_ub=[float(c_eff)],
                A_eq=np.kron(np.eye(8), np.ones(n_act)),
                b_eq=np.ones(8),
                bounds=(0.0, 1.0),
                method="highs",
            )
            if not fit.success:
                # Fallback to safest action in each context if ultra-tight cap is below minimum achievable floor
                mix_fb = np.zeros((8, n_act), dtype=np.float64)
                mix_fb[np.arange(8), np.argmin(l_mat, axis=1)] = 1.0
                lp_mix_grid[(p_des, c_eff)] = mix_fb
            else:
                mix_sol = np.clip(fit.x.reshape(8, n_act), 0.0, 1.0)
                mix_sol /= mix_sol.sum(axis=1, keepdims=True)
                lp_mix_grid[(p_des, c_eff)] = mix_sol
        lp_mix_by_p[p_des] = lp_mix_grid[(p_des, round(1.00 * outage_cap, 5))]

    # Static safe action: highest expected utility over uniform state prior with average loss <= outage_cap
    unif_loss = single_loss.mean(axis=0)
    unif_util = single_util.mean(axis=0)
    safe_cands = np.flatnonzero(unif_loss <= outage_cap)
    static_safe_act = int(safe_cands[np.argmax(unif_util[safe_cands])]) if len(safe_cands) else 0

    return {
        "outage_cap": float(outage_cap),
        "cap_grid": cap_grid,
        "static_safe_act": static_safe_act,
        "single_util": single_util,
        "single_loss": single_loss,
        "dual_util": dual_util,
        "dual_enh_loss": dual_enh_loss,
        "dual_total_null": dual_total_null,
        "lp_mix_by_p": lp_mix_by_p,
        "lp_mix_grid": lp_mix_grid,
    }


def _interp_lp_mix(
    lp_mix_grid: dict[tuple[float, float], np.ndarray],
    cap_grid: tuple[float, ...],
    p_val: float,
    cap_eff: float,
) -> np.ndarray:
    """Piecewise-linear simplex interpolation of exact `val256` HiGHS LP mixtures at `(p_val, cap_eff)`."""
    c_clamped = float(np.clip(cap_eff, cap_grid[0], cap_grid[-1]))
    if c_clamped <= cap_grid[0]:
        return lp_mix_grid[(p_val, cap_grid[0])]
    if c_clamped >= cap_grid[-1]:
        return lp_mix_grid[(p_val, cap_grid[-1])]
    idx_hi = int(np.searchsorted(cap_grid, c_clamped, side="right"))
    c_lo, c_hi = cap_grid[idx_hi - 1], cap_grid[idx_hi]
    w_hi = (c_clamped - c_lo) / max(c_hi - c_lo, 1e-8)
    return (1.0 - w_hi) * lp_mix_grid[(p_val, c_lo)] + w_hi * lp_mix_grid[(p_val, c_hi)]


def simulate_persist_amc_trajectories(
    eval_bank: dict[str, np.ndarray],
    cal_tables: dict,
    *,
    regime: str,
    seed: int = 1001,
    n_episodes: int = 20,
    episode_len: int = 120,
    jitter_std_db: float = 1.0,
    return_trajectory: bool = False,
) -> dict:
    """Run live slot-by-slot delayed-feedback Markov transmission over `n_episodes x episode_len` frames
    on held-out `eval_bank` (`common500` or `kodak24`) with SHARED channel states, jitter, and CRC uniform draws!
    """
    rng = np.random.default_rng(seed)
    n_imgs = eval_bank["psnr_db"].shape[0]
    outage_cap = float(cal_tables["outage_cap"])
    cap_grid = tuple(cal_tables["cap_grid"])
    lp_mix_grid = cal_tables["lp_mix_grid"]
    single_util = cal_tables["single_util"]
    single_loss = cal_tables["single_loss"]
    dual_util = cal_tables["dual_util"]
    dual_enh_loss = cal_tables["dual_enh_loss"]
    lp_slow = cal_tables["lp_mix_by_p"][0.95]  # `582038a` slow-persistence LP policy (`p_design = 0.95`)
    static_act = int(cal_tables["static_safe_act"])

    policies = (
        "static_safe",
        "stale_greedy",
        "fixed_margin_amc",
        "slow_lp_582038a",
        "persistguard_single",
        "persistguard_duallayer",
    )
    metric_names = ("psnr_db", "ms_ssim", "lpips_alex", "dists", "dinov2_cos")
    acc = {pol: {k: [] for k in (*metric_names, "rejected_null", "prefix_m")} for pol in policies}

    traj_true_p = np.zeros(episode_len, dtype=np.float64)
    traj_est_p = np.zeros(episode_len, dtype=np.float64)
    traj_rej_slow = np.zeros(episode_len, dtype=np.float64)
    traj_rej_pg_s = np.zeros(episode_len, dtype=np.float64)
    traj_rej_pg_d = np.zeros(episode_len, dtype=np.float64)
    traj_dino_slow = np.zeros(episode_len, dtype=np.float64)
    traj_dino_pg_s = np.zeros(episode_len, dtype=np.float64)
    traj_dino_pg_d = np.zeros(episode_len, dtype=np.float64)

    for ep in range(n_episodes):
        # Construct persistence schedule p_t across `episode_len = 120` frames (plus 4 warmup slots)
        total_t = episode_len + 4
        p_sched = np.zeros(total_t, dtype=np.float64)
        if regime == "stationary_slow_p095":
            p_sched[:] = 0.95
        elif regime in ("stationary_fast_p050", "mismatch_slow_to_fast"):
            p_sched[:] = 0.50
        elif regime == "nonstationary_regime_switch":
            # Abrupt persistence switch: Slow (0.95) -> Fast Cliff (0.45) -> Recovery Slow (0.95) -> Fast Cliff (0.50)!
            p_sched[:34] = 0.95
            p_sched[34:64] = 0.45
            p_sched[64:94] = 0.95
            p_sched[94:] = 0.50
        else:
            raise ValueError(f"Unknown regime: {regime}")

        # Generate shared physical state trajectory, small-scale SNR jitter, CRC uniform draws, and public sync coin
        states = np.zeros(total_t, dtype=np.int64)
        states[0] = int(rng.integers(0, 4))
        for t in range(1, total_t):
            if rng.random() < p_sched[t]:
                states[t] = states[t - 1]
            else:
                states[t] = int(rng.integers(0, 4))
        snrs_realized = np.array([STATES_SNR_DB[s] for s in states], dtype=np.float64) + rng.normal(
            0.0, jitter_std_db, size=total_t
        )
        u_crc_main = rng.random(total_t)
        u_crc_base = rng.random(total_t)
        u_pub_mix = rng.random(total_t)
        img_order = rng.permutation(n_imgs)

        # Online state for `persistguard_single` & `persistguard_duallayer`:
        #   - `post_p`: 5-grid Joint CQI + Delayed-NACK Bayesian posterior over `P_GRID = (0.45, 0.60, 0.75, 0.88, 0.95)`
        #   - `cap_aci`: Adaptive Conformal Inference (ACI) risk budget in `[0.45*cap, 1.00*cap]`
        post_p = np.ones(len(P_GRID), dtype=np.float64) / float(len(P_GRID))
        cap_aci = 0.92 * outage_cap
        lam_dual = 0.65
        single_fail_hist = np.zeros(total_t, dtype=np.float64)
        single_act_hist = np.zeros(total_t, dtype=np.int64)
        dual_enh_fail_hist = np.zeros(total_t, dtype=np.float64)

        for t in range(4, total_t):
            slot_k = t - 4
            img_idx = int(img_order[slot_k % n_imgs])
            phase = t % 2
            age = 2 + phase
            rep_idx = t - age
            rep_state = int(states[rep_idx])
            prev_rep_idx = rep_idx - 2

            # 1a. Per-slot BOCPD/Wonham hazard mixing (`h = 0.025` once per slot) + CQI transition update on even slots!
            post_p = 0.975 * post_p + 0.025 * (np.ones_like(post_p) / len(P_GRID))
            if phase == 0 and prev_rep_idx >= 0:
                prev_rep = int(states[prev_rep_idx])
                lik_cqi = np.array(
                    [transition_law_from_report(pg, 2, prev_rep)[rep_state] for pg in P_GRID],
                    dtype=np.float64,
                )
                post_p = post_p * np.maximum(lik_cqi, 1e-6)
                post_p /= post_p.sum()

            # 1b. Causal 1-bit HARQ ACK/NACK likelihood from slot `t - 2`!
            if t >= 6:
                t_past = t - 2
                ph_past = t_past % 2
                age_past = 2 + ph_past
                rep_past = int(states[t_past - age_past])
                act_past = int(single_act_hist[t_past])
                f_past = float(single_fail_hist[t_past])
                lik_ack = np.array(
                    [
                        (
                            float(transition_law_from_report(pg, age_past, rep_past) @ single_loss[:, act_past])
                            if f_past > 0.5
                            else 1.0 - float(transition_law_from_report(pg, age_past, rep_past) @ single_loss[:, act_past])
                        )
                        for pg in P_GRID
                    ],
                    dtype=np.float64,
                )
                post_p = post_p * np.clip(lik_ack, 1e-4, 1.0)
                post_p /= post_p.sum()

                # 2. Adaptive Conformal Inference (ACI) update on LP risk budget `cap_aci`
                cap_target = 0.88 * outage_cap
                cap_aci = float(
                    np.clip(
                        cap_aci + 0.16 * (cap_target - f_past),
                        0.45 * outage_cap,
                        1.00 * outage_cap,
                    )
                )

                err_d = dual_enh_fail_hist[t_past] - 0.035
                lam_dual = float(np.clip(lam_dual + (1.20 if err_d > 0 else 0.12) * err_d, 0.15, 8.0))

            p_est_mean = float(np.dot(post_p, np.asarray(P_GRID, dtype=np.float64)))
            pred_law_bayes = sum(
                float(post_p[i_p]) * transition_law_from_report(pg, age, rep_state)
                for i_p, pg in enumerate(P_GRID)
            )

            # --- Select actions for all 6 policies ---
            # 1. Static Safe
            a_static = static_act

            # 2. Stale Throughput Greedy
            valid_g = np.flatnonzero(single_loss[rep_state] <= outage_cap + 0.01)
            a_greedy = int(valid_g[np.argmax(single_util[rep_state, valid_g])]) if len(valid_g) else 0

            # 3. Classical Fixed-Margin AMC (`3.5 dB` backoff from `STATES_SNR_DB[rep_state]`)
            snr_backoff = STATES_SNR_DB[rep_state] - 3.5
            blers_bo = np.array([finite_blocklength_bler(snr_backoff, m[1], m[3]) for m in MCS_CATALOG])
            valid_bo = np.flatnonzero(blers_bo <= outage_cap)
            a_margin = int(valid_bo[-1]) if len(valid_bo) else 0

            # 4. `Slow_LP_582038a` (`p_design = 0.95` LP mixture via shared public coin `u_pub_mix[t]`)
            ctx_row = phase * 4 + rep_state
            probs_lp = lp_slow[ctx_row]
            a_slow_lp = min(int(np.searchsorted(np.cumsum(probs_lp), u_pub_mix[t], side="right")), len(MCS_CATALOG) - 1)

            # 5. `PersistGuard_SinglePacket` (Bayesian-Posterior Conformal LP Mixture — SAME catalog & public coin!)
            probs_pg = sum(
                float(post_p[i_p]) * _interp_lp_mix(lp_mix_grid, cap_grid, pg, cap_aci)[ctx_row]
                for i_p, pg in enumerate(P_GRID)
            )
            probs_pg = np.clip(probs_pg, 0.0, 1.0)
            probs_pg /= probs_pg.sum()
            a_pg_single = min(int(np.searchsorted(np.cumsum(probs_pg), u_pub_mix[t], side="right")), len(MCS_CATALOG) - 1)
            single_act_hist[t] = a_pg_single

            # 6. `PersistGuard_DualLayer_Anchor` (Base m=4 QPSK r=1/4 guarantees <0.1% NULLs; Enhancement maximizes Bayesian expected utility!)
            exp_du = pred_law_bayes @ dual_util
            a_pg_dual = int(np.argmax(exp_du))

            # --- Realize physical over-the-air CRC pass/fail at actual `snrs_realized[t]`! ---
            snr_now = float(snrs_realized[t])
            single_choices = {
                "static_safe": a_static,
                "stale_greedy": a_greedy,
                "fixed_margin_amc": a_margin,
                "slow_lp_582038a": a_slow_lp,
                "persistguard_single": a_pg_single,
            }
            for pol_name, act_id in single_choices.items():
                _, mod_m, _, rho_bps, m_sc = MCS_CATALOG[act_id]
                bler_now = float(finite_blocklength_bler(snr_now, mod_m, rho_bps, n_syms=N_PAYLOAD_SYMS))
                failed = u_crc_main[t] < bler_now
                m_deliv = 0 if failed else m_sc
                if pol_name == "persistguard_single":
                    single_fail_hist[t] = 1.0 if failed else 0.0
                for k in metric_names:
                    acc[pol_name][k].append(float(eval_bank[k][img_idx, m_deliv]))
                acc[pol_name]["rejected_null"].append(1.0 if failed else 0.0)
                acc[pol_name]["prefix_m"].append(float(m_deliv))

            # Realize Dual-Layer frame (`384` Base syms + `576` Enhancement syms = `960` payload syms, total `1,024`!)
            bler_b_now = float(finite_blocklength_bler(snr_now, 4, 0.50, n_syms=N_DUAL_BASE_SYMS))
            base_failed = u_crc_base[t] < bler_b_now
            _, mod_e, _, rho_e, m_enh = DUAL_ENH_CATALOG[a_pg_dual]
            bler_e_now = float(finite_blocklength_bler(snr_now, mod_e, rho_e, n_syms=N_DUAL_ENH_SYMS))
            enh_failed = u_crc_main[t] < bler_e_now
            dual_enh_fail_hist[t] = 1.0 if (base_failed or (a_pg_dual > 0 and enh_failed)) else 0.0

            if base_failed:
                m_dual = 0
                is_null = 1.0
            elif enh_failed:
                m_dual = 4
                is_null = 0.0
            else:
                m_dual = m_enh
                is_null = 0.0

            for k in metric_names:
                acc["persistguard_duallayer"][k].append(float(eval_bank[k][img_idx, m_dual]))
            acc["persistguard_duallayer"]["rejected_null"].append(is_null)
            acc["persistguard_duallayer"]["prefix_m"].append(float(m_dual))

            traj_true_p[slot_k] += p_sched[t] / float(n_episodes)
            traj_est_p[slot_k] += p_est_mean / float(n_episodes)
            traj_rej_slow[slot_k] += acc["slow_lp_582038a"]["rejected_null"][-1] / float(n_episodes)
            traj_rej_pg_s[slot_k] += acc["persistguard_single"]["rejected_null"][-1] / float(n_episodes)
            traj_rej_pg_d[slot_k] += is_null / float(n_episodes)
            traj_dino_slow[slot_k] += acc["slow_lp_582038a"]["dinov2_cos"][-1] / float(n_episodes)
            traj_dino_pg_s[slot_k] += acc["persistguard_single"]["dinov2_cos"][-1] / float(n_episodes)
            traj_dino_pg_d[slot_k] += acc["persistguard_duallayer"]["dinov2_cos"][-1] / float(n_episodes)

    summary = {
        pol: {k: float(np.mean(vals)) for k, vals in m_dict.items()}
        for pol, m_dict in acc.items()
    }
    if return_trajectory:
        return {
            "summary": summary,
            "trajectory": {
                "slot": list(range(1, episode_len + 1)),
                "true_p": traj_true_p.tolist(),
                "est_p_bayes": traj_est_p.tolist(),
                "slow_lp_rej": traj_rej_slow.tolist(),
                "pg_single_rej": traj_rej_pg_s.tolist(),
                "pg_dual_rej": traj_rej_pg_d.tolist(),
                "slow_lp_dino": traj_dino_slow.tolist(),
                "pg_single_dino": traj_dino_pg_s.tolist(),
                "pg_dual_dino": traj_dino_pg_d.tolist(),
            },
        }
    return summary
