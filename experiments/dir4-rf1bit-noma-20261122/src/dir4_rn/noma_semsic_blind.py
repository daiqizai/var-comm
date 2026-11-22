"""Dir-4B: SemSIC-NOMA-JSCC — Grant-Free Zero-Pilot Non-Orthogonal Multi-User Uplink
& Overloaded (3-User -> 2-BS-Antenna) Semantic Multiple Access via Collision-Free
Walsh Signatures and Two-Stage Physical-Semantic Iterative Interference Cancellation (Sem-SIC).
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
)
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb
from dir3_mh.mimo_blind import build_walsh_k_basis
from dir4_rn.rf_1bit_blind import newman_rb_phases


def vec_to_real_ring(g_r: torch.Tensor, g_i: torch.Tensor) -> torch.Tensor:
    """Map complex vector [..., Nr] -> Real Ring Isomorphism [..., 2*Nr, 2]:
       [[g_r, -g_i],
        [g_i,  g_r]]
    """
    col0 = torch.cat([g_r, g_i], dim=-1)
    col1 = torch.cat([-g_i, g_r], dim=-1)
    return torch.stack([col0, col1], dim=-1)


def sample_multiuser_uplink_channels(
    batch_size: int,
    n_users: int,
    n_rx: int,
    num_rbs: int = 32,
    rho_f: float = 0.85,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent frequency-selective Rayleigh channels [B, U, Nr, M] for U autonomous edge users."""
    total = batch_size * n_users * n_rx
    hr, hi = sample_correlated_rbs(total, num_rbs, rho_f, device=device, dtype=dtype, generator=generator)
    return hr.reshape(batch_size, n_users, n_rx, num_rbs), hi.reshape(batch_size, n_users, n_rx, num_rbs)


def physical_grantfree_uplink_mac(
    s_users_iq: torch.Tensor,
    user_snrs_db: torch.Tensor,
    *,
    n_rx: int = 2,
    rho_f: float = 0.85,
    h_mu_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_iq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Physical Grant-Free Non-Orthogonal Uplink Multiple-Access Channel (MAC):
       s_users_iq: [B, U, M=32, N_b=384, 2] (each user strictly normalized to E[|s_u|^2] = 1.0)
       user_snrs_db: [B, U] per-user average received SNR (dB) per BS antenna (captures Near-Far path loss!)
       Returns y_rx_iq: [B, Nr, M=32, N_b=384, 2] with unit reference AWGN variance sigma_n^2 = 1.0.
    """
    B, U, M, N_b, _ = s_users_iq.shape
    device, dtype = s_users_iq.device, s_users_iq.dtype

    # Strict per-user unit transmit power enforcement
    u_pwr = (s_users_iq[..., 0].square() + s_users_iq[..., 1].square()).mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    s_norm = s_users_iq / u_pwr.unsqueeze(-1).sqrt()

    if h_mu_override is not None:
        hr, hi = h_mu_override[0].to(device=device, dtype=dtype), h_mu_override[1].to(device=device, dtype=dtype)
    else:
        hr, hi = sample_multiuser_uplink_channels(B, U, n_rx, M, rho_f, device=device, dtype=dtype)

    snr_u = user_snrs_db.to(device=device, dtype=dtype).reshape(B, U)
    amp_u = torch.pow(10.0, snr_u / 20.0)[:, :, None, None]  # [B, U, 1, 1] -> sqrt(beta_u)
    gr = hr * amp_u  # [B, U, Nr, M] effective complex channel Re
    gi = hi * amp_u  # [B, U, Nr, M] effective complex channel Im

    sr = s_norm[..., 0]  # [B, U, M, N_b]
    si = s_norm[..., 1]  # [B, U, M, N_b]

    # Superpose all U users over the air simultaneously onto Nr BS antennas!
    yr_clean = torch.einsum("burm,bumn->brmn", gr, sr) - torch.einsum("burm,bumn->brmn", gi, si)
    yi_clean = torch.einsum("burm,bumn->brmn", gr, si) + torch.einsum("burm,bumn->brmn", gi, sr)

    if standard_noise_iq is not None:
        eps_r = standard_noise_iq[..., 0].to(device=device, dtype=dtype)
        eps_i = standard_noise_iq[..., 1].to(device=device, dtype=dtype)
    else:
        eps_r = torch.randn((B, n_rx, M, N_b), device=device, dtype=dtype)
        eps_i = torch.randn((B, n_rx, M, N_b), device=device, dtype=dtype)

    inv_s2 = 1.0 / math.sqrt(2.0)
    yr = yr_clean + inv_s2 * eps_r
    yi = yi_clean + inv_s2 * eps_i

    return torch.stack([yr, yi], dim=-1), {"g_true_r": gr, "g_true_i": gi}


class SemanticLatentPriorNet(nn.Module):
    """Lightweight 2D Spatial-Channel Semantic Latent Denoiser on [B, C=96, 16, 16] inside the Sem-SIC Loop!
    Exploits natural image 2D spatial structure + hierarchical channel energy + post-LMMSE SINR maps
    to suppress multi-user collision interference BEFORE physical I/Q re-modulation and subtraction.
    """

    def __init__(self, latent_ch: int = 96, hidden_ch: int = 160):
        super().__init__()
        self.latent_ch = int(latent_ch)
        # Input channels: u_restored (96) + [alpha_map_r, alpha_map_i, log_sigma_r, log_sigma_i] -> 96 + 4 = 100
        self.in_proj = nn.Conv2d(latent_ch + 4, hidden_ch, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, hidden_ch),
                nn.GELU(),
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=8),
                nn.GELU(),
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            )
            for _ in range(3)
        ])
        # Global 16x16 spatial self-attention block for non-local semantic consistency
        self.attn_norm = nn.LayerNorm(hidden_ch)
        self.attn = nn.MultiheadAttention(hidden_ch, num_heads=4, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, latent_ch, kernel_size=1),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(
        self,
        u_rest: torch.Tensor,
        alpha_rb: torch.Tensor,
        sigma_post_rb: torch.Tensor,
        cr_db: torch.Tensor,
    ) -> torch.Tensor:
        """u_rest: [B, L=256, C=96], alpha_rb: [B, M=32], sigma_post_rb: [B, M=32], cr_db: [B]."""
        B, L, C = u_rest.shape
        K = (L * C) // 2
        H_s = W_s = 16

        # Map RB-level physical reliability to exact [B, 1, 16, 16] spatial token reliability maps
        alp_iq = expand_rb_to_symbols(alpha_rb, K)  # [B, K]
        sig_iq = expand_rb_to_symbols(torch.log(sigma_post_rb.clamp_min(1e-4)).div(4.0), K)
        alp_tok = torch.cat([alp_iq, alp_iq], dim=1).reshape(B, L, C).mean(dim=-1).reshape(B, 1, H_s, W_s)
        sig_tok = torch.cat([sig_iq, sig_iq], dim=1).reshape(B, L, C).mean(dim=-1).reshape(B, 1, H_s, W_s)
        cr_map = ((cr_db - SNR_CENTER) / SNR_SCALE).reshape(B, 1, 1, 1).expand(B, 1, H_s, W_s)
        e_tok = u_rest.square().mean(dim=-1).sqrt().div(3.0).reshape(B, 1, H_s, W_s)

        z_2d = u_rest.transpose(1, 2).reshape(B, C, H_s, W_s)
        feat = self.in_proj(torch.cat([z_2d, alp_tok, sig_tok, cr_map, e_tok], dim=1))
        for blk in self.blocks:
            feat = feat + blk(feat)

        tok = feat.flatten(2).transpose(1, 2)  # [B, 256, hidden_ch]
        tok_n = self.attn_norm(tok)
        attn_out, _ = self.attn(tok_n, tok_n, tok_n, need_weights=False)
        feat = feat + attn_out.transpose(1, 2).reshape(B, -1, H_s, W_s)

        delta = self.out_proj(feat).flatten(2).transpose(1, 2)  # [B, 256, 96]
        return u_rest + delta


class SemSICGrantFreeModem(nn.Module):
    """Collision-Free 8-Walsh Subspace Multi-User Modem + 2-Round Soft Covariance-Updated Semantic SIC!
       Supports up to U_max = 3 simultaneous grant-free users:
         - u_0 & u_7: Universal Quiet Nulls across ALL users (exact blind thermal noise variance!)
         - u_{1+u} (u=0,1,2): User-u Orthogonal Channel Probe (ZERO inter-user pilot contamination even at 12 dB Near-Far!)
         - u_{4+u} (u=0,1,2): User-u Orthogonal atan2 Latent Scale Signature
         - span(u_0..u_7)^perp: 376/384 = 97.92% Shared Non-Orthogonal Semantic Payload Subspace!
    """

    def __init__(self, num_rbs: int = 32, hidden_ch: int = 160, enable_semsic: bool = True):
        super().__init__()
        self.num_rbs = int(num_rbs)
        self.enable_semsic = bool(enable_semsic)

        init_eta_ch = 0.055
        inv_s = math.log((init_eta_ch - 0.025) / (0.12 - init_eta_ch))
        self.eta_ch_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta_sc = 0.018

        self.rb_refiner = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        nn.init.zeros_(self.rb_refiner[-1].weight)
        nn.init.zeros_(self.rb_refiner[-1].bias)

        if self.enable_semsic:
            self.latent_prior = SemanticLatentPriorNet(latent_ch=96, hidden_ch=hidden_ch)
            # Soft SIC cancellation reliability MLP per (User, RB): maps [alpha_{u,m}, bar_alpha_u, round_id] -> omega_{u,m} in [0, 0.98]
            self.sic_gate_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.sic_gate_mlp[-1].weight)
            # Start conservative at Step 0 (sigmoid(-2.2) ~ 0.10) so Step-0 never over-subtracts noisy estimates!
            nn.init.constant_(self.sic_gate_mlp[-1].bias, -2.2)

    @property
    def eta_ch(self) -> torch.Tensor:
        return 0.025 + 0.095 * torch.sigmoid(self.eta_ch_raw)

    def project_data_perp(self, z_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normalized latent [B, L=256, C=96] onto span(u_0..u_7)^perp with exact data_pwr."""
        B, L, C = z_norm.shape
        M = self.num_rbs
        N_b = 384
        K = M * N_b
        device, dtype = z_norm.device, z_norm.dtype

        basis = build_walsh_k_basis(8, N_b, device=device, dtype=dtype)  # [8, 384]
        flat = z_norm.reshape(B, 2 * K)
        dr = flat[:, :K].reshape(B, M, N_b)
        di = flat[:, K:].reshape(B, M, N_b)

        pr = torch.einsum("bmn,kn->bmk", dr, basis)
        pi = torch.einsum("bmn,kn->bmk", di, basis)
        dpr = dr - torch.einsum("bmk,kn->bmn", pr, basis)
        dpi = di - torch.einsum("bmk,kn->bmn", pi, basis)

        data_pwr = 1.0 - self.eta_ch - self.eta_sc
        rms = (dpr.square() + dpi.square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()
        scale = data_pwr.sqrt() / rms
        return dpr * scale, dpi * scale

    def embed_user_tx(self, z_tx: torch.Tensor, user_slot: int) -> torch.Tensor:
        """Encode single autonomous edge device u in {0, 1, 2} -> [B, M=32, N_b=384, 2]."""
        assert 0 <= user_slot < 3
        B, L, C = z_tx.shape
        M = self.num_rbs
        N_b = 384
        device, dtype = z_tx.device, z_tx.dtype

        p_img = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
        z_norm = z_tx / p_img[:, None, None]
        sr, si = self.project_data_perp(z_norm)

        basis = build_walsh_k_basis(8, N_b, device=device, dtype=dtype)
        u_ch = basis[1 + user_slot]  # u_1, u_2, or u_3
        u_sc = basis[4 + user_slot]  # u_4, u_5, or u_6
        n_cos, n_sin = newman_rb_phases(M, device=device, dtype=dtype)

        amp_ch = (self.eta_ch * float(N_b)).sqrt()
        pch_r = amp_ch * n_cos[None, :, None] * u_ch[None, None, :]
        pch_i = amp_ch * n_sin[None, :, None] * u_ch[None, None, :]

        theta = 1.25 * torch.tanh(0.40 * (p_img - 2.82))
        amp_sc = math.sqrt(self.eta_sc * float(N_b))
        c_th = torch.cos(theta)[:, None] * n_cos[None, :] - torch.sin(theta)[:, None] * n_sin[None, :]
        s_th = torch.sin(theta)[:, None] * n_cos[None, :] + torch.cos(theta)[:, None] * n_sin[None, :]
        psc_r = amp_sc * c_th[:, :, None] * u_sc[None, None, :]
        psc_i = amp_sc * s_th[:, :, None] * u_sc[None, None, :]

        return torch.stack([sr + pch_r + psc_r, si + pch_i + psc_i], dim=-1)

    def _demix_single_stage(
        self,
        y_perp_r: torch.Tensor,
        y_perp_i: torch.Tensor,
        ghat_r: torch.Tensor,
        ghat_i: torch.Tensor,
        p_hat_u: torch.Tensor,
        sigma_n_hat: torch.Tensor,
        sigma_e_u: torch.Tensor,
        nu_res_um: torch.Tensor,
        latent_ch: int,
    ) -> list[tuple[ChannelV2Output, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Exact Per-User Covariance-Aware Real-Ring-Isomorphic Blind LMMSE Demixing!
           y_perp_r/i: [B, U_target, Nr, M, N_b] (for Stage 1 all U share y_perp; for Stage 2 each user u has its SIC-cleaned y_perp^(u)!)
           ghat_r/i: [B, U, Nr, M]
           nu_res_um: [B, U, M] uncancelled residual power fraction in [0, 1] for each interferer u' on RB m.
        """
        B, U, Nr, M = ghat_r.shape
        N_b = y_perp_r.shape[-1]
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = ghat_r.device, ghat_r.dtype

        data_pwr = 1.0 - self.eta_ch - self.eta_sc
        dof_comp = math.sqrt(float(N_b - 8) / float(N_b))
        data_scale = (1.0 / data_pwr).sqrt() * dof_comp

        # Precompute Real-Ring matrices Phi(g_{u,m}): [B, U, M, 2*Nr, 2]
        gr_bm = ghat_r.permute(0, 1, 3, 2)  # [B, U, M, Nr]
        gi_bm = ghat_i.permute(0, 1, 3, 2)
        phi_all = vec_to_real_ring(gr_bm, gi_bm)  # [B, U, M, 2*Nr, 2]
        # Per-user rank-2 outer product G_outer[b, u, m] = data_pwr * Phi @ Phi^T: [B, U, M, 2*Nr, 2*Nr]
        g_outer = data_pwr * torch.matmul(phi_all, phi_all.transpose(-1, -2))

        eye_2nr = torch.eye(2 * Nr, device=device, dtype=dtype)[None, None, :, :]
        outputs: list[tuple[ChannelV2Output, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for u in range(U):
            # Total covariance for User u: own full signal + residual uncancelled interferers + thermal/channel-error noise
            cov_u = g_outer[:, u].clone()  # [B, M, 2*Nr, 2*Nr]
            for up in range(U):
                if up == u:
                    continue
                cov_u = cov_u + nu_res_um[:, up, :, None, None] * g_outer[:, up]

            # Include channel estimation error variance scaled by sum of residual powers
            tot_sig_var = sigma_n_hat[:, None] + sigma_e_u[:, u, None] * float(Nr)
            cov_u = cov_u + tot_sig_var[:, :, None, None] * eye_2nr

            # Solve Cov_u @ X = data_pwr * Phi_u  -> X = Cov_u^{-1} (data_pwr * Phi_u): [B, M, 2*Nr, 2]
            rhs_u = data_pwr * phi_all[:, u]
            sol_u = torch.linalg.solve(cov_u, rhs_u)  # [B, M, 2*Nr, 2]
            w_u = sol_u.transpose(-1, -2)  # [B, M, 2, 2*Nr]

            # Effective scalar capture alpha_{u,m} = (W_u @ Phi_u)[0, 0] in (0, 1)
            f_u = torch.matmul(w_u, phi_all[:, u])  # [B, M, 2, 2]
            alpha_um = 0.5 * (f_u[:, :, 0, 0] + f_u[:, :, 1, 1]).clamp(1e-5, 0.9999)  # [B, M]
            alpha_bar_u = alpha_um.mean(dim=1, keepdim=True).clamp_min(0.03)  # [B, 1]

            # Apply W_u to User u's received (or SIC-cleaned) stacked vector [Re(Y), Im(Y)]: [B, M, 2*Nr, N_b]
            yu_stack = torch.cat([y_perp_r[:, u], y_perp_i[:, u]], dim=1).permute(0, 2, 1, 3)  # [B, M, 2*Nr, N_b]
            s_est = torch.matmul(w_u, yu_stack)  # [B, M, 2, N_b]

            gain_ub = (data_scale / alpha_bar_u)[:, :, None]  # [B, 1, 1]
            sh_r = s_est[:, :, 0, :] * gain_ub
            sh_i = s_est[:, :, 1, :] * gain_ub

            u_norm = torch.cat([sh_r.reshape(B, K), sh_i.reshape(B, K)], dim=1).reshape(B, L, latent_ch)
            u_rest = u_norm * p_hat_u[:, u, None, None]

            sinr_wb = alpha_bar_u.squeeze(1) / (1.0 - alpha_bar_u.squeeze(1)).clamp_min(1e-4)
            cr_db = (10.0 * torch.log10(sinr_wb)).clamp(-6.0, 22.0)
            sigma_post_rb = 0.5 * ((1.0 - alpha_um) / alpha_bar_u).clamp_min(1e-6)
            g_abs_um = (gr_bm[:, u].square() + gi_bm[:, u].square()).sum(dim=-1).clamp_min(1e-8).sqrt()

            ch_out = ChannelV2Output(
                u_restored=u_rest,
                u_mmse_restored=u_rest,
                u_norm=u_norm,
                z_norm=u_norm,
                pwr=p_hat_u[:, u].square(),
                gamma_db=cr_db,
                c_r_db=cr_db,
                h_abs_iq=expand_rb_to_symbols(g_abs_um, K),
                alpha_ratio_iq=expand_rb_to_symbols(alpha_um / alpha_bar_u, K),
                sigma_post_iq=expand_rb_to_symbols(sigma_post_rb, K),
                phase_unc_iq=expand_rb_to_symbols((sigma_e_u[:, u, None] / (g_abs_um.square() + 1e-5)).sqrt(), K),
                psi_csi=(alpha_um - alpha_bar_u).square().mean(dim=1) + sigma_e_u[:, u],
            )
            outputs.append((ch_out, cr_db, alpha_um, sigma_post_rb))

        return outputs

    def blind_demix_users(
        self,
        y_rx_iq: torch.Tensor,
        n_users: int,
        latent_ch: int,
    ) -> tuple[list[tuple[ChannelV2Output, torch.Tensor]], list[torch.Tensor]]:
        """Demix U simultaneous autonomous users from y_rx_iq: [B, Nr, M=32, N_b=384, 2].
           Returns (per_user_channel_outputs, denoised_latents_for_aux_loss).
        """
        B, Nr, M, N_b, _ = y_rx_iq.shape
        U = int(n_users)
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        yr = y_rx_iq[..., 0]  # [B, Nr, M, N_b]
        yi = y_rx_iq[..., 1]
        basis = build_walsh_k_basis(8, N_b, device=device, dtype=dtype)  # [8, 384]
        n_cos, n_sin = newman_rb_phases(M, device=device, dtype=dtype)

        # 1. Universal Quiet Nulls (u_0 and u_7) -> Exact Blind Noise Variance (immune to U colliding users!)
        q0_r = torch.einsum("brmn,n->brm", yr, basis[0])
        q0_i = torch.einsum("brmn,n->brm", yi, basis[0])
        q7_r = torch.einsum("brmn,n->brm", yr, basis[7])
        q7_i = torch.einsum("brmn,n->brm", yi, basis[7])
        sigma_n_hat = 0.5 * (q0_r.square() + q0_i.square() + q7_r.square() + q7_i.square()).mean(dim=(1, 2)).clamp(1e-4, 15.0)

        # 2. Collision-Free Per-User Blind Channel Estimation from u_{1+u} + Coherence-Adaptive Wiener Smoothing!
        eta_ch = self.eta_ch
        amp_ch = (eta_ch * float(N_b)).sqrt()
        sigma_e_raw = sigma_n_hat / (eta_ch * float(N_b))  # [B]

        ghat_r_list, ghat_i_list, sige_u_list = [], [], []
        e_rb_ant = (yr.square() + yi.square()).mean(dim=-1)  # [B, Nr, M]

        for u in range(U):
            raw_r = torch.einsum("brmn,n->brm", yr, basis[1 + u]) / amp_ch
            raw_i = torch.einsum("brmn,n->brm", yi, basis[1 + u]) / amp_ch
            # Undo Newman phase
            gt_r = raw_r * n_cos[None, None, :] + raw_i * n_sin[None, None, :]
            gt_i = raw_i * n_cos[None, None, :] - raw_r * n_sin[None, None, :]

            flat_r = gt_r.reshape(B * Nr, M)
            flat_i = gt_i.reshape(B * Nr, M)
            sige_rep = sigma_e_raw[:, None].expand(B, Nr).reshape(B * Nr)
            gw_r, gw_i, sige_eff_flat, df_flat = coherence_adaptive_wiener_rb(flat_r, flat_i, sige_rep)

            log_sig_f = torch.log(sigma_n_hat.clamp_min(1e-4))[:, None, None].expand(B, Nr, M).reshape(B * Nr, M).div(4.0)
            df_f = df_flat[:, None].expand(B * Nr, M)
            ref_in = torch.stack([flat_r, flat_i, e_rb_ant.reshape(B * Nr, M).div(10.0), log_sig_f, gw_r, gw_i, df_f], dim=1)
            dg = self.rb_refiner(ref_in)
            ghat_r_list.append((gw_r + dg[:, 0, :]).reshape(B, Nr, M))
            ghat_i_list.append((gw_i + dg[:, 1, :]).reshape(B, Nr, M))
            sige_u_list.append(sige_eff_flat.reshape(B, Nr).mean(dim=1))

        ghat_r = torch.stack(ghat_r_list, dim=1)  # [B, U, Nr, M]
        ghat_i = torch.stack(ghat_i_list, dim=1)
        sigma_e_u = torch.stack(sige_u_list, dim=1)  # [B, U]

        # 3. Collision-Free Per-User atan2 Latent Scale Recovery via Nr x M MRC on u_{4+u}
        p_hat_list = []
        for u in range(U):
            qsc_r = torch.einsum("brmn,n->brm", yr, basis[4 + u])
            qsc_i = torch.einsum("brmn,n->brm", yi, basis[4 + u])
            usc_r = qsc_r * n_cos[None, None, :] + qsc_i * n_sin[None, None, :]
            usc_i = qsc_i * n_cos[None, None, :] - qsc_r * n_sin[None, None, :]
            z_r = (ghat_r[:, u] * usc_r + ghat_i[:, u] * usc_i).sum(dim=(1, 2))
            z_i = (ghat_r[:, u] * usc_i - ghat_i[:, u] * usc_r).sum(dim=(1, 2))
            th_u = torch.atan2(z_i, z_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
            p_u = (2.82 + torch.atanh(th_u / 1.25) / 0.40).clamp(0.4, 7.0)
            p_hat_list.append(p_u)
        p_hat_u = torch.stack(p_hat_list, dim=1)  # [B, U]

        # 4. Exact Orthogonal Projection onto Shared Semantic Payload Subspace span(u_0..u_7)^perp
        proj_yr = torch.einsum("brmn,kn->brmk", yr, basis)
        proj_yi = torch.einsum("brmn,kn->brmk", yi, basis)
        y_perp0_r = yr - torch.einsum("brmk,kn->brmn", proj_yr, basis)  # [B, Nr, M, N_b]
        y_perp0_i = yi - torch.einsum("brmk,kn->brmn", proj_yi, basis)

        # Stage 1: Joint Multi-User Blind LMMSE (nu_res = 1.0 for all interferers)
        y_s1_r = y_perp0_r[:, None].expand(B, U, Nr, M, N_b)
        y_s1_i = y_perp0_i[:, None].expand(B, U, Nr, M, N_b)
        nu_ones = torch.ones((B, U, M), device=device, dtype=dtype)
        stage_outs = self._demix_single_stage(
            y_s1_r, y_s1_i, ghat_r, ghat_i, p_hat_u, sigma_n_hat, sigma_e_u, nu_ones, latent_ch
        )

        if not self.enable_semsic:
            return [(o[0], o[1]) for o in stage_outs], []

        # Stage 2: 2-Round Nonlinear Physical-Semantic Iterative Interference Cancellation (Sem-SIC)!
        denoised_latents: list[torch.Tensor] = []
        for sic_round in range(2):
            s_remod_r_list, s_remod_i_list, omega_list, z_clean_round = [], [], [], []
            round_feat = float(sic_round)

            for u in range(U):
                ch_u, cr_u, alp_um, sigp_um = stage_outs[u]
                z_clean_u = self.latent_prior(ch_u.u_restored, alp_um, sigp_um, cr_u)  # [B, 256, 96]
                z_clean_round.append(z_clean_u)

                # Re-normalize cleaned semantic latent and project onto span(u_0..u_7)^perp
                p_clean_u = (2.0 * z_clean_u.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
                sr_u, si_u = self.project_data_perp(z_clean_u / p_clean_u[:, None, None])  # [B, M, N_b]
                s_remod_r_list.append(sr_u)
                s_remod_i_list.append(si_u)

                # Compute per-RB soft cancellation weight omega_{u,m} in [0, 0.96]
                alp_bar_u = alp_um.mean(dim=1, keepdim=True).expand(B, M)
                r_col = torch.full_like(alp_um, round_feat)
                u_norm_diff = (z_clean_u - ch_u.u_restored).detach().square().mean(dim=(1, 2), keepdim=True).squeeze(-1).expand(B, M).clamp(0.0, 5.0)
                gate_in = torch.stack([alp_um, alp_bar_u, r_col, u_norm_diff], dim=-1)  # [B, M, 4]
                # Product with sqrt(alp_um) guarantees weak noisy users are never over-cancelled!
                omega_um = 0.96 * torch.sigmoid(self.sic_gate_mlp(gate_in).squeeze(-1)) * alp_um.sqrt()
                omega_list.append(omega_um)

            denoised_latents = z_clean_round
            s_remod_r = torch.stack(s_remod_r_list, dim=1)  # [B, U, M, N_b]
            s_remod_i = torch.stack(s_remod_i_list, dim=1)
            omega_all = torch.stack(omega_list, dim=1)  # [B, U, M]

            # Subtract interfering users' reconstructed I/Q waveforms for each target user u!
            # Contribution of user up on BS antennas: ghat_{up} * (omega_{up} * s_remod_{up})
            ws_r = omega_all.unsqueeze(-1) * s_remod_r  # [B, U, M, N_b]
            ws_i = omega_all.unsqueeze(-1) * s_remod_i
            contrib_r = ghat_r.unsqueeze(-1) * ws_r[:, :, None, :, :] - ghat_i.unsqueeze(-1) * ws_i[:, :, None, :, :]  # [B, U, Nr, M, N_b]
            contrib_i = ghat_r.unsqueeze(-1) * ws_i[:, :, None, :, :] + ghat_i.unsqueeze(-1) * ws_r[:, :, None, :, :]

            tot_contrib_r = contrib_r.sum(dim=1, keepdim=True)  # [B, 1, Nr, M, N_b]
            tot_contrib_i = contrib_i.sum(dim=1, keepdim=True)
            # For user u, subtract all up != u -> total - contrib_u!
            y_sic_r = y_perp0_r[:, None] - (tot_contrib_r - contrib_r)
            y_sic_i = y_perp0_i[:, None] - (tot_contrib_i - contrib_i)

            # Residual uncancelled variance fraction nu_{u, m} = (1 - omega_{u,m})^2 (clamped >= 0.03 for stability)
            nu_res = (1.0 - omega_all).square().clamp(0.03, 1.0)
            stage_outs = self._demix_single_stage(
                y_sic_r, y_sic_i, ghat_r, ghat_i, p_hat_u, sigma_n_hat, sigma_e_u, nu_res, latent_ch
            )

        return [(o[0], o[1]) for o in stage_outs], denoised_latents


class BlackBoxMultiUserModem(nn.Module):
    """Pure Neural Multi-User Blind Demixing Baseline (0 Pilots, without Walsh Subspace Signatures or Sem-SIC)."""

    def __init__(self, max_users: int = 3, n_rx: int = 2, num_rbs: int = 32):
        super().__init__()
        self.max_users = int(max_users)
        self.n_rx = int(n_rx)
        self.num_rbs = int(num_rbs)

        # User-specific learned linear spreading phase/gain at TX
        self.tx_phase = nn.Parameter(torch.linspace(0.0, 2.0, max_users))
        in_ch = 3 * n_rx + 1  # [mean_r, mean_i, energy] per antenna + user_id
        self.demix_net = nn.Sequential(
            nn.Conv1d(in_ch, 96, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv1d(96, 4 * n_rx + 2, kernel_size=1),
        )
        nn.init.zeros_(self.demix_net[-1].weight)
        nn.init.zeros_(self.demix_net[-1].bias)

    def embed_user_tx(self, z_tx: torch.Tensor, user_slot: int) -> torch.Tensor:
        B, L, C = z_tx.shape
        K = (L * C) // 2
        pwr = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8)
        flat = (z_tx / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
        sr = flat[:, :K].reshape(B, 32, 384)
        si = flat[:, K:].reshape(B, 32, 384)
        ph = self.tx_phase[user_slot]
        c, s = torch.cos(ph), torch.sin(ph)
        return torch.stack([c * sr - s * si, s * sr + c * si], dim=-1)

    def blind_demix_users(
        self,
        y_rx_iq: torch.Tensor,
        n_users: int,
        latent_ch: int,
    ) -> tuple[list[tuple[ChannelV2Output, torch.Tensor]], list[torch.Tensor]]:
        B, Nr, M, N_b, _ = y_rx_iq.shape
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        yr = y_rx_iq[..., 0]
        yi = y_rx_iq[..., 1]
        mr = yr.mean(dim=-1)
        mi = yi.mean(dim=-1)
        er = (yr.square() + yi.square()).mean(dim=-1).div(10.0).clamp(0, 5.0)
        base_f = torch.cat([mr, mi, er], dim=1)  # [B, 3*Nr, M]
        y_stack = torch.cat([yr, yi], dim=1).permute(0, 2, 1, 3)  # [B, M, 2*Nr, N_b]

        outs: list[tuple[ChannelV2Output, torch.Tensor]] = []
        ones_iq = torch.ones((B, K), device=device, dtype=dtype)
        for u in range(n_users):
            u_col = torch.full((B, 1, M), float(u) / 2.0, device=device, dtype=dtype)
            raw = self.demix_net(torch.cat([base_f, u_col], dim=1))  # [B, 4*Nr+2, M]
            w_u = raw[:, : 4 * Nr, :].permute(0, 2, 1).reshape(B, M, 2, 2 * Nr)
            # Identity fallback on antenna u % Nr
            ant_idx = u % Nr
            w_u = w_u.clone()
            w_u[:, :, 0, ant_idx] = w_u[:, :, 0, ant_idx] + 0.35
            w_u[:, :, 1, Nr + ant_idx] = w_u[:, :, 1, Nr + ant_idx] + 0.35

            s_est = torch.matmul(w_u, y_stack)  # [B, M, 2, N_b]
            cr_db = (5.0 + 5.0 * raw[:, 4 * Nr, :].mean(dim=-1)).clamp(-5.0, 20.0)
            p_hat = (2.82 * torch.exp(raw[:, 4 * Nr + 1, :].mean(dim=-1).clamp(-1.0, 1.0))).clamp(0.5, 6.0)

            u_norm = torch.cat([s_est[:, :, 0, :].reshape(B, K), s_est[:, :, 1, :].reshape(B, K)], dim=1).reshape(B, L, latent_ch)
            u_rest = u_norm * p_hat[:, None, None]
            outs.append((
                ChannelV2Output(
                    u_restored=u_rest,
                    u_mmse_restored=u_rest,
                    u_norm=u_norm,
                    z_norm=u_norm,
                    pwr=p_hat.square(),
                    gamma_db=cr_db,
                    c_r_db=cr_db,
                    h_abs_iq=ones_iq,
                    alpha_ratio_iq=ones_iq,
                    sigma_post_iq=0.35 * ones_iq,
                    phase_unc_iq=0.2 * ones_iq,
                    psi_csi=torch.full((B,), 0.25, device=device, dtype=dtype),
                ),
                cr_db,
            ))
        return outs, []


class GrantFreeNOMATransceiver(nn.Module):
    """Unified Dir-4B Multi-User Grant-Free Uplink Transceiver:
      - semsic_noma: Proposed Collision-Free Walsh Signatures + 2-Round Physical-Semantic Iterative IC (Sem-SIC)
      - semsic_stage1_only: Controlled Ablation WITHOUT Stage-2 Semantic SIC (Collision-Free Blind Linear LMMSE only)
      - blackbox_noma: End-to-End Neural Multi-User Blind Demixer (0-Pilot)
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "semsic_noma",
        n_rx: int = 2,
        num_rbs: int = 32,
        base_c0_db: float = 7.0,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.method = method.lower()
        self.n_rx = int(n_rx)
        self.num_rbs = int(num_rbs)
        self.base_c0_db = float(base_c0_db)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx = OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
        self.blind_dec = BlindMultiScaleDecoderCalibrator(self.latent_channels, dec_dims, hidden_ch=hidden_ch)

        if self.method == "semsic_noma":
            self.modem: nn.Module = SemSICGrantFreeModem(num_rbs=num_rbs, hidden_ch=hidden_ch, enable_semsic=True)
        elif self.method == "semsic_stage1_only":
            self.modem = SemSICGrantFreeModem(num_rbs=num_rbs, hidden_ch=hidden_ch, enable_semsic=False)
        elif self.method == "blackbox_noma":
            self.modem = BlackBoxMultiUserModem(max_users=3, n_rx=n_rx, num_rbs=num_rbs)
        else:
            raise ValueError(f"Unknown Dir-4B method: {self.method}")

    @torch.no_grad()
    def extract_encoder_stem(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encoder
        x = enc.patch_embed(image)
        for layer in enc.layers:
            x = layer(x)
        return enc.norm(x)

    def encode_single_user_stem(self, x_stem: torch.Tensor, user_slot: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        s_u = self.modem.embed_user_tx(z_tx, user_slot)
        return s_u, z_tx

    def _decode_from_channel_out(self, ch_v2: ChannelV2Output, cr_hat_db: torch.Tensor) -> torch.Tensor:
        dec = self.backbone.decoder
        B = cr_hat_db.shape[0]
        L = ch_v2.u_restored.shape[1]
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

        return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()

    def forward_noma_uplink(
        self,
        user_images_or_stems: list[torch.Tensor],
        user_snrs_db: torch.Tensor,
        *,
        rho_f: float = 0.85,
        h_mu_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        U = len(user_images_or_stems)
        s_list, ztx_list = [], []
        for u in range(U):
            stem_u = user_images_or_stems[u] if input_is_stem else self.extract_encoder_stem(user_images_or_stems[u])
            su, zu = self.encode_single_user_stem(stem_u, u)
            s_list.append(su)
            ztx_list.append(zu)

        s_users_iq = torch.stack(s_list, dim=1)  # [B, U, 32, 384, 2]
        y_rx_iq, diag = physical_grantfree_uplink_mac(
            s_users_iq,
            user_snrs_db,
            n_rx=self.n_rx,
            rho_f=rho_f,
            h_mu_override=h_mu_override,
            standard_noise_iq=standard_noise_iq,
        )
        demixed, denoised_z = self.modem.blind_demix_users(y_rx_iq, U, self.latent_channels)
        recons = [self._decode_from_channel_out(ch_u, cr_u) for (ch_u, cr_u) in demixed]
        crs = [cr_u for (_, cr_u) in demixed]
        return {
            "recons": recons,
            "cr_hats": crs,
            "z_tx_users": ztx_list,
            "z_denoised_users": denoised_z,
            **diag,
        }


@torch.no_grad()
def forward_pilot4_or_genie_linear_noma_reference(
    backbone: nn.Module,
    user_images: list[torch.Tensor],
    user_snrs_db: torch.Tensor,
    *,
    n_rx: int = 2,
    h_mu_override: tuple[torch.Tensor, torch.Tensor],
    standard_noise_iq: torch.Tensor,
    mode: str = "genie_linear_mmse",
    pilot_seed: int = 4099,
) -> list[torch.Tensor]:
    """Closed-Loop Linear Multi-User Reference Baselines:
       - genie_linear_mmse: Exact Oracle CSI Multi-User Linear MMSE + Genie per-user SNR & latent norm!
         (Demonstrates the fundamental rank-deficiency ceiling of linear reception on Overloaded 3-User -> 2-BS-Antenna NOMA!)
       - pilot4_linear_mmse: 4 Orthogonal Pilot Symbols per User per RB (4 * U * 32 = 256..384 dedicated pilot slots!) + Linear LMMSE.
    """
    U = len(user_images)
    B = user_images[0].shape[0]
    device, dtype = user_images[0].device, user_images[0].dtype
    enc, dec = backbone.encoder, backbone.decoder

    snr_u = user_snrs_db.to(device=device, dtype=dtype).reshape(B, U)
    s_list, pwr_list = [], []
    L = 256
    C = int(enc.head_list.out_features)
    K = (L * C) // 2

    for u in range(U):
        img_u = user_images[u]
        x = enc.patch_embed(img_u)
        for layer in enc.layers:
            x = layer(x)
        x = enc.norm(x)
        su_in = snr_u[:, u : u + 1]
        temp = None
        for i in range(enc.layer_num):
            temp = enc.sm_list[i](x) if i == 0 else enc.sm_list[i](temp)
            temp = temp * enc.bm_list[i](su_in).unsqueeze(1).expand(-1, L, -1)
        zu = enc.head_list(x * enc.sigmoid(enc.sm_list[-1](temp)))
        pw_u = (2.0 * zu.square().mean(dim=(1, 2))).clamp_min(1e-8)
        pwr_list.append(pw_u)
        fl = (zu / pw_u.sqrt()[:, None, None]).reshape(B, 2 * K)
        s_list.append(torch.stack([fl[:, :K].reshape(B, 32, 384), fl[:, K:].reshape(B, 32, 384)], dim=-1))

    s_users_iq = torch.stack(s_list, dim=1)
    y_rx_iq, diag = physical_grantfree_uplink_mac(
        s_users_iq,
        snr_u,
        n_rx=n_rx,
        h_mu_override=h_mu_override,
        standard_noise_iq=standard_noise_iq,
    )

    gr_true, gi_true = diag["g_true_r"], diag["g_true_i"]  # [B, U, Nr, M]
    if mode == "genie_linear_mmse":
        gr_est, gi_est = gr_true, gi_true
        sig_est = 0.0
    elif mode == "pilot4_linear_mmse":
        p_gen = torch.Generator(device="cpu").manual_seed(pilot_seed)
        # 4 orthogonal pilots per user per RB -> estimation variance 1.0 / 4.0 = 0.25
        pe_r = torch.randn(gr_true.shape, generator=p_gen, dtype=torch.float32).to(device=device, dtype=dtype)
        pe_i = torch.randn(gr_true.shape, generator=p_gen, dtype=torch.float32).to(device=device, dtype=dtype)
        std_p = math.sqrt(0.5 / 4.0)
        gr_est = gr_true + std_p * pe_r
        gi_est = gi_true + std_p * pe_i
        sig_est = 0.25
    else:
        raise ValueError(f"Unknown NOMA reference mode: {mode}")

    # Multi-User Real-Ring Linear MMSE
    M, N_b = 32, 384
    phi_all = vec_to_real_ring(gr_est.permute(0, 1, 3, 2), gi_est.permute(0, 1, 3, 2))  # [B, U, M, 2*Nr, 2]
    g_outer = torch.matmul(phi_all, phi_all.transpose(-1, -2))  # [B, U, M, 2*Nr, 2*Nr]
    cov_tot = g_outer.sum(dim=1) + (1.0 + U * sig_est) * torch.eye(2 * n_rx, device=device, dtype=dtype)[None, None]

    y_stack = torch.cat([y_rx_iq[..., 0], y_rx_iq[..., 1]], dim=1).permute(0, 2, 1, 3)  # [B, M, 2*Nr, N_b]
    recons: list[torch.Tensor] = []

    for u in range(U):
        sol_u = torch.linalg.solve(cov_tot, phi_all[:, u])  # [B, M, 2*Nr, 2]
        w_u = sol_u.transpose(-1, -2)  # [B, M, 2, 2*Nr]
        f_u = torch.matmul(w_u, phi_all[:, u])
        alp_um = 0.5 * (f_u[:, :, 0, 0] + f_u[:, :, 1, 1]).clamp(1e-5, 0.9999)
        alp_bar = alp_um.mean(dim=1, keepdim=True).clamp_min(0.03)

        s_est = torch.matmul(w_u, y_stack) / alp_bar[:, :, None, None]
        u_rest = torch.cat([s_est[:, :, 0, :].reshape(B, K), s_est[:, :, 1, :].reshape(B, K)], dim=1).reshape(B, L, C)
        u_rest = u_rest * pwr_list[u].sqrt()[:, None, None]

        eff_sinr_db = (10.0 * torch.log10(alp_bar / (1.0 - alp_bar).clamp_min(1e-4))).clamp(-6.0, 22.0)
        xd = dec.head_list(u_rest)
        temp = None
        for i in range(dec.layer_num):
            temp = dec.sm_list[i](xd) if i == 0 else dec.sm_list[i](temp)
            temp = temp * dec.bm_list[i](eff_sinr_db).unsqueeze(1).expand(-1, L, -1)
        xd = xd * dec.sigmoid(dec.sm_list[-1](temp))
        for layer in dec.layers:
            xd = layer(xd)
        recons.append(xd.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous())

    return recons
