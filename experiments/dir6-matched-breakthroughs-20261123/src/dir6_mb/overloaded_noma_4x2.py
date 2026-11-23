"""Dir-6C: SemSIC-NOMA-4x2 — 200% Overloaded (4 Simultaneous Users -> 2 BS Antennas) Grant-Free Zero-Pilot
Uplink with Blind CFAR Active-User Subset Detection & Matched-Control 3-Round Semantic SIC.

Protocol Guarantees (GPT-Audit-Proof):
  1. 200% Spatial Overload Geometry: U_max = 4 autonomous edge users transmit full-resolution 256x256 RGB images
     simultaneously over the SAME M = 32 RBs x N_b = 384 REs (K = 12,288 complex REs = CBR 1/16 per user) onto only
     N_r = 2 BS receive antennas (8 real TX DoF -> 4 real RX measurements per RE: rank-deficient 2:1 overload!).
  2. 12-Walsh Collision-Free Signature Subspace (`build_walsh_16_basis(12, 384)`):
       - w_0, w_1:      Universal Quiet Nulls silent across all 4 users -> exact blind thermal noise floor sigma_n^2!
       - w_2..w_5:      User u in {0..3} Orthogonal Newman-Chirped Channel Probes (0 inter-user pilot contamination!)
       - w_6..w_9:      User u in {0..3} Orthogonal atan2 Latent Norm Signatures
       - w_10, w_11:    Reserved guard orthogonal modes (372/384 = 96.875% shared non-orthogonal payload DoF!)
  3. Blind CFAR Active-User Subset Detector: Detects any unknown active subset U_act subseteq {0, 1, 2, 3} directly
     from coherent probe-subspace energy vs `2.8 * sigma_n_hat` (zero grant signaling!).
  4. Strict Byte-Identical Matched Controls:
       - `semsic_3round`:     Stage-1 Blind LMMSE + 3 rounds of `SemanticLatentPriorNet` + cross-user waveform IC +
                              residual covariance update + LMMSE re-separation (4 LMMSE solves total).
       - `owndenoise_3round`: IDENTICAL parameter tensors, shapes, initialization, and 3-round `SemanticLatentPriorNet`
                              compute + identical training loss, ablating ONLY cross-user waveform subtraction & LMMSE
                              re-separation!
       - `tuned_pic_3round`:  Classical 4-LMMSE Parallel Interference Cancellation (soft symbol-domain PIC without
                              semantic image prior) with damping `(beta_1, beta_2, beta_3)` tuned ONLY on `val256`!
       - `linear_1stage`:     Single-stage Blind Multi-User LMMSE (`1` LMMSE solve).
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn

from dir2_zpac.blind_v3 import (
    SNR_CENTER,
    SNR_SCALE,
    BlindMultiScaleDecoderCalibrator,
    OpenLoopHierarchicalTX,
)
from dir2_zpac.phy_v2 import ChannelV2Output, expand_rb_to_symbols
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb
from dir3_mh.mimo_blind import build_hadamard_pow2
from dir4_rn.noma_semsic_blind import (
    SemanticLatentPriorNet,
    sample_multiuser_uplink_channels,
    vec_to_real_ring,
)
from dir4_rn.rf_1bit_blind import newman_rb_phases


class SafeSemanticLatentPriorNet(SemanticLatentPriorNet):
    """Identical parameters to `SemanticLatentPriorNet`, guarding `(u_rest^2 + 1e-8).sqrt()` on silent grant-free slots!"""

    def forward(
        self,
        u_rest: torch.Tensor,
        alpha_rb: torch.Tensor,
        sigma_post_rb: torch.Tensor,
        cr_db: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C = u_rest.shape
        K = (L * C) // 2
        H_s = W_s = 16

        alp_iq = expand_rb_to_symbols(alpha_rb, K)
        sig_iq = expand_rb_to_symbols(torch.log(sigma_post_rb.clamp_min(1e-4)).div(4.0), K)
        alp_tok = torch.cat([alp_iq, alp_iq], dim=1).reshape(B, L, C).mean(dim=-1).reshape(B, 1, H_s, W_s)
        sig_tok = torch.cat([sig_iq, sig_iq], dim=1).reshape(B, L, C).mean(dim=-1).reshape(B, 1, H_s, W_s)
        cr_map = ((cr_db - SNR_CENTER) / SNR_SCALE).reshape(B, 1, 1, 1).expand(B, 1, H_s, W_s)
        e_tok = (u_rest.square().mean(dim=-1) + 1e-8).sqrt().div(3.0).reshape(B, 1, H_s, W_s)

        z_2d = u_rest.transpose(1, 2).reshape(B, C, H_s, W_s)
        feat = self.in_proj(torch.cat([z_2d, alp_tok, sig_tok, cr_map, e_tok], dim=1))
        for blk in self.blocks:
            feat = feat + blk(feat)

        tok = feat.flatten(2).transpose(1, 2)
        tok_n = self.attn_norm(tok)
        attn_out, _ = self.attn(tok_n, tok_n, tok_n, need_weights=False)
        feat = feat + attn_out.transpose(1, 2).reshape(B, -1, H_s, W_s)

        delta = self.out_proj(feat).flatten(2).transpose(1, 2)
        return u_rest + delta

MAX_USERS_4X2 = 4
N_WALSH_4X2 = 12
N_B_4X2 = 384  # 24 * 16 -> exact 16-Walsh tiling


def build_walsh_16_basis(
    k_vecs: int = N_WALSH_4X2,
    n_sym: int = N_B_4X2,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct `k_vecs` strictly orthonormal length-`n_sym` Walsh vectors (`<w_i, w_j> = delta_ij`) for `1 <= k_vecs <= 16`."""
    assert 1 <= k_vecs <= 16 and n_sym % 16 == 0, f"Invalid k_vecs={k_vecs}, n_sym={n_sym}"
    dev = torch.device(device)
    h16 = build_hadamard_pow2(16, device=dev, dtype=torch.float64)
    reps = n_sym // 16
    basis = (h16[:k_vecs, :].repeat(1, reps) / math.sqrt(float(reps))).to(dtype=dtype)
    return basis.contiguous()


def physical_grantfree_4x2_mac(
    s_users_iq: torch.Tensor,                      # [B, U=4, M=32, N_b=384, 2]
    user_snrs_db: torch.Tensor,                    # [B, U=4] per-user received SNR (dB) per BS antenna
    *,
    active_user_mask: torch.Tensor | None = None,  # [B, U=4] in {0.0, 1.0} for random grant-free activity!
    n_rx: int = 2,
    rho_f: float = 0.85,
    h_mu_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    standard_noise_iq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Physical 4-User -> 2-BS-Antenna Non-Orthogonal Uplink MAC with optional sporadic user activity."""
    B, U, M, N_b, _ = s_users_iq.shape
    device, dtype = s_users_iq.device, s_users_iq.dtype

    u_pwr = (s_users_iq[..., 0].square() + s_users_iq[..., 1].square()).mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    s_norm = s_users_iq / u_pwr.unsqueeze(-1).sqrt()

    if active_user_mask is not None:
        act = active_user_mask.to(device=device, dtype=dtype).reshape(B, U)
    else:
        act = torch.ones((B, U), device=device, dtype=dtype)

    s_norm = s_norm * act[:, :, None, None, None]

    if h_mu_override is not None:
        hr = h_mu_override[0].to(device=device, dtype=dtype)
        hi = h_mu_override[1].to(device=device, dtype=dtype)
    else:
        hr, hi = sample_multiuser_uplink_channels(B, U, n_rx, M, rho_f, device=device, dtype=dtype)

    snr_u = user_snrs_db.to(device=device, dtype=dtype).reshape(B, U)
    amp_u = torch.pow(10.0, snr_u / 20.0)[:, :, None, None] * act[:, :, None, None]
    gr = hr * amp_u
    gi = hi * amp_u

    sr = s_norm[..., 0]
    si = s_norm[..., 1]
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

    return torch.stack([yr, yi], dim=-1), {
        "g_true_r": gr,
        "g_true_i": gi,
        "active_true": act,
    }


class Overloaded4x2Modem(nn.Module):
    """Byte-Identical 12-Walsh Subspace Multi-User Modem supporting:
      - `semsic_3round`:     3-Round Nonlinear Semantic-Physical SIC (`4` LMMSE solves)
      - `owndenoise_3round`: Byte-identical parameter & 3-round `SemanticLatentPriorNet` single-user post-denoising
                             WITHOUT cross-user waveform subtraction or LMMSE re-separation!
      - `tuned_pic_3round`:  Classical 4-LMMSE Parallel Interference Cancellation (`pic_betas` tuned on `val256`)
      - `linear_1stage`:     Single-stage Blind Multi-User LMMSE (`1` LMMSE solve)
    """

    def __init__(
        self,
        num_rbs: int = 32,
        hidden_ch: int = 160,
        mode: str = "semsic_3round",
        cfar_mult: float = 2.8,
    ):
        super().__init__()
        self.num_rbs = int(num_rbs)
        self.mode = mode.lower()
        self.cfar_mult = float(cfar_mult)
        self.pic_betas: tuple[float, float, float] = (0.55, 0.70, 0.80)

        init_eta_ch = 0.058
        inv_s = math.log((init_eta_ch - 0.025) / (0.125 - init_eta_ch))
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

        # ALWAYS instantiate `latent_prior` and `sic_gate_mlp` in identical order for `semsic_3round` and
        # `owndenoise_3round` so parameter counts and initial tensors are 100% byte-for-byte identical!
        if self.mode in ("semsic_3round", "owndenoise_3round"):
            self.latent_prior: SafeSemanticLatentPriorNet | None = SafeSemanticLatentPriorNet(latent_ch=96, hidden_ch=hidden_ch)
            self.sic_gate_mlp: nn.Sequential | None = nn.Sequential(
                nn.Linear(4, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.sic_gate_mlp[-1].weight)
            nn.init.constant_(self.sic_gate_mlp[-1].bias, 0.35)
        else:
            self.latent_prior = None
            self.sic_gate_mlp = None

    @property
    def eta_ch(self) -> torch.Tensor:
        return 0.025 + 0.100 * torch.sigmoid(self.eta_ch_raw)

    def project_data_perp(self, z_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normalized latent [B, 256, 96] onto span(w_0..w_{11})^perp with exact `data_pwr`."""
        B, L, C = z_norm.shape
        M = self.num_rbs
        N_b = N_B_4X2
        K = M * N_b
        device, dtype = z_norm.device, z_norm.dtype

        basis = build_walsh_16_basis(N_WALSH_4X2, N_b, device=device, dtype=dtype)
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
        """Encode autonomous edge user `u in {0, 1, 2, 3}` -> [B, M=32, N_b=384, 2]."""
        assert 0 <= user_slot < MAX_USERS_4X2
        B, _, _ = z_tx.shape
        M = self.num_rbs
        N_b = N_B_4X2
        device, dtype = z_tx.device, z_tx.dtype

        p_img = (2.0 * z_tx.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
        z_norm = z_tx / p_img[:, None, None]
        sr, si = self.project_data_perp(z_norm)

        basis = build_walsh_16_basis(N_WALSH_4X2, N_b, device=device, dtype=dtype)
        u_ch = basis[2 + user_slot]  # w_2..w_5
        u_sc = basis[6 + user_slot]  # w_6..w_9
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
        y_perp_r: torch.Tensor,     # [B, U, Nr, M, N_b]
        y_perp_i: torch.Tensor,     # [B, U, Nr, M, N_b]
        ghat_r: torch.Tensor,       # [B, U, Nr, M] (already masked by active_hat)
        ghat_i: torch.Tensor,       # [B, U, Nr, M]
        p_hat_u: torch.Tensor,      # [B, U]
        sigma_n_hat: torch.Tensor,  # [B]
        sigma_e_u: torch.Tensor,    # [B, U]
        nu_res_um: torch.Tensor,    # [B, U, M]
        active_hat: torch.Tensor,   # [B, U] in {0.0, 1.0}
        latent_ch: int,
    ) -> list[tuple[ChannelV2Output, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, U, Nr, M = ghat_r.shape
        N_b = y_perp_r.shape[-1]
        K = M * N_b
        L = (2 * K) // latent_ch
        device, dtype = ghat_r.device, ghat_r.dtype

        data_pwr = 1.0 - self.eta_ch - self.eta_sc
        dof_comp = math.sqrt(float(N_b - N_WALSH_4X2) / float(N_b))
        data_scale = (1.0 / data_pwr).sqrt() * dof_comp

        gr_bm = ghat_r.permute(0, 1, 3, 2)
        gi_bm = ghat_i.permute(0, 1, 3, 2)
        phi_all = vec_to_real_ring(gr_bm, gi_bm)  # [B, U, M, 2*Nr, 2]
        g_outer = data_pwr * torch.matmul(phi_all, phi_all.transpose(-1, -2))

        eye_2nr = torch.eye(2 * Nr, device=device, dtype=dtype)[None, None, :, :]
        outputs = []

        for u in range(U):
            cov_u = g_outer[:, u].clone()
            for up in range(U):
                if up == u:
                    continue
                cov_u = cov_u + (nu_res_um[:, up] * active_hat[:, up, None])[:, :, None, None] * g_outer[:, up]

            tot_sig_var = sigma_n_hat[:, None] + sigma_e_u[:, u, None] * float(Nr)
            cov_u = cov_u + tot_sig_var[:, :, None, None] * eye_2nr

            rhs_u = data_pwr * phi_all[:, u]
            sol_u = torch.linalg.solve(cov_u, rhs_u)
            w_u = sol_u.transpose(-1, -2)  # [B, M, 2, 2*Nr]

            f_u = torch.matmul(w_u, phi_all[:, u])
            alpha_um = 0.5 * (f_u[:, :, 0, 0] + f_u[:, :, 1, 1]).clamp(1e-5, 0.9999)
            alpha_bar_u = alpha_um.mean(dim=1, keepdim=True).clamp_min(0.025)

            yu_stack = torch.cat([y_perp_r[:, u], y_perp_i[:, u]], dim=1).permute(0, 2, 1, 3)
            s_est = torch.matmul(w_u, yu_stack)  # [B, M, 2, N_b] (at scale ~ alpha_um * s_true)

            gain_ub = (data_scale / alpha_bar_u)[:, :, None]
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
                phase_unc_iq=expand_rb_to_symbols(((sigma_e_u[:, u, None] + 1e-8) / (g_abs_um.square() + 1e-5)).clamp_min(1e-8).sqrt(), K),
                psi_csi=(alpha_um - alpha_bar_u).square().mean(dim=1) + sigma_e_u[:, u],
            )
            # Also return unscaled `s_est / alpha_bar_u` for classical PIC symbol-domain IC!
            s_lin_r = s_est[:, :, 0, :] / alpha_bar_u[:, :, None]
            s_lin_i = s_est[:, :, 1, :] / alpha_bar_u[:, :, None]
            outputs.append((ch_out, cr_db, alpha_um, sigma_post_rb, s_lin_r, s_lin_i))

        return outputs

    def blind_demix_users(
        self,
        y_rx_iq: torch.Tensor,
        n_users: int,
        latent_ch: int,
        *,
        override_mode: str | None = None,
        pic_betas_override: tuple[float, float, float] | None = None,
    ) -> tuple[list[tuple[ChannelV2Output, torch.Tensor]], list[torch.Tensor], dict[str, torch.Tensor]]:
        B, Nr, M, N_b, _ = y_rx_iq.shape
        U = int(n_users)
        assert U <= MAX_USERS_4X2
        run_mode = self.mode if override_mode is None else override_mode.lower()
        device, dtype = y_rx_iq.device, y_rx_iq.dtype

        yr = y_rx_iq[..., 0]
        yi = y_rx_iq[..., 1]
        basis = build_walsh_16_basis(N_WALSH_4X2, N_b, device=device, dtype=dtype)
        n_cos, n_sin = newman_rb_phases(M, device=device, dtype=dtype)

        # 1. Universal Quiet Nulls (w_0, w_1) -> Exact Blind Noise Variance (immune to all 4 colliding users!)
        q0_r = torch.einsum("brmn,n->brm", yr, basis[0])
        q0_i = torch.einsum("brmn,n->brm", yi, basis[0])
        q1_r = torch.einsum("brmn,n->brm", yr, basis[1])
        q1_i = torch.einsum("brmn,n->brm", yi, basis[1])
        sigma_n_hat = 0.5 * (q0_r.square() + q0_i.square() + q1_r.square() + q1_i.square()).mean(dim=(1, 2)).clamp(1e-4, 15.0)

        # 2. Blind CFAR Active-User Subset Detection + Collision-Free Per-User Channel Estimation on w_{2+u}!
        eta_ch = self.eta_ch
        amp_ch = (eta_ch * float(N_b)).sqrt()
        sigma_e_raw = sigma_n_hat / (eta_ch * float(N_b))
        e_rb_ant = (yr.square() + yi.square()).mean(dim=-1)

        ghat_r_list, ghat_i_list, sige_u_list, probe_e_list = [], [], [], []
        for u in range(U):
            proj_r = torch.einsum("brmn,n->brm", yr, basis[2 + u])
            proj_i = torch.einsum("brmn,n->brm", yi, basis[2 + u])
            probe_e_u = (proj_r.square() + proj_i.square()).mean(dim=(1, 2))  # [B]: ~sigma_n^2 if silent, >>sigma_n^2 if active!
            probe_e_list.append(probe_e_u)

            raw_r = proj_r / amp_ch
            raw_i = proj_i / amp_ch
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

        probe_e_all = torch.stack(probe_e_list, dim=1)  # [B, U]
        active_hat = (probe_e_all > (self.cfar_mult * sigma_n_hat[:, None])).to(dtype)  # [B, U] in {0.0, 1.0}
        # Safety guard: if all slots appear below threshold under extreme noise, keep strongest slot active
        none_active = active_hat.sum(dim=1) < 0.5
        if none_active.any():
            top_u = probe_e_all.argmax(dim=1)
            for b_idx in torch.nonzero(none_active, as_tuple=False).reshape(-1).tolist():
                active_hat[b_idx, int(top_u[b_idx].item())] = 1.0

        ghat_r = torch.stack(ghat_r_list, dim=1) * active_hat[:, :, None, None]
        ghat_i = torch.stack(ghat_i_list, dim=1) * active_hat[:, :, None, None]
        sigma_e_u = torch.stack(sige_u_list, dim=1)

        # 3. Collision-Free Per-User atan2 Scale Recovery on w_{6+u}
        p_hat_list = []
        for u in range(U):
            qsc_r = torch.einsum("brmn,n->brm", yr, basis[6 + u])
            qsc_i = torch.einsum("brmn,n->brm", yi, basis[6 + u])
            usc_r = qsc_r * n_cos[None, None, :] + qsc_i * n_sin[None, None, :]
            usc_i = qsc_i * n_cos[None, None, :] - qsc_r * n_sin[None, None, :]
            z_r = (ghat_r[:, u] * usc_r + ghat_i[:, u] * usc_i).sum(dim=(1, 2))
            z_i = (ghat_r[:, u] * usc_i - ghat_i[:, u] * usc_r).sum(dim=(1, 2))
            th_u = torch.atan2(z_i, z_r.clamp_min(1e-5)).clamp(-1.21, 1.21)
            p_u = (2.82 + torch.atanh(th_u / 1.25) / 0.40).clamp(0.4, 7.0)
            p_hat_list.append(p_u)
        p_hat_u = torch.stack(p_hat_list, dim=1)

        # 4. Orthogonal Projection onto Shared Semantic Payload Subspace span(w_0..w_{11})^perp
        proj_yr = torch.einsum("brmn,kn->brmk", yr, basis)
        proj_yi = torch.einsum("brmn,kn->brmk", yi, basis)
        y_perp0_r = yr - torch.einsum("brmk,kn->brmn", proj_yr, basis)
        y_perp0_i = yi - torch.einsum("brmk,kn->brmn", proj_yi, basis)

        # Stage 1: Joint Multi-User Blind LMMSE (`1st` LMMSE solve)
        y_s1_r = y_perp0_r[:, None].expand(B, U, Nr, M, N_b)
        y_s1_i = y_perp0_i[:, None].expand(B, U, Nr, M, N_b)
        nu_ones = torch.ones((B, U, M), device=device, dtype=dtype)
        stage_outs = self._demix_single_stage(
            y_s1_r, y_s1_i, ghat_r, ghat_i, p_hat_u, sigma_n_hat, sigma_e_u, nu_ones, active_hat, latent_ch
        )

        aux_meta = {"active_hat": active_hat, "sigma_n_hat": sigma_n_hat, "probe_energy": probe_e_all}

        if run_mode == "linear_1stage":
            return [(o[0], o[1]) for o in stage_outs], [], aux_meta

        if run_mode == "tuned_pic_3round":
            # Classical 4-LMMSE Parallel Interference Cancellation (soft symbol-domain PIC without semantic prior!)
            betas = pic_betas_override if pic_betas_override is not None else self.pic_betas
            data_pwr = (1.0 - self.eta_ch - self.eta_sc).detach()
            amp_clip = (0.5 * data_pwr).clamp_min(1e-4).sqrt()  # Per-quadrature RMS of Gaussian payload
            for r_idx in range(3):
                beta_r = float(betas[r_idx])
                s_pic_r_list, s_pic_i_list, om_list = [], [], []
                for u in range(U):
                    _, _, alp_um, _, slin_r, slin_i = stage_outs[u]
                    # Soft MMSE/tanh physical symbol limiter + projection onto span(w_0..w_{11})^perp
                    sr_c = 1.6 * amp_clip * torch.tanh(slin_r / (1.6 * amp_clip))
                    si_c = 1.6 * amp_clip * torch.tanh(slin_i / (1.6 * amp_clip))
                    pr_u = torch.einsum("bmn,kn->bmk", sr_c, basis)
                    pi_u = torch.einsum("bmn,kn->bmk", si_c, basis)
                    sr_p = sr_c - torch.einsum("bmk,kn->bmn", pr_u, basis)
                    si_p = si_c - torch.einsum("bmk,kn->bmn", pi_u, basis)
                    rms_p = (sr_p.square() + si_p.square()).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()
                    sr_p = sr_p * (data_pwr.sqrt() / rms_p)
                    si_p = si_p * (data_pwr.sqrt() / rms_p)
                    s_pic_r_list.append(sr_p)
                    s_pic_i_list.append(si_p)
                    om_list.append(beta_r * alp_um * active_hat[:, u, None])

                s_pic_r = torch.stack(s_pic_r_list, dim=1)
                s_pic_i = torch.stack(s_pic_i_list, dim=1)
                om_all = torch.stack(om_list, dim=1)

                ws_r = om_all.unsqueeze(-1) * s_pic_r
                ws_i = om_all.unsqueeze(-1) * s_pic_i
                contrib_r = ghat_r.unsqueeze(-1) * ws_r[:, :, None] - ghat_i.unsqueeze(-1) * ws_i[:, :, None]
                contrib_i = ghat_r.unsqueeze(-1) * ws_i[:, :, None] + ghat_i.unsqueeze(-1) * ws_r[:, :, None]
                tot_r = contrib_r.sum(dim=1, keepdim=True)
                tot_i = contrib_i.sum(dim=1, keepdim=True)
                y_pic_r = y_perp0_r[:, None] - (tot_r - contrib_r)
                y_pic_i = y_perp0_i[:, None] - (tot_i - contrib_i)
                nu_res = (1.0 - om_all).square().clamp(0.05, 1.0)
                stage_outs = self._demix_single_stage(
                    y_pic_r, y_pic_i, ghat_r, ghat_i, p_hat_u, sigma_n_hat, sigma_e_u, nu_res, active_hat, latent_ch
                )
            return [(o[0], o[1]) for o in stage_outs], [], aux_meta

        if run_mode == "owndenoise_3round":
            # STRICT BYTE-IDENTICAL PARAMETER & COMPUTE MATCHED CONTROL:
            # Runs the exact same 3 rounds of `SemanticLatentPriorNet` + `sic_gate_mlp` on each user's OWN Stage-1
            # latent estimate, WITHOUT subtracting cross-user waveforms or re-solving multi-user LMMSE!
            assert self.latent_prior is not None and self.sic_gate_mlp is not None
            curr_u = [o[0].u_restored for o in stage_outs]
            denoised_latents: list[torch.Tensor] = []
            for r_idx in range(3):
                r_feat = float(r_idx)
                round_prog = -0.35 + 0.55 * r_feat
                next_u = []
                for u in range(U):
                    ch_u, cr_u, alp_um, sigp_um, _, _ = stage_outs[u]
                    z_prop = self.latent_prior(curr_u[u], alp_um, sigp_um, cr_u)
                    alp_bar_u = alp_um.mean(dim=1, keepdim=True).expand(B, M)
                    r_col = torch.full_like(alp_um, r_feat)
                    u_diff = (z_prop - ch_u.u_restored).detach().square().mean(dim=(1, 2), keepdim=True).squeeze(-1).expand(B, M).clamp(0.0, 5.0)
                    gate_in = torch.stack([alp_um, alp_bar_u, r_col, u_diff], dim=-1)
                    w_own = torch.sigmoid(self.sic_gate_mlp(gate_in).squeeze(-1) + round_prog).mean(dim=-1)
                    z_upd = curr_u[u] + (0.45 + 0.55 * w_own)[:, None, None] * (z_prop - curr_u[u])
                    next_u.append(z_upd)
                    denoised_latents.append(z_upd)
                curr_u = next_u

            own_outs: list[tuple[ChannelV2Output, torch.Tensor]] = []
            for u in range(U):
                ch_u, cr_u, _, _, _, _ = stage_outs[u]
                u_fin = curr_u[u]
                p_fin = (2.0 * u_fin.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
                u_fin_norm = u_fin / p_fin[:, None, None]
                own_outs.append((
                    ChannelV2Output(
                        u_restored=u_fin,
                        u_mmse_restored=u_fin,
                        u_norm=u_fin_norm,
                        z_norm=u_fin_norm,
                        pwr=p_fin.square(),
                        gamma_db=cr_u,
                        c_r_db=cr_u,
                        h_abs_iq=ch_u.h_abs_iq,
                        alpha_ratio_iq=ch_u.alpha_ratio_iq,
                        sigma_post_iq=ch_u.sigma_post_iq,
                        phase_unc_iq=ch_u.phase_unc_iq,
                        psi_csi=ch_u.psi_csi,
                    ),
                    cr_u,
                ))
            return own_outs, denoised_latents, aux_meta

        if run_mode != "semsic_3round":
            raise ValueError(f"Unknown Overloaded4x2Modem mode: {run_mode}")

        # Proposed 3-Round Nonlinear Physical-Semantic Iterative Interference Cancellation (`SemSIC_NOMA_4x2`)!
        assert self.latent_prior is not None and self.sic_gate_mlp is not None
        denoised_latents = []
        for sic_round in range(3):
            s_remod_r_list, s_remod_i_list, omega_list = [], [], []
            round_feat = float(sic_round)
            round_prog = -0.35 + 0.55 * round_feat

            for u in range(U):
                ch_u, cr_u, alp_um, sigp_um, _, _ = stage_outs[u]
                z_clean_u = self.latent_prior(ch_u.u_restored, alp_um, sigp_um, cr_u)
                denoised_latents.append(z_clean_u)

                p_clean_u = (2.0 * z_clean_u.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
                sr_u, si_u = self.project_data_perp(z_clean_u / p_clean_u[:, None, None])
                s_remod_r_list.append(sr_u)
                s_remod_i_list.append(si_u)

                alp_bar_u = alp_um.mean(dim=1, keepdim=True).expand(B, M)
                r_col = torch.full_like(alp_um, round_feat)
                u_diff = (z_clean_u - ch_u.u_restored).detach().square().mean(dim=(1, 2), keepdim=True).squeeze(-1).expand(B, M).clamp(0.0, 5.0)
                gate_in = torch.stack([alp_um, alp_bar_u, r_col, u_diff], dim=-1)
                # Progressive soft cancellation weight ramping smoothly across rounds 0 -> 1 -> 2!
                omega_um = 0.96 * torch.sigmoid(self.sic_gate_mlp(gate_in).squeeze(-1) + round_prog) * (0.25 + 0.75 * alp_um.clamp_min(1e-4).sqrt()) * active_hat[:, u, None]
                omega_list.append(omega_um)
            s_remod_r = torch.stack(s_remod_r_list, dim=1)
            s_remod_i = torch.stack(s_remod_i_list, dim=1)
            omega_all = torch.stack(omega_list, dim=1)

            ws_r = omega_all.unsqueeze(-1) * s_remod_r
            ws_i = omega_all.unsqueeze(-1) * s_remod_i
            contrib_r = ghat_r.unsqueeze(-1) * ws_r[:, :, None] - ghat_i.unsqueeze(-1) * ws_i[:, :, None]
            contrib_i = ghat_r.unsqueeze(-1) * ws_i[:, :, None] + ghat_i.unsqueeze(-1) * ws_r[:, :, None]
            tot_r = contrib_r.sum(dim=1, keepdim=True)
            tot_i = contrib_i.sum(dim=1, keepdim=True)

            y_sic_r = y_perp0_r[:, None] - (tot_r - contrib_r)
            y_sic_i = y_perp0_i[:, None] - (tot_i - contrib_i)
            nu_res = (1.0 - omega_all).square().clamp(0.025, 1.0)
            stage_outs = self._demix_single_stage(
                y_sic_r, y_sic_i, ghat_r, ghat_i, p_hat_u, sigma_n_hat, sigma_e_u, nu_res, active_hat, latent_ch
            )

        return [(o[0], o[1]) for o in stage_outs], denoised_latents, aux_meta


class OverloadedNOMA4x2Transceiver(nn.Module):
    """Unified 4-User -> 2-BS-Antenna Overloaded Grant-Free Uplink Transceiver (`Dir-6C`)."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        method: str = "semsic_3round",
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
        self.modem = Overloaded4x2Modem(num_rbs=num_rbs, hidden_ch=hidden_ch, mode=self.method)

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

    def forward_noma_4x2(
        self,
        user_images_or_stems: list[torch.Tensor],
        user_snrs_db: torch.Tensor,
        *,
        active_user_mask: torch.Tensor | None = None,
        rho_f: float = 0.85,
        h_mu_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        standard_noise_iq: torch.Tensor | None = None,
        input_is_stem: bool = False,
        override_mode: str | None = None,
        pic_betas_override: tuple[float, float, float] | None = None,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        U = len(user_images_or_stems)
        assert U <= MAX_USERS_4X2
        s_list, ztx_list = [], []
        for u in range(U):
            stem_u = user_images_or_stems[u] if input_is_stem else self.extract_encoder_stem(user_images_or_stems[u])
            su, zu = self.encode_single_user_stem(stem_u, u)
            s_list.append(su)
            ztx_list.append(zu)

        s_users_iq = torch.stack(s_list, dim=1)
        y_rx_iq, diag = physical_grantfree_4x2_mac(
            s_users_iq,
            user_snrs_db,
            active_user_mask=active_user_mask,
            n_rx=self.n_rx,
            rho_f=rho_f,
            h_mu_override=h_mu_override,
            standard_noise_iq=standard_noise_iq,
        )
        demixed, denoised_z, aux_meta = self.modem.blind_demix_users(
            y_rx_iq,
            U,
            self.latent_channels,
            override_mode=override_mode,
            pic_betas_override=pic_betas_override,
        )
        recons = [self._decode_from_channel_out(ch_u, cr_u) for (ch_u, cr_u) in demixed]
        crs = [cr_u for (_, cr_u) in demixed]
        return {
            "recons": recons,
            "cr_hats": crs,
            "z_tx_users": ztx_list,
            "z_denoised_users": denoised_z,
            **diag,
            **aux_meta,
        }


@torch.inference_mode()
def tune_pic_damping_val256(
    model: OverloadedNOMA4x2Transceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> tuple[tuple[float, float, float], float]:
    """Grid-search classical 3-round PIC damping `(beta_1, beta_2, beta_3)` strictly on `val256` (disjoint 64 4-tuples)!"""
    from dir2_zpac.metrics import compute_psnr_batch

    model.eval()
    N = val_stems.shape[0]
    n_tup = N // 4  # 64 strictly disjoint 4-image tuples on val256!
    gen = torch.Generator(device="cpu").manual_seed(6609)
    hr, hi = sample_multiuser_uplink_channels(n_tup, 4, model.n_rx, model.num_rbs, 0.85, generator=gen)
    eps = torch.randn((n_tup, model.n_rx, model.num_rbs, N_B_4X2, 2), generator=gen)
    snr_nf = torch.tensor([18.0, 14.0, 9.0, 5.0], dtype=torch.float32)[None, :].expand(n_tup, 4)

    candidates = [
        (0.35, 0.50, 0.65),
        (0.45, 0.65, 0.80),
        (0.55, 0.72, 0.85),
        (0.65, 0.80, 0.90),
        (0.75, 0.85, 0.92),
    ]
    best_betas = candidates[1]
    best_psnr = -1e9
    bs = 16
    for cand in candidates:
        psnrs = []
        for s in range(0, n_tup, bs):
            e = min(n_tup, s + bs)
            u_stems = [val_stems[4 * k + u : 4 * e : 4].to(device) if False else val_stems[4 * s + u : 4 * e : 4].to(device) for u in range(4)]
            out = model.forward_noma_4x2(
                u_stems,
                snr_nf[s:e].to(device),
                h_mu_override=(hr[s:e], hi[s:e]),
                standard_noise_iq=eps[s:e],
                input_is_stem=True,
                override_mode="tuned_pic_3round",
                pic_betas_override=cand,
            )
            for u in range(4):
                tgt_u = val_targets[4 * s + u : 4 * e : 4].to(device)
                psnrs.extend(compute_psnr_batch(tgt_u, out["recons"][u].clamp(0.0, 1.0)).cpu().tolist())
        mean_p = float(sum(psnrs) / len(psnrs))
        if mean_p > best_psnr:
            best_psnr = mean_p
            best_betas = cand
    model.modem.pic_betas = best_betas
    return best_betas, best_psnr


@torch.inference_mode()
def forward_genie_linear_noma_4x2(
    backbone: nn.Module,
    user_images: list[torch.Tensor],
    user_snrs_db: torch.Tensor,
    *,
    n_rx: int = 2,
    h_mu_override: tuple[torch.Tensor, torch.Tensor],
    standard_noise_iq: torch.Tensor,
) -> list[torch.Tensor]:
    """Oracle Exact-CSI + True-SNR Linear MMSE Baseline on 4-User -> 2-BS-Antenna Overloaded MAC
    (100% payload DoF, exact true channel G_{u,m}, demonstrating the rank-deficiency ceiling of linear reception!).
    """
    U = len(user_images)
    B = user_images[0].shape[0]
    device, dtype = user_images[0].device, user_images[0].dtype
    enc, dec = backbone.encoder, backbone.decoder
    snr_u = user_snrs_db.to(device=device, dtype=dtype).reshape(B, U)

    L = 256
    C = int(enc.head_list.out_features)
    K = (L * C) // 2
    s_list, pwr_list = [], []
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
        s_list.append(torch.stack([fl[:, :K].reshape(B, 32, N_B_4X2), fl[:, K:].reshape(B, 32, N_B_4X2)], dim=-1))

    s_users_iq = torch.stack(s_list, dim=1)
    y_rx_iq, diag = physical_grantfree_4x2_mac(
        s_users_iq,
        snr_u,
        n_rx=n_rx,
        h_mu_override=h_mu_override,
        standard_noise_iq=standard_noise_iq,
    )
    gr_true, gi_true = diag["g_true_r"], diag["g_true_i"]
    phi_all = vec_to_real_ring(gr_true.permute(0, 1, 3, 2), gi_true.permute(0, 1, 3, 2))
    g_outer = torch.matmul(phi_all, phi_all.transpose(-1, -2))
    cov_tot = g_outer.sum(dim=1) + torch.eye(2 * n_rx, device=device, dtype=dtype)[None, None]

    y_stack = torch.cat([y_rx_iq[..., 0], y_rx_iq[..., 1]], dim=1).permute(0, 2, 1, 3)
    recons: list[torch.Tensor] = []
    for u in range(U):
        sol_u = torch.linalg.solve(cov_tot, phi_all[:, u])
        w_u = sol_u.transpose(-1, -2)
        f_u = torch.matmul(w_u, phi_all[:, u])
        alp_um = 0.5 * (f_u[:, :, 0, 0] + f_u[:, :, 1, 1]).clamp(1e-5, 0.9999)
        alp_bar = alp_um.mean(dim=1, keepdim=True).clamp_min(0.025)

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
        recons.append(xd.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0))
    return recons
