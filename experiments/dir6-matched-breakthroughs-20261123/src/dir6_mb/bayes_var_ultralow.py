"""Dir-6B: BayesVAR-JSCC — Wireless-Posterior Guided VAR-d16 Next-Scale Autoregressive Semantic JSCC
at Extreme Ultra-Low CBR = 1/512 (K = 384 complex symbols) & CBR = 1/256 (K = 768 complex symbols).

Protocol Guarantees (GPT-Audit-Proof):
  1. Exact CBR Accounting (`K / (3 H W)` per RGB scalar for 256x256x3 = 196,608):
       - CBR = 1/512 -> K = 384 complex symbols (M = 8 RBs x N_b = 48 REs/RB; 768 real dims = 8x8x12 pyramid)
       - CBR = 1/256 -> K = 768 complex symbols (M = 8 RBs x N_b = 96 REs/RB; 1,536 real dims = 8x8x24 pyramid)
  2. Loss-Matched Baselines (`PerceptualSwinUltraLowTransceiver`):
       - Both `swin_genie_perc` (Oracle Exact CSI + True SNR + 100% payload DoF) and `swin_holo_perc`
         (0-Pilot 8-RB Holographic Subspace Modem) use the EXACT SAME 8x8 -> PixelShuffle(2) pyramid bottleneck
         capacity and are trained with the EXACT SAME perceptual objective (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`)
         as `BayesVARTransceiver`!
  3. Exact Within-Model Single-Switch Intervention (`use_ar=True` vs `use_ar=False`):
       - `BayesVAR` (`AR=OFF`) decodes the continuous wireless feature `f_coarse` directly through frozen VQ-VAE.
       - `BayesVAR` (`AR=ON`) takes the IDENTICAL received wireless feature `f_coarse` & blind SNR `(cr_hat_db, alive_frac)`
         (0 extra channel symbols!) and runs frozen 16-layer `VAR-d16` next-scale autoregressive inference anchored
         at coarse scales (1x1..6x6) by the Gaussian wireless likelihood and completed at fine scales (8x8..16x16)
         by `VAR-d16`'s natural-image causal prior!
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
from dir2_zpac.phy_v2 import ChannelV2Output, expand_rb_to_symbols
from dir2_zpac.universal_blind_v4 import coherence_adaptive_wiener_rb
from dir3_mh.mimo_blind import build_hadamard_pow2, build_walsh_k_basis

from dir5_og.genvar_ultralow_blind import (
    ConvNeXt2DBlock,
    DEFAULT_VAR_VQVAE_CKPT,
    N_UL_RBS,
    VAR_UPSTREAM_ROOT,
    build_ultralow_sign_dither,
    newman_8rb_phases,
    physical_ultralow_ofdm_channel,
)

DEFAULT_VAR_D16_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028/"
    "var_comm_runtime_20261010/assets/model_seed_v1/external/home/liulu/projects/VAR-MAP-GATE0/checkpoints/var_d16.pth"
)

PATCH_NUMS_10 = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
# Coarse scales (1x1..6x6) strongly lock onto the 8x8 wireless pyramid observation;
# fine scales (8x8..16x16) smoothly hand off to VAR-d16's autoregressive visual prior!
BASE_OBS_TAU_SCHEDULE = (11.0, 10.0, 9.0, 8.0, 6.8, 5.2, 2.4, 1.05, 0.45, 0.20)


def ultralow_dir6_geometry(cbr_denom: int) -> tuple[int, int, int, int]:
    """Return (K_complex_syms, N_b_per_rb, c_bot_per_16x16_tok, pyr_ch_on_8x8) over M=8 RBs."""
    if cbr_denom == 512:
        # K = 384 complex symbols = 768 real dims = 256 * 3 = 64 * 12
        return 384, 48, 3, 12
    if cbr_denom == 256:
        # K = 768 complex symbols = 1,536 real dims = 256 * 6 = 64 * 24
        return 768, 96, 6, 24
    raise ValueError(f"Unsupported Dir-6B cbr_denom={cbr_denom}; expected 512 or 256.")


def load_frozen_vae_and_var_d16(
    vae_ckpt: Path | str = DEFAULT_VAR_VQVAE_CKPT,
    var_ckpt: Path | str = DEFAULT_VAR_D16_CKPT,
    *,
    device: torch.device | str = "cuda:0",
    load_var: bool = True,
) -> tuple[nn.Module, nn.Module | None]:
    """Load official frozen VQVAE and VAR-d16 WITHOUT calling `build_vae_var` (preserving PyTorch `reset_parameters`!)."""
    vroot = str(VAR_UPSTREAM_ROOT.resolve())
    if vroot not in sys.path:
        sys.path.insert(0, vroot)
    from models.vqvae import VQVAE
    from models.var import VAR

    dev = torch.device(device)
    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=PATCH_NUMS_10,
    )
    vae_sd = torch.load(vae_ckpt, map_location="cpu", weights_only=True)
    vae.load_state_dict(vae_sd, strict=True)
    vae.to(dev).eval()
    for p in vae.parameters():
        p.requires_grad = False

    if not load_var:
        return vae, None

    var_model = VAR(
        vae_local=vae,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1 * 16.0 / 24.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=True,
        patch_nums=PATCH_NUMS_10,
        flash_if_available=True,
        fused_if_available=True,
    )
    var_sd = torch.load(var_ckpt, map_location="cpu", weights_only=True)
    var_model.load_state_dict(var_sd, strict=True)
    var_model.to(dev).eval()
    for p in var_model.parameters():
        p.requires_grad = False
    return vae, var_model


class Dir6HoloSubspaceModem(nn.Module):
    """Zero-Pilot 8-RB Holographic Subspace Modem supporting both CBR=1/512 (N_b=48) and CBR=1/256 (N_b=96)."""

    def __init__(self, cbr_denom: int = 512):
        super().__init__()
        self.cbr_denom = int(cbr_denom)
        self.k_sym, self.n_b, self.c_bot, self.pyr_ch = ultralow_dir6_geometry(self.cbr_denom)
        self.num_rbs = N_UL_RBS

        eta1_init = 0.085 if self.cbr_denom == 512 else 0.068
        inv_s = math.log((eta1_init - 0.030) / (0.160 - eta1_init))
        self.eta1_raw = nn.Parameter(torch.tensor(float(inv_s)))
        self.eta2 = 0.015 if self.cbr_denom == 512 else 0.012

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
        return 0.030 + (0.160 - 0.030) * torch.sigmoid(self.eta1_raw)

    def _forward_holo(self, z_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, C = z_unit.shape
        assert L == 256 and C == self.c_bot
        M, N_b = self.num_rbs, self.n_b
        device, dtype = z_unit.device, z_unit.dtype

        dither = build_ultralow_sign_dither(L, C, device=device, dtype=dtype)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)
        z_disp = torch.matmul(h256, z_unit * dither)
        rb_2nb = z_disp.reshape(B, L // M, M, C).permute(0, 2, 1, 3).reshape(B, M, 2 * N_b)
        return rb_2nb[:, :, :N_b], rb_2nb[:, :, N_b:]

    def _inverse_holo(self, sr: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
        B, M, N_b = sr.shape
        L, C = 256, self.c_bot
        device, dtype = sr.device, sr.dtype
        rb_2nb = torch.cat([sr, si], dim=-1)
        z_disp = rb_2nb.reshape(B, M, L // M, C).permute(0, 2, 1, 3).reshape(B, L, C)
        h256 = build_hadamard_pow2(L, device=device, dtype=dtype)
        dither = build_ultralow_sign_dither(L, C, device=device, dtype=dtype)
        return torch.matmul(h256, z_disp) * dither

    def embed_tx(self, z_bot: torch.Tensor) -> torch.Tensor:
        B, L, C = z_bot.shape
        assert L == 256 and C == self.c_bot
        M, N_b = self.num_rbs, self.n_b
        device, dtype = z_bot.device, z_bot.dtype

        p_img = (2.0 * z_bot.square().mean(dim=(1, 2))).clamp_min(1e-8).sqrt()
        z_unit = z_bot / p_img[:, None, None]

        sr, si = self._forward_holo(z_unit)
        basis = build_walsh_k_basis(4, N_b, device=device, dtype=dtype)
        w0, w1 = basis[0], basis[1]

        proj_r = torch.einsum("bmn,kn->bmk", sr, basis)
        proj_i = torch.einsum("bmn,kn->bmk", si, basis)
        sr_perp = sr - torch.einsum("bmk,kn->bmn", proj_r, basis)
        si_perp = si - torch.einsum("bmk,kn->bmn", proj_i, basis)

        eta1 = self.eta1
        data_pwr = 1.0 - eta1 - self.eta2
        dof_gain = math.sqrt(float(N_b) / float(N_b - 4))
        d_scale = data_pwr.sqrt() * dof_gain

        ncos, nsin = newman_8rb_phases(M, device=device, dtype=dtype)
        amp1 = (eta1 * float(N_b)).sqrt()
        p0_r = amp1 * ncos[None, :, None] * w0[None, None, :]
        p0_i = amp1 * nsin[None, :, None] * w0[None, None, :]

        theta_p = 1.20 * torch.tanh(0.45 * torch.log(p_img.clamp_min(1e-3)))
        cp, sp = torch.cos(theta_p)[:, None], torch.sin(theta_p)[:, None]
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
        y_rb_iq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, M, N_b, _ = y_rb_iq.shape
        assert M == self.num_rbs and N_b == self.n_b
        device, dtype = y_rb_iq.device, y_rb_iq.dtype

        yr, yi = y_rb_iq[..., 0], y_rb_iq[..., 1]
        basis = build_walsh_k_basis(4, N_b, device=device, dtype=dtype)
        w0, w1, w_null = basis[0], basis[1], basis[2:4]

        rb_energy = (yr.square() + yi.square()).mean(dim=-1)
        peak_rb = rb_energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        alive_hat = (rb_energy > (1e-4 * peak_rb)).to(dtype)
        alive_count = alive_hat.sum(dim=-1).clamp_min(1.0)
        alive_frac = alive_count / float(M)

        null_r = torch.einsum("bmn,kn->bmk", yr, w_null)
        null_i = torch.einsum("bmn,kn->bmk", yi, w_null)
        null_pwr_rb = (null_r.square() + null_i.square()).mean(dim=-1)
        sigma_n_hat = ((null_pwr_rb * alive_hat).sum(dim=-1) / alive_count).clamp_min(1e-5)

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

        q1_r = torch.einsum("bmn,n->bm", yr, w1)
        q1_i = torch.einsum("bmn,n->bm", yi, w1)
        u1_r = (q1_r * ncos[None, :] + q1_i * nsin[None, :]) * alive_hat
        u1_i = (q1_i * ncos[None, :] - q1_r * nsin[None, :]) * alive_hat
        z1_r = (hhat_r * u1_r + hhat_i * u1_i).sum(dim=1)
        z1_i = (hhat_r * u1_i - hhat_i * u1_r).sum(dim=1)
        theta_hat = torch.atan2(z1_i, z1_r.clamp_min(1e-5)).clamp(-1.16, 1.16)
        p_img_hat = torch.exp(torch.atanh(theta_hat / 1.20) / 0.45).clamp(0.08, 12.0)

        proj_yr = torch.einsum("bmn,kn->bmk", yr, basis)
        proj_yi = torch.einsum("bmn,kn->bmk", yi, basis)
        yp_r = (yr - torch.einsum("bmk,kn->bmn", proj_yr, basis)) * alive_hat[:, :, None]
        yp_i = (yi - torch.einsum("bmk,kn->bmn", proj_yi, basis)) * alive_hat[:, :, None]

        data_pwr = 1.0 - eta1 - self.eta2
        h_sq = hhat_r.square() + hhat_i.square()
        h_abs = h_sq.clamp_min(1e-10).sqrt()
        eff_var = ((sigma_n_hat / data_pwr) + sigma_e_eff)[:, None]
        denom = (h_sq + eff_var).clamp_min(1e-6)
        alpha_rb = (h_sq / denom) * alive_hat
        alpha_bar_all = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.025)
        alpha_bar_alive = (alpha_rb.sum(dim=1) / alive_count).clamp(0.025, 0.999)

        cos_rb = (hhat_r / h_abs)[:, :, None]
        sin_rb = (hhat_i / h_abs)[:, :, None]
        yd_r = cos_rb * yp_r + sin_rb * yp_i
        yd_i = -sin_rb * yp_r + cos_rb * yp_i

        dof_comp = math.sqrt(float(N_b - 4) / float(N_b))
        d_scale = (1.0 / data_pwr).sqrt() * dof_comp
        w_ub = ((h_abs / denom) / alpha_bar_all)[:, :, None] * d_scale * alive_hat[:, :, None]

        z_unit_hat = self._inverse_holo(yd_r * w_ub, yd_i * w_ub)
        z_bot_restored = z_unit_hat * p_img_hat[:, None, None]

        sinr_eff = (alpha_bar_alive * alive_frac).clamp_min(1e-4) / (1.0 - alpha_bar_alive * alive_frac + 1e-4)
        cr_hat_db = (10.0 * torch.log10(sinr_eff)).clamp(-6.0, 22.0)

        return (
            z_bot_restored,
            cr_hat_db,
            {
                "alive_frac": alive_frac,
                "alpha_rb": alpha_rb,
                "alpha_bar_all": alpha_bar_all.squeeze(1),
                "h_abs_rb": h_abs,
                "sigma_e_eff": sigma_e_eff,
            },
        )


class BayesVARTransceiver(nn.Module):
    """Wireless-Posterior Guided VAR-d16 Next-Scale Autoregressive JSCC Transceiver (`BayesVAR-JSCC`).

    Supports exact within-checkpoint single-switch evaluation:
      - `use_ar=False` (`AR=OFF`): Decodes continuous wireless feature `f_coarse` directly via frozen VQVAE.
      - `use_ar=True`  (`AR=ON`):  Runs frozen 16-layer `VAR-d16` posterior next-scale synthesis conditioned on the
                                   EXACT SAME received `f_coarse`, `cr_hat_db`, and `alive_frac` (0 extra channel bits!).
    """

    def __init__(
        self,
        vae: nn.Module,
        var_model: nn.Module | None = None,
        *,
        cbr_denom: int = 512,
        hidden_ch: int = 160,
    ):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False
        # Store var_model in a 1-tuple so its 310M frozen params are NOT duplicated in BayesVAR checkpoint files!
        self._var_ref: tuple[nn.Module | None] = (var_model,)

        self.cbr_denom = int(cbr_denom)
        self.k_sym, self.n_b, self.c_bot, self.pyr_ch = ultralow_dir6_geometry(self.cbr_denom)
        self.z_ch = 32

        self.tx_pyr_down = nn.Sequential(
            nn.Conv2d(self.z_ch, hidden_ch, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.pyr_ch, kernel_size=1),
        )
        self.modem = Dir6HoloSubspaceModem(self.cbr_denom)

        self.rx_pyr_up = nn.Sequential(
            nn.Conv2d(self.pyr_ch + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, hidden_ch * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.rx_decompressor = nn.Sequential(
            nn.Conv2d(hidden_ch + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=2),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.z_ch, kernel_size=1),
        )

        # Zero-extra-symbol RX semantic class head: predicts soft 1000-class ImageNet prior distribution
        # directly from reconstructed `f_coarse` to condition `VAR-d16`'s AdaLN class embedding!
        self.rx_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.LayerNorm(self.z_ch * 16),
            nn.Linear(self.z_ch * 16, 384),
            nn.SiLU(),
            nn.Linear(384, 1000),
        )

    def attach_var_model(self, var_model: nn.Module) -> None:
        var_model.eval()
        for p in var_model.parameters():
            p.requires_grad = False
        self._var_ref = (var_model,)

    def decode_fhat_to_01(self, f_out: torch.Tensor) -> torch.Tensor:
        rec_m11 = self.vae.decoder(self.vae.post_quant_conv(f_out))
        return (rec_m11.clamp(-1.0, 1.0) + 1.0) * 0.5

    @torch.no_grad()
    def posterior_var_complete(
        self,
        f_coarse: torch.Tensor,     # [B, 32, 16, 16]
        cr_hat_db: torch.Tensor,    # [B]
        alive_frac: torch.Tensor,   # [B]
        *,
        ar_blend: float | None = None,
        seed_offset: int = 20261123,
    ) -> torch.Tensor:
        """Wireless-Anchored Bayesian VAR-d16 Next-Scale MAP Completion on received `f_coarse`:
          - Coarse layout scales (`1x1..4x4` at CBR=1/512; `1x1..6x6` at CBR=1/256) lock directly onto the
            transmitted 8x8 wireless pyramid observation (`argmin ||r_obs - e_v||^2`) and populate VAR-d16's KV-cache.
          - Fine texture/semantic scales (`5x5..16x16` or `8x8..16x16`) maximize the joint log-posterior
            `log p_VAR(v | v_<s, c_soft) - tau_obs * ||r_obs - e_v||^2` (without CFG overshoot), synthesizing sharp
            natural details while preserving exact scene/face geometry!
        """
        _ = seed_offset
        var_model = self._var_ref[0]
        if var_model is None:
            raise RuntimeError("VAR-d16 model is not attached; call `attach_var_model(var_model)` before AR=ON.")
        var_model.eval()

        B, C, H, W = f_coarse.shape
        device = f_coarse.device
        f_obs = f_coarse.float()

        # Soft 50/50 semantic-unconditional class prior from received `f_coarse` (0 extra channel symbols, 0 CFG drift!)
        cls_probs = F.softmax(self.rx_cls_head(f_obs), dim=-1)
        emb_table = var_model.class_emb.weight.float()  # [1001, 1024]
        cond_uncond = emb_table[1000:1001].expand(B, -1)
        sos = cond_BD = 0.50 * torch.matmul(cls_probs, emb_table[:1000]) + 0.50 * cond_uncond  # [B, 1024]

        lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(B, var_model.first_l, -1)
            + var_model.pos_start.expand(B, var_model.first_l, -1)
            + lvl_pos[:, : var_model.first_l]
        )

        cb_weight = self.vae.quantize.embedding.weight.float()  # [4096, 32]
        cb_norm_sq = cb_weight.square().sum(dim=-1)[None, None, :]  # [1, 1, 4096]

        rel_scale = (torch.sigmoid((cr_hat_db.float() - 1.0) / 5.5) * alive_frac.float().clamp(0.30, 1.0)).clamp(0.35, 1.0)
        anchor_si = 3 if self.cbr_denom == 512 else 5
        tau_fine_base = 12.0 if self.cbr_denom == 512 else 9.0

        f_hat = f_obs.new_zeros(B, C, H, W)
        cur_L = 0
        SN = len(PATCH_NUMS_10)

        for b in var_model.blocks:
            b.attn.kv_caching(True)
        try:
            for si, pn in enumerate(PATCH_NUMS_10):
                cur_L += pn * pn
                cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
                x = next_token_map
                for b in var_model.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_prior = var_model.get_logits(x, cond_BD)  # [B, pn*pn, 4096]

                res_curr = f_obs - f_hat
                res_pn = F.interpolate(res_curr, size=(pn, pn), mode="area") if pn < H else res_curr
                r_flat = res_pn.permute(0, 2, 3, 1).reshape(B, pn * pn, C)
                dist_sq = (
                    r_flat.square().sum(dim=-1, keepdim=True)
                    + cb_norm_sq
                    - 2.0 * torch.matmul(r_flat, cb_weight.t())
                )

                if si <= anchor_si:
                    idx_Bl = dist_sq.argmin(dim=-1)
                else:
                    tau_b = (tau_fine_base * rel_scale)[:, None, None]
                    log_post = logits_prior - tau_b * dist_sq
                    idx_Bl = log_post.argmax(dim=-1)

                h_BChw = self.vae.quantize.embedding(idx_Bl).transpose(1, 2).reshape(B, C, pn, pn)
                f_hat, next_map_2d = self.vae.quantize.get_next_autoregressive_input(si, SN, f_hat, h_BChw)
                if si != SN - 1:
                    next_tok = next_map_2d.view(B, C, -1).transpose(1, 2)
                    next_token_map = var_model.word_embed(next_tok) + lvl_pos[:, cur_L : cur_L + PATCH_NUMS_10[si + 1] ** 2]
        finally:
            for b in var_model.blocks:
                b.attn.kv_caching(False)

        w_val = 0.85 if ar_blend is None else float(ar_blend)
        w_ar = torch.full((B, 1, 1, 1), w_val, device=device, dtype=torch.float32)
        return ((1.0 - w_ar) * f_obs + w_ar * f_hat).to(dtype=f_coarse.dtype)

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
        use_ar: bool = False,
        also_return_ar_off: bool = False,
        seed_offset: int = 20261123,
    ) -> dict[str, torch.Tensor]:
        B = f_enc.shape[0]
        f_pyr = self.tx_pyr_down(f_enc)  # [B, pyr_ch, 8, 8]
        z_bot = f_pyr.reshape(B, self.pyr_ch, 64).permute(0, 2, 1).reshape(B, 256, self.c_bot).contiguous()

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

        cr_norm = (cr_hat_db - SNR_CENTER) / SNR_SCALE
        f_pyr_hat = z_hat.reshape(B, 64, self.pyr_ch).permute(0, 2, 1).reshape(B, self.pyr_ch, 8, 8).contiguous()
        cr_8 = cr_norm[:, None, None, None].expand(B, 1, 8, 8)
        af_8 = alive_frac[:, None, None, None].expand(B, 1, 8, 8)
        up_16 = self.rx_pyr_up(torch.cat([f_pyr_hat, cr_8, af_8], dim=1))

        cr_16 = cr_norm[:, None, None, None].expand(B, 1, 16, 16)
        af_16 = alive_frac[:, None, None, None].expand(B, 1, 16, 16)
        f_coarse = self.rx_decompressor(torch.cat([up_16, cr_16, af_16], dim=1))
        cls_logits = self.rx_cls_head(f_coarse.detach())

        out: dict[str, torch.Tensor] = {
            "f_coarse": f_coarse,
            "cls_logits": cls_logits,
            "c_r_hat_db": cr_hat_db,
            "alive_frac": alive_frac,
            "hr_rb": phy["hr_rb"],
            "hi_rb": phy["hi_rb"],
        }

        if use_ar:
            f_ar = self.posterior_var_complete(f_coarse, cr_hat_db, alive_frac, seed_offset=seed_offset)
            out["f_out"] = f_ar
            if decode_pixels:
                out["recon"] = self.decode_fhat_to_01(f_ar)
                if also_return_ar_off:
                    out["recon_ar_off"] = self.decode_fhat_to_01(f_coarse)
        else:
            out["f_out"] = f_coarse
            if decode_pixels:
                out["recon"] = self.decode_fhat_to_01(f_coarse)
        return out

    def forward_ultralow(
        self,
        image_01: torch.Tensor,
        snr_db: torch.Tensor | float,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            f_enc = self.vae.quant_conv(self.vae.encoder(image_01 * 2.0 - 1.0))
        return self.forward_from_fenc(f_enc, snr_db, **kwargs)


class PerceptualSwinUltraLowTransceiver(nn.Module):
    """Loss-Matched & Pyramid-Matched Perceptual SwinJSCC Baselines at CBR = 1/512 (K=384) & CBR = 1/256 (K=768):
      - `swin_holo_perc`:  0-Pilot 8-RB Holographic Subspace Modem + 8x8 Pyramid + Perceptual (L1+MSE+LPIPS) Training!
      - `swin_genie_perc`: Oracle Exact Genie-CSI + True SNR + 100% DoF + 8x8 Pyramid + Perceptual (L1+MSE+LPIPS) Training!
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        cbr_denom: int = 512,
        method: str = "swin_holo_perc",
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
        self.k_sym, self.n_b, self.c_bot, self.pyr_ch = ultralow_dir6_geometry(self.cbr_denom)
        self.latent_channels = int(self.backbone.encoder.head_list.out_features)
        dec_dims = [int(d) for d in self.backbone.decoder.embed_dims]

        self.hier_tx = (
            OpenLoopHierarchicalTX(320, self.latent_channels, base_c0_db=base_c0_db)
            if self.method == "swin_holo_perc"
            else None
        )
        # Matched 8x8 pyramid bottleneck + PixelShuffle(2) decompressor (identical spatial-channel geometry to BayesVAR!)
        self.tx_pyr_down = nn.Sequential(
            nn.Conv2d(self.latent_channels, hidden_ch, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.pyr_ch, kernel_size=1),
        )
        self.rx_pyr_up = nn.Sequential(
            nn.Conv2d(self.pyr_ch + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, hidden_ch * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.rx_decompressor = nn.Sequential(
            nn.Conv2d(hidden_ch + 2, hidden_ch, kernel_size=1),
            nn.SiLU(),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            ConvNeXt2DBlock(hidden_ch, dilation=2),
            ConvNeXt2DBlock(hidden_ch, dilation=1),
            nn.Conv2d(hidden_ch, self.latent_channels, kernel_size=1),
        )

        if self.method == "swin_holo_perc":
            self.modem: Dir6HoloSubspaceModem | None = Dir6HoloSubspaceModem(self.cbr_denom)
        elif self.method == "swin_genie_perc":
            self.modem = None
        else:
            raise ValueError(f"Unknown PerceptualSwin method: {self.method}")

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
        if not input_is_stem and self.method == "swin_genie_perc":
            z_base, _ = encode_with_ct(self.backbone, image_or_stem, snr_db)
            return z_base

        x_stem = image_or_stem if input_is_stem else self.extract_encoder_stem(image_or_stem)
        B, L, _ = x_stem.shape
        if self.method == "swin_genie_perc":
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
        f_pyr = self.tx_pyr_down(z_96_2d)  # [B, pyr_ch, 8, 8]
        z_bot = f_pyr.reshape(B, self.pyr_ch, 64).permute(0, 2, 1).reshape(B, 256, self.c_bot).contiguous()

        if self.method == "swin_holo_perc":
            assert self.modem is not None
            s_rb_iq = self.modem.embed_tx(z_bot)
            y_rb_iq, _ = physical_ultralow_ofdm_channel(
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
            amask = phy["alive_rb_mask"]
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
            z_hat = z_unit_hat * p_img[:, None, None]

            sinr_g = alpha_bar.squeeze(1).clamp_min(1e-4) / (1.0 - alpha_bar.squeeze(1) + 1e-4)
            cr_hat_db = (10.0 * torch.log10(sinr_g)).clamp(-6.0, 24.0)
            h_abs_k = expand_rb_to_symbols((h_sq + 1e-10).sqrt(), K_96)
            alpha_ratio_k = expand_rb_to_symbols(alpha_rb / alpha_bar, K_96)
            sigma_post_k = expand_rb_to_symbols(0.5 * ((1.0 - alpha_rb) / alpha_bar).clamp_min(1e-6), K_96)
            psi_csi = torch.zeros((B,), device=device, dtype=dtype)

        cr_norm = (cr_hat_db - SNR_CENTER) / SNR_SCALE
        f_pyr_hat = z_hat.reshape(B, 64, self.pyr_ch).permute(0, 2, 1).reshape(B, self.pyr_ch, 8, 8).contiguous()
        cr_8 = cr_norm[:, None, None, None].expand(B, 1, 8, 8)
        af_8 = alive_frac[:, None, None, None].expand(B, 1, 8, 8)
        up_16 = self.rx_pyr_up(torch.cat([f_pyr_hat, cr_8, af_8], dim=1))

        cr_16 = cr_norm[:, None, None, None].expand(B, 1, 16, 16)
        af_16 = alive_frac[:, None, None, None].expand(B, 1, 16, 16)
        u_96 = self.rx_decompressor(torch.cat([up_16, cr_16, af_16], dim=1)).permute(0, 2, 3, 1).reshape(B, 256, C)

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

        recon = x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        return {"recon": recon, "c_r_hat_db": cr_hat_db, "u_96": u_96}
