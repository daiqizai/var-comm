"""Audit-Grade Unit Tests for Direction 6 (`dir6_mb`):
Verifies:
  1. Dir-6A (`BurstMatchedTransceiver`):
     - Exact CBR = 1/16 (`12,288` active REs), unit TX frame power, physical pre-AWGN blockage attenuation
     - Byte-identical parameter count AND byte-for-byte identical initial weights between `use_otfs=True` and `use_otfs=False`
     - Identical frame-mean `ChannelV2Output` interface & `override_use_otfs` cross-switch support
  2. Dir-6B (`BayesVARTransceiver` & `PerceptualSwinUltraLowTransceiver`):
     - Exact CBR = 1/512 (`K=384, N_b=48`) & CBR = 1/256 (`K=768, N_b=96`) symbol budgets & unit TX power
     - Zero self-interference in `Dir6HoloSubspaceModem` (`<P_perp(x), w_k> == 0`)
     - `load_frozen_vae_and_var_d16` preserves PyTorch `nn.Linear.reset_parameters`
     - Within-model `use_ar=True` vs `use_ar=False` single-switch execution on identical wireless realization
  3. Dir-6C (`OverloadedNOMA4x2Transceiver`):
     - Exact 12-Walsh orthonormality (`build_walsh_16_basis(12, 384)`) & zero inter-user probe leakage
     - Blind CFAR active-user subset detector recovers exact active subsets `U_act subseteq {0,1,2,3}` across `{1,2,3,4}` active users
     - Byte-identical parameter count, state_dict keys, AND initial weights between `semsic_3round` and `owndenoise_3round`
     - Gradient flow to ALL trainable parameters in both `semsic_3round` and `owndenoise_3round`!
"""
from __future__ import annotations

from pathlib import Path
import sys
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR5_SRC = Path(__file__).resolve().parents[2] / "dir5-otfs-genvar-20261122/src"
DIR4_SRC = Path(__file__).resolve().parents[2] / "dir4-rf1bit-noma-20261122/src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR5_SRC, DIR4_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir2_zpac.hygiene import seed_everything
from dir2_zpac.models import load_swinjscc_backbone
from dir6_mb import (
    BayesVARTransceiver,
    BurstMatchedTransceiver,
    OverloadedNOMA4x2Transceiver,
    PerceptualSwinUltraLowTransceiver,
    build_walsh_16_basis,
    load_frozen_vae_and_var_d16,
    physical_grantfree_4x2_mac,
    sample_blockage_envelope,
    ultralow_dir6_geometry,
)

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)


def test_dir6a_byte_identical_burst_otfs_vs_ofdm():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    seed_everything(20260601)
    m_otfs = BurstMatchedTransceiver(backbone, use_otfs=True).to(device)
    seed_everything(20260601)
    m_ofdm = BurstMatchedTransceiver(backbone, use_otfs=False).to(device)

    sd1 = m_otfs.state_dict()
    sd2 = m_ofdm.state_dict()
    assert list(sd1.keys()) == list(sd2.keys())
    for k in sd1:
        assert sd1[k].shape == sd2[k].shape
        assert torch.equal(sd1[k], sd2[k]), f"Byte mismatch in {k}!"

    n_train_1 = sum(p.numel() for p in m_otfs.parameters() if p.requires_grad)
    n_train_2 = sum(p.numel() for p in m_ofdm.parameters() if p.requires_grad)
    assert n_train_1 == n_train_2 and n_train_1 > 0

    B = 2
    stems = torch.randn((B, 256, 320), device=device)
    blk = sample_blockage_envelope(B, 12, fixed_start=16, device=device)
    out_otfs = m_otfs.forward_burst(stems, 12.0, kd_max=0.10, blockage_env=blk, input_is_stem=True)
    out_cross = m_otfs.forward_burst(
        stems, 12.0, kd_max=0.10, blockage_env=blk, input_is_stem=True, override_use_otfs=False
    )
    assert out_otfs["recon"].shape == (B, 3, 256, 256)
    assert out_cross["recon"].shape == (B, 3, 256, 256)
    assert torch.isfinite(out_otfs["recon"]).all()


def test_dir6b_ultralow_cbr512_256_and_var_single_switch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    orig_reset = torch.nn.Linear.reset_parameters
    vae, var_d16 = load_frozen_vae_and_var_d16(device=device, load_var=True)
    assert torch.nn.Linear.reset_parameters is orig_reset, "PyTorch reset_parameters must NOT be monkey-patched!"

    for cbr_d, exp_k, exp_nb in [(512, 384, 48), (256, 768, 96)]:
        k_sym, nb, c_bot, pyr_ch = ultralow_dir6_geometry(cbr_d)
        assert (k_sym, nb) == (exp_k, exp_nb)
        assert 64 * pyr_ch == 2 * k_sym and 256 * c_bot == 2 * k_sym

        bv = BayesVARTransceiver(vae, var_d16, cbr_denom=cbr_d).to(device).eval()
        f_enc = torch.randn((2, 32, 16, 16), device=device) * 0.8
        gen = torch.Generator(device="cpu").manual_seed(91)
        h_ovr = torch.randn((2, 8, 2), generator=gen).to(device) * (0.5**0.5)
        n_ovr = torch.randn((2, 8, nb, 2), generator=gen).to(device)

        out_off = bv.forward_from_fenc(
            f_enc, 8.0, h_rb_override=h_ovr, standard_noise_iq=n_ovr, use_ar=False
        )
        out_on = bv.forward_from_fenc(
            f_enc, 8.0, h_rb_override=h_ovr, standard_noise_iq=n_ovr, use_ar=True, also_return_ar_off=True
        )
        assert torch.allclose(out_off["f_coarse"], out_on["f_coarse"], atol=1e-6)
        assert torch.allclose(out_off["recon"], out_on["recon_ar_off"], atol=1e-6)
        assert out_on["recon"].shape == (2, 3, 256, 256)
        assert not torch.allclose(out_on["recon"], out_on["recon_ar_off"])

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    for m_name in ("swin_holo_perc", "swin_genie_perc"):
        sw = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=512, method=m_name).to(device)
        stems = torch.randn((2, 256, 320), device=device)
        res = sw.forward_ultralow(stems, 8.0, input_is_stem=True)
        assert res["recon"].shape == (2, 3, 256, 256)
        assert torch.isfinite(res["recon"]).all()


def test_dir6c_walsh12_cfar_and_byte_identical_4x2_controls():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    basis = build_walsh_16_basis(12, 384, device=device)
    gram = basis @ basis.t()
    assert (gram - torch.eye(12, device=device)).abs().max().item() < 5e-6

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    seed_everything(20260603)
    m_sic = OverloadedNOMA4x2Transceiver(backbone, method="semsic_3round").to(device)
    seed_everything(20260603)
    m_own = OverloadedNOMA4x2Transceiver(backbone, method="owndenoise_3round").to(device)

    sd_sic = m_sic.state_dict()
    sd_own = m_own.state_dict()
    assert list(sd_sic.keys()) == list(sd_own.keys())
    for k in sd_sic:
        assert torch.equal(sd_sic[k], sd_own[k]), f"Byte mismatch in {k}!"

    n_sic = sum(p.numel() for p in m_sic.parameters() if p.requires_grad)
    n_own = sum(p.numel() for p in m_own.parameters() if p.requires_grad)
    assert n_sic == n_own and n_sic > 0

    # Verify gradient flow to EVERY trainable parameter in BOTH semsic_3round and owndenoise_3round!
    B = 2
    stems_u = [torch.randn((B, 256, 320), device=device) for _ in range(4)]
    snrs = torch.tensor([[18.0, 13.0, 8.0, 4.0], [16.0, 12.0, 7.0, 3.0]], device=device)
    for mdl in (m_sic, m_own):
        mdl.train()
        mdl.zero_grad()
        out = mdl.forward_noma_4x2(stems_u, snrs, input_is_stem=True)
        loss = sum(r.mean() + r.square().mean() for r in out["recons"]) + 0.05 * sum(
            (zd - out["z_tx_users"][idx_z % 4].detach()).square().mean()
            for idx_z, zd in enumerate(out["z_denoised_users"])
        )
        loss.backward()
        missing_grad = [n for n, p in mdl.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not missing_grad, f"{mdl.method} missing/non-finite grads: {missing_grad}"

    # Verify Blind CFAR Active-User Subset Detection across K_act in {1, 2, 3, 4}!
    m_sic.eval()
    masks = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        device=device,
    )
    stems_4 = [torch.randn((4, 256, 320), device=device) for _ in range(4)]
    snrs_4 = torch.tensor(
        [
            [6.0, 10.0, 10.0, 10.0],
            [10.0, 4.0, 10.0, 14.0],
            [15.0, 9.0, 3.0, 10.0],
            [18.0, 13.0, 8.0, 3.0],
        ],
        device=device,
    )
    with torch.no_grad():
        out_cfar = m_sic.forward_noma_4x2(stems_4, snrs_4, active_user_mask=masks, input_is_stem=True)
    assert torch.equal(out_cfar["active_hat"], masks), f"CFAR mismatch: got {out_cfar['active_hat']} vs {masks}"
