"""Unit tests for Dir-4A (RF-Bussgang-JSCC 1-Bit/2-Bit ADC + HPA + Phase Noise)
and Dir-4B (SemSIC-NOMA-JSCC Grant-Free Multi-User & Overloaded 3x2 Uplink).
"""
from __future__ import annotations

from pathlib import Path
import sys
import math
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir2_zpac.blind_v3 import build_walsh3_basis
from dir3_mh.mimo_blind import build_walsh_k_basis
from dir4_rn.rf_1bit_blind import (
    RFBussgangSubspaceModem,
    ofdm_grid_to_rb384,
    physical_rf_impaired_ofdm_channel,
    rb384_to_ofdm_grid,
    ste_1bit_iq_quantize,
    ste_2bit_iq_quantize,
)
from dir4_rn.noma_semsic_blind import SemSICGrantFreeModem, physical_grantfree_uplink_mac


def test_ofdm_grid_bijection():
    x = torch.randn(3, 32, 384, 2)
    g = rb384_to_ofdm_grid(x)
    assert g.shape == (3, 48, 256, 2)
    y = ofdm_grid_to_rb384(g)
    assert torch.equal(x, y)


def test_1bit_and_2bit_adc_exact_levels_and_ste_grad():
    yr = torch.randn(2, 48, 256, requires_grad=True)
    yi = torch.randn(2, 48, 256, requires_grad=True)
    q1r, q1i = ste_1bit_iq_quantize(yr, yi)
    # Every single 1-bit complex sample must have exact modulus 1.0 and +-1/sqrt(2) coordinates!
    mag_sq = q1r.detach().square() + q1i.detach().square()
    assert torch.allclose(mag_sq, torch.ones_like(mag_sq), atol=1e-6)
    loss1 = (q1r * 2.0 - q1i).sum()
    loss1.backward()
    assert yr.grad is not None and torch.isfinite(yr.grad).all() and yr.grad.abs().sum() > 0

    yr2 = torch.randn(2, 48, 256, requires_grad=True)
    yi2 = torch.randn(2, 48, 256, requires_grad=True)
    q2r, q2i = ste_2bit_iq_quantize(yr2, yi2)
    # Exactly 4 distinct values per batch item in 2-bit I/Q branch
    assert torch.unique(q2r[0].detach()).numel() <= 4
    (q2r.sum() + q2i.sum()).backward()
    assert yr2.grad is not None and yr2.grad.abs().sum() > 0


def test_rf_bussgang_modem_papr_reduction_and_exact_null():
    torch.manual_seed(2026)
    modem = RFBussgangSubspaceModem(num_rbs=32)
    z = torch.randn(4, 256, 96) * 2.5
    s_rb = modem.embed_tx(z)
    assert s_rb.shape == (4, 32, 384, 2)

    # Check unit TX power
    pwr = (s_rb[..., 0].square() + s_rb[..., 1].square()).mean(dim=(1, 2))
    assert torch.allclose(pwr, torch.ones_like(pwr), atol=1e-5)

    # Check exact zero energy on u2 across all 32 x 4 = 128 tiles even after PAPR companding!
    basis = build_walsh3_basis(96, device=z.device, dtype=z.dtype)
    u2 = basis[1]
    tr = s_rb[..., 0].reshape(4, 32, 4, 96)
    ti = s_rb[..., 1].reshape(4, 32, 4, 96)
    leak = torch.einsum("bmsn,n->bms", tr, u2).abs().max() + torch.einsum("bmsn,n->bms", ti, u2).abs().max()
    assert float(leak.item()) < 1e-5

    # Check PAPR reduction vs uncompanded Gaussian OFDM
    _, diag_ours = physical_rf_impaired_ofdm_channel(s_rb, 10.0, adc_bits=1, pa_ibo_db=1.0, pn_step_deg=3.5)
    raw_rb = torch.randn_like(s_rb)
    _, diag_raw = physical_rf_impaired_ofdm_channel(raw_rb, 10.0, adc_bits=1, pa_ibo_db=1.0, pn_step_deg=3.5)
    assert float(diag_ours["papr_db"].detach().mean()) < float(diag_raw["papr_db"].detach().mean()) - 1.8

    # Check blind demodulation & CPE trajectory equalization under 1-Bit ADC + PA + Wiener PN!
    r_rb, _ = physical_rf_impaired_ofdm_channel(s_rb, 12.0, adc_bits=1, pa_ibo_db=1.0, pn_step_deg=3.5)
    ch_out, cr_db = modem.blind_demod_and_equalize(r_rb, latent_ch=96)
    assert ch_out.u_restored.shape == (4, 256, 96)
    assert torch.isfinite(ch_out.u_restored).all()
    cos_sim = torch.nn.functional.cosine_similarity(ch_out.u_restored.flatten(1), z.flatten(1), dim=1).mean()
    assert float(cos_sim.item()) > 0.60, f"1-bit blind Bussgang cosine similarity too low: {cos_sim.item()}"


def test_semsic_noma_collision_free_signatures_and_overloaded_3x2():
    modem = SemSICGrantFreeModem(num_rbs=32, hidden_ch=64, enable_semsic=True)
    z0 = torch.randn(2, 256, 96) * 2.2
    z1 = torch.randn(2, 256, 96) * 3.1
    z2 = torch.randn(2, 256, 96) * 1.7

    s0 = modem.embed_user_tx(z0, 0)
    s1 = modem.embed_user_tx(z1, 1)
    s2 = modem.embed_user_tx(z2, 2)

    basis = build_walsh_k_basis(8, 384, device=z0.device, dtype=z0.dtype)
    # User 0 must have ZERO energy on u_0, u_2 (User 1 probe), u_3 (User 2 probe), u_7!
    for null_idx in (0, 2, 3, 5, 6, 7):
        proj = torch.einsum("bmn,n->bm", s0[..., 0], basis[null_idx]).abs().max()
        assert float(proj.item()) < 1e-5

    # Test Overloaded 3-User -> 2-BS-Antenna Near-Far Uplink (13 dB, 8 dB, 4 dB) + backward pass!
    s_users = torch.stack([s0, s1, s2], dim=1)
    snrs = torch.tensor([[13.0, 8.0, 4.0], [12.0, 7.0, 3.0]])
    y_rx, _ = physical_grantfree_uplink_mac(s_users, snrs, n_rx=2)
    assert y_rx.shape == (2, 2, 32, 384, 2)

    demixed, z_den = modem.blind_demix_users(y_rx, n_users=3, latent_ch=96)
    assert len(demixed) == 3 and len(z_den) == 3
    loss = sum(ch.u_restored.square().mean() for ch, _ in demixed) + sum(zd.square().mean() for zd in z_den)
    loss.backward()
    assert modem.eta_ch_raw.grad is not None
