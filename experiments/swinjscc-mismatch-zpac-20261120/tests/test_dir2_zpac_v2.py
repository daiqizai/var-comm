"""Unit tests for V2 wireless channels (AWGN, Rayleigh, 32-RB OFDM) and MS-ZPAC / TR-ZPAC structural guarantees."""
from __future__ import annotations

from pathlib import Path
import sys
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.phy_v2 import awgn_v2_forward, ofdm_fading_forward, rayleigh_v2_forward, simulate_ofdm_ar1_chain
from dir2_zpac.models_v2 import IQConjugatePairBlock, StageZeroLockedAdapter, UncertaintyAwareTXAdapter


def test_awgn_v2_strict_zero_psi() -> None:
    z = torch.randn((3, 256, 96))
    out = awgn_v2_forward(z, gamma_db=7.0, c_r_db=7.0)
    assert torch.all(out.psi_csi == 0.0)
    assert torch.all(out.phase_unc_iq == 0.0)
    assert torch.allclose(out.alpha_ratio_iq, torch.ones_like(out.alpha_ratio_iq))


def test_ofdm_unbiased_mmse_shapes_and_tau() -> None:
    steps_d0 = simulate_ofdm_ar1_chain(4, avg_snr_db=10.0, rho_c=0.90, delay_blocks=0, seed=11)
    steps_d2 = simulate_ofdm_ar1_chain(4, avg_snr_db=10.0, rho_c=0.90, delay_blocks=2, seed=11)
    assert all(s.tau_unc == 0.0 for s in steps_d0)
    assert all(abs(s.tau_unc - (1.0 - 0.90**4)) < 1e-6 for s in steps_d2)
    z = torch.randn((4, 256, 96))
    out = ofdm_fading_forward(z, steps_d2)
    assert out.u_restored.shape == (4, 256, 96)
    assert torch.all(out.psi_csi > 0.0)


def test_ua_tx_zero_lock_at_tau_zero() -> None:
    ua = UncertaintyAwareTXAdapter(embed_dim=320, latent_ch=96)
    for p in ua.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    ua.eval()
    x_stem = torch.randn((4, 256, 320))
    c_raw = torch.tensor([1.0, 4.0, 7.0, 13.0])
    tau = torch.zeros(4)
    prior = torch.full((4,), 7.0)
    c_eff, g_unc = ua.compute_c_eff(c_raw, tau, prior)
    assert torch.allclose(c_eff, c_raw, atol=0.0)
    assert torch.all(g_unc == 0.0)
    dz = ua.forward_delta_z(x_stem, c_raw, c_eff, prior, tau, g_unc)
    assert torch.all(dz == 0.0)


def test_stage_adapter_zero_lock() -> None:
    adp = StageZeroLockedAdapter(dim=256, cond_dim=6)
    for p in adp.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    adp.eval()
    x = torch.randn((2, 1024, 256))
    phy = torch.randn((2, 3, 16, 16))
    cond = torch.randn((2, 6))
    g0 = torch.zeros((2, 1))
    y = adp(x, 32, 32, phy, cond, g0)
    assert torch.equal(x, y)
