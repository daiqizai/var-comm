"""Unit tests for V3 CSI-Free / Blind Self-Probing Subspace Modem (SPH-ZPAC)."""
from __future__ import annotations

from pathlib import Path
import sys
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.blind_v3 import SelfProbingSubspaceModem, build_walsh3_basis, physical_blind_channel_forward


def test_walsh3_orthonormality() -> None:
    basis = build_walsh3_basis(384, device=torch.device("cpu"), dtype=torch.float32)
    gram = basis @ basis.T
    assert torch.allclose(gram, torch.eye(3), atol=1e-6)


def test_sph_modem_exact_unit_power_and_null_subspace() -> None:
    modem = SelfProbingSubspaceModem(num_rbs=32).eval()
    z = torch.randn((4, 256, 96)) * 2.5
    s_tx = modem.embed_tx(z)
    assert s_tx.shape == (4, 12288, 2)
    # Per-image average complex power must equal 1.000000 identically!
    pwr = (s_tx[:, :, 0].square() + s_tx[:, :, 1].square()).mean(dim=1)
    assert torch.allclose(pwr, torch.ones_like(pwr), atol=1e-5)

    # Projection onto u2 must be strictly zero on both Re and Im across all 32 RBs!
    u2 = build_walsh3_basis(384, device=torch.device("cpu"), dtype=torch.float32)[1]
    q_re = torch.einsum("bmn,n->bm", s_tx[:, :, 0].reshape(4, 32, 384), u2)
    q_im = torch.einsum("bmn,n->bm", s_tx[:, :, 1].reshape(4, 32, 384), u2)
    assert torch.allclose(q_re, torch.zeros_like(q_re), atol=1e-5)
    assert torch.allclose(q_im, torch.zeros_like(q_im), atol=1e-5)


def test_sph_blind_demod_shapes() -> None:
    modem = SelfProbingSubspaceModem(num_rbs=32).eval()
    z = torch.randn((2, 256, 96)) * 2.8
    s_tx = modem.embed_tx(z)
    y_rx, _ = physical_blind_channel_forward(s_tx, snr_db=10.0, channel_type="ofdm_complex", num_rbs=32)
    ch_out, cr_hat = modem.blind_demod_and_equalize(y_rx, latent_ch=96)
    assert ch_out.u_restored.shape == (2, 256, 96)
    assert cr_hat.shape == (2,)
