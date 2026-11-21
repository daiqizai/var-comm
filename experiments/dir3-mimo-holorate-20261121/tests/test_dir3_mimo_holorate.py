"""Unit tests for Direction 3A (OS-MIMO-JSCC 2x2/4x4) and Direction 3B (HoloRate-JSCC Rateless/Erasure/Anti-Jamming)."""
from __future__ import annotations

from pathlib import Path
import sys
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir3_mh.mimo_blind import (
    OrthogonalSpaceFreqMIMOModem,
    build_walsh_k_basis,
    physical_mimo_channel_forward,
)
from dir3_mh.holorate_blind import (
    HoloBlindAntiJamModem,
    physical_hostile_ofdm_channel,
)


def test_os_mimo_2x2_and_4x4_unit_total_power_and_recovery() -> None:
    torch.manual_seed(42)
    z = torch.randn((3, 256, 96)) * 2.8
    for n_ant, expected_slots in [(2, 6144), (4, 3072)]:
        m_rbs = 32 // n_ant  # Fixed 384 REs per 5G NR PRB!
        modem = OrthogonalSpaceFreqMIMOModem(n_ant=n_ant, num_rbs=m_rbs).eval()
        x_tx = modem.embed_tx(z)
        assert x_tx.shape == (3, n_ant, expected_slots, 2)
        # Total power summed across all Nt antennas per slot must equal 1.000000!
        slot_pwr = (x_tx[:, :, :, 0].square() + x_tx[:, :, :, 1].square()).sum(dim=1).mean(dim=1)
        assert torch.allclose(slot_pwr, torch.ones_like(slot_pwr), atol=1e-5)

        y_rx, _ = physical_mimo_channel_forward(x_tx, snr_db=35.0, num_rbs=m_rbs, n_rx=n_ant, rho_ant=0.5)
        ch_out, cr_hat = modem.blind_demod_and_equalize(y_rx, latent_ch=96)
        assert ch_out.u_restored.shape == (3, 256, 96)
        assert cr_hat.shape == (3,)
        # High-SNR blind MIMO restoration correlation must be > 0.98!
        cos_sim = torch.nn.functional.cosine_similarity(ch_out.u_restored.flatten(1), z.flatten(1), dim=1)
        assert float(cos_sim.min().item()) > 0.98, f"cos_sim={cos_sim}"


def test_holorate_exact_isometry_and_jamming_erasure_excision() -> None:
    torch.manual_seed(99)
    modem = HoloBlindAntiJamModem(num_rbs=32, use_holographic_dispersion=True).eval()
    z = torch.randn((2, 256, 96))
    sr, si = modem._forward_holo(z)
    z_rec = modem._inverse_holo(sr, si, latent_ch=96)
    assert torch.allclose(z, z_rec, atol=1e-5)

    # Test under simultaneous 25% burst erasure (RBs 8..15 dead) + 25% +18 dB hostile jamming (RBs 20..27)!
    s_tx_rb = modem.embed_tx(z * 2.82)
    alive = torch.ones((2, 32))
    alive[:, 8:16] = 0.0
    jam = torch.zeros((2, 32))
    jam[:, 20:28] = 1.0
    y_rx_rb, _ = physical_hostile_ofdm_channel(
        s_tx_rb, snr_db=12.0, alive_rb_mask=alive, jam_rb_mask=jam, jam_inr_db=18.0
    )
    ch_out, cr_hat, r_ch = modem.blind_demod_and_equalize(y_rx_rb, latent_ch=96)
    assert ch_out.u_restored.shape == (2, 256, 96)
    # Erased channels [24..47] must have exact 0 reliability; +18 dB jammed channels [60..83] must have < 0.05 reliability!
    assert float(r_ch[:, 24:48].max().item()) == 0.0
    assert float(r_ch[:, 60:84].max().item()) < 0.05
    # Clean Rayleigh channels [0..23] must have mean reliability > 0.75 (vs <0.05 jammed, 0.0 erased)!
    assert float(r_ch[:, 0:24].mean().item()) > 0.75
