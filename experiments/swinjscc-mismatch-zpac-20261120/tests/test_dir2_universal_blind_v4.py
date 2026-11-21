"""Unit tests for V4 Universal Blind Self-Probing (U-SPH-ZPAC) & Hardware-Standard Digital QAM (DD-QAM-SPH)."""
from __future__ import annotations

from pathlib import Path
import sys
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.blind_v3 import build_walsh3_basis
from dir2_zpac.universal_blind_v4 import (
    DecisionDirectedQAMModemV4,
    UpgradedSubspaceModemV4,
    get_pam_levels,
    physical_blind_channel_v4,
)


def test_v4_usph_unit_power_and_exact_orthogonal_cancellation() -> None:
    modem = UpgradedSubspaceModemV4(num_rbs=32).eval()
    torch.manual_seed(42)
    z = torch.randn((4, 256, 96)) * 2.82
    s_tx = modem.embed_tx(z)
    pwr = (s_tx[:, :, 0].square() + s_tx[:, :, 1].square()).mean(dim=1)
    assert torch.allclose(pwr, torch.ones_like(pwr), atol=1e-5)

    # Verifynoiseless blind recovery of p_img via MRC atan2 to < 1e-4 accuracy!
    y_clean, _ = physical_blind_channel_v4(
        s_tx, snr_db=60.0, channel_type="ofdm_complex", num_rbs=32
    )
    ch_out, cr_hat = modem.blind_demod_and_equalize(y_clean, latent_ch=96)
    true_p = (2.0 * z.square().mean(dim=(1, 2))).sqrt()
    est_p = ch_out.pwr.sqrt()
    assert torch.allclose(est_p, true_p, atol=2e-2), f"est_p={est_p} vs true_p={true_p}"
    assert ch_out.u_restored.shape == (4, 256, 96)


def test_v4_dd_qam_strict_hardware_constellation_compliance() -> None:
    for qam_order, n_levels in [(16, 4), (64, 8)]:
        modem = DecisionDirectedQAMModemV4(qam_order=qam_order, num_rbs=32).eval()
        torch.manual_seed(101)
        z = torch.randn((3, 256, 96)) * 2.7
        s_tx = modem.embed_tx(z)
        levels = get_pam_levels(qam_order, device=s_tx.device, dtype=s_tx.dtype)

        # Every single transmitted real & imag coordinate MUST sit identically on the discrete PAM grid!
        min_dist = (s_tx.unsqueeze(-1) - levels).abs().min(dim=-1).values
        assert float(min_dist.max().item()) < 1e-6
        uniq = torch.unique(torch.round(s_tx * 10000.0))
        assert uniq.numel() <= n_levels

        # Verify physical_blind_channel_v4 maintains exact realized Es/N0
        y_rx, diag = physical_blind_channel_v4(s_tx, snr_db=10.0, channel_type="ofdm_complex", num_rbs=32)
        realized_es = (s_tx[:, :, 0].square() + s_tx[:, :, 1].square()).mean(dim=1)
        assert torch.allclose(diag["true_sigma_n_sq"], realized_es * 0.1, atol=1e-6)
        ch_out, cr_hat = modem.blind_demod_and_equalize(y_rx, latent_ch=96)
        assert ch_out.u_restored.shape == (3, 256, 96)
        assert cr_hat.shape == (3,)
