"""Asset-free CPU unit tests verifying link hygiene, paired noise, AWGN/fading math, and ZPAC zero-lock identity."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import torch
import torch.nn as nn

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.hygiene import verify_no_links
from dir2_zpac.phy_channel import awgn_forward, fading_equalized_forward, make_paired_noise, simulate_ar1_fading_chain
from dir2_zpac.models import DualConditionLinear, ReceiverFrontendCalibrator, ZeroLockedModNetDelta
from dir2_zpac.stats import paired_bootstrap_ci


def test_verify_no_links_detects_clean_and_rejects_links():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        f1 = root / "a.txt"
        f1.write_text("hello", encoding="utf-8")
        res = verify_no_links([root])
        assert res["status"] == "PASS" and res["symlink_count"] == 0 and res["hardlink_count"] == 0

        # Hardlink must fail
        f2 = root / "b.txt"
        f2.hardlink_to(f1)
        failed = False
        try:
            verify_no_links([root])
        except RuntimeError:
            failed = True
        assert failed
        f2.unlink()

        # Symlink must fail
        s1 = root / "s.txt"
        s1.symlink_to(f1)
        failed_sym = False
        try:
            verify_no_links([root])
        except RuntimeError:
            failed_sym = True
        assert failed_sym


def test_zpac_strict_zero_locked_identity_at_matched_snr():
    torch.manual_seed(42)
    B, L, C = 4, 256, 96
    cal = ReceiverFrontendCalibrator(C, "P_zpac", hidden_ch=64).eval()
    # Randomize all internal weights so zero-init alone isn't masking bugs
    for p in cal.parameters():
        if p.ndim >= 2:
            nn.init.normal_(p, 0.0, 0.1)
        else:
            nn.init.uniform_(p, 0.1, 0.5)

    u_rest = torch.randn(B, L, C)
    u_norm = torch.randn(B, L, C) * 0.7
    snr = torch.tensor([1.0, 4.0, 7.0, 13.0])
    sigma0 = 0.5 * torch.pow(10.0, -snr / 10.0)

    u_cal, _ = cal(u_rest, u_norm, snr, snr, sigma0)
    assert torch.equal(u_cal, u_rest), f"Max diff: {(u_cal - u_rest).abs().max().item()}"

    # Mismatched SNR must produce non-zero calibration residual once weights are non-zero
    ct_mis = torch.tensor([13.0, 10.0, 1.0, 4.0])
    u_cal_mis, _ = cal(u_rest, u_norm, ct_mis, snr, sigma0)
    assert float((u_cal_mis - u_rest).abs().max().item()) > 1e-4


def test_dual_condition_linear_zero_init_equivalence():
    orig = nn.Linear(1, 32)
    dual = DualConditionLinear(orig)
    ct = torch.tensor([[13.0], [1.0], [7.0]])
    cr = torch.tensor([[4.0], [10.0], [7.0]])
    ref = orig(cr)
    got = dual(torch.cat([ct, cr], dim=-1))
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-6)


def test_zero_locked_modnet_delta_identity():
    delta_mod = ZeroLockedModNetDelta(num_layers=3, hidden_dim=48, out_dim=32, rank_hidden=16)
    for p in delta_mod.parameters():
        nn.init.normal_(p, 0.0, 0.1)
    cr_norm = torch.tensor([[-0.5], [0.0], [1.0]])
    zeros = torch.zeros_like(cr_norm)
    for idx in range(3):
        d = delta_mod.delta_for_layer(idx, cr_norm, zeros, zeros)
        assert torch.equal(d, torch.zeros_like(d))
    x = torch.randn(3, 16, 32)
    assert torch.equal(delta_mod.modulate_post_head(x, cr_norm, zeros, zeros), x)


def test_awgn_and_fading_shapes_and_paired_noise():
    z = torch.randn(3, 256, 96)
    n1 = make_paired_noise((3, 256, 96), [10, 20, 30], 9101, "cpu")
    n2 = make_paired_noise((2, 256, 96), [20, 30], 9101, "cpu")
    assert torch.equal(n1[1:], n2)

    out = awgn_forward(z, torch.tensor([4.0, 7.0, 10.0]), standard_noise=n1)
    # Unit complex power means mean(z_norm^2) == 0.5 per sample
    assert torch.allclose(out.z_norm.square().mean(dim=(1, 2)), torch.full((3,), 0.5), atol=1e-5)

    steps = simulate_ar1_fading_chain(3, avg_snr_db=10.0, rho_c=0.9, delay_blocks=2, seed=1234)
    fout = fading_equalized_forward(z, steps, standard_noise=n1)
    assert fout.u_restored.shape == z.shape and torch.isfinite(fout.u_restored).all()

    ci = paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0], num_resamples=500, seed=7)
    assert ci["ci95_lo"] <= ci["mean"] <= ci["ci95_hi"]
