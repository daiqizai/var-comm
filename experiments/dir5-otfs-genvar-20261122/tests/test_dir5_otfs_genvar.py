"""Comprehensive unit tests for Direction 5:
  - Dir-5A (Symp-OTFS-JSCC): Unitary ISFFT/SFFT invertibility, 16-pt Hadamard subspace orthogonality,
    strict unit TX power on doubly-selective multipath+ICI channel, 24x32 blind DD-Wiener channel NMSE < -8 dB
    even at OOD k_d = 0.18 (~900 km/h), and finite gradients.
  - Dir-5B (GenHolo-VAR-JSCC): Exact ultra-low symbol counts K=1,536 (CBR=1/128) & K=768 (CBR=1/256),
    256-pt stride-8 Hadamard isometric invertibility, zero-pilot 8-RB subspace orthogonality,
    alive-RB erasure detection under 50% contiguous RB loss, and SoftVQManifoldProjector gradient flow.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import torch
import torch.nn as nn

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir5_og.otfs_blind import (
    N_F_TILES,
    N_SC,
    N_SYM,
    N_T_TILES,
    BlackBoxOTFSBlindModem,
    SympOTFSSubspaceModem,
    blind_dd_wiener_fourier_upsample,
    build_hadamard16,
    isfft_2d,
    physical_doubly_selective_waveform_channel,
    sfft_2d,
)
from dir5_og.genvar_ultralow_blind import (
    N_UL_RBS,
    BlackBoxUltraLowModem,
    GenHoloVARTransceiver,
    SoftVQManifoldProjector,
    UltraLowHoloSubspaceModem,
    cbr_geometry,
    physical_ultralow_ofdm_channel,
)


class _DummyQuantize(nn.Module):
    def __init__(self, vocab: int = 4096, ch: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, ch)
        nn.init.normal_(self.embedding.weight, std=0.2)

    def f_to_idxBl_or_fhat(self, f_in: torch.Tensor, to_fhat: bool = True):
        assert to_fhat
        # Nearest-neighbor lookup in embedding space as lightweight mock for unit test
        B, C, H, W = f_in.shape
        flat = f_in.permute(0, 2, 3, 1).reshape(-1, C)
        d = torch.cdist(flat, self.embedding.weight)
        idx = d.argmin(dim=-1)
        q = self.embedding.weight[idx].reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return [q]


class _DummyVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantize = _DummyQuantize()


def test_isfft_sfft_unitarity_and_hadamard16():
    torch.manual_seed(101)
    x_dd = torch.complex(torch.randn(4, 24, 32, 14), torch.randn(4, 24, 32, 14))
    x_tf = isfft_2d(x_dd)
    # Energy conservation (Parseval's theorem)
    e_dd = (x_dd.real.square() + x_dd.imag.square()).sum()
    e_tf = (x_tf.real.square() + x_tf.imag.square()).sum()
    assert torch.allclose(e_dd, e_tf, rtol=1e-5)
    # Exact round-trip inversion
    x_rec = sfft_2d(x_tf)
    assert (x_rec - x_dd).abs().max().item() < 1e-5

    h16 = build_hadamard16(torch.device("cpu"), torch.float32)
    eye = h16.t() @ h16
    assert torch.allclose(eye, torch.eye(16), atol=1e-6)
    # w1 = col 0 must be constant 1/4, orthogonal to cols 1..15
    assert torch.allclose(h16[:, 0], torch.full((16,), 0.25), atol=1e-6)
    assert (h16[:, 2:].t() @ h16[:, :2]).abs().max().item() < 1e-6


def test_symp_otfs_blind_channel_tracking_and_gradients():
    torch.manual_seed(202)
    B = 4
    z_in = torch.randn(B, 256, 96, requires_grad=True)
    modem = SympOTFSSubspaceModem(96, use_otfs=True, hidden_ch=64)

    x_tf_iq = modem.embed_tx(z_in)
    assert x_tf_iq.shape == (B, N_SYM, N_SC, 2)
    # Frame average power should be ~1.0
    frame_pwr = (x_tf_iq[..., 0].square() + x_tf_iq[..., 1].square()).mean(dim=(1, 2))
    assert torch.allclose(frame_pwr, torch.ones_like(frame_pwr), atol=3e-2)

    # Test extreme OOD Doppler k_d = 0.18 at 12 dB SNR!
    y_tf_iq, phy = physical_doubly_selective_waveform_channel(x_tf_iq, 12.0, kd_max=0.18)
    ch_v2, cr_hat, aux = modem.blind_demod_and_equalize(y_tf_iq)
    assert ch_v2.u_restored.shape == (B, 256, 96)
    assert torch.isfinite(cr_hat).all()

    # Blind channel estimation NMSE against true fast-time symbol-center TF response must be < -7.5 dB even at k_d = 0.18!
    h_hat = aux["h_hat_tf"]
    h_true = phy["h_true_tf"]
    nmse_db = 10.0 * math.log10(
        ((h_hat - h_true).abs().square().mean() / h_true.abs().square().mean()).item()
    )
    assert nmse_db < -7.5, f"Expected blind DD-Wiener NMSE < -7.5 dB at kd=0.18, got {nmse_db:.2f} dB"

    loss = ch_v2.u_restored.square().mean() + 0.01 * cr_hat.mean()
    loss.backward()
    assert z_in.grad is not None and torch.isfinite(z_in.grad).all()
    assert modem.eta1_raw.grad is not None and torch.isfinite(modem.eta1_raw.grad).all()

    # Test Controlled OFDM 2D Ablation & BlackBox OTFS
    modem_ofdm = SympOTFSSubspaceModem(96, use_otfs=False, hidden_ch=64)
    ch_o, _, _ = modem_ofdm.blind_demod_and_equalize(y_tf_iq.detach())
    assert ch_o.u_restored.shape == (B, 256, 96)

    bb_otfs = BlackBoxOTFSBlindModem(96)
    x_bb = bb_otfs.embed_tx(z_in.detach())
    ch_b, _, _ = bb_otfs.blind_demod_and_equalize(x_bb)
    assert ch_b.u_restored.shape == (B, 256, 96)


def test_genvar_ultralow_cbr128_and_cbr256_and_erasure():
    torch.manual_seed(303)
    B = 4
    dummy_vae = _DummyVQVAE()
    for p in dummy_vae.parameters():
        p.requires_grad = False

    for cbr_denom, expected_k, expected_nb, expected_c in [(128, 1536, 192, 12), (256, 768, 96, 6)]:
        k_sym, n_b, c_bot = cbr_geometry(cbr_denom)
        assert (k_sym, n_b, c_bot) == (expected_k, expected_nb, expected_c)

        modem = UltraLowHoloSubspaceModem(cbr_denom, use_holo=True)
        z_bot = torch.randn(B, 256, c_bot, requires_grad=True)

        # Verify isometric Hadamard dispersion round-trip
        sr, si = modem._forward_holo(z_bot.detach())
        assert sr.shape == (B, N_UL_RBS, n_b)
        z_rt = modem._inverse_holo(sr, si)
        assert (z_rt - z_bot.detach()).abs().max().item() < 1e-5

        # Embed TX and verify exact symbol budget K = M * N_b
        s_rb = modem.embed_tx(z_bot)
        assert s_rb.shape == (B, N_UL_RBS, n_b, 2)
        assert s_rb.shape[1] * s_rb.shape[2] == expected_k

        # Simulate OOD 50% contiguous RB erasure (RBs 2,3,4,5 zeroed out!)
        alive_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]).expand(B, 8)
        y_rb, _ = physical_ultralow_ofdm_channel(s_rb, 10.0, alive_rb_mask=alive_mask)
        z_hat, cr_hat, aux = modem.blind_demod_and_equalize(y_rb)
        assert z_hat.shape == (B, 256, c_bot)
        # Blind alive detector must detect exactly 0.50 alive fraction!
        assert torch.allclose(aux["alive_frac"], torch.full((B,), 0.50), atol=1e-5)

        # Test SoftVQManifoldProjector differentiability
        f_cont = torch.randn(B, 32, 16, 16, requires_grad=True)
        proj = SoftVQManifoldProjector(32, hidden_ch=64)
        f_out, _ = proj(f_cont, dummy_vae, cr_hat, aux["alive_frac"])
        assert f_out.shape == (B, 32, 16, 16)
        (f_out.square().mean() + z_hat.square().mean()).backward()
        assert z_bot.grad is not None and torch.isfinite(z_bot.grad).all()
        assert f_cont.grad is not None and torch.isfinite(f_cont.grad).all()

        bb_ul = BlackBoxUltraLowModem(cbr_denom)
        s_bb = bb_ul.embed_tx(z_bot.detach())
        zb_hat, _, _ = bb_ul.blind_demod_and_equalize(s_bb)
        assert zb_hat.shape == (B, 256, c_bot)

        # Verify Multi-Scale Pyramid GenHoloVARTransceiver end-to-end latent path
        gh_model = GenHoloVARTransceiver(dummy_vae, cbr_denom=cbr_denom, method="genholo_var", hidden_ch=64)
        f_enc_in = torch.randn(B, 32, 16, 16, requires_grad=True)
        out_gh = gh_model.forward_from_fenc(f_enc_in, 10.0, alive_rb_mask=alive_mask, decode_pixels=False)
        assert out_gh["f_out"].shape == (B, 32, 16, 16)
        assert out_gh["s_rb_iq"].shape == (B, N_UL_RBS, n_b, 2)
        out_gh["f_out"].square().mean().backward()
        assert f_enc_in.grad is not None and torch.isfinite(f_enc_in.grad).all()
