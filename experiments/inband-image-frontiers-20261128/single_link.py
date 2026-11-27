"""Shared frozen codec and physical-channel draws for the four single-link questions."""
import hashlib
import math
from dataclasses import replace
from unittest.mock import patch

import torch

from study import BACKBONE, CFG, CHECK, digest
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4, build_walsh3_basis

MODELS = {
    "continuous": ("u_sph", "u_sph_awgn", CFG["continuous"]),
    "qam64": ("dd_qam64_sph", "dd_qam64_sph", "3403b632bb98e2e9a41dc9d291c430a1763ff74e34529a33ffe919adddbc6043"),
}
BLOCKS, WIDTH = 32, 384
BATCH = 8
DEV_SEED = 281099


def codec(variant):
    method, folder, expected = MODELS[variant]
    path = CHECK / "dir2_universal_blind_v4" / folder / "best_v4.pt"
    assert digest(path) == expected
    backbone = load_swinjscc_backbone(BACKBONE, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device="cuda:0")
    model = UniversalBlindTransceiverV4(backbone, method=method, num_rbs=32, base_c0_db=7., hidden_ch=160).cuda()
    state = torch.load(path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("backbone.") for k in missing), (missing, unexpected)
    return model.eval()


def batches(n):
    for start in range(0, n, BATCH):
        yield slice(start, min(n, start+BATCH))


def rgb(images, sl):
    return images[sl].cuda().float().div(255.)


def latents(model, images):
    bank = []
    handle = model.hier_tx.register_forward_hook(lambda module, inputs, output: bank.append(output.detach().cpu()))
    try:
        for sl in batches(len(images)):
            model.encode_blind_tx(model.extract_encoder_stem(rgb(images, sl)))
    finally:
        handle.remove()
    z = torch.cat(bank)
    assert z.shape == (len(images), 256, 96) and torch.isfinite(z).all()
    return z


def tx(model, z):
    return torch.view_as_complex(model.modem.embed_tx(z).contiguous()).reshape(-1, BLOCKS, WIDTH)


def iq(y):
    return torch.view_as_real(y.contiguous()).reshape(-1, BLOCKS*WIDTH, 2)


def power(y):
    return y.abs().square().mean((1, 2))


def source_scale(z):
    return (2*z.square().mean((1, 2))).sqrt()


def theta(scale):
    return 1.25 * torch.tanh(.4 * (scale - 2.82))


def basis():
    return build_walsh3_basis(WIDTH, device="cuda", dtype=torch.float32).to(torch.complex64)


def draws(n, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def gaussian(shape):
        return torch.view_as_complex(torch.randn((*shape, 2), generator=gen)) / math.sqrt(2)
    g = gaussian((n, BLOCKS))
    rho = CFG["siso"]["rho"]
    h = g.clone()
    for m in range(1, BLOCKS):
        h[:, m] = rho*h[:, m-1] + math.sqrt(1-rho*rho)*g[:, m]
    noise = gaussian((n, BLOCKS, WIDTH))
    signs = torch.randint(0, 2, (n,), generator=gen)*2 - 1
    return h, noise, signs


def draw_identity(bank):
    return [hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for v in bank]


def retrieve(model, y):
    """Read the actual native channel estimate after learned residual refinement."""
    obs = iq(y)
    captured = []
    continuous = model.method == "u_sph"
    def channel_hook(module, inputs, output):
        p = inputs[0]
        k = 4 if continuous else 2
        captured.append(torch.complex(p[:, k] + output[:, 0], p[:, k+1] + output[:, 1]))
    handle = model.modem.rb_refiner.register_forward_hook(channel_hook)
    try:
        ch, cr = model.modem.blind_demod_and_equalize(obs, model.latent_channels)
    finally:
        handle.remove()
    assert len(captured) == 1
    return obs, ch, cr, captured[0]


def render(model, state, scale=None):
    obs, ch, cr, _ = state
    if scale is not None:
        restored = ch.u_norm * scale[:, None, None]
        ch = replace(ch, pwr=scale.square(), u_restored=restored, u_mmse_restored=restored)
    with patch.object(model.modem, "blind_demod_and_equalize", return_value=(ch, cr)):
        return model.decode_blind_rx(obs)[0].clamp(0, 1)


def nmse(estimated, truth):
    return (estimated-truth).abs().square().mean(1)/truth.abs().square().mean(1).clamp_min(1e-8)

