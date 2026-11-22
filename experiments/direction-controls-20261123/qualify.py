"""CPU qualification of the two registered interventions using the real modem implementations."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

from study import D2, D3, D4, STUDY, TEMP, dump_once


def main():
    import torch
    src = TEMP / "source/experiments"
    sys.path[:0] = [str(src / p / "src") for p in (D2, D3, D4)]
    from dir3_mh.holorate_blind import HoloBlindAntiJamModem
    from dir4_rn.noma_semsic_blind import SemSICGrantFreeModem
    torch.manual_seed(17117)
    torch.set_num_threads(2)
    identity = json.loads((STUDY / "source_identity.json").read_text())
    for name, h in identity.items():
        assert hashlib.sha256((TEMP / "source" / name).read_bytes()).hexdigest() == h["run"]
    z = torch.randn(2, 256, 96, dtype=torch.float64)
    holo = HoloBlindAntiJamModem().double()
    errs = {}
    for key in (False, True):
        holo.identity_spatial = key
        real, imag = holo._forward_holo(z)
        round_trip = holo._inverse_holo(real, imag, 96)
        err = float((round_trip-z).abs().max())
        if err > 1e-11:
            raise RuntimeError("Packing should be invertible")
        keep = torch.zeros_like(real)
        keep[:, :8] = 1
        restored = holo._inverse_holo(real*keep, imag*keep, 96)
        expected = z.clone()
        expected[:, :, 24:] = 0
        assert (restored-expected).abs().max() < 1e-11
        errs[str(key)] = err
    modem = SemSICGrantFreeModem()
    # Nonzero deterministic head offsets test effective gradient paths beyond the identity init.
    with torch.no_grad():
        for p in modem.latent_prior.parameters():
            p.add_(torch.randn_like(p)*0.001)
    y = torch.randn(2, 2, 32, 384, 2)
    counts = {}
    for own in (False, True):
        modem.own_denoise_only = own
        modem.zero_grad()
        with patch.object(modem, "_demix_single_stage", wraps=modem._demix_single_stage) as call:
            demixed, zhat = modem.blind_demix_users(y, 3, 96)
            counts[str(own)] = call.call_count
        assert len(demixed) == len(zhat) == 3 and call.call_count == (1 if own else 3)
        loss = sum(item[0].u_restored.square().mean() for item in demixed)
        loss.backward()
        for module in (modem.latent_prior, modem.sic_gate_mlp):
            grad = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            assert 0 < grad < float("inf")
    dump_once(STUDY / "qualification.json", {"case": "CPU engineering check", "spatial_max_error": errs,
              "LMMSE_calls_full_then_local": counts, "local_gate_and_prior_image_path_gradients": True})
    print("PASS: matched packing, permutation, demixing call counts, finite nonzero prior/gate gradients.")


if __name__ == "__main__":
    main()
