"""Reuse the frozen PNOMA projection codec and the measured 4-by-2 receiver."""
import json
import sys
import torch
from study import CFG, PRIOR, SNAP, TMP, digest

sys.path.insert(0,str(TMP / "pnoma_matched_20261127_v1/upstream"))
sys.path.insert(0,str(PRIOR))
import common as prior
from adapter import ReferenceReceiver
from dir6_mb.overloaded_noma_4x2 import physical_grantfree_4x2_mac, build_walsh_16_basis


def load_external():
    assert digest(PRIOR/"adapter.py")=="98b225d33486dcbe73077a4639dceadd6aedaea02886c3ac9c09df47a1044140"
    assert digest(PRIOR/"common.py")=="17bd2bd9cbc03ee263e918314c9e244b4a258e4bd701713d64438ccf61c7933f"
    assert digest(PRIOR/"train/freeze.json")==CFG["prior_pnoma_freeze"]
    prior.verify_inputs()
    return prior.chosen_model().eval().requires_grad_(False)


def waveform(tx,bank,sl,mask=None):
    hr,hi,eps,levels=bank
    return physical_grantfree_4x2_mac(tx,levels[sl].cuda(),n_rx=2,active_user_mask=mask,
          h_mu_override=(hr[sl],hi[sl]),standard_noise_iq=eps[sl])[0]


def reference_stats(y, basis):
    # nr x blocks independent samples per signature under the zero-user Gaussian model.
    proj=torch.matmul(torch.view_as_complex(y.contiguous()),basis.mT.to(torch.complex64))
    energy=proj.abs().square().mean(dim=(1,2))
    return energy[:,2:6]/energy[:,:2].mean(1,keepdim=True).clamp_min(1.e-8)

