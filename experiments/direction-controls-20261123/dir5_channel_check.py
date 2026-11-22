"""Independent CPU check of the published 5A transform, symbol accounting, and fast-time Doppler."""
import hashlib
import math
import sys
from study import D2, D3, TEMP, STUDY, dump_once

import torch


def main():
    torch.set_num_threads(2)
    project = TEMP/"producer_snapshots/observed_v1/experiments/dir5-otfs-genvar-20261122"
    sys.path[:0] = [str(project/"src"), *(str(TEMP/"source/experiments"/d/"src") for d in (D2,D3))]
    from dir5_og import otfs_blind as ot
    seed = 20261123
    rng = torch.Generator().manual_seed(seed)
    real = torch.randn(2,24,32,14,generator=rng,dtype=torch.float64)
    imag = torch.randn(real.shape,generator=rng,dtype=torch.float64)
    x = torch.complex(real,imag)
    tf = ot.isfft_2d(x)
    back_error = float((ot.sfft_2d(tf)-x).abs().max())
    energy_error = float((tf.abs().square().sum()/x.abs().square().sum()-1).abs())
    basis = ot.build_hadamard16(torch.device("cpu"),torch.float64)
    orth_error = float((basis.T@basis-torch.eye(16)).abs().max())
    fp = []
    for doppler in (0.,.1,.18):
        # One unit-power occupied tone and one unit-gain ray. Off-tone energy is physical ICI.
        grid = torch.zeros((1,48,256,2),dtype=torch.float64)
        grid[:,:,17,0] = math.sqrt(256)
        gain = torch.tensor([[[1.,0.]]],dtype=torch.float64)
        nu = torch.tensor([[doppler]],dtype=torch.float64)
        y,_ = ot.physical_doubly_selective_waveform_channel(grid,10.,taus=(0,),
                multipath_override=(gain,nu),standard_noise_td_iq=torch.zeros_like(grid))
        energy = y.square().sum(dim=-1)
        carrier_frac = float(energy[:,:,17].sum()/energy.sum())
        predicted = 1. if not doppler else (math.sin(math.pi*doppler)/(256*math.sin(math.pi*doppler/256)))**2
        fp.append({"normalized_doppler":doppler,"measured_off_tone_fraction":1.-carrier_frac,
                   "theoretical_off_tone_fraction":1.-predicted})
        assert abs(carrier_frac-predicted)<1e-10
    assert max(energy_error,back_error,orth_error)<1e-11
    pixels=256*256*3
    data_re=ot.N_SYM*ot.N_SC
    td=ot.N_SYM*(ot.N_SC+ot.N_CP)
    rec={"source_sha256":hashlib.sha256((project/"src/dir5_og/otfs_blind.py").read_bytes()).hexdigest(),
         "transform_size":[24,32,14],"inverse_max_error":back_error,"relative_energy_error":energy_error,
         "walsh_orthogonality_max_error":orth_error,"single_tone_checks":fp,"input_real_rgb_scalars":pixels,
         "active_complex_RE":data_re,"payload_complex_dimensions":ot.N_T_TILES*ot.N_F_TILES*ot.PAYLOAD_MODES,
         "physical_samples_including_CP":td,"cbr_useful_RE_per_RGB_scalar":data_re/pixels,
         "cbr_including_CP_per_RGB_scalar":td/pixels,"cp_fraction_of_useful":ot.N_CP/ot.N_SC,
         "seed":seed,"device":"CPU; synthetic complex128 waveforms; zero image/model inference"}
    dump_once(STUDY/"dir5/channel_check.json",rec)
    print(rec)


if __name__=="__main__":
    main()
