"""Small CPU checks for the two causal rules and the published 5A processing path."""
from pathlib import Path
import numpy as np
import torch
from common import D5, PRIOR, STUDY, TEMP, digest, include_sources, read, record

include_sources()
from rf import modulated_grid
from dir4_rn.rf_1bit_blind import rb384_to_ofdm_grid
from feedback import DELAY, WIDTH, estimate, history_states, quantize


def main():
    torch.set_num_threads(2)
    gen=torch.Generator().manual_seed(6112311)
    r,i=(torch.randn(2,48,256,generator=gen,dtype=torch.float64) for _ in range(2))
    a,b=r.clone(),i.clone()
    a[:,19:]*=6
    b[:,19:]*=6
    delta=read(STUDY/"rf/calibration.json")["delta0"]
    def samples(rr,ii,mode):
        fd=rb384_to_ofdm_grid(modulated_grid(rr,ii,mode,delta))
        return torch.fft.ifft(torch.view_as_complex(fd.contiguous()),dim=-1,norm="ortho")
    causal=(samples(r,i,"causal")[:,:19]-samples(a,b,"causal")[:,:19]).abs().max().item()
    fixed=(samples(r,i,"fixed")[:,:19]-samples(a,b,"fixed")[:,:19]).abs().max().item()
    frame=(samples(r,i,"frame_rms")[:,:19]-samples(a,b,"frame_rms")[:,:19]).abs().max().item()
    first=(samples(r,i,"causal")[:,0]-samples(r,i,"fixed")[:,0]).abs().max().item()
    assert causal<1e-13 and fixed<1e-13 and first<1e-13 and frame>.01
    fit=read(STUDY/"feedback/estimator.json")
    from dir2_zpac.phy_v2 import simulate_ofdm_ar1_chain
    variants,orig,seed=history_states(48,7.,.75,2019,fit)
    chain=simulate_ofdm_ar1_chain(80,7.,.75,0,seed,warmup=0)
    q=quantize(np.array([s.c_r_db for s in chain]))
    # The report just received remains the original model input. Perturb it and
    # all future reports to check that none enter the auxiliary statistics.
    max_diff=0.
    for pos in range(48):
        t=32+pos-DELAY
        revised=q.copy()
        revised[t:]=np.flip(revised[t:])+3
        after=estimate(revised[t-WIDTH:t],fit)
        max_diff=max(max_diff,max(abs(x-y) for x,y in zip(orig[pos],after)))
        old=variants["known"][pos]
        for mode in ("history","constant"):
            changed=variants[mode][pos]
            for name in ("hr_rb","hi_rb","hhat_r_rb","hhat_i_rb"):
                assert getattr(old,name) is getattr(changed,name)
            for name in ("sigma_n_sq","sigma_e_sq","c_t_raw_db","c_r_db"):
                assert getattr(old,name)==getattr(changed,name)
    assert max_diff==0
    src=TEMP/"source"/D5/"src/dir5_og/otfs_blind.py"
    active=[l.strip() for l in src.read_text().splitlines() if "self.use_otfs" in l and not l.strip().startswith("#")]
    assert len(active)==3 and any("isfft_2d" in l for l in active) and any("sfft_2d" in l for l in active)
    record(STUDY/"causal_checks.json",{"adc_future_changed_after_symbol_18_max_change_in_earlier_codes":{
        "fixed":fixed,"causal":causal,"full_frame_rms_positive_control":frame},
        "adc_first_symbol_causal_vs_fixed":first,"cqi_current_and_future_changed_max_change_to_estimator":max_diff,
        "cqi_unchanged_physical_arrays_and_noise_variances":True,"synthetic_checks_only":True,
        "waveform_switch_sites":active,"wave_model_source_sha256":digest(src),
        "previous_wave_energy_and_budget_validation_sha256":digest(PRIOR/"dir5/channel_check.json")})
    print("ADC, CQI chronology and waveform switches verified.")


if __name__=="__main__":
    main()
