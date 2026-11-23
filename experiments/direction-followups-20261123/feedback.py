"""Replace known channel statistics by delayed feedback available to both terminals in V2."""
import csv
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import torch
from common import D2, OLD, PRIMARY, PRIOR, STUDY, SWIN, SWIN_HASH, TEMP, USER, digest, include_sources, load_state, read, record, table, tensor_hash

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.phy_v2 import simulate_ofdm_ar1_chain

ROOT = STUDY/"feedback"
COMMIT = "7a2d1dec5cc19b958c9ccc68d4894cb8d82595b9"
METHODS = {"rx":"MS_ZPAC","joint":"TR_ZPAC"}
MODES = ("known","constant","history")
SCENARIOS = {"snr10_rho090":(10,.90),"snr4_rho085":(4,.85)}
WIDTH, DELAY = 24, 2


def original(split,file):
    path = f"results/swinjscc_mismatch_zpac_20261120/v2_eval/awgn_{split}/{file}"
    return subprocess.check_output(["git","-C",str(USER/"code/VAR_COMM"),"show",COMMIT+":"+path]).decode()


def specification():
    z = json.loads(original("common500","eval_v2_summary_awgn_common500.json"))
    return {"producer":COMMIT,"weights":{k:z["loaded_checkpoints"][v] for k,v in METHODS.items()},
        "models":"Frozen RX-only and joint TX/RX calibrators, AWGN-pretrained Swin backbone.",
        "scenarios":SCENARIOS,"delay":DELAY,"history_width":WIDTH,"modes":list(MODES),
        "constant":"Long-term 7 dB, nominal correlation 0.85, known delay 2. Same encoded delayed 5-bit CQI in every arm.",
        "history":"Predict long-term SNR and innovation fraction using three statistics of the 24 previously delivered, delayed 5-bit CQIs: mean, variance, and lag-one normalized covariance. The current report remains the original TX input. RX can reproduce the estimate from its feedback records under the original reliable feedback assumption.",
        "fit":"Image-free ridge regression on 4 independent, 192-block Rayleigh trajectories per grid cell; SNR 1/4/7/10/13 dB and rho 0.60/0.75/0.90/0.97, standardized features, unpenalized intercept, 0.01 mean-square penalty. Clip SNR to 1..13 and innovation to 0..1. No holdout images or evaluation channels in fitting.",
        "primary":"At 10 dB compare JOINT minus RX-only with history-derived parameters; also report this difference with known or fixed statistics, each model's change and paired differences of differences.",
        "test":"common500 and Kodak24 separately, prior channel seeds 2001,2002,2003; same four pilot-symbol estimator and complex noise. Only gamma_avg_db and tau_unc supplied to the calibrators change. The original PHY, pilot CSI, raw CQI and RX current effective SNR remain in each event.",
        "intervals":"PSNR and DISTS on source means after averaging three channel draws, 5000 direct paired draws. Source bootstrap and sensitivity to length-eight circular moving-block bootstrap reported separately. Conditional on stationary model and the three trajectories, not a field trial."}


def quantize(cqi):
    return -5+.8*np.round((np.clip(cqi,-5,19.8)+5)/.8).clip(0,31)


def features(x):
    x = np.asarray(x,dtype=float)
    assert x.shape == (WIDTH,)
    mu = x.mean()
    xc = x-mu
    var = (xc*xc).mean()
    ac = (xc[1:]*xc[:-1]).mean()/max(var,1e-8)
    return np.array([mu,var,ac])


def calibrate():
    record(ROOT/"protocol.json",specification())
    xx, yy, seeds = [],[],[]
    for i,gamma in enumerate([1,4,7,10,13]):
        for j,rho in enumerate([.60,.75,.90,.97]):
            for k in range(4):
                key = 6112310+10000*i+100*j+k
                seeds.append(key)
                st = simulate_ofdm_ar1_chain(192,gamma,rho,DELAY,key,warmup=32)
                arr = np.array([s.c_t_raw_db for s in st])
                for pos in range(WIDTH,192,4):
                    xx.append(features(arr[pos-WIDTH:pos]))
                    yy.append([gamma,1-rho**(2*DELAY)])
    x,y = np.asarray(xx),np.asarray(yy)
    mu,sd = x.mean(0),x.std(0)
    x = (x-mu)/sd
    ym = y.mean(0)
    beta = np.linalg.solve(x.T@x/len(x)+.01*np.eye(3), x.T@(y-ym)/len(x))
    record(ROOT/"estimator.json",{"feature_mean":mu.tolist(),"feature_std":sd.tolist(),
          "target_mean":ym.tolist(),"beta":beta.tolist(),"windows":len(x),"channel_seeds":seeds})
    print("Fit on",len(seeds),"independent channel trajectories; no images.",flush=True)


def estimate(x,fit):
    z = (features(x)-fit["feature_mean"])/fit["feature_std"]
    g,t = z@np.asarray(fit["beta"])+fit["target_mean"]
    return float(np.clip(g,1,13)),float(np.clip(t,0,1))


def history_states(n,gamma,rho,seed,fit):
    key = int((seed*104729+round(gamma*10)*97+round(rho*100)*31+DELAY*7)&0x7fffffff)
    assert key not in fit["channel_seeds"]
    physical = simulate_ofdm_ar1_chain(n,gamma,rho,DELAY,key,warmup=32)
    all_steps = simulate_ofdm_ar1_chain(n+32,gamma,rho,0,key,warmup=0)
    q = quantize(np.array([s.c_r_db for s in all_steps]))
    variants = {k:[] for k in MODES}
    predictions = []
    for j,st in enumerate(physical):
        # t is the block whose delayed report has just arrived. Fit inputs end at t-1.
        t = 32+j-DELAY
        assert abs(q[t]-st.c_t_raw_db)<1e-6 and np.array_equal(st.hr_rb,all_steps[t+DELAY].hr_rb)
        g,tau = estimate(q[t-WIDTH:t],fit)
        predictions.append((g,tau))
        variants["known"].append(st)
        variants["constant"].append(replace(st,gamma_avg_db=7.,tau_unc=1-.85**(2*DELAY)))
        variants["history"].append(replace(st,gamma_avg_db=g,tau_unc=tau))
    return variants,predictions,key


@torch.inference_mode()
def evaluate():
    protocol = read(ROOT/"protocol.json")
    assert json.loads(json.dumps(specification()))==protocol
    fit = read(ROOT/"estimator.json")
    configure_runtime_env(cache_root=TEMP/"cache/feedback",num_threads=4)
    seed_everything(20261121)
    configure_torch_determinism()
    assert digest(SWIN)==SWIN_HASH
    bb = load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0")
    models = {}
    for tag,old in METHODS.items():
        cp = protocol["weights"][tag]
        assert digest(cp["path"])==cp["sha256"]
        m = TransceiverZPACv2(bb,method=old.lower(),hidden_ch=160)
        load_state(m,cp["path"])
        models[tag] = m.cuda().eval()
    metric = OfflineVisualEvaluator(device="cuda:0",include_lpips=False)
    checks, estimates, counts = [],[],{}
    for split,n in (("common500",500),("kodak24",24)):
        ds = load_split_u8(split,allow_holdout=True)
        assert ds.is_holdout and len(set(ds.source_ids))==n
        assert read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))==[
            {"source_id":k,"chw_u8_sha256":tensor_hash(x)} for k,x in zip(ds.source_ids,ds.images_u8)]
        targets = ds.images_u8.float()/255.
        stems = torch.cat([models["rx"].extract_encoder_stem(x.cuda()).cpu() for x in targets.split(32)])
        rows = []
        for scenario,(gamma,rho) in SCENARIOS.items():
            for seed in (2001,2002,2003):
                histories,preds,ch_seed = history_states(n,gamma,rho,seed,fit)
                for id_,(g,tau) in zip(ds.source_ids,preds):
                    estimates.append(dict(split=split,scenario=scenario,seed=seed,source_id=id_,
                                          estimated_longterm_snr_db=g,estimated_innovation=tau,
                                          actual_longterm_snr_db=gamma,actual_innovation=1-rho**(2*DELAY)))
                for a in range(0,n,25):
                    b = min(n,a+25)
                    xb,sb = targets[a:b].cuda(),stems[a:b].cuda()
                    eps = make_paired_noise((b-a,256,96),list(range(a,b)),seed,device=torch.device("cuda:0"))
                    for tag,m in models.items():
                        for mode in MODES:
                            out=m.forward_channel_v2(sb,0.,0.,0.,channel_mode="ofdm_fading",
                                ofdm_steps=histories[mode][a:b],standard_noise=eps,equalizer_mode="unbiased_mmse",
                                rate_channels=None,input_is_stem=True)
                            vals=metric.evaluate_batch(xb,out["recon"])
                            for j,id_ in enumerate(ds.source_ids[a:b]):
                                rows.append(dict(split=split,scenario=scenario,seed=seed,source_id=id_,
                                    model=tag,mode=mode,psnr_db=vals["psnr_db"][j],dists=vals["dists"][j]))
                print(split,scenario,seed,"complete",flush=True)
        assert len(rows)==n*2*3*2*3 and all(np.isfinite([r["psnr_db"],r["dists"]]).all() for r in rows)
        table(ROOT/(split+"_frames.csv"),rows)
        original_means=list(csv.DictReader(original(split,f"ofdm_fading_results_awgn_{split}.csv").splitlines()))
        for scenario in SCENARIOS:
            for tag,old in METHODS.items():
                row=next(z for z in original_means if z["scenario"]=="ofdm_"+scenario+"_d2" and z["model"]==old)
                for mk in PRIMARY:
                    actual = np.mean([z[mk] for z in rows if z["scenario"]==scenario and z["model"]==tag and z["mode"]=="known"])
                    checks.append(dict(split=split,scenario=scenario,model=tag,metric=mk,
                                       original=float(row[mk+"_mean"]),new=float(actual)))
        counts[split]=len(rows)
    table(ROOT/"history_estimates.csv",estimates)
    record(ROOT/"evaluation.json",{"counts":counts,"native_replay":checks,"estimator_sha256":digest(ROOT/"estimator.json")})


if __name__=="__main__":
    if sys.argv[1]=="calibrate":
        calibrate()
    else:
        evaluate()
