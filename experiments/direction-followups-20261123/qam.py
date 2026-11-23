"""V4 QAM power control with a fixed noise floor and a validation-frozen radio gain."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from common import (D2, OLD, PRIMARY, PRIOR, STUDY, SWIN, SWIN_HASH, TEMP, USER,
                    digest, include_sources, load_state, read, record, table, tensor_hash)

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import encode_with_ct, load_swinjscc_backbone
from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4, get_pam_levels, ste_pam_quantize

ROOT = STUDY / "qam"
REV = "65a5a8e14b92eb7fad92eb6e6444418e52d9a073"
EX = "experiments/" + D2
RESULT = "results/swinjscc_mismatch_zpac_20261120/v4_universal_eval/"
MODELS = {"dd":"dd_qam64_sph","blind":"tcom25_qam64"}
HISTORICAL = {"dd":"DD_QAM64_SPH","blind":"TCOM25_QAM64_Blind","pilot":"Pilot4_QAM64_MMSE"}
K, M = 12288, 32


def blob(rel):
    return subprocess.check_output(["git","-C",str(USER/"code/VAR_COMM"),"show",REV+":"+rel])


def specification():
    related=[EX+"/scripts/evaluate_blind_v3.py",EX+"/scripts/evaluate_universal_blind_v4.py",
             EX+"/src/dir2_zpac/universal_blind_v4.py"]
    identity=json.loads(blob(RESULT+"eval_v4_summary_common500.json"))
    return {"producer":REV,"bound_sources":{f:hashlib.sha256(blob(f)).hexdigest() for f in related},
            "weights":{k:identity["loaded_checkpoints"][v] for k,v in MODELS.items()},
            "condition":"64-QAM, correlated OFDM (rho_f=0.85), 10 dB, 32 resource blocks. Both blind methods send 12288 symbols. The reference has 128 additional unit-energy pilots in native units and source RMS available to its decoder.",
            "calibration":"For each frozen transmitter average energy over all transmitted symbols on the same 256 validation images, including the explicit pilots when present. Freeze a single public analog gain g=1/sqrt(E_val) per method. No decoder runs or held-out examples to fit gains.",
            "fixed":"Multiply the native constellation and pilots by method gain g and use common physical complex noise variance 0.1. The receiver can undo the public gain. This preserves constellation cardinality and charges pilot energy and duration. All models apply their own original channel estimation and decoder.",
            "native":"Unchanged frame-relative noise using each frame's native mean data-symbol energy, from producer. Same source, fading and standard noise across arms. No per-image gain in the fixed arm.",
            "primary":"Fixed-floor DD minus explicit 4-pilot and DD minus the local trained blind QAM reference on common500. Also report native contrasts, change in contrasts, Kodak24, both metrics and actual test-power distribution.",
            "seeds":[2001,2002,2003],"batch":50,"methods":list(HISTORICAL),"draws":5000,
            "resampling":"Source mean over three draws, then direct source-paired pointwise 95% percentile intervals with seed 2026112308."}


def producer_functions():
    """Split the exact conventional reference at its wireless interface; retain its pilot estimator."""
    v3=blob(EX+"/scripts/evaluate_blind_v3.py").decode()
    v4=blob(EX+"/scripts/evaluate_universal_blind_v4.py").decode()
    def function(src,name):
        node=next(f for f in ast.parse(src).body if isinstance(f,ast.FunctionDef) and f.name==name)
        return ast.get_source_segment(src,node)
    pilot=function(v4,"forward_pilot4_qam_reference")
    before=pilot.split("    B = xb.shape[0]",1)[1].split("    y_rx, diag =",1)[0]
    after=pilot.split('    sigma_n_sq = diag["true_sigma_n_sq"]  # (B,)',1)[1]
    # These functions are numeric code from the pinned, user-owned Git repository.
    code="\n\n".join(function(v3,name) for name in ("make_paired_iq_noise","make_paired_h_rb"))
    code+="\n\ndef pilot_tx(backbone,xb):\n    qam_order=64\n    B = xb.shape[0]"+before+"    return s_tx, p_img, g0, L, C\n"
    code+="\ndef pilot_rx(backbone,xb,y_rx,sigma_n_sq,h_rb,pilot_seed,p_img,g0,L,C):\n"
    code+="    B,device,dtype=xb.shape[0],xb.device,xb.dtype\n    M=32\n    K=L*C//2\n    N_b=K//M\n"
    code+="    channel_type,pilot_symbols='ofdm_complex',4\n"+after
    namespace=dict(torch=torch,math=math,encode_with_ct=encode_with_ct,
                   get_pam_levels=get_pam_levels,ste_pam_quantize=ste_pam_quantize)
    exec(compile(code,"pinned_v4_pilot_and_noise","exec"),namespace)
    return namespace


def transmit(name,bb,models,func,x,stems):
    if name=="pilot":
        wave,*aux=func["pilot_tx"](bb,x)
        return wave,aux
    return models[name].encode_blind_tx(stems),[]


@torch.inference_mode()
def run():
    spec=read(ROOT/"protocol.json")
    assert spec==specification() and digest(SWIN)==SWIN_HASH
    code_path=OLD/"source"/EX/"src/dir2_zpac/universal_blind_v4.py"
    assert digest(code_path)==spec["bound_sources"][EX+"/src/dir2_zpac/universal_blind_v4.py"]
    configure_runtime_env(cache_root=TEMP/"cache/qam",num_threads=4)
    seed_everything(20261123)
    configure_torch_determinism()
    bb=load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device="cuda:0").eval()
    models={}
    for name,method in MODELS.items():
        cp=spec["weights"][name]
        assert digest(cp["path"])==cp["sha256"]
        model=UniversalBlindTransceiverV4(bb,method=method)
        load_state(model,cp["path"])
        models[name]=model.cuda().eval()
    func=producer_functions()
    levels=get_pam_levels(64,torch.device("cuda:0"),torch.float32)
    scorer=OfflineVisualEvaluator(device="cuda:0",include_lpips=False)
    native_check,eff_power,count=[],[],{}
    for split in ("val256","common500","kodak24"):
        cal=split=="val256"
        ds=load_split_u8(split,allow_holdout=not cal)
        n=len(ds.source_ids)
        assert n=={"val256":256,"common500":500,"kodak24":24}[split] and ds.is_holdout!=cal
        source_identity=[{"source_id":k,"chw_u8_sha256":tensor_hash(x)} for k,x in zip(ds.source_ids,ds.images_u8)]
        if cal:
            assert source_identity==[{"source_id":r["id"],"chw_u8_sha256":r["sha256"]}
                                     for r in read(STUDY/"rf/calibration_sources.json")]
        else:
            assert source_identity==read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))
            gain=read(ROOT/"frozen_gains.json")["gains"]
        energy={m:[] for m in HISTORICAL}
        rows=[]
        for a in range(0,n,50):
            b=min(n,a+50)
            x=ds.images_u8[a:b].cuda().float()/255
            stem=models["dd"].extract_encoder_stem(x)
            for method in HISTORICAL:
                tx,aux=transmit(method,bb,models,func,x,stem)
                assert tx.shape==(b-a,K,2)
                if a==0:
                    assert (tx.unsqueeze(-1)-levels).abs().amin(-1).amax().item()<1e-5
                e=tx.square().sum(-1).mean(1)
                pilots=128 if method=="pilot" else 0
                packet_es=(K*e.double()+pilots)/(K+pilots)
                energy[method].extend(packet_es.cpu().tolist())
                if cal: continue
                for seed in spec["seeds"]:
                    index=list(range(a,b))
                    h=func["make_paired_h_rb"](M,.85,index,seed,device=x.device)
                    eps=func["make_paired_iq_noise"](K,index,seed,device=x.device)
                    hr,hi=h.repeat_interleave(K//M,dim=1).unbind(-1)
                    re,im=tx.unbind(-1)
                    clean=torch.stack((hr*re-hi*im,hr*im+hi*re),-1)
                    for mode in ("native","fixed"):
                        if mode=="native":
                            var=e*10**(-10/10)
                            y=clean+eps*(.5*var).sqrt()[:,None,None]
                        else:
                            scale=gain[method]
                            y=(scale*clean+math.sqrt(.05)*eps)/scale
                            var=torch.full_like(e,.1/scale**2)
                        if method=="pilot":
                            rec=func["pilot_rx"](bb,x,y,var,h,seed*997+a,*aux)
                        else:
                            rec,_=models[method].decode_blind_rx(y)
                        scores=scorer.evaluate_batch(x,rec)
                        rows.extend(dict(split=split,source_id=ds.source_ids[a+j],seed=seed,method=method,
                                         mode=mode,psnr_db=scores["psnr_db"][j],dists=scores["dists"][j])
                                    for j in range(b-a))
        if cal:
            avg={m:float(np.mean(e)) for m,e in energy.items()}
            record(ROOT/"frozen_gains.json",{"gains":{m:1/math.sqrt(val) for m,val in avg.items()},
                    "mean_packet_es_before_gain":avg,"validation_sources":n,"image_decodes":0,
                    "source_identity":digest(STUDY/"rf/calibration_sources.json")})
            print("Validation gains frozen.",flush=True)
            continue
        for method,values in energy.items():
            for sid,e_i in zip(ds.source_ids,values):
                eff_power.append(dict(split=split,source_id=sid,method=method,native_packet_es=e_i,
                                      fixed_packet_es=e_i*gain[method]**2))
        assert len(rows)==n*18 and all(np.isfinite([r[k] for k in PRIMARY]).all() for r in rows)
        old=list(csv.DictReader(blob(RESULT+f"v4_s3_digital_qam_{split}.csv").decode().splitlines()))
        for method,label in HISTORICAL.items():
            ref=next(z for z in old if z["model"]==label and z["condition"]=="ofdm_10dB_rhof085")
            native=[z for z in rows if z["method"]==method and z["mode"]=="native"]
            for k in PRIMARY:
                native_check.append(dict(split=split,model=method,metric=k,original=float(ref[k+"_mean"]),
                                         new=float(np.mean([z[k] for z in native]))))
        table(ROOT/(split+"_frames.csv"),rows)
        count[split]=len(rows)
        print(split,len(rows),"complete",flush=True)
    table(ROOT/"test_source_power.csv",eff_power)
    record(ROOT/"evaluation.json",{"native_replay":native_check,"logical_scores":count,
        "validation_transmitted_frames":3*256,"frozen_gain_sha256":digest(ROOT/"frozen_gains.json"),
        "fixed_physical_complex_noise_variance":.1,"data_symbols":K,"extra_reference_pilots":128})


if __name__=="__main__":
    if sys.argv[1]=="register":
        producer_functions()
        record(ROOT/"protocol.json",specification())
    else:
        run()
