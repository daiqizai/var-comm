"""A shared fixed or past-symbol-only 2-bit ADC threshold on both published RF receivers."""
import csv
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from common import (D4, PRIMARY, PRIOR, STUDY, SWIN, SWIN_HASH, TEMP, USER, digest, include_sources,
                    load_state, read, record, table, tensor_hash)

include_sources()
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import configure_runtime_env, configure_torch_determinism
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir4_rn import rf_1bit_blind as rf

SHA = "ec608c7d89d78704b4efa7d32004fafec85f6766"
REPO = USER/"code/VAR_COMM"
MODES = ("frame_rms", "fixed", "causal")
METHODS = ("rf_bussgang", "blackbox_rf")
SOURCE_HASH = "83b4863a0fa206a35d7e7604347e70f1a32e0b85f09076e932cd7622aba67a2d"


def git_result(name):
    path = "results/dir4_rf1bit_noma_20261122/eval/" + name
    return subprocess.check_output(["git","-C",str(REPO),"show",SHA+":"+path]).decode()


def spec():
    import json
    assets = json.loads(git_result("eval_dir4_summary_common500.json"))["loaded_checkpoints"]
    weights = {k:assets[k+"_2bit"] for k in METHODS}
    return {
        "weights":weights,"producer_git":SHA,"source_sha256":SOURCE_HASH,"backbone_sha256":SWIN_HASH,
        "train":0,"adc":"2 bits per real or imaginary component; identical four output levels and 0.9957 multiplier",
        "nominal":"10 dB, PA input backoff 1 dB, oscillator 3.5 degrees/symbol, 32 RBs, same 48x256 complex observations. No extra pilot, sample or signalling.",
        "calibration":"val256, new draw 6112309, each frozen method once for each source. A common delta0 is 0.9957 times sqrt of the pooled per-I/Q second moment. No reconstruction/quality loss in calibration.",
        "receiver_controls":["Original analog RMS of the entire current frame","Fixed delta0 for every symbol","Causal RMS of all strictly earlier symbols of the same frame; delta0 for the first symbol"],
        "test":"common500 and Kodak24 at nominal gain. Additional Kodak24 sensitivities at -6 and +6 dB analog gain applied to signal and noise before the ADC. No oracle gain correction after the ADC.",
        "primary":"RF minus BlackBox under fixed threshold on common500, 10 dB; report same pair under causal and original, changes per model and gain, both PSNR and DISTS.",
        "seeds":[3001,3002,3003],"interval":"5000 source-paired bootstraps after averaging 3 draws, seed 2026112308, datasets separate",
    }


def load_models(device, protocol):
    assert digest(Path(rf.__file__)) == SOURCE_HASH
    assert digest(SWIN) == SWIN_HASH
    bb = load_swinjscc_backbone(SWIN,model_mode="SwinJSCC_w/_SA",c_bottleneck=96,device=device)
    models = {}
    for name, cp in protocol["weights"].items():
        assert digest(cp["path"]) == cp["sha256"]
        model = rf.RFImpairedBlindTransceiver(bb,method=name,num_rbs=32,hidden_ch=160)
        load_state(model,cp["path"])
        models[name] = model.to(device).eval()
    return models


def channels(n,seed):
    gen = torch.Generator().manual_seed(seed*1301+2*53+35)
    eps = torch.randn((n,48,256,2),generator=gen,dtype=torch.float32)
    hr,hi = sample_correlated_rbs(n,32,.85,device="cpu",generator=gen)
    dphi = torch.randn((n,48),generator=gen,dtype=torch.float32)*(3.5*math.pi/180)
    dphi[:,0] = 0
    return eps,torch.stack((hr,hi),dim=-1),torch.cumsum(dphi,dim=1)


def observe(tx,draws):
    raw = []
    native_quantizer = rf.ste_2bit_iq_quantize
    def capture(r,i):
        raw.extend((r,i))
        return native_quantizer(r,i)
    rf.ste_2bit_iq_quantize = capture
    try:
        eps,h,phi = draws
        native,_ = rf.physical_rf_impaired_ofdm_channel(tx,10.0,adc_bits=2,pa_ibo_db=1.0,
                    pn_step_deg=3.5,h_rb_override=h,phase_walk_override=phi,standard_noise_td_iq=eps)
    finally:
        rf.ste_2bit_iq_quantize = native_quantizer
    assert len(raw) == 2
    return raw[0], raw[1], native


def modulated_grid(r,i,mode,delta0):
    if mode == "frame_rms":
        qr,qi = rf.ste_2bit_iq_quantize(r,i)
    else:
        d = torch.full((len(r),1,1),delta0,dtype=r.dtype,device=r.device)
        if mode == "causal":
            # Moment from symbols 0..j-1 affects the threshold at j, never the current symbol.
            power = .5*(r.square()+i.square()).mean(dim=-1,keepdim=True)
            count = torch.arange(1,r.shape[1],device=r.device).reshape(1,-1,1)
            history = (power[:,:-1].cumsum(dim=1)/count).clamp_min(1e-6).sqrt()
            d = torch.cat((d,0.9957*history),dim=1)
        elif mode != "fixed":
            raise ValueError(mode)
        def quant(x):
            return ((torch.floor(x/d)+2).clamp(0,3)-1.5)*d
        qr,qi = quant(r),quant(i)
    fd = torch.fft.fft(torch.complex(qr,qi),dim=-1,norm="ortho")
    return rf.ofdm_grid_to_rb384(torch.stack((fd.real,fd.imag),dim=-1))


@torch.inference_mode()
def execute(stage, device):
    root = STUDY / "rf"
    protocol = read(root/"protocol.json")
    assert protocol == spec()
    configure_runtime_env(cache_root=TEMP/("cache/rf_"+stage),num_threads=12 if device=="cpu" else 4)
    torch.set_num_threads(12 if device=="cpu" else 4)
    configure_torch_determinism()
    models = load_models(device,protocol)
    bs = 8 if device=="cpu" else 24
    counts,draws_record = {},[]
    if stage=="calibrate":
        ds = load_split_u8("val256")
        assert not ds.is_holdout and len(ds.source_ids)==256
        images = [{"id":k,"sha256":tensor_hash(x)} for k,x in zip(ds.source_ids,ds.images_u8)]
        other = {item["chw_u8_sha256"] for split in ("common500","kodak24")
                 for item in read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json"))}
        assert not other.intersection(z["sha256"] for z in images)
        draws = channels(256,6112309)
        squares = {k:0. for k in METHODS}
        samples = 0
        for a in range(0,256,bs):
            x = ds.images_u8[a:a+bs].to(device).float()/255
            stem = models[METHODS[0]].extract_encoder_stem(x)
            for tag,m in models.items():
                y,i,_ = observe(m.encode_rf_tx(stem),[z[a:a+bs].to(device) for z in draws])
                squares[tag] += float((y.double().square()+i.double().square()).sum())
                samples += 2*y.numel()
        delta = .9957*math.sqrt(sum(squares.values())/samples)
        record(root/"calibration_sources.json",images)
        record(root/"calibration.json",{"delta0":delta,"per_method_sum_squares":squares,"total_IQ_samples":samples,
               "sources":256,"analog_frames":512,"new_decodes":0,"device":device,"draw_sha256":[tensor_hash(z) for z in draws]})
        print("Fixed delta",delta,flush=True)
        return
    delta0 = read(root/"calibration.json")["delta0"]
    evaluator = OfflineVisualEvaluator(device=device,include_lpips=False)
    replay = []
    for split,n,gains in (("kodak24",24,(0,-6,6)),("common500",500,(0,))):
        ds = load_split_u8(split,allow_holdout=True)
        ids = ds.source_ids
        assert len(set(ids)) == n and ds.is_holdout
        assert read(PRIOR/"evaluation/noma"/(split+"_source_manifest.json")) == [
            {"source_id":i,"chw_u8_sha256":tensor_hash(x)} for i,x in zip(ids,ds.images_u8)]
        xs = ds.images_u8.float()/255
        stems = torch.cat([models[METHODS[0]].extract_encoder_stem(x.to(device)).cpu() for x in xs.split(bs)])
        rows = []
        for seed in protocol["seeds"]:
            draws = channels(n,seed)
            draws_record.append({"split":split,"seed":seed,"noise_fading_phase":[tensor_hash(z) for z in draws]})
            for a in range(0,n,bs):
                b = min(a+bs,n)
                x,stem = xs[a:b].to(device),stems[a:b].to(device)
                for tag,m in models.items():
                    r,i,native = observe(m.encode_rf_tx(stem),[z[a:b].to(device) for z in draws])
                    for gain in gains:
                        g = 10.**(gain/20.)
                        for mode in MODES:
                            grid = modulated_grid(r*g,i*g,mode,delta0)
                            if gain==0 and mode=="frame_rms":
                                assert torch.equal(grid,native)
                            y,_ = m.decode_rf_rx(grid)
                            score = evaluator.evaluate_batch(x,y)
                            for j,k in enumerate(ids[a:b]):
                                rows.append(dict(split=split,gain_db=gain,seed=seed,source_id=k,model=tag,mode=mode,
                                    psnr_db=score["psnr_db"][j],dists=score["dists"][j]))
            print(split,seed,"all ADC modes complete",flush=True)
        assert len(rows) == 3*n*len(gains)*2*3 and all(np.isfinite([r["psnr_db"],r["dists"]]).all() for r in rows)
        table(root/(split+"_frames.csv"),rows)
        counts[split]={"source_count":n,"per_model_channel_observations":3*n,"quality_pairs":len(rows)}
        old = list(csv.DictReader(git_result("dir4a_rf1bit_results_"+split+".csv").splitlines()))
        for tag,previous in (("rf_bussgang","RF_Bussgang_JSCC"),("blackbox_rf","BlackBox_RF_Blind")):
            where = next(g for g in old if g["model"]==previous and g["condition"]=="adc2bit_snr10dB")
            for mk in PRIMARY:
                value = np.mean([z[mk] for z in rows if z["mode"]=="frame_rms" and z["gain_db"]==0 and z["model"]==tag])
                replay.append(dict(split=split,model=tag,metric=mk,original=float(where[mk+"_mean"]),new=float(value)))
    record(root/"evaluation.json",{"counts":counts,"native_replay":replay,"draws":draws_record,"device":device,
           "native_reference_grids_bitwise_equal_all_nominal_batches":True})


if __name__=="__main__":
    root = STUDY/"rf"
    if sys.argv[1]=="register":
        record(root/"protocol.json",spec())
    else:
        execute(sys.argv[1],sys.argv[2])
