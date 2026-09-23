from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from latent_enhancement.latent import complete_latent,enhancement_noise
from latent_enhancement.runtime import model_paths,settings
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_eval.runner import load_targets,raw_receive_budget,raw_transmit_budget
from var_comm.next_scale_prior import load_models
from var_comm.progressive import split_prefix
from var_comm.quality import load_quality_models,quality_metrics
from var_comm.scale_channel import channel_evidence, crc_accepts, decode_map, encode_packet, indices_to_bits, rate_match_indices
from var_comm.study import seeded_noise

PROJECT=Path(__file__).resolve().parents[5];VAR=PROJECT;ROOT=VAR/'outputs/VAR-LATENT-ENHANCEMENT-20260917';EXP=VAR/'experiments/var-latent-enhancement-20260917'
ENHANCEMENT_USES=1024
CONTROL_USES=32
CONTROL_BITS=8
CONTROL_CRC_BITS=16
CONTROL_TAIL_BITS=6
CONTROL_INFORMATION_BITS=CONTROL_BITS + CONTROL_CRC_BITS + CONTROL_TAIL_BITS
MEASUREMENT_USES=ENHANCEMENT_USES - CONTROL_USES
MEASUREMENT_DIM=2 * MEASUREMENT_USES


def load_protocol_config(path=None):
    """Load and validate the finite-resource measurement protocol."""
    path = Path(path) if path is not None else EXP / "mechanisms/linear_measurement_config.json"
    record = json.loads(path.read_text())
    budget = record.get("budget", {})
    quantization = record.get("norm_quantization", {})
    if int(budget.get("enhancement", -1)) != ENHANCEMENT_USES:
        raise ValueError("linear protocol enhancement budget must be 1024 complex uses")
    if int(budget.get("base", -1)) <= 0 or int(budget.get("total", -1)) != int(budget["base"]) + ENHANCEMENT_USES:
        raise ValueError("linear protocol base/total budget is inconsistent")
    control_uses = int(record.get("control_uses", CONTROL_USES))
    if control_uses != CONTROL_USES or control_uses >= ENHANCEMENT_USES:
        raise ValueError("unsupported linear control budget")
    dimension = int(record.get("measurement_real_coordinates", -1))
    if dimension != 2 * (ENHANCEMENT_USES - control_uses):
        raise ValueError("measurement coordinates do not subtract paid control uses")
    bits = int(quantization.get("bits", -1))
    low, high = float(quantization.get("log_min", np.nan)), float(quantization.get("log_max", np.nan))
    if bits != CONTROL_BITS or not np.isfinite([low, high]).all() or not low < high:
        raise ValueError("invalid norm quantization range")
    if int(record.get("measurement_uses", -1)) != MEASUREMENT_USES:
        raise ValueError("measurement uses do not match the enhancement ledger")
    snrs = [float(value) for value in record.get("snrs_db", [])]
    seeds = [int(value) for value in record.get("noise_seeds", [])]
    w_clip = tuple(float(value) for value in record.get("W_clip", [-4.0, 4.0]))
    if not snrs or not np.isfinite(snrs).all() or not seeds or len(set(seeds)) != len(seeds) or len(w_clip) != 2 or not np.isfinite(w_clip).all() or not w_clip[0] < w_clip[1]:
        raise ValueError("linear protocol must register finite SNRs and unique noise seeds")
    return {"base_uses": int(budget["base"]), "total_uses": int(budget["total"]),
            "control_uses": control_uses, "measurement_uses": MEASUREMENT_USES,
            "control_payload_bits": int(record.get("control_payload_bits", CONTROL_BITS)),
            "control_crc_bits": int(record.get("control_crc_bits", CONTROL_CRC_BITS)),
            "control_tail_bits": int(record.get("control_tail_bits", CONTROL_TAIL_BITS)),
            "measurement_dim": dimension, "norm_bits": bits, "norm_log_min": low,
            "norm_log_max": high, "control_information_bits": CONTROL_INFORMATION_BITS,
            "snrs_db": snrs, "noise_seeds": seeds, "w_clip": w_clip}


def fit_training_norm_range(norms, *, quantiles=(0.001, 0.999)):
    """Derive the registered log-norm range from training samples only."""
    values = torch.as_tensor(norms, dtype=torch.float64).reshape(-1)
    if not len(values) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("training norm range requires finite nonnegative norms")
    values = values[values > 0]
    if not len(values):
        raise ValueError("training norm range has no positive measurements")
    low_q, high_q = quantiles
    if not 0 <= low_q < high_q <= 1:
        raise ValueError("invalid training norm quantiles")
    logarithms = values.log().cpu().numpy()
    low, high = np.quantile(logarithms, [low_q, high_q])
    if not np.isfinite([low, high]).all() or not low < high:
        raise ValueError("training norms do not define a finite range")
    return float(low), float(high)

def quantize_log_norm(norm, *, protocol=None):
    """Quantize a norm to a paid side-information code; code zero is zero norm."""
    protocol = protocol or load_protocol_config()
    values = torch.as_tensor(norm)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise FloatingPointError("projection norm must be finite and nonnegative")
    levels = (1 << protocol["norm_bits"]) - 1
    result = torch.zeros_like(values, dtype=torch.long)
    positive = values > 0
    if positive.any():
        logarithms = values[positive].log()
        clipped = logarithms.clamp(protocol["norm_log_min"], protocol["norm_log_max"])
        result[positive] = 1 + torch.round((clipped - protocol["norm_log_min"]) /
                                           (protocol["norm_log_max"] - protocol["norm_log_min"]) * (levels - 1)).to(torch.long)
    return result

def dequantize_log_norm(code, *, protocol=None):
    """Decode a checked norm code; zero is a registered no-enhancement value."""
    protocol = protocol or load_protocol_config()
    values = torch.as_tensor(code)
    levels = (1 << protocol["norm_bits"]) - 1
    if values.ndim == 0:
        values = values.reshape(1)
    if torch.is_floating_point(values):
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError("norm code must be integral")
    elif values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("norm code must be integral")
    if (values < 0).any() or (values > levels).any():
        raise ValueError("norm code outside registered range")
    output = torch.zeros_like(values, dtype=torch.float32)
    positive = values > 0
    output[positive] = (protocol["norm_log_min"] + (values[positive].to(torch.float32) - 1) /
                        (levels - 1) * (protocol["norm_log_max"] - protocol["norm_log_min"])).exp()
    return output

def channel_apply(signal, noise, snr_db):
    signal = torch.as_tensor(signal)
    noise = torch.as_tensor(noise, device=signal.device, dtype=signal.dtype)
    if signal.shape != noise.shape or signal.ndim != 3 or signal.shape[1:] != (ENHANCEMENT_USES, 2):
        raise ValueError("linear channel requires [batch,1024,2] signal and noise")
    if not torch.isfinite(signal).all() or not torch.isfinite(noise).all():
        raise FloatingPointError("nonfinite linear waveform or noise")
    snr = torch.as_tensor(snr_db, device=signal.device, dtype=signal.dtype)
    if not torch.isfinite(snr).all():
        raise ValueError("SNR must be finite")
    scale = torch.pow(torch.tensor(10.0, device=signal.device, dtype=signal.dtype), snr / 10).sqrt()
    return signal + noise / scale.reshape((-1,) + (1,) * (signal.ndim - 1)) if snr.ndim else signal + noise / scale


def _control_symbols(code):
    payload = indices_to_bits([int(code)], CONTROL_BITS)
    return encode_packet(payload, CONTROL_USES)["symbols"].reshape(-1)


def transmit_projection(measurement, noise, snr_db, *, protocol=None):
    """Transmit a normalized projection plus quantized norm over paid PHY uses."""
    protocol = protocol or load_protocol_config()
    values = torch.as_tensor(measurement)
    if values.ndim != 2 or values.shape[1] != protocol["measurement_dim"]:
        raise ValueError("measurement must use the configured post-control dimension")
    if not torch.isfinite(values).all():
        raise FloatingPointError("nonfinite projection measurement")
    norm = torch.linalg.vector_norm(values, dim=1)
    code = quantize_log_norm(norm, protocol=protocol)
    normalized = values / norm.clamp_min(1e-20).unsqueeze(1) * np.sqrt(protocol["measurement_dim"])
    normalized = torch.where((norm > 0).unsqueeze(1), normalized, torch.ones_like(normalized))
    controls = torch.as_tensor(np.stack([_control_symbols(value) for value in code.cpu().tolist()]),
                               dtype=values.dtype, device=values.device)
    signal = torch.cat((controls, normalized), dim=1).reshape(-1, ENHANCEMENT_USES, 2)
    observation = channel_apply(signal, noise, snr_db)
    return signal, observation, {"norm": norm, "norm_code": code,
                                "norm_saturated": ((norm > 0) & ((norm.log() < protocol["norm_log_min"]) |
                                                                  (norm.log() > protocol["norm_log_max"]))),
                                "measurement_dim": protocol["measurement_dim"],
                                "control_uses": protocol["control_uses"]}


def receive_projection(observation, snr_db, *, protocol=None):
    """Decode the paid norm packet and recover measurement using no TX state."""
    protocol = protocol or load_protocol_config()
    values = torch.as_tensor(observation)
    if values.ndim != 3 or values.shape[1:] != (ENHANCEMENT_USES, 2) or not torch.isfinite(values).all():
        raise ValueError("observation must be finite [batch,1024,2]")
    flat = values.detach().cpu().numpy().reshape(len(values), -1)
    codes, accepted = [], []
    mapping = rate_match_indices(2 * protocol["control_information_bits"], 2 * protocol["control_uses"])
    for frame in flat[:, :2 * protocol["control_uses"]]:
        try:
            evidence = channel_evidence(frame, mapping, protocol["control_information_bits"], float(snr_db))
            decoded, _ = decode_map(evidence)
            valid = bool(len(decoded) == protocol["control_information_bits"] and
                         np.array_equal(decoded[-protocol["control_tail_bits"]:], np.zeros(protocol["control_tail_bits"], dtype=np.uint8)) and
                         crc_accepts(decoded[:-protocol["control_tail_bits"]]))
            code = int(np.dot(decoded[:protocol["norm_bits"]], 1 << np.arange(protocol["norm_bits"] - 1, -1, -1))) if valid else 0
        except (ValueError, RuntimeError, FloatingPointError):
            valid, code = False, 0
        codes.append(code); accepted.append(valid)
    code_tensor = torch.as_tensor(codes, device=values.device, dtype=torch.long)
    norms = dequantize_log_norm(code_tensor, protocol=protocol)
    data = values[:, protocol["control_uses"]:, :].reshape(len(values), protocol["measurement_dim"])
    measurement = data * norms.to(values.device).unsqueeze(1) / np.sqrt(protocol["measurement_dim"])
    measurement = torch.where(torch.as_tensor(accepted, device=values.device).unsqueeze(1), measurement, torch.zeros_like(measurement))
    return {"measurement": measurement, "norm": norms.to(values.device), "norm_code": code_tensor,
            "control_ok": torch.as_tensor(accepted, device=values.device, dtype=torch.bool),
            "control_fallback": ~torch.as_tensor(accepted, device=values.device, dtype=torch.bool)}


def fit_scalar_w(target, correction, valid=None, *, clip=(-4.0, 4.0)):
    """Fit one global W from pooled coordinates, then apply no per-batch clipping."""
    target = torch.as_tensor(target)
    correction = torch.as_tensor(correction, device=target.device, dtype=target.dtype)
    if target.shape != correction.shape or target.ndim < 2 or not torch.isfinite(target).all() or not torch.isfinite(correction).all():
        raise ValueError("W fit requires matching finite target and correction arrays")
    mask = torch.ones(target.shape[0], device=target.device, dtype=torch.bool) if valid is None else torch.as_tensor(valid, device=target.device, dtype=torch.bool)
    if mask.shape != (target.shape[0],):
        raise ValueError("W validity mask must select complete samples")
    if not mask.any():
        return 0.0
    numerator = (target[mask] * correction[mask]).sum()
    denominator = correction[mask].square().sum()
    if float(denominator) <= 0:
        return 0.0
    if len(clip) != 2 or not np.isfinite(clip).all() or not clip[0] < clip[1]:
        raise ValueError("invalid W clip bounds")
    value = numerator / denominator
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite pooled W estimate")
    return float(value.clamp(float(clip[0]), float(clip[1])))
def correction_coordinates(estimate, base_projection, control_ok, method):
    if method not in ('source','residual'): raise ValueError('unknown projection method')
    delta=estimate-base_projection if method=='source' else estimate
    # Mask the complete innovation, including subtraction of the RX base.
    return torch.where(control_ok[:,None],delta,torch.zeros_like(delta))

def validate_coverage(rows, sources, snrs, seeds):
    keys=[(int(r['source_index']),r['method'],float(r['snr_db']),int(r['seed'])) for r in rows]
    expected={(i,m,float(s),int(n)) for i in range(sources) for m in ('source','residual') for s in snrs for n in seeds}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise RuntimeError('incomplete or duplicate linear evaluation coverage')

def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def main():
    from latent_enhancement.runtime import configure, write_json, digest, snapshot, require_available
    configure(); require_available(); protocol=load_protocol_config()
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--sources',type=int,default=100);p.add_argument('--calibration-sources',type=int,default=1000);p.add_argument('--projection-from');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);device=torch.device('cuda:0')
    identity={'source_snapshot':snapshot([__file__,EXP/'mechanisms/linear_measurement_config.json']), 'sources':args.sources,'calibration_sources':args.calibration_sources,'GPU':'real_weights'}
    write_json(out/'registration.json',identity)
    if args.projection_from:A=torch.load(args.projection_from,map_location='cpu',weights_only=True)
    else:
        torch.manual_seed(2026091901);A,_=torch.linalg.qr(torch.randn(8192,protocol['measurement_dim']),mode='reduced')
    if A.shape!=(8192,protocol['measurement_dim']) or not torch.allclose(A.T@A,torch.eye(protocol['measurement_dim']),atol=2e-5):raise RuntimeError('invalid orthonormal projection')
    torch.save(A,out/'A.pt');A=A.to(device)
    vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);train=MatchedPopulation('train');cal=MatchedPopulation('calibration');targets=load_targets()[:args.sources]
    quality=yaml.safe_load((VAR/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(quality,device)
    norms=[]
    with torch.no_grad():
        for start in range(0,len(train),256):
            f=train.source.values['F'][start:start+256].reshape(-1,8192).to(device);b=train.source.values['Fb_TX'][start:start+256].reshape(-1,8192).to(device)
            norms.append(torch.linalg.vector_norm(torch.cat((f@A,(f-b)@A)),dim=1).cpu())
    protocol['norm_log_min'],protocol['norm_log_max']=fit_training_norm_range(torch.cat(norms));del train
    totals={str(s):{m:{'num':0.,'den':0.,'frames':0,'valid_control_frames':0,'seeds':set(),'sources':set()} for m in ('source','residual')} for s in protocol['snrs_db']}
    with torch.no_grad():
        for start in range(0,args.calibration_sources,4):
            ids=torch.arange(start,min(start+4,args.calibration_sources));f=cal.source.values['F'][ids].reshape(len(ids),8192).to(device);tx=cal.source.values['Fb_TX'][ids].reshape(len(ids),8192).to(device)
            for si,snr in enumerate(protocol['snrs_db']):
                for ni,seed in enumerate(protocol['noise_seeds']):
                    batch=cal.batch(ids,torch.full_like(ids,si),torch.full_like(ids,ni),[seed]*len(ids),device);rx=batch['Fb_RX'].reshape(len(ids),8192);valid_header=batch['rx_status'][:,0]>.5
                    for method,vec in [('source',f),('residual',f-tx)]:
                        _,observed,_=transmit_projection(vec@A,batch['standard_noise'],snr,protocol=protocol);r=receive_projection(observed,snr,protocol=protocol)
                        delta=correction_coordinates(r['measurement'],rx@A,r['control_ok'],method);valid=valid_header & r['control_ok'];target=(f-rx)@A
                        t=totals[str(snr)][method];t['num']+=float((target[valid].double()*delta[valid].double()).sum());t['den']+=float(delta[valid].double().square().sum());t['frames']+=len(ids);t['valid_control_frames']+=int(valid.sum());t['seeds'].add(seed);t['sources'].update(ids.tolist())
            if start%100==0:print('linear calibration',start,flush=True)
    W={}
    for snr,methods in totals.items():
        W[snr]={}
        for method,t in methods.items():
            assert t['frames']==args.calibration_sources*len(protocol['noise_seeds'])
            assert t['seeds']==set(protocol['noise_seeds'])
            t['seeds']=sorted(t['seeds']);t['sources']=sorted(t['sources'])
            W[snr][method]=float(np.clip(t['num']/max(t['den'],1e-8),*protocol['w_clip']))
    write_json(out/'projection.json',{'protocol':protocol,'A_sha256':digest(out/'A.pt'),'W':W,'fit_coverage':totals,'training_norm_range_sources':20000})
    del cal
    final=[]
    with torch.no_grad():
        for i,r in enumerate(targets):
            image=torch.from_numpy(r['pixels'][None].astype(np.float32)/127.5-1).to(device);label=int(r['target']['class_index']);source=split_prefix(r['tokens'],10)
            f=vae.quant_conv(vae.encoder(image))[0].reshape(8192);tx=complete_latent(vae,var,source[:8],label,device)[0].reshape(8192);records=[];images=[]
            base,_=raw_transmit_budget(source,label,8,protocol['base_uses'])
            for snr in protocol['snrs_db']:
                for seed in protocol['noise_seeds']:
                    observation=base+seeded_noise(r['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(observation,snr,protocol['base_uses']);ok=phy['label'] is not None
                    rx=complete_latent(vae,var,phy['prefix'],phy['label'],device)[0].reshape(8192) if ok else torch.zeros_like(tx)
                    noise=torch.as_tensor(enhancement_noise(r['target']['image_id'],seed,1024),device=device,dtype=torch.float32).unsqueeze(0)
                    for method,vec in [('source',f),('residual',f-tx)]:
                        wave,y,_=transmit_projection(vec[None]@A,noise,snr,protocol=protocol);received=receive_projection(y,snr,protocol=protocol)
                        delta=correction_coordinates(received['measurement'],rx[None]@A,received['control_ok'],method)
                        latent=(rx+W[str(snr)][method]*(delta@A.T)).reshape(1,32,16,16)
                        img=decoder(latent)[0].cpu().numpy() if ok else np.full((3,256,256),.5,np.float32)
                        images.append(img);records.append({'method':method,'source_index':i,'image_id':r['target']['image_id'],'snr_db':snr,'seed':seed,'N':protocol['total_uses'],'E':2*protocol['total_uses'],'header_ok':int(ok),'body_crc_ok':int(phy.get('body_crc_accepted',False)),'norm_control_ok':bool(received['control_ok'][0]),'norm_code':int(received['norm_code'][0]),'enhancement_energy':float(wave.square().sum())})
            metrics,_,_=quality_metrics(r['pixels'].astype(np.float32)/255.,images,lp,dino,device)
            final.extend({**row,**q} for row,q in zip(records,metrics));write_rows(out/'partial.csv',final);write_json(out/'status.json',{'status':'RUNNING','sources_done':i+1,'expected_sources':len(targets),'rows':len(final)})
            print('linear development',i+1,flush=True)
    validate_coverage(final,len(targets),protocol['snrs_db'],protocol['noise_seeds']);write_rows(out/'per_frame.csv',final)
    write_json(out/'completion.json',{'status':'LINEAR_MEASUREMENT_DEVELOPMENT_COMPLETE' if args.sources==100 and args.calibration_sources==1000 else 'REAL_WEIGHT_SMOKE_COMPLETE','rows':len(final),'sources':len(targets),'calibration_sources':args.calibration_sources,'registration_sha256':digest(out/'registration.json'),'per_frame_sha256':digest(out/'per_frame.csv'),'new_holdout_used':False,'control_uses':32,'measurement_uses':992})
if __name__=='__main__':main()
