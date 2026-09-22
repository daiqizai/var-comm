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
def write_rows(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 protocol = load_protocol_config()
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'followup/linear_measurement_v1'));args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0');
 torch.manual_seed(2026091901);A,_=torch.linalg.qr(torch.randn(8192,protocol['measurement_dim']),mode='reduced');torch.save(A,out/'A.pt');A=A.to(device)
 vae,var=load_models(model_paths(),device);decoder=load_decoder(vae,device);train=MatchedPopulation('train');cal=MatchedPopulation('calibration');targets=load_targets();quality=yaml.safe_load((VAR/'configs/progressive_channel.yaml').read_text())['quality'];lp,dino,_=load_quality_models(quality,device)
 training_norms=[]
 with torch.no_grad():
  for start in range(0,len(train),256):
   values=train.source.values['F'][start:start + 256].reshape(-1,8192).to(device)
   base=train.source.values['Fb_TX'][start:start + 256].reshape(-1,8192).to(device)
   training_norms.append(torch.linalg.vector_norm(torch.cat((values@A,(values-base)@A),dim=0),dim=1).cpu())
 protocol['norm_log_min'], protocol['norm_log_max'] = fit_training_norm_range(torch.cat(training_norms))
 # Per-frame energy is part of the transmitted signal.  The old implementation
 # used a global mean scale (and even took the norm along the sample axis).
 W={str(s):{'source':[0.0,0.0],'residual':[0.0,0.0]} for s in protocol['snrs_db']}
 with torch.no_grad():
  for start in range(0,1000,4):
   ids=torch.arange(start,min(start+4,1000));f=cal.source.values['F'][ids].reshape(len(ids),8192).to(device);fbtx=cal.source.values['Fb_TX'][ids].reshape(len(ids),8192).to(device)
   for si,snr in enumerate(protocol['snrs_db']):
    for ni,seed in enumerate(protocol['noise_seeds']):
     batch=cal.batch(ids,torch.full((len(ids),),si,dtype=torch.long),torch.full((len(ids),),ni,dtype=torch.long),[seed]*len(ids),device);fbrx=batch['Fb_RX'].reshape(len(ids),8192);status=batch['rx_status'][:,0]>0.5;noise=batch['standard_noise'];
     if not status.any():continue
    for method,vec in [('source',f),('residual',f-fbtx)]:
      measurement=vec@A;signal,observed,_=transmit_projection(measurement,noise,snr,protocol=protocol);received=receive_projection(observed,snr,protocol=protocol);estimate=received['measurement'];valid=status & received['control_ok'];delta_coord=estimate-(fbrx@A if method=='source' else torch.zeros_like(estimate));delta=delta_coord@A.T;target=f-fbrx;num=(target[valid]*delta[valid]).sum();den=delta[valid].square().sum();W[str(snr)][method][0]+=float(num);W[str(snr)][method][1]+=float(den)
 W={snr:{m:(float(np.clip(v[0]/max(v[1],1e-8),protocol['w_clip'][0],protocol['w_clip'][1]))) for m,v in d.items()} for snr,d in W.items()};(out/'projection.json').write_text(json.dumps({'A_shape':[8192,protocol['measurement_dim']],'seed':2026091901,'W':W,'W_clip':protocol['w_clip'],'base_uses':protocol['base_uses'],'total_uses':protocol['total_uses'],'norm_side_information_bits':protocol['norm_bits'],'norm_log_min':protocol['norm_log_min'],'norm_log_max':protocol['norm_log_max'],'control_uses':protocol['control_uses'],'measurement_uses':protocol['measurement_uses'],'enhancement_uses':ENHANCEMENT_USES,'per_frame_energy':True,'side_information_over_channel':True,'A_column_orthonormal':True},indent=2))
 rows=[]
 for done,r in enumerate(targets):
  image=torch.from_numpy(r['pixels'][None].astype(np.float32)/127.5-1).to(device);label=int(r['target']['class_index']);source=split_prefix(r['tokens'],10)
  with torch.no_grad():f=vae.quant_conv(vae.encoder(image))[0].reshape(8192);fbtx=complete_latent(vae,var,source[:8],label,device)[0].reshape(8192)
  for snr in protocol['snrs_db']:
   for seed in protocol['noise_seeds']:
    base,_=raw_transmit_budget(source,label,8,protocol['base_uses']);received=base+seeded_noise(r['target']['image_id'],seed,base.shape)/np.sqrt(10**(snr/10));phy=raw_receive_budget(received,snr,protocol['base_uses'])
    if phy['label'] is None: fbrx=torch.zeros_like(fbtx);status=0
    else:fbrx=complete_latent(vae,var,phy['prefix'],phy['label'],device)[0].reshape(8192);status=1
    noise=torch.as_tensor(enhancement_noise(r['target']['image_id'],seed,1024),device=device,dtype=torch.float32).unsqueeze(0)
    for method,vec in [('source',f),('residual',f-fbtx)]:
      measurement=vec.reshape(1,-1)@A;signal,observed,_=transmit_projection(measurement,noise,snr,protocol=protocol);received=receive_projection(observed,snr,protocol=protocol);estimate=received['measurement'];delta=estimate-(fbrx.reshape(1,-1)@A if method=='source' else torch.zeros((1,protocol['measurement_dim']),device=device));latent=(fbrx+W[str(snr)][method]*(delta@A.T)).reshape(1,32,16,16);img=decoder(latent)[0].cpu().numpy() if status else np.full((3,256,256),.5,np.float32);rows.append({'method':method,'source_index':done,'image_id':r['target']['image_id'],'snr_db':snr,'seed':seed,'N':protocol['total_uses'],'E':2*protocol['total_uses'],'header_ok':status,'body_crc_ok':int(phy.get('body_crc_accepted',False)),'norm_control_ok':bool(received['control_ok'][0]),'norm_code':int(received['norm_code'][0]),'image':img})
  if done%10==0:print('linear',done+1,flush=True)
 final=[]
 for source_index in range(100):
  subset=[r for r in rows if r['source_index']==source_index];images=[r['image'] for r in subset];target=targets[source_index]['pixels'].astype(np.float32)/255.;q,_,_=quality_metrics(target,images,lp,dino,device)
  for row,metric in zip(subset,q):row.pop('image');row.update(metric);final.append(row)
  write_rows(out/'per_frame.csv',final);(out/'completion.json').write_text(json.dumps({'status':'LINEAR_MEASUREMENT_DEVELOPMENT_COMPLETE','rows':len(final),'sources':100,'new_holdout_used':False,'A_column_orthonormal':True,'control_uses':protocol['control_uses'],'measurement_uses':protocol['measurement_uses']},indent=2));print('complete',len(final))
if __name__=='__main__':main()
