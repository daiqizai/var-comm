from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from latent_enhancement.latent import complete_latent, original_rgb
from latent_enhancement.runtime import digest, model_paths, settings
from latent_enhancement_b.common import load_decoder, scale_statistics
from latent_enhancement_b.model import build_arms, render_received
from latent_enhancement_eval.runner import (
    arithmetic_receive_budget, arithmetic_transmit_budget, load_targets, raw_receive_budget,
    raw_transmit_budget,
)
from latent_enhancement_eval import runner as eval_runner
from latent_enhancement_eval.deployment import (
    channel_apply, cpu_rgb_to_device, device_rgb_to_cpu, rx_base, rx_continuous,
    synchronize, tx_encode,
)
from var_comm.next_scale_prior import load_models
from var_comm.progressive import receive_whole, transmit_whole
from var_comm.whole_entropy import encode_prefixes

PROJECT = Path(__file__).resolve().parents[5]
VAR_COMM = PROJECT
EXP = VAR_COMM / "experiments/var-latent-enhancement-20260917"
OUT_ROOT = VAR_COMM / "outputs/VAR-LATENT-ENHANCEMENT-20260917"
CONFIG = EXP / "followup/config.json"
TIMING_CONFIG = EXP / "timing/config.json"


def read_json(path): return json.loads(Path(path).read_text())

def configure():
    torch.set_num_threads(6); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False

def source_prepare(vae, image):
    with torch.no_grad():
        f=vae.quant_conv(vae.encoder(image)); scales=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False)
    return f,[scale[0].cpu().numpy() for scale in scales]

def load_arm(name, shape, scale, device):
    selected=read_json(OUT_ROOT/f"stage_B_v1/training/selected_{name}.json")
    checkpoint=Path(selected["checkpoint"])
    if digest(checkpoint)!=selected["checkpoint_sha256"]: raise RuntimeError(f"checkpoint changed: {name}")
    arms=build_arms(shape,scale,settings()["stage_B"]).to(device)
    arms.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["arms"],strict=True)
    return arms[name].eval().requires_grad_(False)

def clean_digital(phy,family,renderer,vae,var,decoder,device):
    if phy["label"] is None: return np.full((3,256,256),.5,dtype=np.float32)
    if family=="raw": prefix=phy["prefix"]
    else:
        decoded=eval_runner.decode_source(phy,vae,var,device,render=False,return_latent=True)
        prefix=decoded["prefix"]
        latent=decoded.get("latent")
        if latent is not None:
            return original_rgb(vae,latent)[0].cpu().numpy() if renderer=="D0" else decoder(latent)[0].cpu().numpy()
    latent=complete_latent(vae,var,prefix,phy["label"],device)
    if renderer=="D0": return original_rgb(vae,latent)[0].cpu().numpy()
    return decoder(latent)[0].cpu().numpy()

def load_clean_sources():
    targets=load_targets()
    with np.load(OUT_ROOT/"development_eval_v1/images/000/reconstructions.npz",allow_pickle=False) as archive:
        stored_methods=archive["method_order"].tolist(); stored_images=archive["images"].copy()
    return targets,stored_methods,stored_images

def verify_reconstruction(vae,var,decoder,device):
    targets,stored_methods,stored_images=load_clean_sources(); record=targets[0]; image_id=record["target"]["image_id"]; label=int(record["target"]["class_index"])
    image=torch.from_numpy(record["pixels"][None].astype(np.float32)/127.5-1).to(device)
    with torch.no_grad(): _,source=source_prepare(vae,image)
    checks=[]
    for family,mode in (("raw",8),("arithmetic",8)):
        for renderer in ("D0","Dc"):
            name=f"{family}_N3060_m8_{renderer}"
            signal=raw_transmit_budget(source,label,mode,3060)[0] if family=="raw" else arithmetic_transmit_budget(encode_prefixes(vae,var,source,label,device,modes=(mode,))[mode],label,mode,3060)[0]
            received=channel_apply(tx_encode(signal), image_id, 2001, 1)["received"]
            phy=raw_receive_budget(received,1,3060) if family=="raw" else arithmetic_receive_budget(received,1,3060)
            clean=clean_digital(phy,family,renderer,vae,var,decoder,device)
            index=stored_methods.index(name)
            old=stored_images[index,0]
            maximum=float(np.abs(clean-old).max())
            checks.append({"method":name,"max_abs_difference":maximum,"old_method_index":index})
            if maximum>2e-5: raise RuntimeError(f"clean reconstruction mismatch {name}: {maximum}")
    return checks

def run(args):
    configure(); config=read_json(TIMING_CONFIG); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0"); vae,var=load_models(model_paths(),device); decoder=load_decoder(vae,device); scale=scale_statistics(device)
    arms={name:load_arm(name,(32,16,16),scale,device) for name in ("enhancement512","enhancement1024","receiver_only_refiner")}
    targets,_,_=load_clean_sources(); checks=verify_reconstruction(vae,var,decoder,device)
    sources=config["timing_sources"]; snrs=config["snrs_db"]; seeds=config["noise_seeds"]; methods=config["methods"]
    warm=targets[sources[0]]; warm_img=torch.from_numpy(warm["pixels"][None].astype(np.float32)/127.5-1).to(device)
    with torch.no_grad(): f,warm_source=source_prepare(vae,warm_img); _=complete_latent(vae,var,warm_source[:8],int(warm["target"]["class_index"]),device)
    rows=[]
    for index in sources:
        record=targets[index]; label=int(record["target"]["class_index"])
        for snr in snrs:
            for seed in seeds:
                for method in methods:
                    # Include CPU RGB input, host-to-device transfer, and
                    # source preparation in the TX endpoint.
                    tx_start=time.perf_counter()
                    image=cpu_rgb_to_device(record["pixels"], device)
                    with torch.no_grad():
                        f,source=source_prepare(vae,image)
                        if method in ("m8_D0","m8_Dc_only","m8_receiver_only_refiner"):
                            signal=transmit_whole(source,label,8); kind="base"
                        elif method.startswith("m8_plus_latent_"):
                            uses=512 if method.endswith("512") else 1024; arm_name="enhancement512" if uses==512 else "enhancement1024"
                            base=transmit_whole(source,label,8); base_latent=complete_latent(vae,var,source[:8],label,device)
                            wave=arms[arm_name].encoder(f-base_latent,base_latent)[0].cpu().numpy(); signal=tx_encode(base, wave); kind=arm_name
                        else:
                            family=method.split("_N",1)[0]; parts=method.split("_"); budget=int(parts[1][1:]); mode=int(parts[2][1:])
                            if family=="raw": signal,_=raw_transmit_budget(source,label,mode,budget)
                            else: signal,_=arithmetic_transmit_budget(encode_prefixes(vae,var,source,label,device,modes=(mode,))[mode],label,mode,budget)
                            kind=family
                    synchronize(device); tx_ms=(time.perf_counter()-tx_start)*1000
                    encoded = signal if isinstance(signal, dict) else tx_encode(signal)
                    channel = channel_apply(encoded, record["target"]["image_id"], seed, snr)
                    rx_start=time.perf_counter()
                    with torch.no_grad():
                        if method in ("m8_D0","m8_Dc_only","m8_receiver_only_refiner") or method.startswith("m8_plus_latent_"):
                            def complete_base(prefix, received_label):
                                return complete_latent(vae,var,prefix,received_label,device)
                            base_result=rx_base(channel["base_received"],snr,complete_fn=complete_base,device=device)
                            base_latent=base_result["latent"]; status=torch.as_tensor(base_result["status"][None],device=device)
                            if method=="m8_D0": final=torch.full((1,3,256,256),.5,device=device) if not base_result["header_ok"] else original_rgb(vae,base_latent)
                            elif method=="m8_Dc_only": final=render_received(decoder,base_latent,status)
                            elif method=="m8_receiver_only_refiner": final=render_received(decoder,arms["receiver_only_refiner"].receiver(None,base_latent,torch.tensor([snr],device=device),status),status)
                            else:
                                uses=512 if method.endswith("512") else 1024; arm_name="enhancement512" if uses==512 else "enhancement1024"
                                latent=rx_continuous(channel["enhancement_received"],base_result,arms[arm_name],snr,device=device)
                                final=render_received(decoder,latent,status)
                        else:
                            family=method.split("_N",1)[0]; parts=method.split("_"); budget=int(parts[1][1:]); mode=int(parts[2][1:]); phy=raw_receive_budget(channel["received"],snr,budget) if family=="raw" else arithmetic_receive_budget(channel["received"],snr,budget)
                            final=torch.as_tensor(clean_digital(phy,family,"Dc",vae,var,decoder,device)[None],device=device)
                    # Include device-to-host conversion in the RX endpoint.
                    synchronize(device); _final_cpu=device_rgb_to_cpu(final); rx_ms=(time.perf_counter()-rx_start)*1000
                    rows.append({"source_index":index,"image_id":record["target"]["image_id"],"snr_db":snr,"noise_seed":seed,"method":method,"tx_ms":tx_ms,"rx_ms":rx_ms,"total_ms":tx_ms+rx_ms,"kind":kind})
    with (output/"per_call.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for method in methods:
        values=[x for x in rows if x["method"]==method]
        summary.append({"method":method,"calls":len(values),"tx_ms_mean":float(np.mean([float(x["tx_ms"]) for x in values])),"rx_ms_mean":float(np.mean([float(x["rx_ms"]) for x in values])),"total_ms_mean":float(np.mean([float(x["total_ms"]) for x in values]))})
    with (output/"summary.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    atomic={"status":"CORRECTED_ENDPOINT_TIMING_COMPLETE","calls":len(rows),"sources":sources,"snrs_db":snrs,"seeds":seeds,"reconstruction_checks":checks,"noise_outside_rx_timer":True,"no_redundant_D0_for_Dc":True,"no_free_Fb_TX_for_digital":True,"new_holdout_used":False}
    (output/"completion.json").write_text(json.dumps(atomic,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default=str(OUT_ROOT/"online_timing_v2"));run(p.parse_args())
