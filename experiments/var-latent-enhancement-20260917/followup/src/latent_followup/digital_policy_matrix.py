from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from latent_enhancement.runtime import configure, digest, model_paths, settings
from latent_enhancement_b.common import load_decoder
from latent_enhancement_eval.runner import (
    arithmetic_receive_budget, arithmetic_transmit_budget, raw_receive_budget,
    raw_transmit_budget,
)
from latent_enhancement.latent import complete_latent, original_rgb
from var_comm.mode_policies import choose_modes
from var_comm.next_scale_prior import load_models
from var_comm.progressive import prefix_key
from var_comm.quality import load_quality_models
from var_comm.prefix_training_data import read_image_population
from var_comm.study import seeded_noise
from var_comm.whole_entropy import decode_source, encode_prefixes, payload_key

PROJECT = Path(__file__).resolve().parents[5]
VAR_COMM = PROJECT
CALIB = VAR_COMM / "outputs/COMMUNICATION-CONVERGENCE-20260915/CALIBRATION_001"
EXP = VAR_COMM / "experiments/var-latent-enhancement-20260917"
OUT = VAR_COMM / "outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/digital_policy_v1"
CONFIG = EXP / "followup/config.json"
DIGITAL_CONFIG = VAR_COMM / "configs/communication_decision_study.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    if not rows: raise RuntimeError("empty rows")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


_CALIB_IMAGES = None
_CALIB_LABELS = None
_CALIB_IDS = None


def load_source(index):
    global _CALIB_IMAGES, _CALIB_LABELS, _CALIB_IDS
    if _CALIB_IMAGES is None:
        _CALIB_IMAGES, _CALIB_LABELS, _CALIB_IDS, _ = read_image_population("calibration")
    population = read_json(CALIB / "population.json")[index]
    with np.load(CALIB / "images" / f"{index:04d}" / "source_tokens.npz", allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(np.int64)
        label = int(archive["label"])
    if _CALIB_IDS[index] != population["image_id"] or int(_CALIB_LABELS[index]) != label:
        raise RuntimeError("calibration source population binding changed")
    source_rgb = _CALIB_IMAGES[index].numpy().astype(np.float32) / 255.0
    source = [part for part in np.split(tokens, np.cumsum([1,4,9,16,25,36,64,100,169,256])[:-1])]
    if [len(part) for part in source] != [1,4,9,16,25,36,64,100,169,256]:
        raise RuntimeError("calibration scale token split changed")
    return {"index": index, "image_id": population["image_id"], "label": label, "tokens": tokens,
            "source": source,
            "source_rgb": source_rgb}


def metric_rows(images, target, perceptual, device):
    target_tensor = torch.as_tensor(target[None], device=device, dtype=torch.float32)
    output = []
    for start in range(0, len(images), 8):
        prediction = torch.as_tensor(np.stack(images[start:start + 8]), device=device, dtype=torch.float32)
        reference = target_tensor.expand(len(prediction), -1, -1, -1)
        psnr = -10 * torch.log10((prediction - reference).square().flatten(1).mean(1))
        lpips = perceptual(prediction * 2 - 1, reference * 2 - 1).reshape(-1)
        output.extend({"psnr_db": float(psnr[i]), "lpips": float(lpips[i])} for i in range(len(prediction)))
    return output


def candidate_key(phy, family):
    if phy["label"] is None: return "header_erasure"
    return prefix_key(phy["prefix"], phy["label"]) if family == "raw" else payload_key(phy)


def clean_render(phy, family, renderer, vae, var, decoder, device, cache):
    key = (family, candidate_key(phy, family))
    if key not in cache:
        if phy["label"] is None:
            cache[key] = {"latent": None, "prefix": [], "source_complete": False}
        elif family == "raw":
            latent = complete_latent(vae, var, phy["prefix"], phy["label"], device)
            cache[key] = {"latent": latent, "prefix": phy["prefix"], "source_complete": True}
        else:
            decoded = decode_source(phy, vae, var, device, render=False)
            latent = complete_latent(vae, var, decoded["prefix"], phy["label"], device)
            cache[key] = {"latent": latent, "prefix": decoded["prefix"], "source_complete": decoded["source_complete"],
                          "source_error": decoded["source_error"]}
    item = cache[key]
    if item["latent"] is None:
        image = np.full((3,256,256), .5, dtype=np.float32)
    elif renderer == "D0":
        image = original_rgb(vae, item["latent"])[0].cpu().numpy()
    else:
        image = decoder(item["latent"])[0].cpu().numpy()
    return image, item


def evaluate_source(record, vae, var, decoder, perceptual, device, budget, seeds, snrs, renderers=("D0","Dc"), dino=None):
    if not renderers or any(r not in ("D0","Dc") for r in renderers):raise ValueError("invalid renderers")
    source, label = record["source"], record["label"]
    payloads = encode_prefixes(vae, var, source, label, device, modes=(7, 8, 9))
    prepared = {}
    for family in ("raw", "arithmetic"):
        for mode in (7,8,9):
            payload = payloads[mode] if family == "arithmetic" else None
            prepared[family,mode] = (raw_transmit_budget(source,label,mode,budget) if family == "raw" else
                                     arithmetic_transmit_budget(payload,label,mode,budget))
    images, rows = {}, []
    candidate_cache = {}
    for family in ("raw", "arithmetic"):
        for mode in (7,8,9):
            for renderer in renderers:
                images[family,budget,mode,renderer] = []
    for snr in snrs:
        for seed in seeds:
            for family in ("raw", "arithmetic"):
                for mode in (7,8,9):
                    signal, ledger = prepared[family,mode]
                    received = signal + seeded_noise(record["image_id"], seed, signal.shape) / np.sqrt(10 ** (snr/10))
                    phy = raw_receive_budget(received,snr,budget) if family == "raw" else arithmetic_receive_budget(received,snr,budget)
                    base_image, candidate = clean_render(phy, family, renderers[0], vae, var, decoder, device, candidate_cache)
                    actual_prefix = candidate.get("prefix", [])
                    correct = bool(phy["header"]["accepted"] and phy.get("body_crc_accepted",False) and
                                   phy["label"] == label and phy["mode"] == mode and len(actual_prefix)==mode and
                                   all(np.array_equal(g,t) for g,t in zip(actual_prefix,source[:mode])))
                    for renderer in renderers:
                        image = base_image if renderer == renderers[0] else clean_render(phy,family,renderer,vae,var,decoder,device,cache=candidate_cache)[0]
                        images[family,budget,mode,renderer].append(image)
                        rows.append({"population":"calibration","image_index":record["index"],"image_id":record["image_id"],
                                     "family":family,"budget":budget,"mode":mode,"renderer":renderer,"snr_db":float(snr),"seed":int(seed),
                                     "raw_payload_bits":ledger["payload_bits"],"actual_payload_bits":ledger["actual_payload_bits"],
                                     "header_accepted":int(phy["header"]["accepted"]),"body_crc_accepted":int(phy.get("body_crc_accepted",False)),
                                     "accepted_correct":int(correct),"source_complete":int(candidate["source_complete"])})
    for family in ("raw","arithmetic"):
        for mode in (7,8,9):
            for renderer in renderers:
                values=metric_rows(images[family,budget,mode,renderer],record["source_rgb"],perceptual,device)
                if dino is not None:
                    from var_comm.quality import quality_metrics
                    actual,_,_=quality_metrics(record["source_rgb"],images[family,budget,mode,renderer],perceptual,dino,device)
                    values=[{"psnr_db":r["psnr_db"],"lpips":r["lpips_alex"],"dino_cosine":r["dino_cosine"]} for r in actual]
                offset=0
                for row in rows:
                    if row["family"]==family and row["mode"]==mode and row["renderer"]==renderer:
                        row.update(values[offset]); offset+=1
    return rows


def aggregate_candidates(rows):
    groups={}
    for row in rows:
        key=(row["family"],int(row["budget"]),row["renderer"],float(row["snr_db"]),int(row["mode"]))
        groups.setdefault(key,[]).append(row)
    summary=[]
    for key,values in sorted(groups.items()):
        family,budget,renderer,snr,mode=key
        success=np.array([int(x["accepted_correct"]) for x in values])
        summary.append({"family":family,"budget":budget,"renderer":renderer,"snr_db":snr,"mode":mode,"frames":len(values),
                        "source_images":len({x["image_id"] for x in values}),"raw_payload_bits":int(values[0]["raw_payload_bits"]),
                        "accepted_correct_probability":float(success.mean()),"source_packet_BLER":float(1-success.mean()),
                        "source_index_goodput":float(values[0]["raw_payload_bits"]*success.mean()),
                        "psnr_db":float(np.mean([float(x["psnr_db"]) for x in values])),
                        "lpips":float(np.mean([float(x["lpips"]) for x in values])),
                        "actual_payload_bits_mean":float(np.mean([int(x["actual_payload_bits"]) for x in values])),
                        "header_failure_probability":float(np.mean([not int(x["header_accepted"]) for x in values])),
                        "body_crc_failure_given_header_accept":float(np.mean([not int(x["body_crc_accepted"]) for x in values if int(x["header_accepted"])])) if any(int(x["header_accepted"]) for x in values) else ""})
    return summary


def freeze_policies(summary, config):
    actions={}
    for family in ("raw","arithmetic"):
        actions[family]={}
        for budget in config["budgets"]:
            actions[family][str(budget)]={}
            for renderer in config["renderers"]:
                actions[family][str(budget)][renderer]={}
                for snr in config["snrs_db"]:
                    candidates=[x for x in summary if x["family"]==family and int(x["budget"])==budget and x["renderer"]==renderer and float(x["snr_db"])==float(snr)]
                    selected=choose_modes(candidates,config["reliability_BLER_target"],config["quality_PSNR_guard_db"],1e-12)
                    actions[family][str(budget)][renderer][str(float(snr))]=selected
    return actions


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default=str(OUT)); parser.add_argument("--max-sources",type=int); args=parser.parse_args()
    config=read_json(CONFIG); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    configure()
    device=torch.device("cuda:0");
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    vae,var=load_models(model_paths(),device); decoder=load_decoder(vae,device)
    quality=yaml.safe_load((VAR_COMM/"configs/progressive_channel.yaml").read_text())["quality"]
    perceptual,_,_=load_quality_models(quality,device)
    rows=[]; sources=read_json(CALIB/"population.json"); limit=args.max_sources or len(sources)
    for index in range(limit):
        directory=output/"images"/f"{index:04d}"; receipt=directory/"receipt.json"
        if receipt.exists():
            with (directory/"per_frame.csv").open(newline="") as h: rows.extend(csv.DictReader(h))
            continue
        record=load_source(index); start=time.perf_counter(); directory.mkdir(parents=True,exist_ok=True)
        current=evaluate_source(record,vae,var,decoder,perceptual,device,config["budgets"][0],config["noise_seeds"]["calibration"],config["snrs_db"])
        # The two budgets are intentionally separate fixed candidate grids.
        for budget in config["budgets"][1:]:
            current.extend(evaluate_source(record,vae,var,decoder,perceptual,device,budget,config["noise_seeds"]["calibration"],config["snrs_db"]))
        write_rows(directory/"per_frame.csv",current); rows.extend(current)
        atomic_json(receipt,{"status":"SOURCE_COMPLETE","source_index":index,"image_id":record["image_id"],"rows":len(current),"elapsed_seconds":time.perf_counter()-start})
        atomic_json(output/"status.json",{"status":"CALIBRATION_DIGITAL_MATRIX_RUNNING","completed_sources":index+1,"total_sources":limit,"rows":len(rows),"timestamp":time.time()})
        print(f"calibration digital {index+1}/{limit} rows={len(current)} seconds={time.perf_counter()-start:.1f}",flush=True)
    write_rows(output/"per_frame.csv",rows); summary=aggregate_candidates(rows); write_rows(output/"candidate_summary.csv",summary)
    actions=freeze_policies(summary,config)
    atomic_json(output/"policies.json",{"status":"CALIBRATION_DIGITAL_POLICIES_FROZEN","config_sha256":digest(CONFIG),"actions":actions,"calibration_sources":limit,"rows":len(rows),"DINO_used_for_selection":False,"m10":"excluded_current_header_legal_mode_domain_7_8_9"})
    atomic_json(output/"completion.json",{"status":"CALIBRATION_DIGITAL_MATRIX_COMPLETE","sources":limit,"rows":len(rows),"policies":"policies.json","new_holdout_used":False})

if __name__=="__main__": main()
