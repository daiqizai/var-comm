#!/usr/bin/env python3
"""Use the existing frozen metric models; no model selection or source reprocessing."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
from pytorch_msssim import ssim
import yaml

from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import sha256, write_csv, write_json


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied; metrics must not interfere with timed reception")
    inputs = json.loads((arguments.inputs / ("calibration_inputs.json" if arguments.calibration else "development_inputs.json")).read_text())
    sources = {record["image_id"]: record for record in inputs}
    rows, bindings = [], {}
    for directory in arguments.runs:
        if not (directory / "completion.json").exists():
            raise RuntimeError("do not score incomplete author populations as completed")
        path = directory / "per_frame.csv"
        bindings[str(path)] = sha256(path)
        for row in read_rows(path):
            if row["performance_population"] != ("calibration" if arguments.calibration else "development"):
                raise RuntimeError("calibration and development cannot be mixed")
            row["generation_run"] = str(directory)
            rows.append(row)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((PROJECT / "configs/progressive_channel.yaml").read_text())["quality"]
    for name in ("alexnet_checkpoint", "dino_checkpoint"):
        if sha256(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError("original metric checkpoint changed")
    perceptual, dino, _weights = load_quality_models(paths, device)
    result = []
    for ordinal, identifier in enumerate(sorted({row["image_id"] for row in rows})):
        source_record = sources[identifier]
        if sha256(source_record["path"]) != source_record["file_sha256"]:
            raise RuntimeError("source RGB cache changed")
        source = np.load(source_record["path"], allow_pickle=False).astype(np.float32) / 255
        selected = [row for row in rows if row["image_id"] == identifier]
        unique = {}
        for row in selected:
            if row["source_pixels_sha256"] != source_record["source_pixels_sha256"]:
                raise RuntimeError("author inference used different source pixels")
            if row["image_sha256"] not in unique:
                with np.load(row["image_archive"], allow_pickle=False) as archive:
                    image = archive["images"][int(row["image_slot"])].copy()
                import hashlib
                if hashlib.sha256(image.tobytes()).hexdigest() != row["image_sha256"]:
                    raise RuntimeError("frozen author RGB changed")
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise RuntimeError("illegal author image cannot disappear from statistics")
                unique[row["image_sha256"]] = image
        keys = list(unique)
        scores, _source_feature, _features = quality_metrics(source, [unique[key] for key in keys], perceptual, dino, device)
        lookup = {}
        reference = torch.from_numpy(source)[None]
        for key, score in zip(keys, scores):
            with torch.no_grad():
                structural = float(ssim(torch.from_numpy(unique[key])[None], reference, data_range=1., size_average=False)[0])
            lookup[key] = {"psnr_db": score["psnr_db"], "ssim": structural, "lpips": score["lpips_alex"], "dino": score["dino_cosine"]}
        for row in selected:
            result.append({**row, **lookup[row["image_sha256"]]})
        if (ordinal + 1) % 10 == 0:
            print(f"Scored {ordinal + 1} sources", flush=True)
    fields = sorted({name for row in result for name in row})
    write_csv(output / "per_frame.csv", [{name: row.get(name, "") for name in fields} for row in result])
    regions = {"primary": [1., 4., 7.], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in (1., 4., 7., 13., 19.)})
    summaries = []
    for method, regime in sorted({(row["method"], row["protocol"]) for row in result}):
        for region, snrs in regions.items():
            selected = [row for row in result if row["method"] == method and row["protocol"] == regime and float(row["snr_db"]) in snrs]
            identifiers = sorted({row["image_id"] for row in selected})
            if not selected:
                continue
            means = {metric: float(np.mean([np.mean([float(row[metric]) for row in selected if row["image_id"] == identifier]) for identifier in identifiers]))
                     for metric in ("psnr_db", "ssim", "lpips", "dino")}
            summaries.append({"method": method, "protocol": regime, "region": region, "sources": len(identifiers),
                              "rows": len(selected), "complex_uses": selected[0]["complex_uses"], **means})
    write_csv(output / "summary.csv", summaries)
    write_json(output / "completion.json", {"status": "AUTHOR_METRICS_COMPLETE", "rows": len(result), "bindings": bindings,
               "new_training_or_holdout": False, "metric_models_selected_with_DINO": False,
               "per_frame_sha256": sha256(output / "per_frame.csv"), "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    run(parser.parse_args())
