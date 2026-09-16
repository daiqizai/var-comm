#!/usr/bin/env python3
"""Hash already-used communication train/calibration images and prior validation sources only."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from benchmark_frozen_systems import digest
from var_comm.next_scale_prior import preprocess
from var_comm.prefix_training_data import IMAGE_CACHE, MANIFEST_SHA
from var_comm.study import create_output, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    inventory = json.loads((arguments.catalog / "inventory.json").read_text())
    if inventory["holdout_pixels_accessed"] or inventory["image_files_opened"]:
        raise RuntimeError("the unused catalog already accessed image content")
    output = create_output(arguments.output_dir)
    torch.set_num_threads(4)
    file_hashes, pixel_hashes = set(inventory["known_file_or_pixel_hashes"]), set(inventory["known_file_or_pixel_hashes"])
    source_manifest = IMAGE_CACHE / "manifest.json"
    if sha256(source_manifest) != MANIFEST_SHA:
        raise RuntimeError("communication training image manifest changed")
    manifest = json.loads(source_manifest.read_text())
    counts, bindings = {}, {str(source_manifest): MANIFEST_SHA}
    for population in manifest["populations"]:
        count = 0
        for descriptor in population["shards"]:
            path = IMAGE_CACHE / descriptor["path"]
            if sha256(path) != descriptor["sha256"]:
                raise RuntimeError("previously used image shard changed")
            bindings[str(path)] = descriptor["sha256"]
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for image in shard["targets_u8"]:
                pixels = image.numpy()
                pixel_hashes.add(digest(pixels))
                if population["name"] == "train":
                    pixel_hashes.add(digest(pixels[:, :, ::-1]))
                count += 1
        counts[population["name"]] = count
        print(f"known {population['name']} content hashes: {count}", flush=True)
    split_path = ROOT.parent / "VAR-MAP-GATE0/manifests/split_manifest.json"
    split = json.loads(split_path.read_text())
    known_indices = set(inventory["used_val_indices"])
    pattern = re.compile(r"ILSVRC2012_val_(\d{8})")
    count = 0
    for path in sorted(Path(split["val_root"]).glob("*/*.JPEG")):
        match = pattern.search(path.name)
        if match is None or match.group(1) not in known_indices:
            continue
        file_hashes.add(sha256(path))
        unused_image, pixel_hash = preprocess(path)
        pixel_hashes.add(pixel_hash)
        count += 1
    write_json(output / "completion.json", {"status": "KNOWN_USED_CONTENT_EXCLUSION_READY", "completed_local": datetime.now().astimezone().isoformat(),
        "catalog_inventory_sha256": sha256(arguments.catalog / "inventory.json"), "counts": counts, "previously_used_val_files_opened": count,
        "new_holdout_files_opened": 0, "training_horizontal_flip_hashes_included": True,
        "file_hashes": sorted(file_hashes), "pixel_hashes": sorted(pixel_hashes), "source_image_bindings": bindings,
        "source_script_sha256": sha256(Path(__file__)), "coverage_caveat": "not a near-duplicate or full visual-pretraining-corpus audit"})
    print(f"known validation content hashes: {count}; new holdout opened: 0", flush=True)


if __name__ == "__main__":
    main()
