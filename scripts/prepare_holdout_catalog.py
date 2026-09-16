#!/usr/bin/env python3
"""Inventory prior source identities and unused image paths without opening holdout images."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
IDENTITY = re.compile(r"ILSVRC2012_val_(\d{8})")
HASH_FIELD = re.compile(r'"(file_sha256|source_file_sha256|preprocessed_rgb_sha256|source_pixels_sha256|rgb_sha256)"\s*:\s*"([a-f0-9]{64})"')
NAME = re.compile(r"(population|manifest|selected|sample|ids|identity).*\.jsonl?$")


def scan_metadata(path):
    digest, indices, hashes = hashlib.sha256(), set(), set()
    trailing = ""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            text = trailing + block.decode("utf-8", errors="replace")
            indices.update(IDENTITY.findall(text))
            hashes.update(value for key, value in HASH_FIELD.findall(text))
            trailing = text[-512:]
    return digest.hexdigest(), indices, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/HOLDOUT_CATALOG_001")
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"):
        raise ValueError("catalog output escaped the study")
    roots = [ROOT / "outputs", WORKSPACE / "VAR-MAP-GATE0/results", WORKSPACE / "VAR-MAP-GATE0/manifests",
             WORKSPACE / "channel-adaptive-semantic-drift-controlled-diffusion-jscc/outputs", WORKSPACE / "var-next-scale-comm/outputs"]
    result = subprocess.run(["rg", "--files", "-uu", *map(str, roots), "-g", "*.json", "-g", "*.jsonl", "-g", "*.csv"], text=True, capture_output=True, check=True)
    paths = sorted({Path(line) for line in result.stdout.splitlines() if NAME.search(Path(line).name) or Path(line).suffix == ".csv"})
    sources, used, known_hashes = [], set(), set()
    for path in paths:
        if "HOLDOUT_CATALOG_" in str(path):
            continue
        digest, identities, hashes = scan_metadata(path)
        sources.append({"path": str(path), "sha256": digest, "prior_val_ids_found": len(identities), "content_hashes_found": len(hashes)})
        used.update(identities)
        known_hashes.update(hashes)
    split_path = WORKSPACE / "VAR-MAP-GATE0/manifests/split_manifest.json"
    split = json.loads(split_path.read_text())
    labels = {entry["synset"]: int(entry["class_index"]) for entry in split["entries"]}
    if len(labels) != 1000 or set(labels.values()) != set(range(1000)):
        raise RuntimeError("original ImageNet class mapping is incomplete")
    image_root = Path(split["val_root"]).resolve()
    candidates, catalog_count = [], 0
    for path in sorted(image_root.glob("*/*.JPEG")):
        match = IDENTITY.search(path.name)
        if match is None or path.parent.name not in labels:
            continue
        catalog_count += 1
        if match.group(1) not in used:
            candidates.append({"path": str(path), "image_id": str(path.relative_to(image_root).with_suffix("")),
                               "class_index": labels[path.parent.name], "val_index": match.group(1)})
    counts = {label: sum(row["class_index"] == label for row in candidates) for label in range(1000)}
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "METADATA_ONLY_HOLDOUT_CANDIDATE_CATALOG_NOT_TEST_DATA_ACCESSED",
              "created_local": datetime.now().astimezone().isoformat(), "catalog_images": catalog_count,
              "known_prior_val_ids": len(used), "eligible_paths": len(candidates), "classes_with_candidates": sum(count > 0 for count in counts.values()),
              "minimum_candidates_per_class": min(counts.values()), "image_files_opened": 0, "holdout_pixels_accessed": False,
              "coverage": "identity-bearing JSON/JSONL plus all CSV artifacts in VAR and historical source projects; not an assertion about undocumented manual uses",
              "method_not_frozen_yet": True, "source_artifacts": sources, "used_val_indices": sorted(used),
              "known_file_or_pixel_hashes": sorted(known_hashes), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "inventory.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    (output / "candidate_paths.json").write_text(json.dumps({"role": "metadata_only_unused_candidates_not_accessed", "images": candidates}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "catalog_images", "known_prior_val_ids", "eligible_paths", "classes_with_candidates", "minimum_candidates_per_class", "image_files_opened")}, indent=2))


if __name__ == "__main__":
    main()
