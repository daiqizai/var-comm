"""Read-only verification of the original SP pilot assets on the shared PFS."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np


def require(value, label):
    if not value:
        raise ValueError(label)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filehash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verified_bytes(entry):
    p = Path(entry["path"])
    raw = p.read_bytes()
    require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], str(p))
    return raw


def rows(file):
    with file.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "science", "quality", "lengths", "model-routes", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    require(not a.receipt.exists(), "receipt already exists")
    start, cpu = time.time(), time.process_time()
    pilot_hash = filehash(a.pilot)
    require(pilot_hash == "b89a964f24127d9fcd2bddd6b7a88d8b366f53764632a71d93dfcf3ffcd94853", "unregistered Pilot50")
    roster = json.loads(a.pilot.read_bytes())["records"]
    require(len(roster) == 50 and len({r["source_id"] for r in roster}) == 50, "roster")
    clean = {(r["source_id"], int(r["m"]), int(r["K"])): r
             for r in rows(a.quality) if r["native_reconstruction_sha256"]}
    split = {(int(r["pilot_slot"]), int(r["j"]), int(r["m"]), int(r["K"])): r
             for r in rows(a.lengths) if r["enhancement_bits"]}
    require(len(clean) == 1200 and len(split) == 3400, "reference tables")
    model_ids = {"e3d757a5a16206f5f9be345d0e6330b0f1fd6c78bf167670e70378b429d931c4",
                 "7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c",
                 "4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb"}
    models = json.loads(a.model_routes.read_bytes())["files"]
    require(len(models) == 3 and {m["sha256"] for m in models} == model_ids, "model routes")
    for item in models:
        p = Path(item["path"])
        require(p.stat().st_size == item["bytes"] and filehash(p) == item["sha256"], str(p))
    stream_manifest = hashlib.sha256()
    for i, r in enumerate(roster):
        require(r["pilot_slot"] == i, "source order")
        verified_bytes(r["checkpoint"])
        with np.load(io.BytesIO(verified_bytes(r["archive"])), allow_pickle=False) as z:
            pixels, tokens = z["pixels"], z["tokens"]
            require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256)
                    and digest(pixels.tobytes()) == r["preprocessing_id"], ("pixels", i))
            require(tokens.dtype == np.int64 and tokens.shape == (680,)
                    and digest(b"int64:680\0" + tokens.astype("<i8", copy=False).tobytes()) == r["tokens_sha256"], ("tokens", i))
        record = json.loads((a.science / "streams" / f"{i:02d}.json").read_bytes())
        require(record["source"]["source_id"] == r["source_id"]
                and record["source"]["tokens_sha256"] == r["tokens_sha256"], ("stream id", i))
        with np.load(io.BytesIO(verified_bytes(record["streams"])), allow_pickle=False) as coded:
            for (s, j, m, k), ref in split.items():
                if s != i:
                    continue
                require(ref["source_id"] == r["source_id"], ("length id", i))
                for name, length in ((f"base_{j}", ref["base_bits"]),
                                     (f"enhancement_{j}_{m}_{k}", ref["enhancement_bits"])):
                    bits = coded[name]
                    require(bits.dtype == np.uint8 and bits.shape == (int(length),)
                            and np.all(bits <= 1), (i, name))
        stream_manifest.update(bytes.fromhex(record["streams"]["sha256"]))
    source_pixels = {r["source_id"]: r["preprocessing_id"] for r in roster}
    images_file = a.science / "all_clean_images.json"
    archived = json.loads(images_file.read_bytes())
    require(len(archived) == len(clean), "clean archive count")
    visited = set()
    for r in archived:
        key = (r["source_id"], int(r["m"]), int(r["K"]))
        require(key in clean and key not in visited, key)
        visited.add(key)
        ref = clean[key]
        require(ref["source_pixel_sha256"] == source_pixels[key[0]]
                and ref["native_reconstruction_sha256"] == r["image_array_sha256"], key)
        with np.load(io.BytesIO(verified_bytes(r["image_archive"])), allow_pickle=False) as z:
            im = np.ascontiguousarray(z[r["image_key"]])
            sha = digest(str(im.dtype).encode() + str(im.shape).encode() + im.tobytes())
            require(im.dtype == np.float32 and im.shape == (3, 256, 256)
                    and np.isfinite(im).all() and im.min() >= 0 and im.max() <= 1
                    and sha == r["image_array_sha256"], key)
    summary = dict(status="PASS_PFS_ASSET_IDENTITY_ONLY", pilot_manifest_sha256=pilot_hash,
                   verified_native_source_pixels=50, verified_native_680_token_arrays=50,
                   verified_per_source_metadata=50, verified_split_archives=50,
                   verified_historical_double_stream_length_rows=len(split),
                   ordered_split_archive_hash_list_sha256=stream_manifest.hexdigest(),
                   verified_native_clean_float32_reconstructions=len(visited),
                   clean_image_manifest_sha256=filehash(images_file), verified_frozen_model_files=models,
                   route_manifest_sha256=filehash(a.model_routes), missing_full_endpoint_lengths_added=0,
                   new_quality_scores=0, new_wireless_frames=0, training_steps=0, gpu_calls=0,
                   utc_finished_unix=time.time(), wall_seconds=time.time()-start,
                   parent_cpu_seconds=time.process_time()-cpu)
    with a.receipt.open("x", encoding="utf8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
    print("PASS: 50 native sources, 50 split archives, 3400 length pairs, 1200 native images, three frozen weight files.")


if __name__ == "__main__":
    main()
