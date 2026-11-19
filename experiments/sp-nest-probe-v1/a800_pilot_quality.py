"""Score all A800 pilot endpoints and the real fallback points of the 300 single policies."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from a800_clean_pilot import RecoveredContext, array_hash, bind_native, write_json
from profiles import OFF, QUARTER, capacity, count
from qualify_native_cuda import snapshot
from split_prefixes import SplitPrefixes
from verify_pfs_cache import digest, filehash, require, verified_bytes


def metric_manifest(routes, assets):
    def read_pinned(pin):
        raw = Path(pin["path"]).read_bytes()
        require(digest(raw) == pin["sha256"] and len(raw) == pin.get("bytes", len(raw)), pin["path"])
        return json.loads(raw)

    spec = read_pinned(assets["overlay_identity"])
    require(isinstance(spec, dict), "metric dependencies")
    pins = routes["metric_config"]
    original = read_pinned(pins["manifest"])
    locations = read_pinned(pins["dino_receipt"])
    weights = read_pinned(pins["weight_receipt"])
    dino = copy.deepcopy(original["dinov2_vitl14"])
    implementation = dino["implementation"]
    root = Path(locations["actual_root"])
    require(len(implementation["files"]) == locations["file_count"] == 157, "DINO implementation count")
    require({p.relative_to(root).as_posix() for p in root.rglob("*.py")} == set(implementation["files"]),
            "DINO source set")
    for rel, sha in implementation["files"].items():
        require(filehash(root / rel) == sha, rel)
    implementation["path"] = str(root)
    sha = dino["weights"]["sha256"]
    require(sha == "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428", "DINO-L/14")
    weight, = [w for w in weights["mapping"] if w["sha256"] == sha]
    registered = Path(weight["actual_path"])
    checkpoint = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028")
    rel = registered.relative_to("/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu")
    actual = registered if registered.is_file() else checkpoint / rel
    require(actual.stat().st_size == weight["bytes"] and filehash(actual) == sha, str(actual))
    dino["weights"]["path"] = str(actual)
    return {"dinov2_vitl14": dino, "dists": assets["dists"], "dreamsim": assets["dreamsim"]}


def needed_single_endpoints(lengths, policies):
    require(len(lengths) == 651 and np.all(np.diff(lengths) >= 0), "actual terminated length order")
    result = set()
    for q, rate, ceiling in policies:
        feasible = min(ceiling - OFF[4] + 1, int(np.searchsorted(lengths, capacity(q, rate), side="right")))
        if feasible:
            n = OFF[4] + feasible - 1
            m = int(np.searchsorted(OFF, n, side="right") - 1)
            result.add((m, int(n - OFF[m])))
    return sorted(result.difference(QUARTER), key=count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("repo", "pilot", "source", "source-control", "routes", "metrics", "policies", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir()
    start = time.time()
    counts = dict(native_renders=0, scored_pairs=0, source_features=0, verified_clean=0, verified_extra_single=0, grays=0)
    try:
        write_json(a.out / "registration.json", dict(inputs="only independently recovered A800 arrays, all 50 in fixed order",
                   metrics=["PSNR","DINOv2 ViT-L/14 cosine","DISTS","DreamSim"], calibration="DINOv2",
                   scientific_calls_while_waiting=0, training_steps=0, start=start,
                   code_sha256=filehash(__file__), prior_source_sha256=filehash(Path(__file__).with_name("a800_clean_pilot.py")),
                   policies_sha256=filehash(a.policies), metrics_sha256=filehash(a.metrics), expected_clean=1250, gray=50,
                   extra_single="all unique non-quarter endpoints selected by original token-level fitting for 12 MCS x 25 ceilings"))
        while not (a.source_control / "actual_wait.json").is_file():
            time.sleep(3)
        w = json.loads((a.source_control / "actual_wait.json").read_bytes())
        require(w["returncode"] == 0 and not w["timed_out"], "source stage exit")
        done = json.loads((a.source / "completion.json").read_bytes())
        require(done["status"] == "A800_PILOT_SOURCE_AND_RX_COMPLETE" and done["sources"] == 50, "complete native pilot")
        gpu_start, cpu_start = time.time(), time.process_time()
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
        os.nice(10)
        require(filehash(a.policies) == "92199ce8fab34a66d4fa3de6c0ddaea5092ca0e68975cd37b6500056cbd2a313",
                "registered single policy set")
        with a.policies.open(encoding="utf-8-sig", newline="") as f:
            policies = [(int(r["q"]), r["rate"], count((int(r["target_m"]),int(r["target_K"])))) for r in csv.DictReader(f)]
        require(len(set(policies)) == 300, "300 genuine NeST candidates")
        ref = json.loads((a.source/"registration.json").read_bytes())
        require(ref["gpu"]["uuid"] == os.environ["CUDA_VISIBLE_DEVICES"] and ref["pilot_sha256"] == filehash(a.pilot)
                and ref["route_sha256"] == filehash(a.routes), "source, receiver and scorer identities")
        source_id = json.loads((a.source/"identity.json").read_bytes())
        routes = json.loads(a.routes.read_bytes())
        require(filehash(a.metrics) == "13dd56073f73b2886e9d0efb1a66e815951108da9ef94161b02f444545ca6e23", "metric asset identity")
        assets = json.loads(a.metrics.read_bytes())
        require(filehash(assets["loader"]["path"]) == assets["loader"]["sha256"] ==
                "dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066", "original metric loader")
        sys.path.insert(0, assets["dependency_overlay"])
        import torch
        torch.set_num_threads(6)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        require(torch.cuda.device_count() == 1 and str(props.uuid) == ref["gpu"]["uuid"][4:]
                and torch.__version__ == source_id["runtime"]["torch"], "same physical A800 and runtime")
        torch.cuda.set_per_process_memory_fraction((12 << 30) / props.total_memory, device)
        loc = importlib.util.spec_from_file_location("_sp_a800_metrics", assets["loader"]["path"])
        mod = importlib.util.module_from_spec(loc)
        loc.loader.exec_module(mod)
        manifest = metric_manifest(routes, assets)
        codec, render, states = bind_native(a.repo, routes, counts, device)
        require(states == source_id["model_states"], "same renderer")
        with mod.no_network():
            evaluator = mod.MetricEvaluator(manifest, device=device, requested=("dinov2_vitl14","dists","dreamsim"))
        write_json(a.out / "identity.json", dict(device=source_id["gpu_uuid"], model_states=states,
                   evaluator=evaluator.metadata, evaluator_id=evaluator.identity(), selected_metric="dinov2_vitl14_cosine",
                   source_wait=w, no_network=True, metric_source=assets["loader"], native_float32=True))
        roster = json.loads(a.pilot.read_bytes())["records"]
        gray = np.full((3,256,256), 0.5, np.float32)
        fields = ["slot","source_id","kind","m","K","source_pixel_sha256","output_array_sha256",
                  "psnr_db","dinov2_cosine","dists","dreamsim"]
        with (a.out / "scores.csv").open("x", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for slot, row in enumerate(roster):
                require(next(gpu for gpu in snapshot() if gpu["uuid"] == ref["gpu"]["uuid"])["free_MiB"] >= 4096,
                        "shared GPU memory")
                folder = a.source / f"slot_{slot:02d}"
                record = json.loads((folder / "done.json").read_bytes())
                require(row["pilot_slot"] == record["slot"] == slot and row["source_id"] == record["source_id"], "source order")
                for name, sha in record["files"].items():
                    require(filehash(folder / name) == sha, (slot,name))
                with np.load(io.BytesIO(verified_bytes(row["archive"])), allow_pickle=False) as z:
                    pixels = z["pixels"]
                require(digest(pixels.tobytes()) == row["preprocessing_id"], "pixels")
                x = pixels.astype(np.float32) / np.float32(255)
                xt = torch.from_numpy(x[None]).to(device)
                src_features = evaluator.prepare_reference(xt)
                counts["source_features"] += 1
                with np.load(folder / "single_tokens.npz", allow_pickle=False) as z:
                    offsets, bits = z["offsets"], z["bits"]
                extra = needed_single_endpoints(np.diff(offsets), policies)
                with np.load(folder / "rx_cdfs.npz", allow_pickle=False) as z:
                    tables = {(s,key): z[f"s{s}"] for s,key in record["independent_rx_cdf_keys"]}
                cached = SplitPrefixes(SimpleNamespace(Encoder=codec.Encoder, Decoder=codec.Decoder,
                    _inference=codec._inference, _context=lambda family: RecoveredContext(tables),
                    _cdf=lambda family, provider, scale: provider.cdf(scale)))
                extra_images = {}
                for end in extra:
                    i = count(end) - OFF[4]
                    with cached.context() as provider:
                        received = cached._read(provider, 0, end, bits[offsets[i]:offsets[i+1]])
                    extra_images[f"e_{end[0]}_{end[1]}"] = render(received, end)
                    counts["verified_extra_single"] += 1
                if extra_images:
                    np.savez(a.out / f"extra_{slot:02d}.npz", **extra_images)
                with np.load(folder / "native_images.npz", allow_pickle=False) as native:
                    for kind,end in [("gray",None)] + [("clean",e) for e in QUARTER] + [("single_fallback",e) for e in extra]:
                        key = None if end is None else f"e_{end[0]}_{end[1]}"
                        im = gray if key is None else native[key] if kind == "clean" else extra_images[key]
                        oid = array_hash(im)
                        require(im.dtype == np.float32 and im.shape == (3,256,256) and
                                np.isfinite(im).all() and im.min() >= 0. and im.max() <= 1., "native image")
                        if kind == "clean":
                            require(record["native_images"][key] == oid, "native render identity")
                            counts["verified_clean"] += 1
                        elif kind == "gray":
                            counts["grays"] += 1
                        yt = torch.from_numpy(im[None]).to(device)
                        q = evaluator.score(xt,yt,[row["evaluation_class_index"]], label_conditioned=False,prepared=src_features)[0]
                        psnr = float(-10 * np.log10(np.mean((x.astype(np.float64)-im.astype(np.float64))**2)))
                        scores = dict(psnr_db=psnr, dinov2_cosine=q["dinov2_vitl14_cosine"], dists=q["dists"], dreamsim=q["dreamsim"])
                        require(np.isfinite(list(scores.values())).all(), "finite four metrics")
                        writer.writerow(dict(slot=slot, source_id=row["source_id"], kind=kind,
                            m="" if end is None else end[0], K="" if end is None else end[1],
                            source_pixel_sha256=row["preprocessing_id"],output_array_sha256=oid,**scores))
                        counts["scored_pairs"] += 1
                f.flush()
                os.fsync(f.fileno())
                write_json(a.out / "progress.json", dict(completed_sources=slot+1, counts=counts), update=True)
                print(f"Pilot quality {slot+1}/50, {counts['scored_pairs']} four-metric pairs",flush=True)
        require(counts["verified_clean"] == 1250 and counts["grays"] == counts["source_features"] == 50 and
                counts["scored_pairs"] == 1300 + counts["verified_extra_single"], "complete SP0 coverage")
        write_json(a.out / "completion.json", dict(status="A800_SP0_CLEAN_FOUR_METRICS_COMPLETE",
                   source_stage=done["status"], counts=counts, scores_sha256=filehash(a.out/"scores.csv"),
                   training_steps=0, wireless_frames=0, gpu_wall_seconds=time.time()-gpu_start,
                   cpu_seconds=time.process_time()-cpu_start, elapsed_including_dependency=time.time()-start,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    except BaseException as error:
        write_json(a.out/"failure.json", dict(error=repr(error), counts=counts, elapsed=time.time()-start))
        raise


if __name__ == "__main__":
    main()
