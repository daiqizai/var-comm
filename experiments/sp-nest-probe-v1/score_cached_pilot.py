"""SP0: score registered native Pilot50 PNG-free outputs and paid gray on CPU."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from verify_pfs_cache import digest, filehash, require, verified_bytes


def save(path, obj):
    with path.open("x", encoding="utf8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def array_id(a):
    return digest(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "audit", "images", "spec", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "this role is CPU only")
    cores = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cores)
    os.nice(10)
    audit = json.loads(args.audit.read_bytes())
    require(audit["status"] == "PASS_PFS_ASSET_IDENTITY_ONLY" and audit["verified_native_clean_float32_reconstructions"] == 1200, "input audit")
    require(filehash(args.pilot) == audit["pilot_manifest_sha256"] and filehash(args.images) == audit["clean_image_manifest_sha256"], "input identity")
    require(filehash(args.spec) == "13dd56073f73b2886e9d0efb1a66e815951108da9ef94161b02f444545ca6e23", "metric asset registration")
    spec = json.loads(args.spec.read_bytes())
    require(filehash(spec["loader"]["path"]) == spec["loader"]["sha256"] == "dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066", "metric loader identity")
    require(filehash(spec["overlay_identity"]["path"]) == spec["overlay_identity"]["sha256"], "overlay manifest")
    rows = json.loads(args.images.read_bytes())
    roster = json.loads(args.pilot.read_bytes())["records"]
    groups = {r["source_id"]: [] for r in roster}
    for row in rows:
        groups[row["source_id"]].append(row)
    require(len(groups) == 50 and all(len(v) == 24 and len({(r["m"],r["K"]) for r in v}) == 24 for v in groups.values()), "score plan coverage")
    args.out.mkdir(exist_ok=False)
    start, cpu = time.time(), time.process_time()
    save(args.out / "registration.json", {"role":"SP0 existing native images and fixed gray only","source_count":50,"native_rows":1200,
         "gray_rows":50,"new_perceptual_metrics":["DISTS official","DreamSim ensemble"],"mode":"batch 1, CPU only, FP32, deterministic, no training, no network",
         "cpu_affinity":cores,"intraop_threads":len(cores),"interop_threads":1,"caps":{"dists_pairs":1250,"dreamsim_pairs":1250,"dreamsim_unique_embeddings":1251},
         "audit_sha256":filehash(args.audit),"spec_sha256":filehash(args.spec),"script_sha256":filehash(__file__),"sha256_input_images_manifest":audit["clean_image_manifest_sha256"],"started_unix":start})
    pairs = 0
    calls = {"dists_pairs":0,"dreamsim_pairs":0,"dreamsim_embeddings":0}
    try:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(len(cores))
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        require(not torch.cuda.is_available() and torch.__version__ == "2.11.0+cu128", "registered torch runtime")
        sys.path.insert(0, spec["dependency_overlay"])
        location = importlib.util.spec_from_file_location("_sp_registered_metric_loader",spec["loader"]["path"])
        met = importlib.util.module_from_spec(location)
        location.loader.exec_module(met)
        with met.no_network(), torch.inference_mode():
            dists=met.load_dists(spec["dists"],torch.device("cpu"))
            dream=met.load_dreamsim(spec["dreamsim"],torch.device("cpu"))
            def embed(t):
                calls["dreamsim_embeddings"]+=1
                return dream.embed(met.preprocess_dreamsim(t))
            gray = np.full((3,256,256),0.5,np.float32)
            gray_id=array_id(gray)
            require(gray_id == "37d8fc9661cfa6fae8cd600a767b7561b0638b50cfeeeeae2d03dd0d4e0c9669", "gray identity")
            gray_t=torch.from_numpy(gray[None])
            gray_emb=embed(gray_t)
            completed = []
            with (args.out/"pairs.csv").open("x",encoding="utf8",newline="") as f:
                keys=["pilot_slot","source_id","kind","m","K","source_pixel_sha256","output_array_sha256","dists","dreamsim","psnr_db"]
                writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");writer.writeheader()
                for slot,src in enumerate(roster):
                    require(src["pilot_slot"]==slot,"roster order")
                    with np.load(io.BytesIO(verified_bytes(src["archive"])),allow_pickle=False) as z:
                        rgb=z["pixels"]
                        require(rgb.dtype==np.uint8 and rgb.shape==gray.shape and digest(rgb.tobytes())==src["preprocessing_id"],"source pixels")
                        x=rgb.astype(np.float32)/np.float32(255)
                    xt=torch.from_numpy(x[None]);xe=embed(xt)
                    for row in [None]+sorted(groups[src["source_id"]],key=lambda r:(r["m"],r["K"])):
                        if row is None:
                            im,yt,ye,oid=gray,gray_t,gray_emb,gray_id
                        else:
                            with np.load(io.BytesIO(verified_bytes(row["image_archive"])),allow_pickle=False) as z:
                                im=np.ascontiguousarray(z[row["image_key"]])
                            oid=array_id(im)
                            require(oid==row["image_array_sha256"] and im.dtype==np.float32 and im.shape==gray.shape,"receiving image identity")
                            yt=torch.from_numpy(im[None]);ye=embed(yt)
                        calls["dists_pairs"]+=1
                        d=float(dists(xt,yt,require_grad=False,batch_average=False).reshape(-1)[0])
                        calls["dreamsim_pairs"]+=1
                        ds=float((1-F.cosine_similarity(xe,ye,dim=-1))[0])
                        psnr=float(-10*np.log10(np.mean((im.astype(np.float64)-x.astype(np.float64))**2)))
                        require(np.isfinite([d,ds,psnr]).all(),"nonfinite score")
                        writer.writerow(dict(pilot_slot=slot,source_id=src["source_id"],kind="gray" if row is None else "clean",
                                             m="" if row is None else row["m"],K="" if row is None else row["K"],
                                             source_pixel_sha256=src["preprocessing_id"],output_array_sha256=oid,dists=d,dreamsim=ds,psnr_db=psnr))
                        pairs+=1
                    f.flush();os.fsync(f.fileno());completed.append(slot)
                    print("completed_sources",len(completed),"completed_pairs",pairs,flush=True)
        require(pairs==1250 and calls=={"dists_pairs":1250,"dreamsim_pairs":1250,"dreamsim_embeddings":1251},"final counting")
        save(args.out/"completion.json",dict(status="PASS_CACHED_ENDPOINT_AND_GRAY_METRICS_ONLY",pairs=pairs,calls=calls,
             new_radio_frames=0,training_steps=0,gpu_hours=0,pairs_csv_sha256=filehash(args.out/"pairs.csv"),
             wall_seconds=time.time()-start,parent_cpu_seconds=time.process_time()-cpu))
    except BaseException as error:
        save(args.out/"failure.json",dict(error=repr(error),complete_rows=pairs,calls=calls,training_steps=0,gpu_hours=0,
                                         elapsed=time.time()-start,parent_cpu_seconds=time.process_time()-cpu))
        raise


if __name__ == "__main__":
    main()
