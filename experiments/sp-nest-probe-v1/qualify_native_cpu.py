"""One preselected real source: may the CPU float path extend archived H800 streams?"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from verify_pfs_cache import digest, filehash, require, verified_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("repo","pilot","audit","science","routes","out"):
        p.add_argument("--"+n,type=Path,required=True)
    a = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "must not allocate a GPU")
    audit=json.loads(a.audit.read_bytes())
    require(audit["status"]=="PASS_PFS_ASSET_IDENTITY_ONLY" and filehash(a.pilot)==audit["pilot_manifest_sha256"]
            and filehash(a.routes)==audit["route_manifest_sha256"],"PFS identities")
    roster=json.loads(a.pilot.read_bytes())["records"]
    source=roster[0]
    require(source["pilot_slot"]==0,"registered fixed slot")
    historical=json.loads((a.science/"streams/00.json").read_bytes())
    require(historical["source"]==source and len(historical["cdf_trace"])==10,"reference")
    routes=json.loads(a.routes.read_bytes())
    code_paths={"experiments/layered-entropy-feasibility-20261014/native.py":"c1e61b582e5c17f02af769c0cf71f028014866b9ecc0a2800bf97a94dc09de08",
        "src/var_comm/next_scale_prior.py":"568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562",
        "experiments/scale-causal-partial-residual-20261002/partial_receiver.py":"46b676720283e3d7e90ef38bab4f4ea102ae51f38f414913d613ccb4b7e01cdc"}
    for path,want in code_paths.items():require(filehash(a.repo/path)==want,path)
    for entry in routes["author_files"]+[routes["dist"]]:
        require(filehash(entry["path"])==entry["sha256"],entry["path"])
    cpus=sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0,cpus);os.nice(10)
    a.out.mkdir(exist_ok=False)
    def save(name,v):
        with (a.out/name).open("x",encoding="utf8") as f:json.dump(v,f,indent=2,allow_nan=False);f.write("\n")
    start,cpu=time.time(),time.process_time()
    save("registration.json",dict(slot=0,source_id=source["source_id"],purpose="before new radio outcomes; compatibility qualification only",
              required="all ten integer CDF hashes equal the historical scientific record for this source",
              on_mismatch="Do not extend or decode archived H800 bitstreams with this CPU provider.",
              caps={"TX_sources":1,"prior_scale":10,"render":0,"wireless_frames":0},device="CPU only",
              cpu_affinity=cpus,script_sha256=filehash(__file__),input_audit_sha256=filehash(a.audit),code_identity=code_paths))
    calls=0
    try:
        import io
        import torch
        from types import SimpleNamespace
        torch.set_num_threads(len(cpus));torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest")
        require(not torch.cuda.is_available() and torch.__version__=="2.11.0+cu128","fixed runtime")
        sys.path.insert(0,str(a.repo/"experiments/layered-entropy-feasibility-20261014"))
        sys.path.insert(0,str(a.repo/"experiments/wcl-evidence-closure-20261009/scripts"))
        from native import definitions
        from t1_entropy_core import load_integer_codec,SIZES,OFFSETS
        entropy,_,_=load_integer_codec(a.repo)
        ns={"torch":torch,"Path":Path,"sys":sys,"np":np,"hashlib":__import__("hashlib"),"PATCH_NUMS":tuple(SIZES),"VOCAB_SIZE":4096}
        definitions(a.repo/"src/var_comm/next_scale_prior.py",{"load_models","state_sha256"},ns)
        modelpaths={"var_source":routes["author_root"]}
        for m in routes["files"]:
            if m["path"].endswith("vae_ch160v4096z32.pth"): modelpaths["vae_checkpoint"]=m["path"]
            if m["path"].endswith("var_d16.pth"): modelpaths["var_checkpoint"]=m["path"]
        vae,var=ns["load_models"](modelpaths,torch.device("cpu"))
        states={k:ns["state_sha256"](v) for k,v in {"var":var,"vae":vae}.items()}
        require(all(states[k]==routes["models"][k] for k in states),"source model state identity")
        pn={"torch":torch,"np":np,"SIZES":tuple(SIZES)}
        definitions(a.repo/"experiments/scale-causal-partial-residual-20261002/partial_receiver.py",{"_Prior"},pn)
        with np.load(io.BytesIO(verified_bytes(source["archive"])),allow_pickle=False) as z: tokens=z["tokens"]
        results=[]
        with torch.no_grad():
            provider=pn["_Prior"](vae,var,torch.device("cpu"))
            try:
                for k,exp in enumerate(historical["cdf_trace"]):
                    require(k==exp["scale"],"reference scale")
                    calls+=1
                    lp=torch.log_softmax(provider.logits(k),dim=-1)[0].cpu().numpy()
                    table=np.ascontiguousarray(entropy.probability_cdf(lp))
                    actual=digest(str(table.dtype).encode()+str(table.shape).encode()+table.tobytes())
                    results.append(dict(scale=k,computed_sha256=actual,expected_sha256=exp["cdf_sha256"],identical=actual==exp["cdf_sha256"]))
                    if k<9:provider.advance(tokens[OFFSETS[k]:OFFSETS[k+1]].copy(),k)
            finally:provider.close()
        same=sum(r["identical"] for r in results)
        save("outcome.json",dict(status="SLOT0_COMPATIBLE" if same==10 else "CPU_NOT_IDENTICAL_TO_HISTORICAL_CDF",
             historical_device="H800",candidate_device="CPU",models=states,source_id=source["source_id"],matching_cdf_tables=same,
             tables=results,scientific_m10_lengths_added=0,new_radio_frames=0,training_steps=0,GPU_hours=0,prior_scale_calls=calls,
             wall_seconds=time.time()-start,parent_cpu_seconds=time.process_time()-cpu))
        print(f"CPU qualification: {same}/10 archived H800 integer CDF tables identical. No cross-route scores or streams admitted.",flush=True)
    except BaseException as e:
        save("failure.json",dict(error=repr(e),prior_scale_calls=calls,wall_seconds=time.time()-start))
        raise


if __name__=="__main__":
    main()
