"""Finite A800 interoperability check against real registered H800 bitstreams and CDFs."""
import argparse
import json
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verify_pfs_cache import digest, filehash, require, verified_bytes


def snapshot():
    s=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    return [{ "index":int(i), "uuid":uuid, "name":model, "free_MiB":int(free), "util_pct":int(util)}
            for i,uuid,model,free,util in ([v.strip() for v in line.split(",")] for line in s.splitlines())]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("repo","pilot","audit","science","routes","out"):
        parser.add_argument("--"+field,required=True,type=Path)
    a=parser.parse_args()
    audit=json.loads(a.audit.read_bytes())
    require(audit["status"]=="PASS_PFS_ASSET_IDENTITY_ONLY" and filehash(a.pilot)==audit["pilot_manifest_sha256"]
            and filehash(a.routes)==audit["route_manifest_sha256"],"input identities")
    source=json.loads(a.pilot.read_bytes())["records"][0]; require(source["pilot_slot"]==0,"fixed slot")
    h=json.loads((a.science/"streams/00.json").read_bytes());require(h["source"]==source,"source-to-stream join")
    code_hashes={"experiments/layered-entropy-feasibility-20261014/native.py":"c1e61b582e5c17f02af769c0cf71f028014866b9ecc0a2800bf97a94dc09de08",
        "src/var_comm/next_scale_prior.py":"568217086c94ee7607abc256415d705a11702ffcaaed9c79ca04d5e9a62da562",
        "experiments/scale-causal-partial-residual-20261002/partial_receiver.py":"46b676720283e3d7e90ef38bab4f4ea102ae51f38f414913d613ccb4b7e01cdc",
        "experiments/content-real-64qam-20261006/h_source_driver.py":"b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce"}
    for name,expected in code_hashes.items():require(filehash(a.repo/name)==expected,name)
    routes=json.loads(a.routes.read_bytes())
    for entry in routes["author_files"]+[routes["dist"]]:require(filehash(entry["path"])==entry["sha256"],entry["path"])
    a.out.mkdir(exist_ok=False)
    def save(name,v):
        with (a.out/name).open("x",encoding="utf8") as f:json.dump(v,f,indent=2,allow_nan=False);f.write("\n")
    caps={"startup_free_MiB":49152,"startup_max_util_pct":10,"running_free_floor_MiB":32768,"torch_allocator_cap_GiB":4,
          "target_sources":1, "TX_prior_scale_calls":10, "RX_prior_scale_calls":0,"training_steps":0}
    save("registration.json",dict(purpose="One fixed native source before SP1; generate all independently terminated streams and compare all archived bytes",
        source_id=source["source_id"],pilot_slot=0,input_audit_sha256=filehash(a.audit),code_sha256=filehash(__file__),reference_software=code_hashes,
        caps=caps,requires_for_H800_CDF_reuse="matching CDF at each scale; a one-source success does not qualify other sources",
        requires_for_archived_payload_reuse="compare exact bit vector for the same source, split, endpoints and model; no length-only matches",
        branch_on_difference="save diagnostics, do not use GPU outputs as historical H800 data; no SP1, no bulk job launched by this process"))
    sampled=[snapshot()]
    for _ in range(2):time.sleep(0.6);sampled.append(snapshot())
    stable=set.intersection(*[{r["uuid"] for r in s if "A800" in r["name"] and r["free_MiB"]>=caps["startup_free_MiB"] and r["util_pct"]<=caps["startup_max_util_pct"]} for s in sampled])
    if not stable:
        save("admission.json",dict(status="NO_IDLE_CAPACITY",samples=sampled));print("No GPU passed the bounded shared-capacity check.");return
    dev=max([r for r in sampled[-1] if r["uuid"] in stable],key=lambda r:(r["free_MiB"],-r["index"]))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";os.environ["CUDA_VISIBLE_DEVICES"]=dev["uuid"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,cpus);os.nice(10)
    save("admission.json",dict(status="ONE_GPU_ADMITTED",chosen=dev,cpu_affinity=cpus,samples=sampled))
    calls=0;begin=time.time();cpu=time.process_time()
    try:
        import torch
        from split_prefixes import SplitPrefixes
        torch.set_num_threads(6);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest")
        require(torch.cuda.device_count()==1 and torch.__version__=="2.11.0+cu128","registered runtime")
        torch.cuda.set_device(0);d=torch.device("cuda:0");props=torch.cuda.get_device_properties(d)
        require("A800" in props.name and "A800" in dev["name"],"unexpected physical GPU product")
        torch.cuda.set_per_process_memory_fraction((4<<30)/props.total_memory,d)
        sys.path.insert(0,str(a.repo/"experiments/layered-entropy-feasibility-20261014"))
        sys.path.insert(0,str(a.repo/"experiments/wcl-evidence-closure-20261009/scripts"))
        sys.path.insert(0,str(a.repo/"experiments/content-real-64qam-20261006"))
        from native import definitions
        from t1_entropy_core import load_integer_codec,SIZES
        from h_source_driver import IndependentProvider
        entropy,encoder,_=load_integer_codec(a.repo)
        ns={"torch":torch,"Path":Path,"sys":sys,"np":np,"hashlib":__import__("hashlib"),"PATCH_NUMS":tuple(SIZES),"VOCAB_SIZE":4096}
        definitions(a.repo/"src/var_comm/next_scale_prior.py",{"load_models","state_sha256"},ns)
        modelpaths={"var_source":routes["author_root"]}
        for m in routes["files"]:
            if m["path"].endswith("vae_ch160v4096z32.pth"):modelpaths["vae_checkpoint"]=m["path"]
            if m["path"].endswith("var_d16.pth"):modelpaths["var_checkpoint"]=m["path"]
        vae,var=ns["load_models"](modelpaths,d)
        states={k:ns["state_sha256"](v) for k,v in {"var":var,"vae":vae}.items()}
        require(all(states[k]==routes["models"][k] for k in states),"source model identity")
        pn={"torch":torch,"np":np,"SIZES":tuple(SIZES)}
        definitions(a.repo/"experiments/scale-causal-partial-residual-20261002/partial_receiver.py",{"_Prior"},pn)
        native=SimpleNamespace(torch=torch,receiver=SimpleNamespace(_Prior=pn["_Prior"]),loaded=dict(vae=vae,var=var,device=d))
        tables=[]
        def cdf(family,provider,k):
            nonlocal calls
            require(family=="EC_VAR_WHOLE" and k==len(tables) and calls<10,"public source protocol")
            if k in (0,5,9):
                state=next(r for r in snapshot() if r["uuid"]==dev["uuid"])
                require(state["free_MiB"]>=caps["running_free_floor_MiB"],"shared memory guard")
            calls+=1
            x=np.ascontiguousarray(entropy.validate_cdf(provider.cdf(),rows=SIZES[k]**2))
            actual=digest(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
            require(h["cdf_trace"][k]["scale"]==k,"reference ordering")
            expected=h["cdf_trace"][k]["cdf_sha256"]
            tables.append(dict(scale=k,observed=actual,historical=expected,identical=actual==expected))
            return x
        codec=SimpleNamespace(Encoder=encoder,_inference=lambda family:torch.no_grad(),
                _context=lambda family:IndependentProvider(native,SimpleNamespace(cdf_function=entropy.probability_cdf)),_cdf=cdf)
        with np.load(io.BytesIO(verified_bytes(source["archive"])),allow_pickle=False) as z:tokens=z["tokens"]
        s,b,e=SplitPrefixes(codec).encode(tokens)
        origin={}
        for (m,k),v in s.items():origin[f"single_{m}_{k}"]=v
        for j,v in b.items():origin[f"base_{j}"]=v
        for (j,m,k),v in e.items():origin[f"enhancement_{j}_{m}_{k}"]=v
        with np.load(io.BytesIO(verified_bytes(h["streams"])),allow_pickle=False) as z:
            require(len(z.files)==96 and all(k in origin for k in z.files),"historical stream set")
            by_name={k:bool(np.array_equal(z[k],origin[k])) for k in sorted(z.files)}
        require(len(origin)==101,"new endpoint coverage")
        with (a.out/"candidate_streams.npz").open("xb") as f:np.savez_compressed(f,**origin)
        c=sum(x["identical"] for x in tables); bits=sum(by_name.values())
        save("outcome.json",dict(status="SLOT0_IDENTICAL" if c==10 and bits==96 else "ARCHIVE_ROUTE_DIFFERENCE",
             device=dev["name"],device_index=dev["index"],gpu_uuid=dev["uuid"],cuda_product=props.name,cuda_uuid=str(props.uuid),
             models=states,source_id=source["source_id"],matching_cdf_tables=c,comparison_bits_exact=bits,comparison_bits_total=96,
             streams_equal=by_name,cdf=tables,new_m10_enhancement_candidate_bits={str(j):len(e[(j,10,0)]) for j in b},
             bytes_archive_sha256=filehash(a.out/"candidate_streams.npz"),peak_reserved_bytes=torch.cuda.max_memory_reserved(d),
             native_prior_scales=calls,actual_TX_sources=1,admitted_as_historical=False,training_steps=0,PHY_frames=0,
             cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-begin,upper_gpu_hours=(time.time()-begin)/3600))
        print(f"A800 #{dev['index']}: ran real VAE/VAR; {c}/10 CDF and {bits}/96 old bitstreams identical; no training.",flush=True)
    except BaseException as ex:
        save("failure.json",dict(error=repr(ex),prior_scales=calls,seconds=time.time()-begin));raise


if __name__=="__main__":
    main()
