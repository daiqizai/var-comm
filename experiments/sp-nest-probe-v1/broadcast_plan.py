"""Preregister the broadcast pilot and map policies to byte-identical physical frames."""
import argparse
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from profiles import ALPHA, BR, EPHY, OFF, QUARTER, bit_bound, count, public_directory
from verify_pfs_cache import filehash, require


def k_info(symbols, q, rate):
    return int(symbols * q * Fraction(rate))


def grids():
    return (("quarter", QUARTER), ("whole", tuple(e for e in QUARTER if not e[1])))


def valid_targets(jcap, es):
    return [e for e in es if e[0] >= 6 and count(e) > OFF[jcap]]


def oma_directory():
    rows = set()
    for nb,jcap,br,(qe,er) in itertools.product((256,384,512,640),(5,6,7),BR,EPHY):
        cb, ce = k_info(nb,2,br)-29, k_info(956-nb,qe,er)-29
        for _, grid in grids():
            for tgt in valid_targets(jcap, grid):
                for j in range(4,jcap+1):
                    if any(bit_bound(OFF[h]) <= cb for h in range(j+1,jcap+1)):
                        continue
                    es = [e for e in grid if OFF[j] < count(e) <= count(tgt)]
                    floor = max([OFF[j]]+[count(e) for e in es if bit_bound(count(e)-OFF[j]) <= ce])
                    rows.update((nb,j,*e,br,qe,er) for e in [(j,0)]+es if count(e) >= floor)
    result = {i+1:r for i,r in enumerate(sorted(rows))}
    require(len(result)+1 == 4048, "public OMA directory capacity")
    return {0:None, **result}


def dump(path, value):
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
        f.write("\n")


def protocol():
    return dict(version="SP_NEST_A800_PILOT50_V1",N=1024,header=68,body=956,Es=2,
        pilot_sources=50,repeats=3,pairs=[[4,19],[4,13]],weights=[0,.25,.5,.75,1],
        noise="Original ep_phy.standard_noise(source_id,seed), per real standard deviation 10**(-SNR/20). "
             "Common waves, counters and noise across policies. Weak 4 dB is shared by both pairs. "
             "Both strong SNRs scale the same standardized strong-receiver draw. Weak and strong banks are independent.",
        weak_seeds=[73101,73102,73103],strong_seeds=[73201,73202,73203],counter="3*pilot_slot+repeat_index",
        single="Original 300 NeST policies, all 651 fallback lengths, actual original method/MCS directory and scrambler",
        oma="Original B/E content and source fitting; public fixed directory of 4048 including NULL, carries split, both MCS, actual endpoints",
        superposition="3805 public profiles incl NULL; unconditional public lookup; both users can attempt verified CRC-SIC",
        decoding="Actual RX header controls decoding, with only verified actual-received B enabling E. All gray/base-only outcomes count.",
        selected_by="highest source mean of lambda*DINO_weak+(1-lambda)*DINO_strong. Exact ties use lexical policy ID.",
        bootstrap=dict(seed=20261119,replicates=10000,unit="fixed ordered Pilot50 source, mean its three repetitions first"),
        primary_gate=dict(both_pairs=True,lambda_weak=.5,DISTS_drop_min=.025,DISTS_drop_relative_strict=.12,
            DreamSim_drop_min=.035,paired_intervals_favorable=True,vs_OMA="DINO >0 and DISTS <0, intervals exclude 0",
            outer_envelope="For lambda extremes include single-user policies as selectable operating modes. "
                           "For an interior outward support, require higher support of the actual SP-only point than every single "
                           "at lambda=0.5. Also report coordinate dominance and independent weak/strong scores."),
        inference="Pilot selection and evaluation use the same 50 sources. CIs are conditional on the selected policies.",
        training_steps=0,common500_requires_observed_SP1_GO=True)


def all_policies(csv_path):
    with csv_path.open(encoding="utf-8-sig",newline="") as f:
        single = [dict(id=r["candidate_id"], family="single",q=int(r["q"]),rate=r["rate"],
                       target=[int(r["target_m"]),int(r["target_K"])]) for r in csv.DictReader(f)]
    require(len(single)==300, "single-codeword roster")
    result = single
    for family, variants in (("sp",ALPHA),("oma",(256,384,512,640))):
        for variant,jcap,br,(qe,er) in itertools.product(variants,(5,6,7),BR,EPHY):
            for g,es in grids():
                for m,k in valid_targets(jcap,es):
                    name=f"{family}_{g}_{variant:03d}_j{jcap}_B{br}_E{qe}r{er}_m{m:02d}k{k:03d}".replace("/","-")
                    result.append(dict(id=name,family=family,variant=variant,jcap=jcap,br=br,
                                       qe=qe,er=er,grid=g,target=[m,k]))
    require(len({r["id"] for r in result})==len(result),"unique policy IDs")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("pilot","single","source","out"):
        ap.add_argument("--"+k,required=True,type=Path)
    a=ap.parse_args()
    require(filehash(a.single)=="92199ce8fab34a66d4fa3de6c0ddaea5092ca0e68975cd37b6500056cbd2a313" and
            filehash(a.pilot)=="b89a964f24127d9fcd2bddd6b7a88d8b366f53764632a71d93dfcf3ffcd94853","fixed identities")
    a.out.mkdir()
    # This stage sees source CODE LENGTHS only. No SP1 receiver or quality output is read.
    reg=protocol()
    dump(a.out/"protocol.json",reg)
    policies=all_policies(a.single)
    dump(a.out/"policies.json",policies)
    sp=public_directory();oma=oma_directory()
    dump(a.out/"sp_profiles.json",sp);dump(a.out/"oma_profiles.json",oma)
    sp_ids={v:k for k,v in sp.items() if v}
    oma_ids={v:k for k,v in oma.items() if v}
    roster=json.loads(a.pilot.read_bytes())["records"]
    bank=np.random.default_rng(reg["bootstrap"]["seed"]).integers(0,50,(10000,50),dtype=np.int16)
    np.save(a.out/"paired_source_bank.npy",bank,allow_pickle=False)
    cases=[]
    lookup=np.empty((50,len(policies)),np.int32)
    for slot,src in enumerate(roster):
        folder=a.source/f"slot_{slot:02d}"
        meta=json.loads((folder/"done.json").read_bytes())
        require(meta["source_id"]==src["source_id"],"roster")
        with np.load(folder/"single_tokens.npz",allow_pickle=False) as z:
            lengths=np.diff(z["offsets"])
        require(len(lengths)==651 and np.all(np.diff(lengths)>=0),"actual single lengths")
        B={r["j"]:r["base_bits"] for r in meta["lengths"]}
        E={(r["j"],r["m"],r["K"]):r["enhancement_bits"] for r in meta["lengths"]}
        seen={}
        for i,policy in enumerate(policies):
            fam=policy["family"];target=tuple(policy["target"])
            if fam=="single":
                q,rate=policy["q"],policy["rate"]
                nfit=min(count(target)-OFF[4]+1,int(np.searchsorted(lengths,k_info(956,q,rate)-29,side="right")))
                pid=nfit-1 if nfit else 651
                key=(fam,q,rate,pid)
            else:
                v,jcap,br,qe,er=(policy[x] for x in ("variant","jcap","br","qe","er"))
                nb,ne=(956,956) if fam=="sp" else (v,956-v)
                cb,ce=k_info(nb,2,br)-29,k_info(ne,qe,er)-29
                j=next((h for h in range(jcap,3,-1) if B[h]<=cb),None)
                if j is None:
                    pid=0
                else:
                    es=QUARTER if policy["grid"]=="quarter" else [e for e in QUARTER if e[1]==0]
                    end=next((e for e in reversed(es) if OFF[j]<count(e)<=count(target) and E[(j,*e)]<=ce),(j,0))
                    pid=sp_ids[(j,*end,br,qe,er,v)] if fam=="sp" else oma_ids[(v,j,*end,br,qe,er)]
                key=(fam,0,"",pid)
            if key not in seen:
                seen[key]=len(cases)
                cases.append(dict(case=len(cases),slot=slot,family=key[0],q=key[1],rate=key[2],pid=key[3]))
            lookup[slot,i]=seen[key]
    np.save(a.out/"policy_to_case.npy",lookup,allow_pickle=False)
    with (a.out/"physical_cases.csv").open("x",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]),lineterminator="\n");w.writeheader();w.writerows(cases)
    population=[r["source_id"] for r in roster]
    require(len(set(population))==50 and lookup.min()==0 and lookup.max()==len(cases)-1,"case map")
    done=dict(status="SP1_PLAN_FIXED_BEFORE_RADIO",roster=population,policy_count=len(policies),
              unique_cases={g:sum(c["family"]==g for c in cases) for g in ("single","sp","oma")},
              policies={g:sum(c["family"]==g for c in policies) for g in ("single","sp","oma")},
              physical_waves=len(cases)*3,receiver_observations=len(cases)*9,bootstrap_hash=filehash(a.out/"paired_source_bank.npy"),
              inputs={p.name:filehash(p) for p in sorted(a.out.iterdir()) if p.is_file()},new_wireless_trials=0)
    dump(a.out/"completion.json",done)
    print({k:done[k] for k in ("policies","unique_cases","physical_waves","receiver_observations")},flush=True)


if __name__=="__main__":
    main()
