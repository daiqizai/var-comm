"""Content-addressed results from independent arithmetic receivers for the pilot."""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from profiles import OFF, count
from split_prefixes import SplitPrefixes
from verify_pfs_cache import digest, filehash, require


def array_hash(x):
    a=np.ascontiguousarray(x)
    return digest(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())


class PrefixCDF:
    def __init__(self, tables):
        self.tables=tables
        self.recovered=bytearray()
        self.scale=0
    def __enter__(self):
        return self
    def __exit__(self,*exc):
        pass
    def advance(self, q):
        self.recovered.extend(np.asarray(q,dtype="<i8").tobytes())
        self.scale+=1
    def lookup(self, scale):
        require(scale==self.scale,"RX coarse scale")
        return self.tables[(scale,digest(self.recovered))]


def byte_key(layer, j, m, k, base_id, bits):
    return (layer,j,m,k,base_id,len(bits),np.packbits(bits,bitorder="big").tobytes())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("repo","source","quality","out"):
        ap.add_argument("--"+key,required=True,type=Path)
    a=ap.parse_args()
    stage=json.loads((a.quality/"completion.json").read_bytes())
    require(stage["status"]=="A800_SP0_CLEAN_FOUR_METRICS_COMPLETE" and filehash(a.quality/"scores.csv")==stage["scores_sha256"],"SP0 complete")
    sys.path.insert(0,str(a.repo/"experiments/wcl-evidence-closure-20261009/scripts"))
    from t1_entropy_core import load_integer_codec
    _,enc,dec=load_integer_codec(a.repo)
    a.out.mkdir()
    with (a.quality/"scores.csv").open() as f:
        scored=list(csv.DictReader(f))
    db=sqlite3.connect(a.out/"independent_rx.sqlite")
    db.execute("create table received(layer text,j int,m int,k int,base_id text,n int,payload blob,token_hash text,primary key(layer,j,m,k,base_id,n,payload)) without rowid")
    counts=dict(base=0,enhancement=0,single=0)
    def insert(key,tok):
        old=db.execute("select token_hash from received where (layer,j,m,k,base_id,n,payload)=(?,?,?,?,?,?,?)",key).fetchone()
        if old:
            require(old[0]==tok,"same actual bits and independent context produced different tokens")
        else:
            db.execute("insert into received values(?,?,?,?,?,?,?,?)",key+(tok,))
    with (a.out/"references.csv").open("x",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(["slot","source_id","m","K","received_token_hash","dino","dists","dreamsim","psnr_db"])
        for slot in range(50):
            folder=a.source/f"slot_{slot:02d}"
            meta=json.loads((folder/"done.json").read_bytes())
            for name,h in meta["files"].items():
                require(filehash(folder/name)==h,(slot,name))
            with np.load(folder/"accepted_rx_tokens.npz",allow_pickle=False) as z:
                recovered=z["values"]
            with np.load(folder/"split.npz",allow_pickle=False) as z:
                for j in (4,5,6,7):
                    bt=recovered[:OFF[j]]
                    b=z[f"base_{j}"]
                    allrows=[r for r in meta["lengths"] if r["j"]==j]
                    require(all(r["base_bits_sha256"]==array_hash(b) for r in allrows),"verified B")
                    b_id=array_hash(bt)
                    insert(byte_key("B",j,j,0,"",b),b_id);counts["base"]+=1
                    for row in allrows:
                        m,k=row["m"],row["K"]
                        e=z[f"enhancement_{j}_{m}_{k}"]
                        t_id=array_hash(recovered[:count((m,k))])
                        require(array_hash(e)==row["enhancement_bits_sha256"] and t_id==row["received_tokens_sha256"],"verified E")
                        insert(byte_key("E",j,m,k,b_id,e),t_id);counts["enhancement"]+=1
            with np.load(folder/"single_tokens.npz",allow_pickle=False) as z:
                offsets,bits=z["offsets"],z["bits"]
            with np.load(folder/"rx_cdfs.npz",allow_pickle=False) as z:
                tables={(s,h):z[f"s{s}"] for s,h in meta["independent_rx_cdf_keys"]}
            rx=SplitPrefixes(SimpleNamespace(Encoder=enc,Decoder=dec,_inference=lambda _:nullcontext(),
                  _context=lambda _:PrefixCDF(tables),_cdf=lambda _,provider,s:provider.lookup(s)))
            for row in [r for r in scored if int(r["slot"])==slot]:
                if row["kind"]=="gray":
                    m,k,t_id=-1,0,"GRAY"
                else:
                    m,k=int(row["m"]),int(row["K"])
                    idx=count((m,k))-OFF[4]
                    payload=bits[offsets[idx]:offsets[idx+1]]
                    with rx.context() as provider:
                        tokens=rx._read(provider,0,(m,k),payload)
                    require(np.array_equal(tokens,recovered[:count((m,k))]),"audit against independently received full stream")
                    t_id=array_hash(tokens);insert(byte_key("S",0,m,k,"",payload),t_id);counts["single"]+=1
                writer.writerow([slot,row["source_id"],m,k,t_id,row["dinov2_cosine"],row["dists"],row["dreamsim"],row["psnr_db"]])
            f.flush();db.commit()
    unique=db.execute("select count(*) from received").fetchone()[0];db.close()
    require(counts==dict(base=200,enhancement=3600,single=len(scored)-50),"receiver catalogue coverage")
    (a.out/"completion.json").write_text(json.dumps(dict(status="INDEPENDENT_RECEIVER_CACHE",counts=counts,unique_keys=unique,
        inputs="received, canonical reconstruction contexts only; no TX CDF",
        database_sha256=filehash(a.out/"independent_rx.sqlite"),reference_sha256=filehash(a.out/"references.csv"),
        unseen="must independently decode an exact miss; never treat a cache miss as a radio failure"),indent=2))
    print(counts,flush=True)


if __name__=="__main__":
    main()
