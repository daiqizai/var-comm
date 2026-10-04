"""Small shared receipts and deterministic identities for the registered UEP study."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

PROTOCOL = 'PRIOR-AWARE-UEP-20261004-V1'
SIZES = (1,2,3,4,5,6,8,10,13,16)
SNRS = (1,4,7,10,13,19)
DEV_SNRS = (4,7,10,13)
DEV_SEEDS = (2001,2002,2003)

def require(ok,message):
    if not ok: raise RuntimeError(message)

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)

def seal(path,value):
    if Path(path).exists():require(read(path)==value,'Immutable receipt differs: '+str(path))
    else:write(path,value)

def csv_write(path,rows):
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})

def rng(*parts):
    # Public experiment/session/frame counters, never payload or receiver noise truth.
    seed=int.from_bytes(hashlib.sha256(json.dumps([PROTOCOL,*parts],sort_keys=True).encode()).digest()[:16],'little')
    return np.random.default_rng(seed)

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
