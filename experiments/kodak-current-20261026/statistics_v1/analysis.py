"""Source statistics only: no model, channel, image or new bootstrap draws."""
from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np

METHODS=('NEST_PARTIAL','NEST_WHOLE','P100000','Swin200000','BPG_ADAPTIVE','ARPC_ADAPTED')
SNRS=(4,10,19)
NOISES=(2001,2002,2003)
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
DIRECTION={'psnr_db':'higher','lpips_alex':'lower','dreamsim':'lower','clip_image_cosine':'higher'}
BANK_SHA='552e5b5aebdfa237ac3a72f9c1329dbdcff67335f4f0ee1fa9d1dc0275ed42d8'

def require(ok,message):
    if not ok: raise ValueError(message)

def validate_bank(bank):
    require(bank.shape==(10000,24) and bank.dtype==np.dtype('int64'),'Original10000x24 int64 bank')
    require(int(bank.min())>=0 and int(bank.max())<24,'Bank source indices')
    require(hashlib.sha256(bank.tobytes()).hexdigest()==BANK_SHA,'Original Kodak bank bytes')

def source_arrays(rows):
    """Reject missing/duplicate cells and never discard a protocol failure."""
    require(len(rows)==1296,'Expected1296 complete logical method frames')
    observed={};images={}
    for row in rows:
        m=str(row['method']);s=int(row['snr_db']);i=int(row['source_index']);n=int(row['noise_seed'])
        key=m,s,i,n
        require(m in METHODS and s in SNRS and i in range(24) and n in NOISES,'Unexpected frame identity')
        require(row['source_id']==f'kodak/kodim{i+1:02d}','Source identity or order changed')
        require(key not in observed,'Duplicate logical frame')
        require(bool(row['rx_status']) and row.get('engineering_error','') in ('',False,None),'Incomplete engineering result')
        observed[key]=tuple(float(row[k]) for k in METRICS)
        require(all(math.isfinite(x) for x in observed[key]),'Nonfinite metric; no silently capped/dropped values')
        for name in ('source_pixel_sha256',):
            if row.get(name):
                require(images.setdefault(i,row[name])==row[name],'Different source pixels across methods')
    arrays={};long=[]
    for m in METHODS:
        for s in SNRS:
            for k,metric in enumerate(METRICS):
                v=np.array([np.mean([observed[m,s,i,n][k] for n in NOISES],dtype=np.float64) for i in range(24)],dtype=np.float64)
                arrays[m,s,metric]=v
                long.extend(dict(method=m,snr_db=s,metric=metric,source_index=i,source_id=f'kodak/kodim{i+1:02d}',mean=float(v[i]),noise_count=3) for i in range(24))
    require(len(long)==1728,'Source mean rows')
    return arrays,long

def interval(values,bank):
    require(values.shape==(24,) and np.isfinite(values).all(),'24 finite source values')
    samples=values[bank].mean(axis=1,dtype=np.float64)
    low,high=np.quantile(samples,[.025,.975],method='linear')
    return dict(mean=float(values.mean(dtype=np.float64)),ci_low=float(low),ci_high=float(high),source_count=24,noise_count=3,frame_count=72,bootstrap_replicates=10000,bootstrap_seed=2026100901,bootstrap_unit='source image after three-noise mean')

def summarize(rows,bank):
    validate_bank(bank)
    arrays,long=source_arrays(rows)
    summary=[dict(method=m,snr_db=s,metric=k,favorable_direction=DIRECTION[k],**interval(arrays[m,s,k],bank)) for m in METHODS for s in SNRS for k in METRICS]
    paired=[dict(method='NEST_PARTIAL',reference=b,snr_db=s,metric=k,delta_definition='method minus reference',favorable_delta='positive' if DIRECTION[k]=='higher' else 'negative',pairing_unit='source image',**interval(arrays['NEST_PARTIAL',s,k]-arrays[b,s,k],bank)) for b in METHODS[1:] for s in SNRS for k in METRICS]
    require(len(summary)==72 and len(paired)==60,'Expected mean and paired condition counts')
    acceptance=[]
    for m in METHODS:
        for s in SNRS:
            rr=[r for r in rows if r['method']==m and int(r['snr_db'])==s]
            require(len(rr)==72,'Complete per-SNR denominator')
            counts={}
            for r in rr:counts[r['rx_status']]=counts.get(r['rx_status'],0)+1
            for state,count in sorted(counts.items()):acceptance.append(dict(method=m,snr_db=s,rx_status=state,count=count,total_logical_frames=72))
    return summary,paired,long,acceptance

def csv_write(path,rows):
    require(bool(rows),'No empty scientific table')
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
