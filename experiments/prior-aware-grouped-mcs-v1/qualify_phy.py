"""Actual Sionna engineering qualification and bounded throughput measurement."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
from uep_common import require,write,seal,rng,sha
from ldpc_backend import SionnaBackend
from uep_phy import (append_crc_batch,crc_accept_batch,simulate_packets,transmit_packet,
                     receive_packet,modulate_torch,demap_torch,Header)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    started=time.perf_counter();b=SionnaBackend(a.device);cases=[]
    # Representative small/large raw packets, filler and both admitted modulations.
    configs=[(360,1128,2),(1092,2216,2),(3060,3824,4),(5088,7920,4),(768,1568,2),(1200,1824,4)]
    for source,n,q in configs:
        plan=b.plan(source+16,n,q);payload=rng('qualify',source,n,q).integers(0,2,(8,source),dtype=np.uint8)
        counters=list(range(8));wave,meta=transmit_packet(b,payload,n,q,counters,session='qualification')
        recovered,accepted,_=receive_packet(b,wave,source+16,n,q,60,counters,session='qualification')
        require(accepted.all() and np.array_equal(recovered,payload),'Actual noiseless LDPC/CRC/map roundtrip failed')
        if q==2:require(np.allclose(meta['actual_E'],2*n/q,atol=0,rtol=0),'QPSK energy differs')
        # A successful payload comparison alone cannot prove CRC or descrambling.
        changed=payload.copy();changed[:,0]^=1
        require(not np.array_equal(append_crc_batch(changed),append_crc_batch(payload)),'CRC payload not bound')
        cases.append(dict(source_bits=source,n=n,q=q,layout=plan,noiseless_frames=8,actual_E=meta['actual_E']))
    h=Header(a.root);headers=[]
    for pid in (0,1,1023,2048,4095):
        x=h.transmit(pid);r=h.receive(x,60,{str(pid):{}})
        require(r['header_ok'] and r['profile_id']==pid and np.square(x).sum()==136,'Paid original header roundtrip failed')
        denied=h.receive(x,60,{str((pid+1)%4096):{}})
        require(not denied['header_ok'],'Receiver accepted an unregistered profile')
        headers.append(pid)
    # Uniform constellation average is 2, not a per-frame 16QAM assertion.
    bits=np.array([[(i>>shift)&1 for shift in range(3,-1,-1)] for i in range(16)],np.float32)
    qam=modulate_torch(torch.tensor(bits,device=a.device),4)
    require(abs(float(qam.square().sum((1,2)).mean())-2)<1e-6,'16QAM average Es differs')
    bench=[]
    for source,n,q in configs[:4]:
        for batch in (32,128,256):
            payload=rng('benchmark',source,n,q,batch).integers(0,2,(batch,source),dtype=np.uint8)
            # Warmup does not count as scientific BLER observations.
            simulate_packets(b,payload[:2],n,q,7,[0,1],session='warmup')
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            begin=time.perf_counter();r=simulate_packets(b,payload,n,q,7,list(range(batch)),session='benchmark')
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            sec=time.perf_counter()-begin
            bench.append(dict(source_bits=source,n=n,q=q,batch=batch,seconds=sec,blocks_per_second=batch/sec,
                correct=int(r['correct'].sum()),reject=int(r['rejected'].sum()),undetected=int(r['undetected'].sum()),
                engineering_only=True))
    result=dict(status='PASS',backend_identity=b.identity,device=a.device,torch=torch.__version__,
        source_bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('ldpc_backend.py').resolve()):sha(Path(__file__).with_name('ldpc_backend.py')),
                         str(Path(__file__).with_name('uep_phy.py').resolve()):sha(Path(__file__).with_name('uep_phy.py'))},
        cases=cases,header_ids=headers,benchmarks=bench,total_seconds=time.perf_counter()-started,
        memory_peak_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,
        new_neural_training_updates=0,scientific_bler_samples=0)
    seal(out/'ldpc_qualification.json',result);print('QUALIFIED',len(cases),len(bench),flush=True)

if __name__=='__main__':main()
