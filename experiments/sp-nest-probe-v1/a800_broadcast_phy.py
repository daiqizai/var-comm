"""Pilot50 broadcast PHY. Public decoded headers drive the independent receivers."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from broadcast_plan import k_info
from profiles import OFF
from sp1_rx_catalog import byte_key
from verify_pfs_cache import filehash, require

SNR=(4,13,19)
LEVELS={2:[1.,-1.],4:[-3.,-1.,3.,1.],6:[-7.,-5.,-1.,-3.,7.,5.,1.,3.]}
NORMS={2:1.,4:5.,6:21.}


def jread(path):
    return json.loads(path.read_bytes())


def jsave(path, value, update=False):
    tmp=path.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    if not update and path.exists():
        raise FileExistsError(path)
    tmp.replace(path)


class PilotData:
    def __init__(self, args):
        self.family=args.family
        self.cats={}
        self.cathashes={}
        if args.family=="single":
            for f in sorted(args.legacy.glob("NeST_q*_r*.json")):
                d=jread(f);key=(d["q"],d["rate"])
                require(d["method"]=="NeST" and d["profile_bits"]==12 and len(d["profiles"])==652, "single public roster")
                self.cats[key]={p["profile_id"]:p for p in d["profiles"]}
                self.cathashes[f.name]=filehash(f)
            require(len(self.cats)==12,"twelve original MCS directories")
        else:
            f=args.plan/("sp_profiles.json" if self.family=="sp" else "oma_profiles.json")
            self.cathashes[f.name]=filehash(f)
            self.cats[None]={int(k):v for k,v in jread(f).items()}
        self.streams=[];self.single=[]
        for slot in range(50):
            folder=args.source/f"slot_{slot:02d}"
            meta=jread(folder/"done.json")
            if args.family=="single":
                require(filehash(folder/"single_tokens.npz")==meta["files"]["single_tokens.npz"], "single input")
                with np.load(folder/"single_tokens.npz",allow_pickle=False) as z:
                    self.single.append((z["offsets"],z["bits"]))
            else:
                require(filehash(folder/"split.npz")==meta["files"]["split.npz"], "two-layer input")
                with np.load(folder/"split.npz",allow_pickle=False) as z:
                    self.streams.append({k:z[k] for k in z.files if not k.startswith("single_")})

    @lru_cache(maxsize=16384)
    def profile(self, fixed, pid):
        cat=self.cats[fixed]
        if pid is None or pid not in cat:
            return None
        if self.family=="single":
            p=cat[pid]
            return (956,0,p["q"],p["nominal_rate"],0,"",100,0,p["m"],p["K"],p["m"]==0)
        if pid==0:
            nb,ne=(956,956) if self.family=="sp" else (256,700)
            return (nb,ne,2,"1/2",2,"2/3",80 if self.family=="sp" else 100,0,0,0,True)
        if self.family=="sp":
            j,m,k,br,qe,er,alpha=cat[pid];nb=ne=956
        else:
            nb,j,m,k,br,qe,er=cat[pid];ne=956-nb;alpha=100
        return (nb,ne,2,br,qe,er,alpha,j,m,k,False)

    @lru_cache(maxsize=4096)
    def tx_payload(self,slot,pid,profile,layer):
        nb,ne,q,br,qe,er,alpha,j,m,k,null=profile
        if null:
            return np.zeros(0 if layer=="E" else 2,dtype=np.uint8)
        if self.family=="single":
            offsets,bits=self.single[slot];return bits[offsets[pid]:offsets[pid+1]]
        if layer=="B":
            return self.streams[slot][f"base_{j}"]
        return np.empty(0,np.uint8) if (j,0)==(m,k) else self.streams[slot][f"enhancement_{j}_{m}_{k}"]


class RadioEngine:
    def __init__(self,args,refs):
        import torch
        self.t=torch
        for lab in ("prior-aware-grouped-mcs-v1","entropy-partial-mismatch-20261010/scripts"):
            sys.path.insert(0,str(args.repo/"experiments"/lab))
        sys.path.insert(0,str(args.repo/"src"))
        import uep_phy, ep_phy, ldpc_backend
        self.legacy=uep_phy
        self.original=ep_phy
        self.h=uep_phy.Header(args.repo)
        self.backend=ldpc_backend.SionnaBackend('cuda:0')
        self.device=torch.device('cuda:0')
        self.data=PilotData(args)
        self.family=args.family
        self.noise={}
        for slot,row in enumerate(refs):
            for rep in range(3):
                for role,base in (("w",73101),("s",73201)):
                    self.noise[(slot,rep,role)]=ep_phy.standard_noise(row["source_id"],base+rep)
        # Payload keys contain actual received bits, public endpoints and already decoded B only.
        # No pilot source label, reference pixels, TX bits or TX probability table enters lookup.
        db=sqlite3.connect("file:"+str(args.rx/"independent_rx.sqlite")+"?mode=ro",uri=True)
        self.received={tuple(r[:-1]):r[-1] for r in db.execute("select layer,j,m,k,base_id,n,payload,token_hash from received")}
        db.close()
        self.qualified={};self.misses=0

    @lru_cache(maxsize=128)
    def levels(self,q,single):
        t=self.t
        return t.tensor(LEVELS[q],dtype=t.float32 if single else t.float64,device=self.device)/np.sqrt(NORMS[q])

    @lru_cache(maxsize=64)
    def code(self,k,n,q):
        e,d=self.backend.codecs(k,n,q)
        require(e.k==k and e.n==n and e.k_ldpc-e.k_filler==k and n<=e.n_cb_comp,("unqualified layout",k,n,q))
        # The established scalar CPU codec and this GPU encoder must emit identical RV0 bits.
        t=self.t
        pattern=np.random.default_rng(2911+k+n+q).integers(0,2,(1,k),dtype=np.uint8)
        cpu=self.backend.Encoder(k,n,num_bits_per_symbol=q,precision="single",device="cpu").eval()
        gpu_bits=e(t.as_tensor(pattern,dtype=t.float32,device=self.device))
        cpu_bits=cpu(t.as_tensor(pattern,dtype=t.float32))
        require(np.array_equal(gpu_bits.cpu().numpy(),cpu_bits.numpy()),("GPU vs original CPU encoded bytes",k,n,q))
        hard=d((2*gpu_bits-1)*15)
        require(np.array_equal(hard.cpu().numpy(),pattern),("GPU noiseless independent LDPC",k,n,q))
        self.qualified[f"{k}_{n}_{q}"]=dict(cpu_gpu_bits_equal=True,hard_roundtrip=True,
                    bg=str(e._bg),z=int(e.z),iterations=self.backend.identity["decoder"])
        return e,d

    @lru_cache(maxsize=8192)
    def public_mask(self,n,counter,layer):
        if self.family=="single":
            return self.original.scramble_mask(n,counter,self.original.WIRE_PROTOCOL)
        b=hashlib.sha256(f"SP_NEST_V1:{layer}:{counter}".encode("ascii")).digest()
        return np.random.Generator(np.random.PCG64(int.from_bytes(b[:16],"big"))).integers(0,2,n,dtype=np.uint8)

    def masks(self,n,counters,layer):
        return self.t.as_tensor(np.stack([self.public_mask(n,c,layer) for c in counters]),device=self.device)

    def modulate(self,bits,q):
        t=self.t
        if self.family=="single" and q==2:
            return (1-2*bits.float()).reshape(len(bits),-1,2)
        symbols=bits.reshape(len(bits),-1,q//2).long()
        idx=symbols[:,:,0]
        for i in range(1,q//2):
            idx=2*idx+symbols[:,:,i]
        return self.levels(q,self.family=="single")[idx].reshape(len(bits),-1,2)

    def llr(self,y,snr,q,qe=None,alpha=1.,scale=1.):
        t=self.t
        n0=2.0/(10**(snr/10))/scale
        is_single=self.family=="single"
        y=y.float() if is_single else y.double()
        if qe is None and q==2:
            return (-4*y/n0).reshape(len(y),-1)
        y=y.reshape(len(y),-1,1)
        levels=self.levels(q,is_single)
        if qe:
            pairs=np.sqrt(alpha)*levels[:,None]+np.sqrt(1-alpha)*self.levels(qe,False)[None,:]
            probs=t.logsumexp(-(y[:,:,:,None]-pairs)**2/n0,dim=-1)
        else:
            probs=-(y-levels)**2/n0
        labels=t.arange(len(levels),device=self.device)
        ll=[]
        for b in reversed(range(q//2)):
            one=((labels>>b)&1).bool()
            ll.append(t.logsumexp(probs[:,:,one],dim=-1)-t.logsumexp(probs[:,:,~one],dim=-1))
        return t.stack(ll,dim=-1).reshape(len(y),-1)

    @lru_cache(maxsize=8192)
    def packed(self,payload,k):
        v=np.frombuffer(payload,dtype=np.uint8);n=len(v)
        require(n<=k-29 and n!=1,"actual protected information capacity")
        pre=np.zeros(k-16,dtype=np.uint8)
        pre[:13]=((n>>np.arange(12,-1,-1))&1).astype(np.uint8);pre[13:13+n]=v
        return self.legacy.append_crc_batch(pre)

    def signal(self,info,n,q,counters,layer):
        t=self.t;k=info.shape[1]
        encoded=self.code(k,n,q)[0](t.as_tensor(info,dtype=t.float32,device=self.device))
        return self.modulate(encoded.to(t.uint8)^self.masks(n,counters,layer),q)

    def decode(self,llrs,k,n,q,counters,layer):
        llrs=llrs.float()*(1-2*self.masks(n,counters,layer).float())
        raw=self.code(k,n,q)[1](llrs).detach().cpu().numpy()
        require(raw.shape==(len(counters),k) and np.isin(raw,[0,1]).all(),"binary LDPC output")
        return raw.astype(np.uint8)

    def payloads(self,hard,empty):
        crc=self.legacy.crc_accept_batch(hard); out=[]
        for row,ok,explicit in zip(hard,crc,empty):
            if not ok:
                out.append(("CRC",None));continue
            length=int(row[:13] @ (1<<np.arange(12,-1,-1)))
            if not (length==0 if explicit else 2<=length<=len(row)-29):
                out.append(("LENGTH",None));continue
            if row[13+length:-16].any():
                out.append(("PADDING",None));continue
            out.append(("OK",row[13:13+length]))
        return out

    @lru_cache(maxsize=4096)
    def transmitted_header(self,pid):
        return self.h.transmit(pid)

    @lru_cache(maxsize=24000)
    def header_observation(self, pid, slot, rep, snr):
        # Only the noisy symbol vector goes into the Viterbi/CRC receiver.
        noise=self.noise[(slot,rep,"w" if snr==4 else "s")]
        y=self.transmitted_header(pid)+noise[:68]*(10**(-snr/20))
        decoded=self.h.receive(y,snr,codebook=None)
        return decoded["profile_id"] if decoded["header_ok"] else None

    def send(self,rows,profile):
        nb,ne,q,br,qe,er,ap=profile[:7]
        c=[slot*3+rep for _,slot,rep,_,_,_ in rows]
        def information(layer,k):
            return np.stack([self.packed(self.data.tx_payload(slot,pid,p,layer).tobytes(),k)
                             for _,slot,rep,pid,p,fixed in rows])
        b=self.signal(information("B",k_info(nb,q,br)),nb*q,q,c,"B").double()
        if self.family=="single":
            body=b
        else:
            e=self.signal(information("E",k_info(ne,qe,er)),ne*qe,qe,c,"E").double()
            body=b*(ap/100)**.5+e*(1-ap/100)**.5 if self.family=="sp" else self.t.cat([b,e],dim=1)
        require(tuple(body.shape)==(len(rows),956,2),"actual fixed PHY symbols")
        Eb=b.square().sum(2).mean(1)
        Ee=self.t.zeros_like(Eb) if self.family=="single" else e.square().sum(2).mean(1)
        Ebody=body.square().sum(2).mean(1)
        # Each QPSK protected header carries exactly 2 energy units/symbol.
        energy=self.t.stack([Eb,Ee,Ebody,(136.+956.*Ebody)/1024.],dim=1).cpu().numpy()
        return body,energy

    def receive_bodies(self,y,profiles,counters,snr):
        # This API takes actual observations, the **received** profiles and public counters only.
        nb,ne,q,br,qe,er,ap=profiles[0][:7]
        alpha=ap/100
        single=self.family=="single";paired=self.family=="sp"
        ll=self.llr(y[:,:nb],snr,q,qe if paired else None,alpha if paired else 1.)
        kb=k_info(nb,q,br)
        hard=self.decode(ll,kb,nb*q,q,counters,"B")
        parsed=self.payloads(hard,[False]*len(hard))
        out=[];next_indices=[]
        for i,((status,bits),p) in enumerate(zip(parsed,profiles)):
            j,m,k=p[7:10]
            if bits is None:
                out.append(("B_"+status,"GRAY",False));continue
            key=byte_key("S",0,m,k,"",bits) if single else byte_key("B",j,j,0,"",bits)
            target=self.received.get(key)
            if target is None:
                out.append(("PENDING_S" if single else "PENDING_B","",False));continue
            if single:
                out.append(("S_ACCEPT",target,False));continue
            out.append(("B_EMPTY" if (j,0)==(m,k) else "B_ACCEPT",target,False))
            if (j,0)!=(m,k):
                next_indices.append(i)
        if next_indices:
            sub=[counters[i] for i in next_indices]
            if paired:
                b=self.signal(hard[next_indices],nb*q,q,sub,"B")
                ye=(y[next_indices]-np.sqrt(alpha)*b)/np.sqrt(1-alpha)
            else:
                ye=y[next_indices,nb:]
            ke=k_info(ne,qe,er)
            ell=self.llr(ye,snr,qe,scale=1-alpha if paired else 1.)
            eh=self.decode(ell,ke,ne*qe,qe,sub,"E")
            for i,(status,bits) in zip(next_indices,self.payloads(eh,[False]*len(eh))):
                _,bid,_=out[i];j,m,k=profiles[i][7:10]
                if bits is None:
                    out[i]=("E_"+status,bid,paired);continue
                target=self.received.get(byte_key("E",j,m,k,bid,bits))
                out[i]=("BOTH_ACCEPT",target,paired) if target else ("PENDING_E","",paired)
        return out

    def measure(self,rows,folder):
        body,energy=self.send(rows,rows[0][4])
        outcomes=[{} for _ in rows]
        for snr in SNR:
            noise=np.stack([self.noise[(slot,rep,"w" if snr==4 else "s")][68:] for _,slot,rep,_,_,_ in rows])
            y=body+self.t.as_tensor(noise,device=self.device)*10**(-snr/20.)
            received={};groups=defaultdict(list)
            for i,(_,slot,rep,pid,_,fixed) in enumerate(rows):
                got=self.header_observation(pid,slot,rep,snr)
                p=self.data.profile(fixed,got)
                if p is None or p[-1]:
                    outcomes[i][snr]=(got,"HEADER" if p is None else "NULL","GRAY",False)
                else:
                    received[i]=(got,p)
                    groups[p[:7]].append(i)
            for indices in groups.values():
                profiles=[received[i][1] for i in indices]
                counters=[rows[i][1]*3+rows[i][2] for i in indices]
                dec=self.receive_bodies(y[indices],profiles,counters,snr)
                for i,(status,target,sic) in zip(indices,dec):
                    got,_=received[i]
                    if status.startswith("PENDING"):
                        # A new valid-looking string needs the original VAR independently. Preserve
                        # the received symbols for exact continuation. It is neither gray nor success.
                        case,slot,rep,pid,_,fixed=rows[i]
                        self.misses+=1
                        header=self.transmitted_header(pid)+self.noise[(slot,rep,"w" if snr==4 else "s")][:68]*10**(-snr/20)
                        np.savez_compressed(folder/f"unresolved_{case}_{rep}_{snr}.npz",header=header,
                            body=y[i].cpu().numpy(),counter=slot*3+rep,snr=snr,received_profile=got)
                    outcomes[i][snr]=(got,status,target,sic)
        return energy,outcomes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","pilot","plan","source","rx","legacy","out"):
        ap.add_argument("--"+k,required=True,type=Path)
    ap.add_argument("--family",choices=("single","sp","oma"),required=True)
    ap.add_argument("--batch",type=int,default=48)
    a=ap.parse_args()
    require(3<=a.batch<=96 and a.batch%3==0,"registered actual wave batch")
    plan=jread(a.plan/"completion.json")
    for n,sha in plan["inputs"].items():
        require(filehash(a.plan/n)==sha,n)
    rc=jread(a.rx/"completion.json");require(filehash(a.rx/"independent_rx.sqlite")==rc["database_sha256"],"canonical RX lookup")
    roster=jread(a.pilot)["records"];require([s["source_id"] for s in roster]==plan["roster"],"roster order")
    a.out.mkdir()
    start=time.time();cpu=time.process_time();nwave=0;rx_count=0;role=a.family
    try:
        import torch
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:3])
        os.nice(10)
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest");torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        require(torch.cuda.device_count()==1,"one physical GPU")
        props=torch.cuda.get_device_properties(0)
        require("A800" in props.name and os.environ["CUDA_VISIBLE_DEVICES"].endswith(str(props.uuid)), "A800 admission")
        torch.cuda.set_per_process_memory_fraction((10<<30)/props.total_memory,0)
        engine=RadioEngine(a,roster)
        jsave(a.out/"registration.json",dict(family=role,device=props.name,uuid=str(props.uuid),batch=a.batch,
               plan_sha256=filehash(a.plan/"completion.json"),code_sha256=filehash(__file__),
               backend=engine.backend.identity,catalogue_hashes=engine.data.cathashes,seed_bank=jread(a.plan/"protocol.json"),
               receiver_no_tx_truth=True,neural_training_steps=0,source_manifest_sha256=filehash(a.pilot)))
        groups=defaultdict(list)
        with (a.plan/"physical_cases.csv").open() as f:
            for row in csv.DictReader(f):
                if row["family"]!=role:continue
                fixed=(int(row["q"]),row["rate"]) if role=="single" else None
                pid=int(row["pid"]);p=engine.data.profile(fixed,pid)
                groups[p[:7]].append((int(row["case"]),int(row["slot"]),pid,p,fixed))
        require(sum(map(len,groups.values()))==plan["unique_cases"][role], "physical case coverage")
        with torch.inference_mode():
            for gi,(geom,cases) in enumerate(sorted(groups.items())):
                name=f"geometry_{gi:03d}"
                expanded=[(case,slot,rep,pid,p,fixed) for case,slot,pid,p,fixed in cases for rep in range(3)]
                with gzip.open(a.out/(name+".csv.gz"),"xt",encoding="utf8",newline="") as f:
                    writer=csv.writer(f,lineterminator="\n")
                    writer.writerow(["case","slot","repeat","tx_pid","energy_B","energy_E","energy_body","energy_frame"]+
                           [f"{k}_{snr}" for snr in SNR for k in ("rx_pid","status","token_hash","sic")])
                    for off in range(0,len(expanded),a.batch):
                        batch=expanded[off:off+a.batch]
                        energy,outcomes=engine.measure(batch,a.out)
                        for (case,slot,rep,pid,_,_),ev,out in zip(batch,energy,outcomes):
                            writer.writerow([case,slot,rep,pid,*ev.tolist(),*[x for snr in SNR for x in out[snr]]])
                        nwave+=len(batch);rx_count+=len(batch)*3
                        if off==0 or off//a.batch%8==0 or off+len(batch)==len(expanded):
                            f.flush()
                            jsave(a.out/"progress.json",dict(family=role,geometry=gi+1,geometries=len(groups),
                                  waves=nwave,expected=plan["unique_cases"][role]*3,receptions=rx_count,
                                  unresolved=engine.misses,elapsed_seconds=time.time()-start),update=True)
                jsave(a.out/(name+".json"),dict(geometry=geom,waves=len(expanded),filename=name+".csv.gz",
                                                sha256=filehash(a.out/(name+".csv.gz"))))
                print(role,gi+1,len(groups),nwave,"true transmitted waveforms",flush=True)
        require(nwave==3*plan["unique_cases"][role] and rx_count==nwave*3, "full registered family")
        jsave(a.out/"completion.json",dict(status="PHY_RECEPTIONS_RECORDED",family=role,waves=nwave,
              new_receiver_observations=rx_count,unresolved_source_inputs=engine.misses,
              equipment=props.name,qualified_layouts=engine.qualified,training_steps=0,common500=0,
              wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
    except BaseException as e:
        jsave(a.out/"failure.json",dict(error=repr(e),waves=nwave,receptions=rx_count,elapsed_seconds=time.time()-start))
        raise


if __name__=="__main__":
    main()
