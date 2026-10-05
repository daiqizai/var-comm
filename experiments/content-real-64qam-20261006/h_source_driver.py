"""Frozen unconditional H source coding; independent receiver histories, no training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
from h64_catalog import SIZES, require, token_count
from h64_source import encode_prefixes, decode_prefix, reference_primitives

STOP = False

def stop(*args):
    global STOP
    STOP = True

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def verify(bindings):
    for p,h in bindings.items(): require(sha(p)==h,'Changed binding: '+str(p))
def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: np.savez(f,**arrays)
    os.replace(tmp,path)
def split_tokens(flat):
    a=np.asarray(flat)
    require(a.shape==(680,) and np.issubdtype(a.dtype,np.integer) and np.all((a>=0)&(a<4096)), 'Invalid source tokens')
    out=[]; offset=0
    for size in SIZES:
        out.append(a[offset:offset+size*size].astype(np.int64,copy=True));offset+=size*size
    return out

class IndependentProvider:
    """No true source is held here; only explicit advance() changes context."""
    def __init__(self,native,primitives):
        self.native=native;self.primitives=primitives;self.scale=0;self.prior=None
        self.model_seconds=0.; self.cdf_seconds=0.
    def __enter__(self):
        n=self.native
        self.prior=n.receiver._Prior(n.loaded['vae'],n.loaded['var'],n.loaded['device'])
        return self
    def cdf(self):
        t=self.native.torch;t.cuda.synchronize();start=time.monotonic()
        lp=t.log_softmax(self.prior.logits(self.scale),dim=-1)[0].cpu().numpy()
        t.cuda.synchronize();self.model_seconds+=time.monotonic()-start
        start=time.monotonic();cdf=self.primitives.cdf_function(lp)
        self.cdf_seconds+=time.monotonic()-start;return cdf
    def advance(self,values):
        t=self.native.torch;t.cuda.synchronize();start=time.monotonic()
        self.prior.advance(np.asarray(values).copy(),self.scale);self.scale+=1
        t.cuda.synchronize();self.model_seconds+=time.monotonic()-start
    def __exit__(self,*args):
        if self.prior is not None:self.prior.close()

def render_received(native,received,m,K=0):
    """Input is exactly the accepted raster prefix, including any received errors."""
    a=np.asarray(received,dtype=np.int64)
    require(a.shape==(token_count(m,K),) and np.all((a>=0)&(a<4096)), 'Received token count/range differs')
    p=native.receiver._Prior(native.loaded['vae'],native.loaded['var'],native.loaded['device'])
    offset=0
    try:
        for s,size in enumerate(SIZES):
            logits=p.logits(s)
            chosen=logits[0].argmax(-1).cpu().numpy()
            count=size*size if s<m else (K if s==m else 0)
            if count: chosen[:count]=a[offset:offset+count];offset+=count
            p.advance(chosen,s)
        require(offset==len(a),'Receiver did not consume accepted tokens')
        result=native.loaded['decoder'](p.fhat)[0].clamp(0,1).cpu().numpy().astype(np.float32)
        require(result.shape==(3,256,256) and np.isfinite(result).all(),'Invalid decoder RGB')
        return result
    finally:p.close()

def pixel_scores(image,target):
    mse=float(np.mean(np.square(image.astype(np.float64)-target.astype(np.float64))))
    require(mse>0 and np.isfinite(mse),'Unexpected zero/nonfinite image MSE')
    return dict(mse=mse,psnr_db=float(-10*np.log10(mse)))

class Driver:
    def __init__(self,args):
        self.cfg=read(args.config);self.stage=args.stage
        self.reg=read(self.cfg['registration']);self.regsha=sha(self.cfg['registration'])
        require(self.reg['status']=='H_EXECUTION_REVISION_REGISTERED','Unregistered execution')
        verify(self.reg['source_bindings']);verify(self.reg['input_bindings'])
        require(self.reg['input_bindings'].get(str(Path(args.config).absolute()))==sha(args.config),'Config not bound')
        self.out=Path(self.cfg['out'])/args.stage;self.out.mkdir(parents=True,exist_ok=True)
        require(not (self.out/'failure.json').exists(),'Previous failure retained; separate recovery required')
        self.started=time.monotonic();self.native=None
        self.source_ids=read(self.cfg['source200'])['source_ids']
        require(len(self.source_ids)==200 and len(set(self.source_ids))==200,'Expected fixed hash200')
        self.assets=Path(self.cfg['S1'])/'export-assets'
        done=read(self.assets/'completion.json');verify(done['outputs'])
        require(done['status']=='S1_EXPORT_ASSETS_COMPLETE','S1 assets not complete')
        require(read(Path(self.cfg['S1'])/'completion.json')['status']=='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION','S1 not complete')
        q=read(self.cfg['qualification_completion'])
        require(q['status']=='H_PHY_QUALIFICATION_PASS' and q['registration_sha256']==self.regsha,'New PHY not qualified')
        verify(q['outputs'])
    def boundary(self):
        require(not STOP and not (self.out/'STOP').exists(),'Requested safe boundary stop')
        require(time.monotonic()-self.started<self.cfg['max_stage_seconds'],'Registered stage time exhausted')
        if self.native:
            import quality_driver
            require(not quality_driver.boundary(self.native),'Original GPU guard requested stop')
    def build(self):
        sys.path.insert(0,self.cfg['uep_runtime']);sys.path.insert(0,str(Path(self.cfg['root'])/'src'))
        import quality_driver
        from var_comm import whole_entropy,entropy
        self.native=quality_driver.build_native(Path(self.cfg['root']),self.cfg['native_runtime'],stop)
        cal=read(self.cfg['calibration_registration'])
        require(self.native.loaded['identity']==cal['identity'],'Frozen visual identity changed')
        self.primitives=reference_primitives(whole_entropy,entropy)
    def source(self,i):
        cp=read(self.assets/'source_checkpoints'/('%04d.json'%i));verify(cp['outputs'])
        require(cp['source_id']==self.source_ids[i],'Source order changed')
        with np.load(cp['archive'],allow_pickle=False) as z:
            flat=z['tokens'].copy();pixels=z['pixels'].copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==cp['preprocessing_id'],'Preprocessing changed')
        return cp,flat,pixels.astype(np.float32)/255.
    def existing(self,i):
        p=self.out/'source_checkpoints'/('%04d.json'%i)
        if not p.exists():return None
        d=read(p);require(d['registration_sha256']==self.regsha and d['source_id']==self.source_ids[i],'Changed checkpoint')
        verify(d['outputs']);return d
    def checkpoint(self,i,outputs,**values):
        d=dict(registration_sha256=self.regsha,source_index=i,source_id=self.source_ids[i],outputs=outputs,**values)
        atomic(self.out/'source_checkpoints'/('%04d.json'%i),d)
        atomic(self.out/'status.json',dict(status='RUNNING',stage=self.stage,completed_sources=i+1,total_sources=200,
            elapsed_seconds=time.monotonic()-self.started,registration_sha256=self.regsha))
    def run(self):
        if (self.out/'completion.json').exists():
            d=read(self.out/'completion.json');require(d['registration_sha256']==self.regsha,'Different completed registration');verify(d['outputs']);return d
        if self.stage=='clean-quality200':
            d=read(Path(self.cfg['out'])/'source200'/'completion.json')
            require(d['status']=='H_SOURCE200_COMPLETE' and d['registration_sha256']==self.regsha,'Source200 not complete');verify(d['outputs'])
        self.build();t=self.native.torch
        with t.no_grad():
            for i in range(200):
                self.boundary()
                if self.existing(i):continue
                cp,flat,target=self.source(i);scales=split_tokens(flat)
                if self.stage=='source200':
                    providers=[]
                    def factory():
                        p=IndependentProvider(self.native,self.primitives);providers.append(p);return p
                    t.cuda.synchronize();start=time.monotonic()
                    encoded=encode_prefixes(scales,factory,self.primitives)
                    t.cuda.synchronize();encode_seconds=time.monotonic()-start
                    rows=[];arrays={};decoded_history=[]
                    for m,rec in encoded.items():
                        t.cuda.synchronize();start=time.monotonic()
                        # Only the bit vector and public m reach the receiver factory.
                        decoded=decode_prefix(rec['bits'].copy(),m,factory,self.primitives)
                        t.cuda.synchronize();seconds=time.monotonic()-start
                        recovered=np.concatenate(decoded['scales'])
                        require(np.array_equal(recovered,flat[:len(recovered)]),'Independent source roundtrip failed')
                        arrays['m%d_bits'%m]=rec['bits'];decoded_history.append(decoded)
                        image=render_received(self.native,recovered,m)
                        arrays['m%d_image'%m]=image
                        rows.append(dict(m=m,raw_bits=rec['raw_bits'],arithmetic_bits=rec['arithmetic_bits'],
                            flush_bits=rec['flush_bits'],zero_extension_reads=decoded['zero_extension_reads'],
                            decode_seconds=seconds,receiver_probability_seconds=providers[-1].model_seconds,
                            receiver_cdf_seconds=providers[-1].cdf_seconds,**pixel_scores(image,target)))
                    archive=self.out/'sources'/('%04d.npz'%i);save_npz(archive,**arrays)
                    self.checkpoint(i,{str(archive):sha(archive)},lengths=rows,gray=pixel_scores(np.full_like(target,.5),target),
                        encode_prefixes_seconds=encode_seconds,tx_probability_seconds=providers[0].model_seconds,
                        tx_cdf_seconds=providers[0].cdf_seconds,independent_roundtrip=True,
                        timing_scope='asset preparation, not exclusive online end-to-end timing')
                else:
                    catalogue=read(self.cfg['catalogue'])
                    states=sorted({(p['m'],p['K']) for p in catalogue['profiles'] if 'H64-RAW-COMPLETE-STATE-CONTROL' in p['families']},key=lambda x:token_count(*x))
                    rows=[]
                    for j,(m,K) in enumerate(states):
                        if j%16==0:self.boundary()
                        image=render_received(self.native,flat[:token_count(m,K)].copy(),m,K)
                        rows.append(dict(m=m,K=K,token_count=token_count(m,K),**pixel_scores(image,target)))
                    output=self.out/'sources'/('%04d.json'%i);atomic(output,rows)
                    self.checkpoint(i,{str(output):sha(output)},clean_states=len(rows),ranking_only=True,
                        all_legal_K_included=True,new_channel_decodes=0)
        outputs={};rows=[]
        for i in range(200):
            cp=self.existing(i);require(cp is not None,'Missing source');p=self.out/'source_checkpoints'/('%04d.json'%i)
            outputs[str(p)]=sha(p);outputs.update(cp['outputs'])
            for r in cp.get('lengths',[]):rows.append(dict(source_id=cp['source_id'],source_index=i,**r))
        if rows:
            table=self.out/'source_lengths_per_image.csv'
            with table.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            outputs[str(table)]=sha(table)
        verify(self.reg['source_bindings']);verify(self.reg['input_bindings'])
        result=dict(status='H_'+self.stage.upper().replace('-','_')+'_COMPLETE',registration_sha256=self.regsha,
            outputs=outputs,source_count=200,source_population='original calibration construction hash200',
            elapsed_seconds=time.monotonic()-self.started,development_used=False,holdout_used=False,training_updates=0)
        atomic(self.out/'completion.json',result);atomic(self.out/'status.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True,choices=['source200','clean-quality200'])
    args=p.parse_args();d=None;signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        d=Driver(args);print(json.dumps(d.run(),ensure_ascii=False))
    except BaseException as error:
        if d is not None and not (d.out/'failure.json').exists():atomic(d.out/'failure.json',dict(status='FAILED',registration_sha256=d.regsha,error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
        raise

if __name__=='__main__':main()
