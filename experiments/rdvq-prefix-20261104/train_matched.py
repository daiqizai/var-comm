"""Finite, checkpointable R1/R2/R3 screening, never a background retry loop."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,random,signal,time
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from training_core import PrefixModel,load_training_lpips,rng_state,restore_rng
from codec_adapter import source_image,encode_all_prefixes

STOP=False
def request_stop(signum,frame):
    global STOP;STOP=True

def save_json(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class FixedOrder:
    """Random crop/flip keyed by seed, epoch, source; identical across arms."""
    def __init__(self,manifest,seed):
        self.rows=json.loads(Path(manifest).read_text())['sources'];assert len(self.rows)==20000
        self.seed=seed;self.orders={}
    def image(self,position):
        epoch,j=divmod(position,len(self.rows))
        if epoch not in self.orders:
            order=list(range(len(self.rows)));random.Random(self.seed+1000003*epoch).shuffle(order);self.orders={epoch:order}
        i=self.orders[epoch][j];r=self.rows[i];g=random.Random(f'{self.seed}:{epoch}:{r["source_id"]}:crop-v1')
        p=Path(r['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256']
        with Image.open(p) as src:im=src.convert('RGB')
        im=TF.resize(im,256,interpolation=TF.InterpolationMode.BICUBIC,antialias=True)
        w,h=im.size;top=g.randrange(h-256+1);left=g.randrange(w-256+1);flip=g.random()<.5
        im=TF.crop(im,top,left,256,256)
        if flip:im=TF.hflip(im)
        x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1
        return x,dict(source_id=r['source_id'],epoch=epoch,top=top,left=left,flip=flip)

def native_state(model):
    d={k:v.detach().cpu() for k,v in model.vq.state_dict().items()}
    for k,v in model.ar.state_dict().items():
        name='quantize.condition_entropy_small.'+k;assert name not in d;d[name]=v.detach().cpu()
    return {'model':d}

def checkpoint(model,opt,gen,step,r,folder):
    # Alternate two complete snapshots. Atomic replace preserves the last good one.
    slot=f'rolling_{(step//50)%2}.pt';p=folder/slot;tmp=folder/(slot+'.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(gen),step=step,
        cursor=step*16,request_sha256=r['_request_sha256'],experiment_sha256=r['experiment_sha256'],arm=r['arm'],seed=r['seed']),tmp)
    tmp.replace(p);identity=dict(path=str(p),sha256=file_sha(p),step=step,cursor=step*16,request_sha256=r['_request_sha256'])
    save_json(folder/'latest.json',identity)
    if step in (2000,r['updates']):
        native=folder/f'native_step{step:05d}.pt'
        if not native.exists():
            torch.save(native_state(model),native)
            save_json(folder/f'native_step{step:05d}.json',dict(path=str(native),sha256=file_sha(native),step=step,arm=r['arm'],seed=r['seed']))
        else:
            saved=json.loads((folder/f'native_step{step:05d}.json').read_text());assert saved['sha256']==file_sha(native) and saved['arm']==r['arm'] and saved['seed']==r['seed']
    return identity

@torch.no_grad()
def actual_rate_audit(model,rows,out,step,arm,gen):
    """Actual independently terminated bytes on fixed dev images; no test input."""
    out.mkdir(exist_ok=False);state=rng_state(gen);was=model.training;model.eval();records=[];diff=[]
    for row in rows:
        sid=row['source_id'];p=Path(row['path']);assert file_sha(p)==row['sha256']
        x,_,identity=source_image(p);src=out/sid;src.mkdir()
        save_json(src/'started.json',dict(source_id=sid,encoder_calls=1,started=time.time()))
        packets,lengths,idx,mask,timing=encode_all_prefixes(model.vq,model.ar,model.codebook,x)
        # Explicit received hard embeddings for the rate proxy. No soft target
        # affects termination or selects a reported evaluation prefix.
        ii=idx.cuda();proxy=model.rate_estimates(model.codebook[ii],ii)[0].cpu().numpy()
        actual=np.array([72]+[v['packet_bits'] for v in lengths],dtype=np.float64)
        diff.append(actual-proxy)
        for t,packet in packets.items():(src/f's{t:02d}.bin').write_bytes(packet.to_bytes())
        save_json(src/'complete.json',dict(identity=identity,lengths=lengths,estimated_packet_bits=proxy.tolist(),timing=timing))
        records.append(dict(source_id=sid,actual=actual.tolist(),estimated=proxy.tolist()))
        save_json(out/'progress.json',dict(sources=len(records),total=len(rows),step=step))
    correction=np.median(np.stack(diff),axis=0);correction[0]=0
    if arm=='R3':model.rate_correction.copy_(torch.tensor(correction,device=model.rate_correction.device,dtype=torch.float32))
    else:assert torch.equal(model.rate_correction,torch.zeros_like(model.rate_correction))
    save_json(out/'complete.json',dict(passed=True,step=step,arm=arm,sources=len(rows),encoder_calls=len(rows),independent_terminations=28*len(rows),
        applied=arm=='R3',median_packet_minus_proxy_bits=correction.tolist(),records=records,scientific_scope='development rate audit; not quality selection'))
    model.train(was);restore_rng(state,gen)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args();rp=Path(args.request);r=json.loads(rp.read_text());r['_request_sha256']=file_sha(rp)
    O=Path(r['output']);assert O.is_dir() and not (O/'training_started.json').exists()
    assert r['updates']==4000 and r['arm'] in ['R1','R2','R3','R3_estimated']
    for p,h in r['bindings'].items():assert file_sha(p)==h,p
    assert json.loads(Path(r['p1_gate']).read_text())['passed']
    data=FixedOrder(r['training_manifest'],r['seed']);audit=json.loads(Path(r['audit_manifest']).read_text())['sources'];assert len(audit)==128
    C=Path(r['checkpoint_dir']);assert C.parent.is_dir();C.mkdir(exist_ok=bool(r.get('resume')))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    save_json(O/'training_started.json',dict(started=time.time(),request_sha256=r['_request_sha256'],resume=r.get('resume',False)))
    model=PrefixModel(r['initialization']['path']);assert file_sha(r['initialization']['path'])==r['initialization']['sha256'];model.train()
    lp=load_training_lpips(json.loads(Path(r['metric_assets']).read_text()));opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95))
    random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True
    gen=torch.Generator().manual_seed(r['seed']+40001)
    step=0
    if r.get('resume'):
        prior=r['resume'];assert file_sha(prior['path'])==prior['sha256']
        state=torch.load(prior['path'],map_location='cpu',weights_only=False)
        assert state['experiment_sha256']==r['experiment_sha256'] and state['arm']==r['arm'] and state['seed']==r['seed']
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],gen)
        step=state['step'];assert state['cursor']==step*16 and step<r['updates'];del state
    else:
        # All arms share the initial codec. Reuse only the identity-checked R0 audit.
        initial=json.loads(Path(r['initial_rate_audit']).read_text());assert initial['checkpoint_sha256']==r['initialization']['sha256'] and initial['source_ids']==[x['source_id'] for x in audit]
        if r['arm']=='R3':model.rate_correction.copy_(torch.tensor(initial['median_packet_minus_proxy_bits'],device='cuda',dtype=torch.float32))
        save_json(O/'initial_rate_reuse.json',dict(path=r['initial_rate_audit'],sha256=file_sha(r['initial_rate_audit']),applied=r['arm']=='R3'))
    start=time.monotonic()
    with (O/'updates.jsonl').open('x') as log:
        segment_end=r.get('segment_end',r['updates']);assert segment_end in (2000,4000) and step<segment_end<=r['updates']
        while step<segment_end:
            if STOP or (Path(r['root'])/'STOP_P2').exists() or time.monotonic()-start>r['training_seconds']-180:
                ident=checkpoint(model,opt,gen,step,r,C);save_json(O/'paused.json',dict(step=step,checkpoint=ident,reason='signal, requested stop or registered wall limit'));return
            opt.zero_grad(set_to_none=True);values=[];crops=[];branches=[];torch.cuda.synchronize();t0=time.perf_counter()
            for micro in range(4):
                batch=[data.image(step*16+micro*4+j) for j in range(4)];x=torch.stack([v[0] for v in batch]).cuda();crops.extend(v[1] for v in batch)
                loss,record=model.objective(x,r['arm'],lp,gen);assert torch.isfinite(loss);(loss/4).backward();values.append(float(loss.detach()));branches.append(record)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();step+=1;torch.cuda.synchronize()
            row=dict(step=step,mean_loss=float(np.mean(values)),seconds=time.perf_counter()-t0,examples=crops,branches=branches,peak_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row)+'\n');log.flush();save_json(O/'progress.json',dict(step=step,target=r['updates'],loss=row['mean_loss'],step_seconds=row['seconds'],elapsed_seconds=time.monotonic()-start))
            if step%500==0:actual_rate_audit(model,audit,O/f'rate_audit_{step:05d}',step,r['arm'],gen)
            if step%50==0:checkpoint(model,opt,gen,step,r,C)
    if step<r['updates']:
        save_json(O/'segment_complete.json',dict(passed=True,step=step,target=r['updates'],planned_development_boundary=True,native_checkpoint=str(C/f'native_step{step:05d}.pt'),resume_checkpoint=json.loads((C/'latest.json').read_text()),seconds=time.monotonic()-start));return
    save_json(O/'complete.json',dict(passed=True,updates=step,image_exposures=step*16,arm=r['arm'],seed=r['seed'],native_checkpoint=str(C/f'native_step{step:05d}.pt'),seconds=time.monotonic()-start,
        final_quality_pending=True,not_convergence_claim=True))

if __name__=='__main__':main()
