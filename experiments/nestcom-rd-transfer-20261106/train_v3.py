"""One finite eight-device VAR/NeST RD-transfer tranche with full-quality guard."""
import argparse,json,os,random,signal,time,math
from pathlib import Path
from runtime import *
setup()
import numpy as np,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model_v2 import Transfer,tensor_hash
from packet import encode_fit,decode
from data import Corpus,image
from perceptual import load as load_lp
STOP=False
RUN=L/'training'
def stop(*args):
    global STOP;STOP=True
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def save(model,opt,step,cursor,states,r,folder,reason,baseline,bad):
    folder.mkdir(parents=True,exist_ok=True);p=folder/f'rolling_{(step//r["checkpoint_steps"])%2}.pt';t=p.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,cursor=cursor,rng_by_rank=states,request=r,world_size=8,reason=reason,original_development_baseline=baseline,consecutive_adverse=bad),t);t.replace(p)
    rec=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,step=step,cursor=cursor,reason=reason)
    write(folder/'latest.json',rec);write(RUN/'latest_checkpoint.json',rec)
@torch.no_grad()
def evaluate(model,lp,rows,rank,step,capacities):
    out=[]
    for j,row in enumerate(rows):
        if j%8!=rank:continue
        x=image(row)[None].cuda();ids,_,_,full,_=model.quantize(x,soft=False)
        target=(x+1)/2;y=model.decoder(full)
        out.append(dict(source_id=row['source_id'],kind='full',psnr=float(-10*torch.log10((y-target).square().mean())),lpips_vgg=float(lp(y*2-1,x).mean())))
        # Fixed first eight development sources cover all four actual budgets.
        if j<8:
            for cap in capacities:
                packet,rec=encode_fit(model,ids,cap);rx=decode(model,packet)
                if len(rx):_,f=model.receive(rx);y=model.decoder(f)
                else:y=torch.full_like(x,.5)
                out.append(dict(source_id=row['source_id'],kind='prefix',psnr=float(-10*torch.log10((y-target).square().mean())),lpips_vgg=float(lp(y*2-1,x).mean()),**rec))
    gathered=[None]*8;dist.all_gather_object(gathered,out);allrows=sum(gathered,[])
    if rank==0:write(RUN/f'dev_step{step:06d}.json',dict(step=step,rows=allrows,heldout500_used=False))
    full=[x for x in allrows if x['kind']=='full'];assert len(full)==32
    return dict(psnr=sum(x['psnr'] for x in full)/32,lpips_vgg=sum(x['lpips_vgg'] for x in full)/32)
def main():
    a=argparse.ArgumentParser();a.add_argument('--request',required=True);args=a.parse_args();r=json.loads(Path(args.request).read_text())
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    os.sched_setaffinity(0,set(r['rank_cpus'][str(rank)]));torch.cuda.set_device(rank);numerical();old_paused()
    for p,h in r['bindings'].items():assert sha(p)==h,p
    dist.init_process_group('nccl');signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    random.seed(r['seed']+rank);np.random.seed(r['seed']+rank);torch.manual_seed(r['seed']+rank);torch.cuda.manual_seed(r['seed']+rank)
    model=Transfer(L/'model_routes.json');lp=load_lp();opt=torch.optim.AdamW([{'params':ps,'lr':r['learning_rates'][k],'base_lr':r['learning_rates'][k]} for k,ps in model.groups().items()],betas=(.9,.95),weight_decay=.01)
    # Main starts from original verified NeST, not the qualification's updated model.
    corpus=Corpus(L/'training_sources.json');dev=json.loads((L/'development_sources.json').read_text())['sources']
    global RUN
    RUN=Path(r.get('output',str(L/'training')));out=RUN;out.mkdir(parents=True,exist_ok=True)
    log=(out/f'updates_rank{rank}.jsonl').open('x');step=cursor=0;bad=0
    if r.get('resume'):
        ident=r['resume'];assert sha(ident['path'])==ident['sha256']
        st=torch.load(ident['path'],map_location='cpu',weights_only=False)
        assert st['world_size']==8
        for key in ['seed','source_capacities','learning_rates','global_batch','training_precision']:
            assert st['request'][key]==r[key],key
        assert st['request']['bindings'][str(L/'training_sources.json')]==sha(L/'training_sources.json')
        model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer'])
        state=st['rng_by_rank'][rank];random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
        step=st['step'];cursor=st['cursor'];baseline=st['original_development_baseline'];bad=st['consecutive_adverse'];del st
    else:baseline=evaluate(model,lp,dev,rank,0,r['source_capacities'])
    ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
    if rank==0:write(out/'training_started.json',dict(time=time.time(),parameters=model.counts(),world_size=8,global_batch=8,baseline=baseline,original_initialization=True,training_sources=len(corpus.rows)))
    frozen=tensor_hash(model.vae.quantize);reason='finite tranche finished';last_saved=-1
    while step<r['max_steps']:
        flag=torch.tensor(int(STOP or (L/'STOP').exists() or time.time()>=r['deadline']-180),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if int(flag):reason='safe user/deadline stop';break
        tick=time.time();x,sid=corpus.get(cursor+rank);x=x[None].cuda();cap=r['source_capacities'][(step+rank)%len(r['source_capacities'])]
        for g in opt.param_groups:g['lr']=g['base_lr']*min(1.,(step+1)/100)
        opt.zero_grad(set_to_none=True);loss,record=ddp(x,cap,lp,rate_encoder=True)
        assert loss.isfinite();loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();step+=1;cursor+=8
        assert tensor_hash(model.vae.quantize)==frozen
        torch.cuda.synchronize();rec=dict(step=step,source_id=sid,seconds=time.time()-tick,loss=float(loss.detach()),gradient_norm=float(norm),**record);log.write(json.dumps(rec)+'\n');log.flush()
        if rank==0:write(out/'progress.json',dict(time=time.time(),step=step,image_exposures=cursor,world=8,deadline=r['deadline'],record=rec))
        if step%r['validation_steps']==0:
            quality=evaluate(model,lp,dev,rank,step,r['source_capacities'])
            adverse=quality['psnr']<baseline['psnr']-.15 or quality['lpips_vgg']>baseline['lpips_vgg']+.01
            bad=bad+1 if adverse else 0
            if rank==0:write(out/'quality_guard.json',dict(step=step,baseline=baseline,current=quality,consecutive_adverse=bad,stop=bad>=2))
        if step==1 or step%r['checkpoint_steps']==0 or bad>=2:
            states=[None]*8 if rank==0 else None;dist.gather_object(rng(),states,dst=0)
            if rank==0:save(model,opt,step,cursor,states,r,CK/'main_v1','periodic' if bad<2 else 'full reconstruction guard',baseline,bad)
            dist.barrier();last_saved=step
        if bad>=2:reason='full reconstruction guard';break
    if last_saved!=step:
        states=[None]*8 if rank==0 else None;dist.gather_object(rng(),states,dst=0)
        if rank==0:save(model,opt,step,cursor,states,r,CK/'main_v1',reason,baseline,bad)
        dist.barrier()
    if rank==0:write(out/'segment_complete.json',dict(step=step,reason=reason,finished_tranche=step>=r['max_steps'],old_four_and_640M_paused=True))
    log.close();dist.destroy_process_group()
if __name__=='__main__':main()
