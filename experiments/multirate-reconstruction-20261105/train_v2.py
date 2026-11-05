"""One recoverable distributed main model, with explicit corpus transitions.

256 and 512 are training stages. A finite allocation bounds each invocation;
the schedule and image cursor survive pauses. Old P2 stop flags are untouched.
"""
from pathlib import Path
import argparse,contextlib,json,math,os,random,signal,time
from runtime import setup,L,OLD,CHECKPOINT_ROOT,sha,write
setup()
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torchvision.transforms import functional as TF
from model import MultiRateCodec
from training_core import load_training_lpips
STOP=False
def request_stop(*args):
    global STOP;STOP=True
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])
class Corpus:
    def __init__(self,path,seed):
        self.path=str(path);self.identity=sha(path);self.rows=json.loads(Path(path).read_text())['sources'];self.seed=seed;self.orders={}
        assert len(self.rows)>=19999
    def image(self,position,size):
        epoch,j=divmod(position,len(self.rows))
        if epoch not in self.orders:
            order=list(range(len(self.rows)));random.Random(self.seed+1000003*epoch).shuffle(order);self.orders={epoch:order}
        row=self.rows[self.orders[epoch][j]];p=Path(row['path']);assert sha(p)==row['sha256']
        g=random.Random(f'{self.seed}:{epoch}:{row["source_id"]}:{size}:crop-v1')
        with Image.open(p) as original:im=original.convert('RGB')
        im=TF.resize(im,size,interpolation=TF.InterpolationMode.BICUBIC,antialias=True)
        w,h=im.size;top=g.randrange(h-size+1);left=g.randrange(w-size+1);flip=g.random()<.5
        im=TF.crop(im,top,left,size,size)
        if flip:im=TF.hflip(im)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1,dict(source_id=row['source_id'],epoch=epoch,top=top,left=left,flip=flip)
def available_corpus(out,current):
    """New immutable manifest at an explicit boundary; never mutate active data."""
    root=L/'corpus_coco116k_v2';journal=root/'journal.jsonl'
    if not journal.exists():return current.path
    base=json.loads((root/'bootstrap_train_manifest.json').read_text())['sources']
    held=set(json.loads((root/'registration.json').read_text())['heldout_native_hashes'])
    rows={r['name']:r for r in base}
    with journal.open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # in-progress final write is excluded
            if r.get('status')=='verified':rows[r['name']]=r
    unique={}
    for row in sorted(rows.values(),key=lambda r:r['name']):
        if row['native_rgb_sha256'] not in held:unique.setdefault(row['native_rgb_sha256'],row)
    if len(unique)<=len(current.rows):return current.path
    if len(unique)<len(current.rows)+1000 and not (root/'complete.json').exists():return current.path
    dest=out/f'corpus_{len(unique):07d}.json'
    assert not dest.exists();write(dest,dict(sources=list(unique.values()),images=len(unique),heldout_excluded=True,origin='verified official corpus journal snapshot'))
    return str(dest)
def save(model,opt,step,cursor,corpus,request,out,rank,world):
    local=rng();states=[None]*world if rank==0 else None;dist.gather_object(local,states,dst=0)
    ident=None
    if rank==0:
        folder=CHECKPOINT_ROOT/'main_v1';folder.mkdir(parents=True,exist_ok=True)
        slot=(step//request['checkpoint_steps'])%2;p=folder/f'rolling_{slot}.pt';temp=p.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng_by_rank=states,world_size=world,
              step=step,cursor=cursor,corpus_path=corpus.path,corpus_sha256=corpus.identity,
              schedule=request['schedule'],seed=request['seed'],global_batch=request['global_batch'],bindings=request['bindings']),temp)
        temp.replace(p);ident=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,step=step,cursor=cursor,corpus=corpus.path)
        write(folder/'latest.json',ident);write(out/'latest_checkpoint.json',ident)
    dist.barrier();return ident
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();reqpath=Path(a.request);r=json.loads(reqpath.read_text())
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    torch.set_num_threads(2);dist.init_process_group('nccl')
    for name,h in r['bindings'].items():assert sha(name)==h,name
    assert (OLD/'STOP_P2').exists();out=Path(r['output'])
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    init=json.loads((OLD/'checkpoint.json').read_text());assert sha(init['path'])==init['sha256']
    model=MultiRateCodec(init['path']);model.train()
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True)
    lp=load_training_lpips(json.loads((OLD/'metric_assets.json').read_text()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=r['learning_rate'],betas=(.9,.95),weight_decay=.01)
    random.seed(r['seed']+rank);np.random.seed(r['seed']+rank);torch.manual_seed(r['seed']+rank);torch.cuda.manual_seed(r['seed']+rank)
    corpus=Corpus(r['training_manifest'],r['seed']);step=cursor=0
    if r.get('resume'):
        prior=r['resume'];assert sha(prior['path'])==prior['sha256'];st=torch.load(prior['path'],map_location='cpu',weights_only=False)
        assert st['global_batch']==r['global_batch'] and st['schedule']==r['schedule'] and st['seed']==r['seed']
        assert st['world_size']==world,'World-size change requires a separately recorded RNG/data migration; no silent restart'
        model.load_state_dict(st['model'],strict=True);optimizer.load_state_dict(st['optimizer']);restore(st['rng_by_rank'][rank])
        step=st['step'];cursor=st['cursor'];corpus=Corpus(st['corpus_path'],r['seed']);assert corpus.identity==st['corpus_sha256'];del st
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    assert r['global_batch']%world==0
    start=time.monotonic();end_step=sum(v['updates'] for v in r['schedule']);last_save=step
    if rank==0:write(out/'training_started.json',dict(time=time.time(),parameters=model.parameter_counts(),world_size=world,global_batch=r['global_batch'],corpus=corpus.path,corpus_images=len(corpus.rows),request_sha256=sha(reqpath)))
    log=(out/f'updates_rank{rank}.jsonl').open('x')
    while step<end_step:
        flag=torch.tensor(int(STOP or (L/'STOP_MAIN').exists() or time.monotonic()-start>r['wall_seconds']-600),device='cuda')
        dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if int(flag):break
        size=256 if step<r['schedule'][0]['updates'] else 512
        microbatch=r['microbatch_256'] if size==256 else r['microbatch_512']
        assert r['global_batch']%(world*microbatch)==0;accum=r['global_batch']//(world*microbatch)
        full=step%4==0;optimizer.zero_grad(set_to_none=True);tick=time.monotonic();records=[];examples=[]
        lr=r['learning_rate']*min(1.,(step+1)/r['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*step/end_step)))
        for group in optimizer.param_groups:group['lr']=lr
        for micro in range(accum):
            positions=[cursor+micro*world*microbatch+rank*microbatch+j for j in range(microbatch)]
            items=[corpus.image(position,size) for position in positions];x=torch.stack([it[0] for it in items]).cuda()
            mode=[(7 if full else position%8)+(0 if size==256 else 8) for position in positions];modes=torch.tensor(mode,device='cuda')
            context=ddp.no_sync() if micro<accum-1 else contextlib.nullcontext()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,rec,reconstruction=ddp(x,modes,full=full,perceptual=lp)
                assert torch.isfinite(loss),f'nonfinite loss at step {step}'
                (loss/accum).backward()
            records.append({k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in rec.items()});examples.extend(dict(it[1],mode=m) for it,m in zip(items,mode))
            del loss,reconstruction,x
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1;cursor+=r['global_batch'];torch.cuda.synchronize()
        log.write(json.dumps(dict(step=step,size=size,seconds=time.monotonic()-tick,lr=lr,grad_norm=float(grad),records=records,examples=examples,corpus_sha256=corpus.identity))+'\n');log.flush()
        if rank==0:write(out/'progress.json',dict(step=step,schedule_target=end_step,resolution=size,step_seconds=time.monotonic()-tick,
            mse=sum(v['mse'] for v in records)/len(records),soft_bpp=sum(v['soft_bpp'] for v in records)/len(records),image_exposures=step*r['global_batch'],corpus_images=len(corpus.rows),elapsed_seconds=time.monotonic()-start))
        if step==1 or step%r['checkpoint_steps']==0:
            path=[available_corpus(out,corpus) if rank==0 else None];dist.broadcast_object_list(path,src=0)
            if path[0]!=corpus.path:
                corpus=Corpus(path[0],r['seed']);cursor=0
                if rank==0:write(out/f'corpus_transition_step{step:07d}.json',dict(step=step,new_manifest=corpus.path,sha256=corpus.identity,images=len(corpus.rows),cursor_reset_for_new_manifest=True))
            # Commit the data transition and model/optimizer in the same snapshot.
            save(model,optimizer,step,cursor,corpus,r,out,rank,world);last_save=step
    if last_save!=step or step==0:save(model,optimizer,step,cursor,corpus,r,out,rank,world)
    if rank==0:write(out/'segment_complete.json',dict(step=step,target=end_step,training_finished=step>=end_step,paused=step<end_step,
        reason='registered window or user stop' if step<end_step else 'schedule complete',old_P2_remains_paused=(OLD/'STOP_P2').exists()))
    log.close();dist.destroy_process_group()
if __name__=='__main__':main()
