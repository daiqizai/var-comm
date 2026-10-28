"""Matched finite arms; durable update checkpoints and actual source-budget evaluation."""
from residual_runtime import *
from residual_model import ResidualCodec
from mixed_arithmetic_v1 import classes
from nested_entropy import encode_candidates,decode_residual
import torch,signal,traceback,random,collections,csv
stop=False
def on_stop(*args):
    global stop
    stop=True
def atomic_torch(path,value):
    tmp=path.with_suffix('.new')
    with tmp.open('wb')as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def schedule(seed,count,updates):
    rng=np.random.default_rng(seed);indices=[]
    while len(indices)<updates*4:indices.extend(rng.permutation(count).tolist())
    probs=np.r_[np.full(255,.75/255),.25];js=rng.choice(np.arange(1,257),updates,p=probs)
    return np.asarray(indices[:updates*4],np.int64).reshape(updates,4),js
class Data:
    def __init__(self,index):
        self.rows=read(checked(index))['sources'];self.by={s['source_id']:s for s in self.rows};self.cache={};self.rgb={};self.checked=set()
    def load(self,d,rgb=False):
        cache=self.rgb if rgb else self.cache;p=d['path']
        if p not in cache:
            if p not in self.checked:checked(d);self.checked.add(p)
            cache[p]=torch.load(p,map_location='cpu',weights_only=True)
        return cache[p]
    def sample(self,s):
        x=self.load(s['latents']);i=s['within_latent'];rgb=self.load(s['rgb_shard'],True);j=s['within_rgb'];assert rgb['image_ids'][j]==s['source_id']
        return x['full'][i],x['coarse'][i],rgb['targets_u8'][j].float()/255,x['tokens']if s['layout']=='single'else x['tokens'][i]
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);cpdir=pathlib.Path(r['checkpoint_dir']);cpdir.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic();signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));lp=ledger.call('lpips_load','once',lambda:load_lpips(r));codec=FitCodec(R,native=n);Encoder,Decoder=classes(R/'src/var_comm/whole_entropy.py')
        data=Data(r['cache_index']);train=[s for s in data.rows if s['split']=='train'];dev=[s for s in data.rows if s['split']=='dev'];assert len(train)==19968 and len(dev)==200
        ids,js=schedule(r['seed'],len(train),5000);schedule_hash=hashlib.sha256(ids.tobytes()+js.tobytes()).hexdigest()
        torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed']);model=ResidualCodec(r['order']).cuda();initial=statehash(model);assert initial==r['initial_state_sha256'];optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        start=0;best=None;history=[]
        if r.get('resume'):
            cp=torch.load(checked(r['resume']),map_location='cpu',weights_only=False);assert cp['order']==r['order']and cp['schedule_hash']==schedule_hash and cp['cache_index']==r['cache_index']and cp['scope']==r['scope']
            model.load_state_dict(cp['model']);optim.load_state_dict(cp['optimizer']);start=cp['step'];best=cp['best'];history=cp['history']
            torch.set_rng_state(cp['cpu_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);assert statehash(model)==cp['model_state_sha256']
            assert ids[start].tolist()==cp['next_indices']and int(js[start])==cp['next_j'];assert start==r['start_step']
            save(out/'resume_proof.json',dict(previous=r['resume'],step=start,next_indices=ids[start].tolist(),next_j=int(js[start]),schedule_hash=schedule_hash,model_state_sha256=statehash(model),optimizer_loaded=True,RNG_restored=True,repeated_updates=0))
        else:assert r['start_step']==0
        receiver_tables={};eval_rows=[]
        for s in dev:
            received=torch.load(checked(s['rx_coarse']),map_location='cpu',weights_only=True)['tokens']
            with np.load(checked(s['rx_cdfs']),allow_pickle=False)as z:
                for scale in range(4):
                    key=(scale,tuple(received[:OFFSETS[scale]]));table=z['cdf'+str(scale)]
                    if key in receiver_tables:assert np.array_equal(receiver_tables[key],table)
                    else:receiver_tables[key]=table
        def receive(bits,j):
            d=Decoder(bits);e=Encoder();tokens=[]
            for scale in range(4):
                key=(scale,tuple(tokens))
                assert key in receiver_tables,'Unseen received coarse context: stop, never borrow TX tables'
                table=receiver_tables[key];v=d.decode(table);e.encode(v,table);tokens+=v.tolist()
            fc=coarse_latent(n,tokens)
            with torch.no_grad():pr=model.probabilities(fc)[0].cpu().numpy()
            q,depth=decode_residual(d,e,j,pr,r['order']);assert np.array_equal(e.finish(),bits)and d.position-len(bits)==30
            return fc,torch.from_numpy(q)[None].cuda(),torch.from_numpy(depth)[None].cuda()
        def evaluate(step):
            model.eval();folder=out/f'eval_{step:05d}';folder.mkdir();rows=[]
            with torch.no_grad():
                for s in dev:
                    full,fc,target,tokens=data.sample(s);full=full[None].cuda();fc=fc[None].cuda();target=target[None].cuda()
                    u=model.encode(full,fc);pr=model.probabilities(fc);values=torch.floor((u+1)*8).clamp(0,15).long()[0].cpu().numpy()
                    with np.load(checked(s['coarse_stream']),allow_pickle=False)as z:tables=[z['cdf'+str(q)]for q in range(4)]
                    parts=[tokens[OFFSETS[q]:OFFSETS[q+1]].numpy()for q in range(4)]
                    streams,_=ledger.call('eval_encode',f'{step}|{s["source_id"]}',lambda:encode_candidates(Encoder,parts,tables,values,pr[0].cpu().numpy(),r['order']))
                    chosen={b:max((j for j,wire in streams.items()if len(wire)<=b),default=None)for b in(449,547,608)};scores={}
                    for j in dict.fromkeys(chosen.values()):
                        key=f'{step}|{s["source_id"]}|{j}'
                        def one():
                            if j is None:image=torch.full_like(target,.5);bits=0
                            else:
                                rxfc,q,depth=receive(streams[j],j);image=dc(model.decode(q,depth,rxfc));bits=len(streams[j])
                            mse=float((image.double()-target.double()).square().mean());pl=float(lp(image*2-1,target*2-1).mean());return dict(j=j,actual_bits=bits,mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=pl,D=mse+.1*pl,status='PAID_NULL'if j is None else'LOSSLESS_RX')
                        scores[j]=ledger.call('eval_rx_image_metric',key,one)
                    rr=[dict(source_id=s['source_id'],source_index=s['source_index'],step=step,order=r['order'],budget=b,**scores[j])for b,j in chosen.items()];rows+=rr;save(folder/(str(s['source_index'])+'.json'),rr)
            value=float(np.mean([v['D']for v in rows]));save(folder/'summary.json',dict(step=step,objective=value,rows=rows,unit='200 calibration sources, three equal actual source budgets',new_wireless=0));model.train();return value
        def checkpoint(step):
            cp=dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=None,amp_scaler=None,step=step,global_samples=step*4,order=r['order'],scope=r['scope'],cache_index=r['cache_index'],schedule_hash=schedule_hash,model_state_sha256=statehash(model),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),best=best,history=history,next_indices=ids[step].tolist()if step<5000 else None,next_j=int(js[step])if step<5000 else None,frozen_models=n.loaded['identity'],format='MIXED_4096_2_AC_V1_FP32')
            atomic_torch(cpdir/'latest.pt',cp)
        model.train();step=start
        for z in range(start,r['stop_step']):
            if stop or(L/'STOP').exists()or time.monotonic()-begin>r['seconds']-180:break
            selected=[train[i]for i in ids[z]];xs=[data.sample(s)for s in selected];full=torch.stack([x[0]for x in xs]).cuda();fc=torch.stack([x[1]for x in xs]).cuda();target=torch.stack([x[2]for x in xs]).cuda();j=int(js[z])
            def update():
                nonlocal step
                optim.zero_grad(set_to_none=True);latent,rate=model(full,fc,j);image=dc(latent);mse=(image-target).square().flatten(1).mean(1);pl=lp(image*2-1,target*2-1).flatten();dist=mse+.1*pl;loss=(dist+1e-5*rate).mean();assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);optim.step()
                item=dict(step=z+1,j=j,loss=float(loss.detach()),D=float(dist.mean().detach()),estimated_bits=float(rate.mean().detach()),gradient_norm=float(norm),source_ids=[s['source_id']for s in selected]);history.append(item);step=z+1
                checkpoint(step) # Mark ledger COMPLETE only after durable model/optimizer/RNG.
                return item
            item=ledger.call('training_update',str(z+1),update)
            if step%1000==0:
                score=evaluate(step)
                if best is None or score<best['objective']:
                    best=dict(step=step,objective=score,path=str(cpdir/f'checkpoint_{step:05d}.pt'))
                checkpoint(step);atomic_torch(cpdir/f'checkpoint_{step:05d}.pt',torch.load(cpdir/'latest.pt',map_location='cpu',weights_only=False))
            if step%25==0:print('RESIDUAL_TRAIN',r['order'],step,item['D'],item['estimated_bits'],flush=True)
        assert statehash(dc)==n.loaded['identity']['models']['decoder']and all(p.grad is None for p in dc.parameters())
        for name in('vae','var'):assert statehash(n.loaded[name])==n.loaded['identity']['models'][name]
        assert step>start
        save(out/'completion.json',dict(status='ARM_COMPLETE'if step==5000 else'RESUMABLE_BOUNDARY',step=step,new_updates=step-start,new_samples=4*(step-start),checkpoint=pin(cpdir/'latest.pt'),best=best,schedule_hash=schedule_hash,frozen_models=n.loaded['identity'],ledger=ledger.summary(),seconds=time.monotonic()-begin,wireless_frames=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
