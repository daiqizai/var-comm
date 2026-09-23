"""One paired training/selection/resume engine for every registered allocation."""
import argparse,copy,io,os,signal,time
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import (write_json,save_torch,digest,image_losses,perceptual_model,model_paths,verify_snapshot,require_available,ResourceBusy)
from latent_enhancement.training import PairedOrder
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.model import render_received
from latent_research.models import PureContinuous
from latent_followup.run_identity import checked_checkpoint
from latent_mechanisms.requalify_predictor import write_rows
from var_comm.next_scale_prior import load_models,state_sha256
from .common import OUT,ROOT,read,configure_runtime,register,identity_files,Safety
from .protocol import config
from .models import matched_models
from .data import Population

PARENT=ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/research_20260923_training_v3'

def forward(model,b):
    if isinstance(model,PureContinuous):return model(b['F'],b['snr'],b['noise'])
    return model(b)

def losses(model,b,decoder,lp,scale):
    z,wave=forward(model,b)
    pure=isinstance(model,PureContinuous)
    pred=decoder(z) if pure else render_received(decoder,z,b['status'])
    mse,percept=image_losses(pred,b['target'],lp)
    aux=((z-b['F'])/scale[None,:,None,None]).square().flatten(1).mean(1)
    if not pure:aux=aux*(b['status'][:,0]>.5)
    utility=mse+.1*percept+.01*aux
    if not torch.isfinite(utility).all():raise FloatingPointError('training utility')
    return utility,mse,percept,aux,wave

def update(models,opts,b,decoder,lp,scale,micro=4):
    size=len(b['F'])
    for name,model in models.items():
        model.train();opt=opts[name];opt.zero_grad(set_to_none=True)
        for start in range(0,size,micro):
            part={k:v[start:start+micro] for k,v in b.items()}
            loss,*_=losses(model,part,decoder,lp,scale);(loss.sum()/size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()

def same_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b

def qualify(models,b,decoder,lp,scale,out):
    frozen=state_sha256(decoder);probe=copy.deepcopy(models);opts={n:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for n,m in probe.items()}
    # Populate both arms' buffers, parameters, step counters and optimizer moments.
    update(probe,opts,b,decoder,lp,scale)
    names=list(probe);b_name=names[-1];before=copy.deepcopy(probe[b_name].state_dict());opt_before=copy.deepcopy(opts[b_name].state_dict())
    if len(names)>1:
        first=names[0];update({first:probe[first]},{first:opts[first]},b,decoder,lp,scale)
        assert same_tree(before,probe[b_name].state_dict()) and same_tree(opt_before,opts[b_name].state_dict())
    for n,m in probe.items():
        _,wave=forward(m,b);torch.testing.assert_close(wave.square().sum((1,2)),torch.full((len(wave),),2.*m.uses,device=wave.device),atol=.02,rtol=1e-5)
        gradient=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert gradient>0
    buffer=io.BytesIO();torch.save({'models':probe.state_dict(),'opts':{n:o.state_dict() for n,o in opts.items()}},buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=True);restored=copy.deepcopy(probe);restored.load_state_dict(saved['models'])
    resumed_opts={n:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4) for n,m in restored.items()}
    for n in names:resumed_opts[n].load_state_dict(copy.deepcopy(saved['opts'][n]))
    update(probe,opts,b,decoder,lp,scale);update(restored,resumed_opts,b,decoder,lp,scale)
    assert same_tree(probe.state_dict(),restored.state_dict())
    assert all(same_tree(opts[n].state_dict(),resumed_opts[n].state_dict()) for n in names)
    assert state_sha256(decoder)==frozen and all(p.grad is None for p in decoder.parameters())
    write_json(out/'qualification.json',{'status':'REAL_GPU_GRADIENT_ENERGY_POPULATED_OPTIMIZER_ISOLATION_AND_BITWISE_RESUME_PASS','arms':names,'decoder_sha256':frozen,'probe_updates_discarded':True,'synthetic':False})

@torch.no_grad()
def calibrate(models,pop,decoder,lp,scale,device,out,step,safety):
    path=out/'calibration'/f'full_{step:05d}.csv';receipt=path.with_suffix('.json')
    path.parent.mkdir(exist_ok=True);started=time.time();rows=[]
    for start in range(0,len(pop),16):
        if safety.check():raise ResourceBusy('sustained thermal throttle during calibration')
        ids=torch.arange(start,min(start+16,len(pop)))
        for si,snr in enumerate(pop.snrs):
            for ni,seed in enumerate(pop.seeds):
                b=pop.batch(ids,torch.full_like(ids,si),torch.full_like(ids,ni),[seed]*len(ids),device)
                for name,model in models.items():
                    loss,mse,percept,aux,_=losses(model.eval(),b,decoder,lp,scale)
                    for j,i in enumerate(ids):
                        rows.append({'method':name,'source_index':int(i),'image_id':pop.ids[int(i)],'snr_db':snr,'seed':seed,'mse':float(mse[j]),'lpips_alex':float(percept[j]),'normalized_latent':float(aux[j]),'utility':float(loss[j]),'U_image':float(mse[j]+.1*percept[j]),'header_ok':1 if pop.m is None else int(b['status'][j,0]),'body_crc_ok':'not_applicable' if pop.m is None else int(b['status'][j,1])})
    expected=len(pop)*5*3*len(models)
    assert len(rows)==expected and len({(r['method'],r['image_id'],r['snr_db'],r['seed']) for r in rows})==expected
    write_rows(path,rows)
    summary={n:float(np.mean([r['utility'] for r in rows if r['method']==n])) for n in models}
    write_json(receipt,{'step':step,'summary':summary,'rows':len(rows),'sha256':digest(path),'seconds':time.time()-started})
    print('full calibration',step,summary,flush=True);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['m6','m7','m8','pure'],required=True);p.add_argument('--seed',type=int,default=2026092304);p.add_argument('--until',type=int,default=20000);p.add_argument('--qualification-only',action='store_true');p.add_argument('--cache-from-preflight',action='store_true');a=p.parse_args()
    configure_runtime();require_available();cfg=config();device=torch.device('cuda:0');m=None if a.group=='pure' else int(a.group[1:])
    out=OUT/'training'/f'{a.group}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    vae,var=load_models(model_paths(),device);del var;decoder=load_decoder(vae,device);del vae
    scale=scale_statistics(device);lp=perceptual_model(device)
    if a.cache_from_preflight and not a.qualification_only:raise ValueError('preflight is qualification only')
    cal=Population('calibration',m,preflight=a.cache_from_preflight);train=None if a.qualification_only else Population('train',m)
    if train and set(train.ids)&set(cal.ids):raise RuntimeError('train/cal overlap')
    pure_parent=None
    if m:
        models=matched_models(m,scale.cpu(),a.seed)
        if m==8:del models['H8-P']
    else:
        if a.seed!=cfg['initialization_seed']:raise ValueError('pure repeat requires fresh registered seed protocol, not relabeled parent')
        rec=read(PARENT/'latest.json');parent_path=checked_checkpoint(rec,ROOT);pure_parent={'checkpoint':str(parent_path),'sha256':digest(parent_path),'registration_sha256':digest(PARENT/'registration.json')}
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        verify_snapshot(read(PARENT/'registration.json')['bindings'])
        if parent['registration_sha256']!=pure_parent['registration_sha256']:raise RuntimeError('original pure resume binding')
        assert parent['state']['step']==10000 and parent['state']['updates']['pure_continuous']==10000
        models=torch.nn.ModuleDict({'P4084':PureContinuous(scale.cpu())})
        models['P4084'].load_state_dict({k[len('pure_continuous.'):]:v for k,v in parent['models'].items() if k.startswith('pure_continuous.')})
    models.to(device);opts={n:torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for n,model in models.items()}
    ids=torch.arange(4);b=cal.batch(ids,torch.zeros_like(ids),torch.zeros_like(ids),[4101]*4,device)
    # Always rerun actual-device qualification when explicitly requested; it never updates real training arms.
    if a.qualification_only:
        qualify(models,b,decoder,lp,scale,out);return
    if not (out/'qualification.json').exists():qualify(models,b,decoder,lp,scale,out)
    registration={'bindings':identity_files([__file__,Path(__file__).with_name('data.py')]),'protocol':cfg,'group':a.group,'seed':a.seed,'decoder_state_sha256':state_sha256(decoder),'cache_train':train.cache_identity,'cache_calibration':cal.cache_identity,'source_image_bindings':{'train':train.image_bindings,'calibration':cal.image_bindings},'parent':pure_parent,'parameter_counts':{n:sum(p.numel() for p in v.parameters()) for n,v in models.items()},'order_seed':cfg['data_seed']+a.seed-cfg['initialization_seed'],'channel_seed':cfg['channel_seed']+a.seed-cfg['initialization_seed']}
    register(out/'registration.json',registration);regsha=digest(out/'registration.json')
    order=PairedOrder(len(train),registration['order_seed']);rng=torch.Generator().manual_seed(registration['channel_seed'])
    state={'step':0,'updates':{n:0 for n in models},'selection':{n:None for n in models},'last_full':-1,'training_seconds':0.,'calibration_seconds':0.}
    if pure_parent:
        opts['P4084'].load_state_dict(copy.deepcopy(parent['optimizers']['pure_continuous']))
        order.load_state_dict(parent['order']);rng.set_state(parent['rng']);state['step']=10000;state['updates']['P4084']=10000
        torch.set_rng_state(parent['torch_rng']);torch.cuda.set_rng_state_all(parent['cuda_rng']);del parent
    if (out/'latest.json').exists():
        rec=read(out/'latest.json');payload=torch.load(checked_checkpoint(rec,ROOT),map_location='cpu',weights_only=True)
        if payload['registration_sha256']!=regsha:raise RuntimeError('resume identity')
        models.load_state_dict(payload['models'])
        for n in models:opts[n].load_state_dict(copy.deepcopy(payload['optimizers'][n]))
        state=payload['state'];order.load_state_dict(payload['order']);rng.set_state(payload['rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    def checkpoint(reason):
        path=out/'checkpoints'/f"step_{state['step']:05d}.pt"
        save_torch(path,{'models':models.state_dict(),'optimizers':{n:o.state_dict() for n,o in opts.items()},'order':order.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,'registration_sha256':regsha})
        sha=digest(path);write_json(out/'latest.json',{'path':str(path),'sha256':sha,'step':state['step'],'reason':reason})
        for n,rec in state['selection'].items():
            if rec and rec['step']==state['step']:write_json(out/f'selected_{n}.json',{**rec,'checkpoint':str(path),'checkpoint_sha256':sha,'arm_key':n,'registration_sha256':regsha,'total_updates':state['updates'][n],'parent_updates':10000 if pure_parent else 0})
    safety=Safety();stop=[False]
    def halt(*_):stop[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def full():
        checkpoint('before_full_calibration');started=time.time();summary=calibrate(models,cal,decoder,lp,scale,device,out,state['step'],safety)
        state['calibration_seconds']+=time.time()-started
        for n,u in summary.items():
            if state['selection'][n] is None or u<state['selection'][n]['utility']:state['selection'][n]={'step':state['step'],'utility':u}
        state['last_full']=state['step'];checkpoint('full_calibration')
    try:
        if state['step']%2500==0 and state['last_full']!=state['step']:full()
        while state['step']<a.until:
            ids=order.next(cfg['batch_size']);si=torch.randint(5,(len(ids),),generator=rng);ni=torch.randint(len(train.seeds),(len(ids),),generator=rng)
            noise_seed=(2026092302 if pure_parent else registration['channel_seed'])+state['step']
            b=train.batch(ids,si,ni,[noise_seed]*len(ids),device);started=time.time();update(models,opts,b,decoder,lp,scale,cfg['microbatch_size'])
            state['training_seconds']+=time.time()-started;state['step']+=1
            for n in models:state['updates'][n]+=1
            if stop[0] or (state['step']%10==0 and safety.check()):raise ResourceBusy('requested pause or sustained thermal condition')
            if state['step']%100==0:
                write_json(out/'status.json',{'status':'TRAINING','pid':os.getpid(),'state':state,'hardware':safety.last});print('train',a.group,state['step'],flush=True)
            if state['step']%2500==0:full()
            elif state['step']%500==0:checkpoint('regular')
        verify_snapshot(registration['bindings']);assert state_sha256(decoder)==registration['decoder_state_sha256']
        write_json(out/'completion.json',{'status':'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE','state':state,'selected':{n:read(out/f'selected_{n}.json') for n in models},'registration_sha256':regsha,'pending':'calibration-only extension decision, development, second budget, seed repeats, new test'})
        write_json(out/'status.json',{'status':'MILESTONE_COMPLETE','state':state})
    except ResourceBusy as e:
        checkpoint('safe_paired_pause');write_json(out/'status.json',{'status':'PAUSED_SAFE_CHECKPOINT','reason':str(e),'state':state});raise SystemExit(75)
if __name__=='__main__':main()
