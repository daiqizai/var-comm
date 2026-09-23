"""Actual N3060 V/P confirmation and fresh P4084 seed repeats; shared original engine."""
from short_prefix.train import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['m6','m7','m8','pure'],required=True);p.add_argument('--seed',type=int,default=2026092304);p.add_argument('--until',type=int,default=20000);p.add_argument('--N',type=int,choices=[3060,4084],required=True);p.add_argument('--qualification-only',action='store_true');p.add_argument('--cache-from-preflight',action='store_true');a=p.parse_args()
    configure_runtime();require_available();cfg=config();device=torch.device('cuda:0');m=None if a.group=='pure' else int(a.group[1:])
    if (a.N==3060 and m not in (6,7)) or (a.N==4084 and m is not None):raise ValueError('scoped trainer is N3060 V/P or freshP4084 repeat only')
    out=OUT/f'scoped_N{a.N}'/'training'/f'{a.group}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    vae,var=load_models(model_paths(),device);del var;decoder=load_decoder(vae,device);del vae
    scale=scale_statistics(device);lp=perceptual_model(device)
    if a.cache_from_preflight and not a.qualification_only:raise ValueError('preflight is qualification only')
    cal=Population('calibration',m,N=a.N,preflight=a.cache_from_preflight);train=None if a.qualification_only else Population('train',m,N=a.N)
    if train and set(train.ids)&set(cal.ids):raise RuntimeError('train/cal overlap')
    pure_parent=None
    if m:
        models=matched_models(m,scale.cpu(),a.seed,N=a.N)
        if m==8:del models['H8-P']
    else:
        if a.N!=4084 or a.seed not in cfg['repeats']['additional_training_seeds']:raise ValueError('fresh pure repeats only; reuse existingP3060')
        torch.manual_seed(a.seed);models=torch.nn.ModuleDict({'P4084':PureContinuous(scale.cpu())})
    models.to(device);opts={n:torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for n,model in models.items()}
    ids=torch.arange(4);b=cal.batch(ids,torch.zeros_like(ids),torch.zeros_like(ids),[4101]*4,device)
    # Always rerun actual-device qualification when explicitly requested; it never updates real training arms.
    if a.qualification_only:
        qualify(models,b,decoder,lp,scale,out);return
    if not (out/'qualification.json').exists():qualify(models,b,decoder,lp,scale,out)
    registration={'bindings':identity_files([__file__,ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py',ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/data.py']),'protocol':cfg,'N':a.N,'fresh_pure_repeat':m is None,'group':a.group,'seed':a.seed,'decoder_state_sha256':state_sha256(decoder),'cache_train':train.cache_identity,'cache_calibration':cal.cache_identity,'source_image_bindings':{'train':train.image_bindings,'calibration':cal.image_bindings},'parent':pure_parent,'parameter_counts':{n:sum(p.numel() for p in v.parameters()) for n,v in models.items()},'order_seed':cfg['data_seed']+a.seed-cfg['initialization_seed'],'channel_seed':cfg['channel_seed']+a.seed-cfg['initialization_seed']}
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
        path=out/'checkpoints'/f"step_{state['step']:05d}_{reason}.pt"
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
        write_json(out/'completion.json',{'status':'REGISTERED_MILESTONE_COMPLETE_NOT_CONVERGENCE','state':state,'selected':{n:read(out/f'selected_{n}.json') for n in models},'registration_sha256':regsha,'synthetic':False,'pending':'calibration-only extension decision, registered development and seed repeats; new holdout deferred'})
        write_json(out/'status.json',{'status':'MILESTONE_COMPLETE','state':state})
    except ResourceBusy as e:
        checkpoint('safe_paired_pause');write_json(out/'status.json',{'status':'PAUSED_SAFE_CHECKPOINT','reason':str(e),'state':state});raise SystemExit(75)

if __name__=='__main__':main()
