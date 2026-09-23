"""New budget orchestration using the existing shared loss/update/calibration engine."""
import argparse,copy,os,signal,time
from pathlib import Path
import torch
from latent_enhancement.runtime import write_json,save_torch,digest,model_paths,perceptual_model,require_available,verify_snapshot,ResourceBusy
from latent_enhancement.training import PairedOrder
from latent_enhancement_b.common import scale_statistics,load_decoder
from latent_followup.run_identity import checked_checkpoint
from var_comm.next_scale_prior import load_models,state_sha256
from short_prefix.common import identity_files
from short_prefix.data import Population
from short_prefix.train import update,qualify,calibrate
from .common import ROOT,EXP,OUT,CONFIG,read,register,config,configure_runtime,Safety
from .models import BudgetContinuous

SEED=2026092304

def main():
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,choices=[2048,3060],required=True);p.add_argument('--qualification-only',action='store_true');a=p.parse_args()
    configure_runtime();require_available();device=torch.device('cuda:0');cfg=config()['continuous']
    out=OUT/'training'/f'P{a.N}_seed{SEED}';out.mkdir(parents=True,exist_ok=True)
    vae,var=load_models(model_paths(),device);del var;decoder=load_decoder(vae,device);del vae
    scale=scale_statistics(device);lp=perceptual_model(device);cal=Population('calibration',N=a.N)
    torch.manual_seed(SEED);models=torch.nn.ModuleDict({f'P{a.N}':BudgetContinuous(scale.cpu(),a.N)}).to(device)
    ids=torch.arange(4);b=cal.batch(ids,torch.zeros_like(ids),torch.zeros_like(ids),[4101]*4,device)
    # A disposable twin is included to exercise populated optimizer isolation even
    # though the real budget model is a single arm. No discarded probe trains it.
    twins=torch.nn.ModuleDict({'A':copy.deepcopy(models[f'P{a.N}']),'B':copy.deepcopy(models[f'P{a.N}'])})
    qualify(twins,b,decoder,lp,scale,out);del twins,b
    if a.qualification_only:return
    targets=read(EXP/'quality_targets.json')
    if targets['status']!='FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT' or targets['source_csv_sha256']!=digest(OUT/'source/calibration/source_codec_per_image.csv'):raise RuntimeError('source A calibration gate')
    train=Population('train',N=a.N)
    if set(train.ids)&set(cal.ids):raise RuntimeError('population overlap')
    old=ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix'
    files=[CONFIG,Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('common.py'),old/'train.py',old/'data.py',ROOT/'experiments/var-latent-enhancement-20260917/research/src/latent_research/models.py',EXP/'quality_targets.json']
    registration={'bindings':identity_files(files),'N':a.N,'initialization_seed':SEED,'parent':None,'decoder_sha256':state_sha256(decoder),'train_files':train.bindings,'calibration_files':cal.bindings,'train_images':train.image_bindings,'calibration_images':cal.image_bindings,'protocol':cfg,'order_seed':2026092301,'channel_seed':2026092302,'noise_namespace':f'VAR-CONTINUOUS-{a.N}','parameter_count':sum(p.numel() for p in models.parameters()),'shared_P3060':a.N==3060}
    register(out/'registration.json',registration);regsha=digest(out/'registration.json')
    opts={n:torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for n,m in models.items()}
    order=PairedOrder(len(train),registration['order_seed']);rng=torch.Generator().manual_seed(registration['channel_seed'])
    state={'step':0,'updates':{n:0 for n in models},'selection':{n:None for n in models},'last_full':-1,'training_seconds':0.,'calibration_seconds':0.}
    if (out/'latest.json').exists():
        payload=torch.load(checked_checkpoint(read(out/'latest.json'),ROOT),map_location='cpu',weights_only=True)
        if payload['registration_sha256']!=regsha:raise RuntimeError('resume identity')
        models.load_state_dict(payload['models'])
        for n in models:opts[n].load_state_dict(copy.deepcopy(payload['optimizers'][n]))
        state=payload['state'];order.load_state_dict(payload['order']);rng.set_state(payload['rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    def checkpoint(reason):
        # Immutable named events prevent changing an already-selected checkpoint
        # when calibration updates bookkeeping at the same training step.
        path=out/'checkpoints'/f"step_{state['step']:05d}_{reason}.pt"
        save_torch(path,{'models':models.state_dict(),'optimizers':{n:o.state_dict() for n,o in opts.items()},'order':order.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,'registration_sha256':regsha})
        sha=digest(path);write_json(out/'latest.json',{'path':str(path),'sha256':sha,'step':state['step'],'reason':reason})
        for n,rec in state['selection'].items():
            if rec and rec['step']==state['step']:write_json(out/f'selected_{n}.json',{**rec,'checkpoint':str(path),'checkpoint_sha256':sha,'arm_key':n,'registration_sha256':regsha,'total_updates':state['updates'][n],'parent_updates':0,'N':a.N})
    safety=Safety();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def full():
        checkpoint('before_calibration');started=time.time();summary=calibrate(models,cal,decoder,lp,scale,device,out,state['step'],safety);state['calibration_seconds']+=time.time()-started
        for n,u in summary.items():
            if state['selection'][n] is None or u<state['selection'][n]['utility']:state['selection'][n]={'step':state['step'],'utility':u}
        state['last_full']=state['step'];checkpoint('calibrated')
    try:
        if state['step']%2500==0 and state['last_full']!=state['step']:full()
        while state['step']<cfg['updates']:
            ids=order.next(cfg['batch_size']);si=torch.randint(5,(len(ids),),generator=rng);ni=torch.randint(len(train.seeds),(len(ids),),generator=rng)
            b=train.batch(ids,si,ni,[registration['channel_seed']+state['step']]*len(ids),device);started=time.time();update(models,opts,b,decoder,lp,scale,cfg['microbatch_size'])
            state['training_seconds']+=time.time()-started;state['step']+=1
            for n in models:state['updates'][n]+=1
            if stop[0] or (state['step']%10==0 and safety.check()):raise ResourceBusy('requested or thermal pause')
            if state['step']%100==0:write_json(out/'status.json',{'status':'TRAINING','pid':os.getpid(),'state':state,'hardware':safety.last});print('P',a.N,state['step'],flush=True)
            if state['step']%2500==0:full()
            elif state['step']%500==0:checkpoint('regular')
        verify_snapshot(registration['bindings']);verify_snapshot(train.bindings);verify_snapshot(cal.bindings)
        if state_sha256(decoder)!=registration['decoder_sha256']:raise RuntimeError('frozen Dc changed')
        write_json(out/'completion.json',{'status':'REGISTERED_20K_MILESTONE_NOT_CONVERGENCE','state':state,'selected':{n:read(out/f'selected_{n}.json') for n in models},'registration_sha256':regsha,'pending':'calibration decision, selected development and shared online timing; no new holdout','synthetic':False})
    except ResourceBusy as e:
        checkpoint('safe_pause');write_json(out/'status.json',{'status':'PAUSED_SAFE_CHECKPOINT','reason':str(e),'state':state});raise SystemExit(75)
if __name__=='__main__':main()
