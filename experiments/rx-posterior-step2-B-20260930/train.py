"""Single-arm low-SNR continuation of the frozen selected P4084 checkpoint."""
import argparse, copy, json, os, signal, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
os.environ['VAR_COMM_DECODER_GATE']=str(ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json')
sys.path.insert(0,str(ROOT/'experiments/rx-posterior-step1-20260929'))
import run_preflight as environment
import numpy as np
import torch
from latent_enhancement.runtime import (write_json,save_torch,digest,verify_snapshot,
    ResourceBusy,require_available,perceptual_model,model_paths)
from latent_enhancement.training import PairedOrder
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_research.models import PureContinuous
from var_comm.next_scale_prior import load_models,state_sha256
from short_prefix.common import Safety,identity_files,register,configure_runtime
from short_prefix.data import Population
from short_prefix.train import losses,update
from qualification import qualify
from token_efficiency.C_evaluate import load_selected
from protocol import (TRAIN_SNRS,CAL_SEEDS,PARENT_STEP,INTERVAL,MILESTONE,
    extension_decision,select_checkpoint)

OUT=ROOT/'outputs/RX-POSTERIOR-STEP2-B-20260930-R1'
TRAIN=OUT/'training'
PARENT=ROOT/'outputs/SHORT-PREFIX-20260923/training/pure_seed2026092304'
PARENT_SHA='cba6190cabff71d2e6906644025a4769a5c2c46ec492bc8fc8d275fa085778af'


def read(path):return json.loads(Path(path).read_text())


def load_for_evaluation(selected_path,scale,device):
    """Read only a fully selected model after the calibration-only stop decision."""
    selected_path=Path(selected_path);folder=selected_path.parent
    rec=read(selected_path);reg=read(folder/'registration.json');done=read(folder/'completion.json')
    verify_snapshot(reg['bindings'])
    if rec!=done['selected']['P_low'] or rec['arm_key']!='P_low':
        raise RuntimeError('P_low selection/completion mismatch')
    if rec['registration_sha256']!=digest(folder/'registration.json'):
        raise RuntimeError('P_low registration hash mismatch')
    cp=Path(rec['checkpoint'])
    if digest(cp)!=rec['checkpoint_sha256']:raise RuntimeError('P_low checkpoint hash mismatch')
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    if payload['registration_sha256']!=rec['registration_sha256'] or payload['state']['step']!=rec['step']:
        raise RuntimeError('P_low selected checkpoint identity')
    candidate=select_checkpoint(done['state']['history'])
    if candidate['step']!=rec['step'] or candidate['utility']!=rec['utility']:
        raise RuntimeError('P_low was not selected by frozen calibration utility')
    verify_snapshot({candidate['calibration_csv']:candidate['calibration_sha256']})
    model=PureContinuous(scale.cpu())
    model.load_state_dict({k[len('P_low.'):]:v for k,v in payload['models'].items() if k.startswith('P_low.')},strict=True)
    meta=dict(method='P_low',arm='P_low',training_seed=2026092304,N=4084,kind='continuous',
        training=str(folder),selected=rec,selected_file=str(selected_path),
        selected_sha256=digest(selected_path),checkpoint=str(cp),
        decoder_sha256=reg['decoder_state_sha256'],training_completed_step=done['state']['step'],
        completed_added_updates=done['state']['step'],
        parent_selected=reg['parent'],training_snrs_db=TRAIN_SNRS,
        total_updates=PARENT_STEP+rec['step'])
    return model.to(device).eval().requires_grad_(False),meta


def calibration(models,pop,decoder,lp,scale,device,step,safety):
    from latent_mechanisms.requalify_predictor import write_rows
    path=TRAIN/'calibration'/f'full_{step:05d}.csv';receipt=path.with_suffix('.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    model_sha=state_sha256(models['P_low'])
    if receipt.exists():
        rec=read(receipt)
        if digest(path)!=rec['sha256'] or rec['step']!=step or rec['rows']!=12000 or rec['model_state_sha256']!=model_sha:
            raise RuntimeError('completed calibration identity mismatch')
        return dict(step=step,utility=rec['utility'],calibration_csv=str(path),calibration_sha256=rec['sha256'])
    rows=[];started=time.time();model=models['P_low'].eval()
    with torch.no_grad():
        for start in range(0,len(pop),16):
            if safety.check():raise ResourceBusy('sustained thermal condition during full calibration')
            ids=torch.arange(start,min(start+16,len(pop)))
            for si,snr in enumerate(TRAIN_SNRS):
                for ni,seed in enumerate(CAL_SEEDS):
                    batch=pop.batch(ids,torch.full_like(ids,si),torch.full_like(ids,ni),[seed]*len(ids),device)
                    utility,mse,percept,aux,_=losses(model,batch,decoder,lp,scale)
                    for j,index in enumerate(ids):
                        rows.append(dict(method='P_low',source_index=int(index),image_id=pop.ids[int(index)],
                            snr_db=snr,seed=seed,mse=float(mse[j]),lpips_alex=float(percept[j]),
                            normalized_latent=float(aux[j]),utility=float(utility[j])))
            write_json(TRAIN/'status.json',dict(status='FULL_CALIBRATION',pid=os.getpid(),step=step,
                sources=min(start+16,len(pop)),total_sources=len(pop),time=time.time()))
    if len(rows)!=12000 or len({(r['image_id'],r['snr_db'],r['seed']) for r in rows})!=12000:
        raise RuntimeError('full1000x4x3 calibration incomplete')
    utility=float(np.mean([r['utility'] for r in rows]))
    write_rows(path,rows)
    write_json(receipt,dict(step=step,utility=utility,rows=len(rows),sha256=digest(path),
        seconds=time.time()-started,source_count=1000,snrs_db=TRAIN_SNRS,seeds=CAL_SEEDS,
        development_read=False,model_state_sha256=model_sha))
    print('full calibration',step,utility,'seconds',time.time()-started,flush=True)
    return dict(step=step,utility=utility,calibration_csv=str(path),calibration_sha256=digest(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification-only',action='store_true');args=parser.parse_args()
    TRAIN.mkdir(parents=True,exist_ok=True)
    if (TRAIN/'completion.json').exists() and not args.qualification_only:return
    configure_runtime();require_available();device=torch.device('cuda:0')
    cfg=read(HERE/'protocol.json');parent_reg=read(PARENT/'registration.json')
    original=parent_reg['protocol']
    for key,value in dict(learning_rate=2e-4,weight_decay=1e-4,batch_size=16,microbatch_size=4).items():
        if original[key]!=value:raise RuntimeError('original optimizer setting changed '+key)
    desc=dict(method='B2',arm='P4084',training_seed=2026092304,N=4084,training=str(PARENT))
    scale=scale_statistics(device);base,parent_meta=load_selected(desc,scale,device)
    selected=parent_meta['selected']
    if selected['step']!=PARENT_STEP or selected['checkpoint_sha256']!=PARENT_SHA:
        raise RuntimeError('wrong parent selected checkpoint')
    parent=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    vae,var=load_models(model_paths(),device);encoder_sha=state_sha256(vae);del var
    decoder=load_decoder(vae,device);del vae
    if state_sha256(decoder)!=parent_meta['decoder_sha256']:raise RuntimeError('frozen Dc mismatch')
    lp=perceptual_model(device)
    models=torch.nn.ModuleDict({'P_low':base.requires_grad_(True)})
    opts={'P_low':torch.optim.AdamW(base.parameters(),lr=2e-4,weight_decay=1e-4)}
    opts['P_low'].load_state_dict(copy.deepcopy(parent['optimizers']['P4084']))
    groups=opts['P_low'].param_groups
    if any(g['lr']!=2e-4 or g['weight_decay']!=1e-4 or tuple(g['betas'])!=(.9,.999) or g['eps']!=1e-8 for g in groups):
        raise RuntimeError('populated parent optimizer settings differ')
    cal=Population('calibration');cal.snrs=TRAIN_SNRS
    if len(cal)!=1000 or cal.seeds!=CAL_SEEDS:raise RuntimeError('calibration identity/count/seeds')
    ids=torch.arange(4);batch=cal.batch(ids,torch.zeros_like(ids),torch.zeros_like(ids),[4101]*4,device)
    if args.qualification_only:
        qualify(models,opts,cal,decoder,lp,scale,parent,TRAIN);return
    qualified=read(TRAIN/'qualification_resumed_parent.json')
    if qualified['parent_checkpoint_sha256']!=PARENT_SHA or not qualified['passed']:
        raise RuntimeError('resumed-parent GPU qualification missing or mismatched')
    if qualified['training_updates']!=0 or not qualified['actual_training_model_optimizer_modes_rng_unchanged']:
        raise RuntimeError('qualification probe altered formal training')
    verify_snapshot(qualified['source_bindings'])
    train=Population('train');train.snrs=TRAIN_SNRS
    if len(train)!=20000 or set(train.ids)&set(cal.ids):raise RuntimeError('train/calibration source overlap/count')
    verify_snapshot({**train.bindings,**cal.bindings})
    own_files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'EXECUTION_PLAN.md']
    bindings=identity_files([*own_files,
        ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/train.py',
        ROOT/'experiments/var-short-prefix-hybrid-20260923/src/short_prefix/data.py',
        ROOT/'experiments/var-latent-enhancement-20260917/research/src/latent_research/models.py',
        ROOT/'experiments/token_channel_efficiency_20260923/src/token_efficiency/C_evaluate.py',
        PARENT/'registration.json',PARENT/'selected_P4084.json'])
    registration=dict(protocol=cfg,bindings=bindings,parent=parent_meta,
        decoder_state_sha256=state_sha256(decoder),encoder_state_sha256=encoder_sha,
        lpips_state_sha256=state_sha256(lp),cache_train=train.cache_identity,cache_calibration=cal.cache_identity,
        latent_cache_bindings={'train':train.bindings,'calibration':cal.bindings},
        source_image_bindings={'train':train.image_bindings,'calibration':cal.image_bindings},
        train_ids=train.ids,calibration_ids=cal.ids,parameter_count=sum(p.numel() for p in base.parameters()),
        optimizer_initialization='RESUME_FULL_SELECTED27500_MOMENTS_AND_STEP',
        order_rng_initialization='RESUME_SELECTED27500_ORDER_CHANNEL_CPU_CUDA_RNG',development_read=False)
    register(TRAIN/'registration.json',registration);regsha=digest(TRAIN/'registration.json')
    order=PairedOrder(len(train),original['data_seed']);order.load_state_dict(parent['order'])
    rng=torch.Generator();rng.set_state(parent['rng'])
    torch.set_rng_state(parent['torch_rng']);torch.cuda.set_rng_state_all(parent['cuda_rng'])
    state=dict(step=0,parent_step=PARENT_STEP,limit=MILESTONE,last_full=-1,history=[],decisions=[],
        training_seconds=0.,calibration_seconds=0.,finished=False)
    if (TRAIN/'latest.json').exists():
        latest=read(TRAIN/'latest.json')
        if digest(latest['path'])!=latest['sha256']:raise RuntimeError('resume checkpoint SHA')
        payload=torch.load(latest['path'],map_location='cpu',weights_only=True)
        if payload['registration_sha256']!=regsha:raise RuntimeError('resume registration mismatch')
        models.load_state_dict(payload['models']);opts['P_low'].load_state_dict(payload['optimizers']['P_low'])
        order.load_state_dict(payload['order']);rng.set_state(payload['rng']);state=payload['state']
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    del parent
    stop=[False]
    def halt(*_):stop[0]=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    safety=Safety()

    def checkpoint(reason):
        folder=TRAIN/'checkpoints';folder.mkdir(exist_ok=True)
        # Unique receipts preserve interrupted and failed candidate states without overwriting.
        path=folder/f"step_{state['step']:05d}_{reason}_{time.time_ns()}.pt"
        save_torch(path,dict(models=models.state_dict(),optimizers={n:o.state_dict() for n,o in opts.items()},
            order=order.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),state=copy.deepcopy(state),registration_sha256=regsha))
        rec=dict(path=str(path),sha256=digest(path),step=state['step'],reason=reason)
        write_json(TRAIN/'latest.json',rec);return rec

    def full():
        before=checkpoint('before_full_calibration');started=time.time()
        row=calibration(models,cal,decoder,lp,scale,device,state['step'],safety)
        state['calibration_seconds']+=time.time()-started
        row.update(checkpoint=before['path'],checkpoint_sha256=before['sha256'])
        if any(r['step']==state['step'] for r in state['history']):raise RuntimeError('duplicate full calibration')
        state['history'].append(row);state['last_full']=state['step']
        if state['step'] in (10000,20000,30000):
            decision=extension_decision(state['history'],state['step']);state['decisions'].append(decision)
            write_json(TRAIN/f"extension_{state['step']:05d}.json",decision)
            state['limit']=decision['next_limit'];state['finished']=not decision['extend']
        checkpoint('full_calibration')
        write_json(TRAIN/'calibration_history.json',state['history'])

    try:
        if state['last_full']!=state['step'] and state['step']%INTERVAL==0:full()
        while not state['finished']:
            if stop[0]:raise ResourceBusy('requested safe pause before next update')
            ids=order.next(16);si=torch.randint(4,(len(ids),),generator=rng)
            ni=torch.randint(len(train.seeds),(len(ids),),generator=rng)
            noise_seed=2026092302+PARENT_STEP+state['step']
            batch=train.batch(ids,si,ni,[noise_seed]*len(ids),device)
            started=time.time();update(models,opts,batch,decoder,lp,scale,4)
            state['training_seconds']+=time.time()-started;state['step']+=1
            if stop[0] or (state['step']%10==0 and safety.check()):raise ResourceBusy('requested pause or sustained thermal condition')
            if state['step']%100==0:
                write_json(TRAIN/'status.json',dict(status='TRAINING',pid=os.getpid(),state=state,hardware=safety.last,time=time.time()))
                print('P_low added updates',state['step'],'limit',state['limit'],'training seconds',state['training_seconds'],flush=True)
            if state['step']%INTERVAL==0:full()
            elif state['step']%500==0:checkpoint('regular')
        verify_snapshot(bindings)
        if state_sha256(decoder)!=registration['decoder_state_sha256'] or state_sha256(lp)!=registration['lpips_state_sha256']:
            raise RuntimeError('frozen loss/decoder model changed')
        if any(p.grad is not None for p in decoder.parameters()) or any(p.grad is not None for p in lp.parameters()):
            raise RuntimeError('frozen model received a gradient')
        best=select_checkpoint(state['history'])
        selected=dict(step=best['step'],utility=best['utility'],checkpoint=best['checkpoint'],
            checkpoint_sha256=best['checkpoint_sha256'],arm_key='P_low',registration_sha256=regsha,
            total_updates=PARENT_STEP+best['step'],parent_updates=PARENT_STEP,
            base_parent_selected=parent_meta['selected'],calibration_csv=best['calibration_csv'],
            calibration_sha256=best['calibration_sha256'])
        register(TRAIN/'selected_P_low.json',selected)
        write_json(TRAIN/'completion.json',dict(status='CALIBRATION_ONLY_SELECTION_COMPLETE',state=state,
            selected={'P_low':selected},registration_sha256=regsha,development_read=False,synthetic=False))
        write_json(TRAIN/'status.json',dict(status='TRAINING_COMPLETE',state=state,time=time.time()))
    except ResourceBusy as exc:
        checkpoint('safe_pause');write_json(TRAIN/'status.json',dict(status='PAUSED_SAFE_CHECKPOINT',reason=str(exc),state=state,time=time.time()))
        raise SystemExit(75)
    except Exception:
        checkpoint('failure');write_json(TRAIN/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),state=state,time=time.time()))
        raise


if __name__=='__main__':
    try:main()
    except ResourceBusy as exc:
        write_json(TRAIN/'status.json',dict(status='WAITING_FOR_SAFE_RESOURCE_BEFORE_LOAD',reason=str(exc),time=time.time()))
        raise SystemExit(75)
