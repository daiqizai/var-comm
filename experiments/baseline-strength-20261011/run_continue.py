"""Finite P/Swin continuation, original losses, calibration-only selection.

R0 is a separate mode; training requires its successful receipt and verifies all
bindings. No legacy controller is launched. Final holdout evaluation is a later,
explicitly frozen stage after both model selections.
"""
from __future__ import annotations
import argparse, copy, csv, fcntl, json, math, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import baseline_runtime as b
from training_rules import P_SNRS,S_SNRS,learning_rate,stability

def optimizer(branch,model,payload):
    if branch=='P':
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
        opt.load_state_dict(copy.deepcopy(payload['optimizers']['P1024']))
        assert {float(v['step']) for v in opt.state.values()}=={40000.}
    else:opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    return opt

def load(branch):
    ns,pins=b.p_math()
    if branch=='P':
        model,dc,lp,scale,payload=b.p_models(ns)
        return dict(model=model,dc=dc,lp=lp,scale=scale,payload=payload,ns=ns,pins=pins)
    sm,sp,sr,ss=b.swin_modules()
    path=b.CP/'parents/swin/model_080000.pt';assert b.sha(path)==b.S_SHA
    payload=torch.load(path,map_location='cpu',weights_only=True);assert payload['step']==80000
    torch.manual_seed(2026101101);torch.cuda.manual_seed_all(2026101101)
    model=sm.build_official(b.TMP/'vendor/SwinJSCC');model.load_state_dict(payload['model'],strict=True)
    # This records the actual new range even though the author native.forward is
    # bypassed by the unchanged explicit-SNR SwinCodec training_forward.
    if hasattr(model.native,'args'):model.native.args.multiple_snr=','.join(map(str,range(1,20)))
    for p in (b.TMP/'vendor/SwinJSCC').rglob('*.py'):pins[str(p)]=b.sha(p)
    for m in [sm,sp,sr,ss]:pins[m.__file__]=b.sha(m.__file__)
    b.definitions(b.ROOT/'experiments/external-comparison-20261004/swin_train.py',{'train_update'},ns,pins)
    ns['RATES']=sp.RATES
    return dict(model=model,payload=payload,ns=ns,pins=pins,sp=sp,sr=sr,ss=ss)

def resource_check(startup=False):
    gpu=os.environ['BASELINE_GPU_ID']
    row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    free,temp=[int(x.strip()) for x in row]
    if free<(20480 if startup else 4096) or temp>=86:raise InterruptedError('Shared GPU free-memory or thermal boundary')

def p_update(state,ids,snrs,seeds,pop):
    ns=state['ns'];batch=pop.p_batch(ids,snrs,seeds,ns['seeded_noise']);captured=[];original=ns['losses']
    def measured(*a,**kw):
        result=original(*a,**kw);captured.extend(result[0].detach().cpu().tolist());return result
    ns['losses']=measured
    try:ns['update']({'P1024':state['model']},{'P1024':state['opt']},batch,state['dc'],state['lp'],state['scale'],4)
    finally:ns['losses']=original
    return math.fsum(captured)/len(captured)

def qualification(branch,state,out):
    pop=b.Population('calibration',latent=branch=='P',limit=16)
    model=state['model'];ns=state['ns'];initial=copy.deepcopy(model.state_dict());rng=b.global_rng()
    opt=optimizer(branch,model,state['payload']);state['opt']=opt
    opt_initial=copy.deepcopy(opt.state_dict());proof={};ids=list(range(16))
    if branch=='P':
        batch=pop.p_batch(ids,[19]*16,[4101]*16,ns['seeded_noise'])
        with torch.no_grad():
            model.eval();z,w=model(batch['F'],batch['snr'],batch['noise'])
            # Original explicit TX/channel/RX path versus inherited forward.
            ref_w=model.transmit(batch['F']);ref_z=model.receive(ref_w+batch['noise']*torch.pow(10.,-batch['snr']/20)[:,None,None],batch['snr'])
            torch.testing.assert_close(z,ref_z,rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(w,ref_w,rtol=1e-5,atol=2e-6)
            assert w.shape==(16,1024,2)
            torch.testing.assert_close(w.square().sum((1,2)),w.new_full((16,),2048.),rtol=1e-5,atol=.02)
        dc_hash=ns['state_sha256'](state['dc']);lp_hash=ns['state_sha256'](state['lp'])
        def step():return p_update(state,ids,[19]*16,[4101]*16,pop)
        proof.update(wave_shape=list(w.shape),energy_min=float(w.square().sum((1,2)).min()),energy_max=float(w.square().sum((1,2)).max()),decoder_sha256=dc_hash,lpips_sha256=lp_hash)
    else:
        image=pop.batch(ids);sp=state['sp'];sp.configure_phy(b.ROOT);sp._PHY.ROOT=b.TMP/'native_build'
        with torch.no_grad():
            model.eval();iq,power,mask=model.encode_data(image,19,6)
            noises=np.zeros((16,1024,2),np.float64)
            physical,ledgers=state['sr'].physical_batch(model,image,1024,19,noises)
            direct=model.training_forward(image,19,6,torch.zeros_like(iq))
            torch.testing.assert_close(physical,direct,rtol=1e-4,atol=2e-5)
            assert all(l['header_accepted'] and abs(l['actual_energy']-2048)<.02 for l in ledgers)
        noise=torch.randn((16,1664,2),generator=torch.Generator().manual_seed(2026101105)).cuda()
        def step():return ns['train_update'](model,opt,image,2048,19,noise,4)
        proof.update(data_shape=list(iq.shape),total_N=1024,header_N=256,physical_zero_noise_header_pass=True,training_SNRS=list(range(1,20)))
    losses=[]
    losses.append(step());first=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=b.global_rng())
    gradient=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None);assert gradient>0
    checkpoint=b.save(b.TMP/branch/'probe_after_one.pt',first)
    losses.append(step());after=copy.deepcopy(model.state_dict());after_opt=copy.deepcopy(opt.state_dict())
    reloaded=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    model.load_state_dict(reloaded['model'],strict=True);opt.load_state_dict(reloaded['optimizer']);b.restore_rng(reloaded['rng'])
    replay_loss=step();assert ns['same_tree'](after,model.state_dict()) and ns['same_tree'](after_opt,opt.state_dict())
    assert replay_loss==losses[1]
    if branch=='P':
        assert ns['state_sha256'](state['dc'])==dc_hash and ns['state_sha256'](state['lp'])==lp_hash
        assert all(p.grad is None and not p.requires_grad for m in [state['dc'],state['lp']] for p in m.parameters())
    model.load_state_dict(initial,strict=True);opt.load_state_dict(opt_initial);opt.zero_grad(set_to_none=True);b.restore_rng(rng)
    assert ns['same_tree'](initial,model.state_dict()) and ns['same_tree'](opt_initial,opt.state_dict())
    result=dict(status='PASS',branch=branch,calibration_sources=16,holdout_read=False,formal_updates=0,probe_updates=3,
        reason_for_three='two isolated updates plus replay of update two after save/reload; all discarded',losses=losses,
        replay_loss=replay_loss,gradient_abs_sum=gradient,same_environment_resume_exact=True,
        checkpoint_restored_after_probe=True,parent_checkpoint_sha256=b.P_SHA if branch=='P' else b.S_SHA,
        source_bindings=state['pins'],data_bindings=pop.bindings,proof=proof)
    b.write(out/'preflight.json',result);return result

def calibrate(branch,s,pop,step,out,stop):
    model=s['model'];model.eval();rng=b.global_rng();rows=[];start=time.monotonic()
    snrs=P_SNRS if branch=='P' else S_SNRS
    with torch.no_grad():
        try:
            for N in ([1024] if branch=='P' else [1024,2048]):
                for snr in snrs:
                    for seed in ([4101,4102,4103] if branch=='P' else [4101]):
                        for first in range(0,len(pop),16):
                            if stop[0]:raise InterruptedError('Safe pause during calibration')
                            ids=list(range(first,min(first+16,len(pop))))
                            if branch=='P':
                                bat=pop.p_batch(ids,[snr]*len(ids),[seed]*len(ids),s['ns']['seeded_noise'])
                                loss,mse,lp,aux,_=s['ns']['losses'](model,bat,s['dc'],s['lp'],s['scale'])
                                values=zip(ids,loss.cpu().tolist(),mse.cpu().tolist(),lp.cpu().tolist(),aux.cpu().tolist())
                                rows.extend(dict(source_index=i,source_id=pop.ids[i],N=N,snr_db=snr,seed=seed,objective=u,mse=m,lpips=l,latent_mse=a,header_accepted=True) for i,u,m,l,a in values)
                            else:
                                image=pop.batch(ids);noise=np.stack([s['sp'].standard_noise(pop.ids[i],seed,N,snr) for i in ids])
                                recon,ledgers=s['sr'].physical_batch(model,image,N,snr,noise)
                                mses=(recon-image).square().mean((1,2,3)).cpu().tolist()
                                rows.extend(dict(source_index=i,source_id=pop.ids[i],N=N,snr_db=snr,seed=seed,objective=m,mse=m,lpips=None,latent_mse=None,header_accepted=l['header_accepted']) for i,m,l in zip(ids,mses,ledgers))
                        b.write(out/'status.json',dict(status='CALIBRATING',global_step=step,added_step=step-(40000 if branch=='P' else 80000),rows=len(rows),total_rows=15000 if branch=='P' else 12000,updated_unix=time.time(),pid=os.getpid()))
        finally:b.restore_rng(rng);model.train()
    assert len(rows)==(15000 if branch=='P' else 12000)
    assert len({(r['source_id'],r['N'],r['snr_db'],r['seed']) for r in rows})==len(rows)
    cells={}
    for N in ([1024] if branch=='P' else [1024,2048]):
        for snr in snrs:
            values=[r['objective'] for r in rows if r['N']==N and r['snr_db']==snr]
            cells[f'N{N}_snr{snr}']=float(np.mean(np.asarray(values,dtype=np.float64)))
    path=out/'calibration'/f'{step:06d}.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(step=step,objective=float(np.mean(list(cells.values()))),cells=cells,rows=len(rows),source_count=1000,
        noise_count=3 if branch=='P' else 1,header_failures=sum(not r['header_accepted'] for r in rows),
        seconds=time.monotonic()-start,csv=str(path),sha256=b.sha(path),holdout_read=False)
    b.write(path.with_suffix('.json'),result);return result

def train(branch,s,out):
    pre=b.read(out/'preflight.json');assert pre['status']=='PASS' and pre['formal_updates']==0
    admission_path=out/'preflight_admission.json'
    admitted=b.read(admission_path) if admission_path.exists() else None
    if admitted:
        assert admitted['preflight_sha256']==b.sha(out/'preflight.json')
        assert admitted['before_bindings']==pre['source_bindings']
    for p,digest in pre['source_bindings'].items():
        expected=admitted['after_bindings'][p] if admitted else digest
        assert b.sha(p)==expected
    train_pop=b.Population('train',latent=branch=='P');cal=b.Population('calibration',latent=branch=='P')
    assert not set(train_pop.ids)&set(cal.ids)
    model=s['model'];opt=optimizer(branch,model,s['payload']);s['opt']=opt
    parent=40000 if branch=='P' else 80000;interval=2500 if branch=='P' else 5000
    maximum=100000 if branch=='P' else min(parent+120000,240000)
    order=s['ns']['PairedOrder'](20000,2026092301) if branch=='P' else s['ss'].SourceOrder(20000,2026101104)
    channel=torch.Generator().manual_seed(2026092302 if branch=='P' else 2026101105)
    if branch=='P':
        order.load_state_dict(s['payload']['order']);channel.set_state(s['payload']['rng'])
        b.restore_rng(dict(cpu=s['payload']['torch_rng'],cuda=s['payload']['cuda_rng']))
    else:torch.manual_seed(2026101101);torch.cuda.manual_seed_all(2026101101)
    state=dict(step=parent,parent=parent,history=[],decisions=[],training_seconds=0.,last_calibration=None)
    protocol=b.read(Path(__file__).with_name('p_continue_protocol.json' if branch=='P' else 'swin_1to19_continue_protocol.json'))
    registration=dict(protocol=protocol,source_bindings=s['pins'],preflight_sha256=b.sha(out/'preflight.json'),
        input_manifest_sha256=b.sha(b.TMP/'manifest.json'),training_sources=20000,calibration_sources=1000,holdout_read=False,
        preflight_admission_sha256=b.sha(admission_path) if admitted else None)
    if (out/'registration.json').exists():assert b.read(out/'registration.json')==registration
    else:b.write(out/'registration.json',registration)
    regsha=b.sha(out/'registration.json');stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    cpdir=b.CP/branch;cpdir.mkdir(exist_ok=True)
    if (out/'latest.json').exists():
        rec=b.read(out/'latest.json');assert rec['reason']!='failure' and b.sha(rec['path'])==rec['sha256']
        v=torch.load(rec['path'],map_location='cpu',weights_only=True);assert v['registration_sha256']==regsha
        model.load_state_dict(v['model'],strict=True);opt.load_state_dict(v['optimizer']);order.load_state_dict(v['order']);channel.set_state(v['channel']);b.restore_rng(v['global_rng']);state=v['state']
    def checkpoint(reason,milestone=False):
        path=cpdir/(f"full_{state['step']:06d}_{reason}.pt" if branch=='P' or milestone else f"resume_{(state['step']//1000)%2}.pt")
        rec=b.save(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),order=order.state_dict(),channel=channel.get_state(),global_rng=b.global_rng(),state=copy.deepcopy(state),registration_sha256=regsha))
        b.write(out/'latest.json',dict(**rec,step=state['step'],reason=reason));return rec
    try:
        while True:
            if stop[0] or (out/'STOP').exists():raise InterruptedError('Requested safe pause')
            if state['step']%25==0:resource_check()
            step=state['step'];u=step-parent
            if u%interval==0 and state['last_calibration']!=step:
                checkpoint('before_calibration',milestone=u in [0,40000,80000,120000] or step in [60000,80000,100000])
                rec=b.save(cpdir/f'model_{step:06d}.pt',dict(model=model.state_dict(),step=step,registration_sha256=regsha))
                result=calibrate(branch,s,cal,step,out,stop);result.update(checkpoint=rec)
                state['history'].append(result);state['last_calibration']=step;b.write(out/'calibration_history.json',state['history'])
                checkpoint('after_calibration')
                eligible=step in [80000,90000,100000] if branch=='P' else u in [80000,100000,120000]
                if eligible:
                    decision=stability(state['history'],step,5000 if branch=='P' else 10000);decision['step']=step
                    state['decisions'].append(decision);b.write(out/'stopping_checks.json',state['decisions'])
                    if decision['stable'] or step>=maximum:
                        state['stop_status']='PLATEAU_UNDER_REGISTERED_RECIPE' if decision['stable'] else ('BUDGET_CAP_WHILE_IMPROVING' if decision['improving'] else 'BUDGET_CAP_NOT_STABLE')
                        break
            assert state['step']<maximum
            lr=learning_rate(branch,state['step']+1,parent,1e-4)
            for group in opt.param_groups:group['lr']=lr
            ids=order.next(16);started=time.monotonic()
            if branch=='P':
                si=torch.randint(5,(16,),generator=channel);torch.randint(2,(16,),generator=channel)
                loss=p_update(s,ids,torch.tensor(P_SNRS)[si],[2026092302+state['step']]*16,train_pop)
                N=1024;snr_value=None
            else:
                N=[1024,2048][state['step']%2];snr_value=1+int(torch.randint(19,(1,),generator=channel))
                noise=torch.randn((16,s['sp'].layout(N)['data_N'],2),generator=channel).cuda()
                loss=s['ns']['train_update'](model,opt,train_pop.batch(ids),N,snr_value,noise,4)
            torch.cuda.synchronize();seconds=time.monotonic()-started;state['step']+=1;state['training_seconds']+=seconds
            row=dict(global_step=state['step'],added_step=state['step']-parent,lr=lr,train_loss=loss,N=N,snr=snr_value,
                     seconds=seconds,seen_sources=16*(state['step']-parent),memory_reserved=torch.cuda.memory_reserved())
            log=b.TMP/branch/'training_history.csv';log.parent.mkdir(exist_ok=True)
            exists=log.exists()
            with log.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if not exists:writer.writeheader()
                writer.writerow(row)
            if state['step']%25==0:
                b.write(out/'status.json',dict(status='TRAINING',pid=os.getpid(),**row,last_full_calibration=state['last_calibration'],updated_unix=time.time()))
                print('TRAIN',branch,state['step'],loss,seconds,flush=True)
            if state['step']%1000==0:checkpoint('regular')
        selected=min(state['history'],key=lambda r:(r['objective'],r['step']))
        state['selected_step']=selected['step'];checkpoint('training_finished',milestone=True)
        b.write(out/'selected_checkpoint.json',dict(**selected,parent_step=parent,completed_step=state['step'],stop_status=state['stop_status'],
            training_snrs=P_SNRS if branch=='P' else list(range(1,20)),no_validation_gain_over_parent=selected['step']==parent,
            selected_model_trained_19db=branch=='P' or selected['step']>parent,registration_sha256=regsha))
        b.write(out/'completion.json',dict(status='TRAINING_AND_CALIBRATION_COMPLETE',state=state,milestone_four_metrics='PENDING',bottleneck_diagnostic='PENDING' if branch=='P' else 'NOT_APPLICABLE',final_evaluation='PENDING',timing='PENDING'))
        b.write(out/'status.json',dict(status='TRAINING_COMPLETE_DIAGNOSTICS_AND_FROZEN_EVALUATION_PENDING',global_step=state['step'],added_step=state['step']-parent))
    except InterruptedError as e:
        checkpoint('safe_pause');b.write(out/'status.json',dict(status='PAUSED_SAFE_CHECKPOINT',step=state['step'],reason=str(e)))
        raise SystemExit(75)
    except BaseException:
        b.write(out/'failure.json',dict(status='STOPPED_ENGINEERING_FAILURE',step=state['step'],traceback=traceback.format_exc(),automatic_retry=False));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['P','Swin']);p.add_argument('--mode',choices=['preflight','train'],required=True);a=p.parse_args()
    out=b.RESULT/('P_continue' if a.branch=='P' else 'Swin_1to19');out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'failure.json').exists(),'Review prior failure before creating a separate attempt'
    assert not (out/'completion.json').exists(),'Already complete; do not duplicate'
    resource_check(startup=True)
    environment=b.configure();b.write(out/f'environment_{a.mode}.json',environment)
    state=load(a.branch)
    for p in Path(__file__).parent.glob('*.py'):state['pins'][str(p)]=b.sha(p)
    for p in Path(__file__).parent.glob('*protocol.json'):state['pins'][str(p)]=b.sha(p)
    if a.mode=='preflight':
        assert not (out/'preflight.json').exists(),'Do not repeat completed preflight'
        qualification(a.branch,state,out)
    else:train(a.branch,state,out)

if __name__=='__main__':main()
