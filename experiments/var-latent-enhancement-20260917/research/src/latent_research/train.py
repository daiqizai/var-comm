"""Finite, checkpointed training of the pure control and matched light-TX pair."""
import argparse,copy,json,signal,time
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import configure,require_available,foreign_gpu_processes,digest,write_json,save_torch,model_paths,settings,image_losses,perceptual_model,verify_snapshot
from latent_enhancement.training import PairedOrder
from latent_enhancement_b.common import bind_files,load_decoder,scale_statistics,decoder_gate_path,OUT_B,CACHE
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms,render_received
from latent_followup.run_identity import checked_checkpoint
from var_comm.next_scale_prior import load_models,state_sha256
from var_comm.study import seeded_noise
from .models import PureContinuous,light_warm_start,prefix_latent,forward_arm

EXP=Path(__file__).resolve().parents[3];PROJECT=EXP.parents[1];ROOT=PROJECT/'outputs/VAR-LATENT-ENHANCEMENT-20260917';CONFIG=EXP/'research/config.json'
NAMES=('pure_continuous','full_tx_control','light_tx')

def read(p):return json.loads(Path(p).read_text())
def make_models(scale,recipe,parent,device):
    arms=build_arms((32,16,16),scale,recipe);arms.load_state_dict(parent['arms'],strict=True);base=arms['enhancement1024']
    torch.manual_seed(read(CONFIG)['initialization_seed']);return torch.nn.ModuleDict({'pure_continuous':PureContinuous(scale.cpu()),'full_tx_control':copy.deepcopy(base),'light_tx':light_warm_start(base)}).to(device)

def prefix_cache(pop,vae,out,device,role):
    folder=out/'prefix_cache'/role;folder.mkdir(parents=True,exist_ok=True);parts=[]
    for start in range(0,len(pop),256):
        path=folder/f'{start:05d}.pt';receipt=path.with_suffix('.json');ids=pop.identifiers[start:start+256];F=pop.source.values['F'][start:start+256]
        identity={'image_ids':ids,'input_F_sha256':__import__('hashlib').sha256(F.numpy().tobytes()).hexdigest(),'mapping':'official10scales, accumulate first8','vae_sha256':model_paths_cached['vae_checkpoint_sha256']}
        if receipt.exists():
            r=read(receipt)
            if r['identity']!=identity or digest(path)!=r['sha256']:raise RuntimeError('prefix cache identity changed')
            value=torch.load(path,map_location='cpu',weights_only=True)
        else:
            values=[]
            with torch.no_grad():
                for offset in range(0,len(F),16):
                    f=F[offset:offset+16].to(device);scales=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=False);partial=prefix_latent(vae,scales)
                    if start==0 and offset==0:
                        expected=vae.quantize.f_to_idxBl_or_fhat(f,to_fhat=True)[7]
                        torch.testing.assert_close(partial,expected,atol=2e-5,rtol=1e-5)
                    values.append(partial.cpu())
            value=torch.cat(values);save_torch(path,value);write_json(receipt,{'identity':identity,'sha256':digest(path)})
        parts.append(value)
    return torch.cat(parts)

def complete_batch(pop,indices,si,ni,seeds,prefix,device):
    b=pop.batch(indices,si,ni,seeds,device);b['F_prefix']=prefix[indices].to(device)
    noises=np.stack([seeded_noise('VAR-CONTINUOUS-4084|'+str(pop.identifiers[int(i)]),int(s),(4084,2)) for i,s in zip(indices,seeds)])
    b['pure_noise']=torch.as_tensor(noises,device=device,dtype=torch.float32);return b

def losses(name,model,b,decoder,perceptual,scale):
    z,w=forward_arm(name,model,b)
    pred=decoder(z) if name=='pure_continuous' else render_received(decoder,z,b['rx_status'])
    mse,lp=image_losses(pred,b['target'],perceptual);aux=((z-b['F'])/scale[None,:,None,None]).square().flatten(1).mean(1)
    if name!='pure_continuous':aux=aux*(b['rx_status'][:,0]>.5)
    return mse+.1*lp+.01*aux,mse,lp,aux,w

@torch.no_grad()
def calibrate(models,names,pop,prefix,decoder,perceptual,scale,device,out,step):
    started=time.time();rows=[]
    for start in range(0,len(pop),16):
        ids=torch.arange(start,min(start+16,len(pop)))
        for si in range(len(pop.snrs)):
            for ni,seed in enumerate(pop.seeds):
                b=complete_batch(pop,ids,torch.full_like(ids,si),torch.full_like(ids,ni),[seed]*len(ids),prefix,device)
                for name in names:
                    loss,mse,lp,aux,_=losses(name,models[name].eval(),b,decoder,perceptual,scale)
                    for j,i in enumerate(ids):rows.append({'method':name,'source_index':int(i),'snr_db':float(pop.snrs[si]),'seed':int(seed),'mse':float(mse[j]),'lpips_alex':float(lp[j]),'normalized_latent':float(aux[j]),'utility':float(loss[j])})
    from latent_mechanisms.requalify_predictor import write_rows
    (out/'calibration').mkdir(exist_ok=True);path=out/'calibration'/f'full_{step:05d}.csv';write_rows(path,rows)
    summary={n:float(np.mean([r['utility'] for r in rows if r['method']==n])) for n in names}
    write_json(path.with_suffix('.json'),{'step':step,'summary':summary,'rows':len(rows),'sha256':digest(path),'seconds':time.time()-started})
    print('calibration',step,summary,flush=True);return summary

def main():
    global model_paths_cached
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--qualification-only',action='store_true');parser.add_argument('--prefix-cache-from');a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;require_available();cfg=read(CONFIG);device=torch.device('cuda:0');model_paths_cached=model_paths();vae,var=load_models(model_paths_cached,device);del var
    scale=scale_statistics(device);decoder=load_decoder(vae,device);perceptual=perceptual_model(device);recipe=settings()['stage_B'];record=read(ROOT/'stage_B_v1/training/selected_enhancement1024.json');parent_path=checked_checkpoint(record,PROJECT);parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    train=MatchedPopulation('train');cal=MatchedPopulation('calibration');assert len(train)==20000 and len(cal)==1000
    cache_root=Path(a.prefix_cache_from) if a.prefix_cache_from else out
    tprefix=prefix_cache(train,vae,cache_root,device,'train');cprefix=prefix_cache(cal,vae,cache_root,device,'calibration')
    sources=[]
    for d in ['src','phase_b/src']:
        sources.extend(sorted((EXP/d).rglob('*.py')))
    sources += [Path(__file__),Path(__file__).with_name('models.py'),CONFIG,EXP/'configs/experiment.json',EXP/'phase_b/config.json',decoder_gate_path(),ROOT/'stage_B_v1/training/selected_enhancement1024.json',CACHE/'training_statistics.json',OUT_B/'rx_cache/completion.json']
    sources += sorted((cache_root/'prefix_cache').rglob('*.json'))
    registration={'bindings':bind_files(sources),'parent_checkpoint_sha256':digest(parent_path),'decoder_state_sha256':state_sha256(decoder),'scope':cfg,'prefix_cache_root':str(cache_root),'source_roles':{'train':len(train),'calibration':len(cal)}}
    regpath=out/'registration.json'
    if regpath.exists():
        if read(regpath)!=registration:raise RuntimeError('training registration changed')
        verify_snapshot(registration['bindings'])
    else:write_json(regpath,registration)
    regsha=digest(regpath);models=make_models(scale,recipe,parent,device);del parent
    optimizers={n:torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for n,m in models.items()}
    order=PairedOrder(len(train),cfg['data_seed']);rng=torch.Generator().manual_seed(cfg['channel_seed']);state={'step':0,'selection':{n:None for n in NAMES},'updates':{n:0 for n in NAMES},'seconds':0.,'last_full':-1}
    if (out/'latest.json').exists():
        r=read(out/'latest.json');path=checked_checkpoint(r,PROJECT);saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['registration_sha256']!=regsha:raise RuntimeError('resume identity mismatch')
        models.load_state_dict(saved['models'],strict=True)
        for n in NAMES:optimizers[n].load_state_dict(copy.deepcopy(saved['optimizers'][n]))
        order.load_state_dict(saved['order']);rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);state=saved['state']
    def checkpoint(reason):
        path=out/'checkpoints'/f"step_{state['step']:05d}.pt";save_torch(path,{'models':models.state_dict(),'optimizers':{n:o.state_dict() for n,o in optimizers.items()},'order':order.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,'registration_sha256':regsha})
        write_json(out/'latest.json',{'path':str(path),'sha256':digest(path),'step':state['step'],'reason':reason})
        for n,r in state['selection'].items():
            if r and r['step']==state['step']:write_json(out/f'selected_{n}.json',{**r,'checkpoint':str(path),'checkpoint_sha256':digest(path),'arm_key':n,'registration_sha256':regsha})
    def full(names):
        summary=calibrate(models,names,cal,cprefix,decoder,perceptual,scale,device,out,state['step'])
        for n,u in summary.items():
            old=state['selection'][n]
            if old is None or u<old['utility']:state['selection'][n]={'step':state['step'],'utility':u}
        state['last_full']=state['step'];checkpoint('full_calibration')
    if not (out/'qualification.json').exists():
        ids=torch.arange(4);b=complete_batch(cal,ids,torch.zeros_like(ids),torch.zeros_like(ids),[4101]*4,cprefix,device)
        frozen_sha=state_sha256(decoder);before={n:state_sha256(m) for n,m in models.items()};candidate_before=copy.deepcopy(models['light_tx'].state_dict());opt_before=copy.deepcopy(optimizers['light_tx'].state_dict())
        probe=copy.deepcopy(models['pure_continuous']);loss,mse,lp,aux,w=losses('pure_continuous',probe,b,decoder,perceptual,scale);loss.mean().backward()
        assert sum(float(p.grad.abs().sum()) for p in probe.parameters() if p.grad is not None)>0
        torch.testing.assert_close(w.square().sum((1,2)),torch.full((4,),8168.,device=device),atol=.01,rtol=1e-5)
        assert state_sha256(decoder)==frozen_sha and all(p.grad is None for p in decoder.parameters())
        # A real update on the full-TX arm must leave every light-TX tensor and optimizer state unchanged.
        snapshot=copy.deepcopy(models['full_tx_control'].state_dict());loss,*_=losses('full_tx_control',models['full_tx_control'],b,decoder,perceptual,scale);loss.mean().backward();optimizers['full_tx_control'].step()
        assert all(torch.equal(v,models['light_tx'].state_dict()[k]) for k,v in candidate_before.items());assert optimizers['light_tx'].state_dict()==opt_before
        models['full_tx_control'].load_state_dict(snapshot);optimizers['full_tx_control']=torch.optim.AdamW(models['full_tx_control'].parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);models.zero_grad(set_to_none=True)
        assert before=={n:state_sha256(m) for n,m in models.items()}
        # Exact resumed Adam behavior on actual latent/image/decoder tensors.
        import io
        probe=copy.deepcopy(models['pure_continuous']);probe_opt=torch.optim.AdamW(probe.parameters(),lr=cfg['learning_rate'])
        def update_probe(m,o):
            o.zero_grad(set_to_none=True);loss,*_=losses('pure_continuous',m,b,decoder,perceptual,scale);loss.mean().backward();o.step()
        update_probe(probe,probe_opt);buffer=io.BytesIO();torch.save({'model':probe.state_dict(),'optimizer':probe_opt.state_dict()},buffer);buffer.seek(0);saved=torch.load(buffer,map_location="cpu",weights_only=True)
        restored=copy.deepcopy(probe);restored.load_state_dict(saved['model']);restored_opt=torch.optim.AdamW(restored.parameters(),lr=cfg['learning_rate']);restored_opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        update_probe(probe,probe_opt);update_probe(restored,restored_opt)
        for k,v in probe.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
        for p1,p2 in zip(probe.parameters(),restored.parameters()):
            for key,value in probe_opt.state[p1].items():
                if torch.is_tensor(value):torch.testing.assert_close(value,restored_opt.state[p2][key],rtol=0,atol=0)
        del probe,restored,probe_opt,restored_opt,saved,buffer
        write_json(out/'qualification.json',{'status':'REAL_MODEL_GRADIENT_ENERGY_PREFIX_AND_ISOLATION_PASS','decoder_sha256':frozen_sha,'prefix_exact_official_mapping':True,'pure_waveform_shape':[4,4084,2],'synthetic_model':False,'probe_updates_discarded':True,'real_tensor_Adam_save_resume_bitwise_equal':True})
    if a.qualification_only:return
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if state['step']%cfg['full_calibration_interval']==0 and state['last_full']!=state['step']:full(NAMES if state['step']<=cfg['paired']['updates'] else ('pure_continuous',))
    while state['step']<cfg['pure']['updates']:
        ids=order.next(cfg['batch_size']);si=torch.randint(len(train.snrs),(len(ids),),generator=rng);ni=torch.randint(len(train.seeds),(len(ids),),generator=rng);seeds=[cfg['channel_seed']+state['step']]*len(ids);b=complete_batch(train,ids,si,ni,seeds,tprefix,device);start=time.time()
        names=NAMES if state['step']<cfg['paired']['updates'] else ('pure_continuous',)
        for n in names:
            models[n].train();opt=optimizers[n];opt.zero_grad(set_to_none=True)
            for j in range(0,len(ids),cfg['microbatch_size']):
                part={k:v[j:j+cfg['microbatch_size']] for k,v in b.items()};loss,*_=losses(n,models[n],part,decoder,perceptual,scale);(loss.sum()/len(ids)).backward()
            torch.nn.utils.clip_grad_norm_(models[n].parameters(),1.,error_if_nonfinite=True);opt.step();state['updates'][n]+=1
        state['step']+=1;state['seconds']+=time.time()-start
        if state['step']%100==0:
            write_json(out/'status.json',{'status':'TRAINING','state':state});print('train',state,flush=True)
        if stop[0] or (state['step']%100==0 and foreign_gpu_processes()):checkpoint('safe_pause');write_json(out/'status.json',{'status':'PAUSED_AT_PAIRED_UPDATE','state':state});return
        if state['step']%cfg['full_calibration_interval']==0:full(names)
        elif state['step']%cfg['checkpoint_interval']==0:checkpoint('regular')
    assert state_sha256(decoder)==registration['decoder_state_sha256']
    write_json(out/'completion.json',{'status':'BOUNDED_TRAINING_COMPLETE_NOT_CONVERGENCE_CLAIM','state':state,'selected':{n:read(out/f'selected_{n}.json') for n in NAMES},'registration_sha256':regsha,'new_holdout_used':False})
if __name__=='__main__':main()
