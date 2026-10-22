"""Matched decoder scale-up on immutable19,968-source caches and validation200."""
from rdscale_common import *
from rdscale_policy import objective
import argparse,copy,csv,signal,sys,traceback,random

def main(regsha,job_name):
    assert sha(O/'registration.json')==regsha
    reg=read(O/'registration.json');job=reg['jobs'][job_name];arm=job['arm'];seed=job['seed']
    for d in reg['sources']:checked(d)
    checked(reg['seed_closure']);checked(reg['seed_index'])
    out=L/'training_v1'/job_name;out.mkdir(parents=True,exist_ok=False)
    for k in ['tmp','inductor','triton']:(out/'runtime_cache'/k).mkdir(parents=True)
    os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
    import tempfile;tempfile.tempdir=os.environ['TMPDIR']
    import numpy as np,torch
    from decoder_model import ProfileDecoder
    from rdscale_runtime import load_lpips
    from pytorch_msssim import ms_ssim
    from PIL import Image
    from pathlib import Path
    started=time.perf_counter();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(6);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0)
    assert torch.cuda.device_count()==1 and sorted(os.sched_getaffinity(0))==job['resource']['cpu_affinity']
    pci_ns=dict(ctypes=__import__('ctypes'));definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py',{'cuda_driver_pci_bus_id'},pci_ns)
    actual_pci=pci_ns['cuda_driver_pci_bus_id']();assert actual_pci.lower().lstrip('0')==job['resource']['pci_bus_id'].lower().lstrip('0')
    torch.cuda.set_per_process_memory_fraction((48<<30)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    ledger=Ledger(out/'ledger.jsonl')
    def boundary():
        if stop[0]:raise InterruptedError('Owned decoder stop requested')
        if time.time()>=reg['absolute_deadline_unix'] or time.perf_counter()-started>=reg['training_walltime_seconds']:raise TimeoutError('Registered bounded decoder window')
        if torch.cuda.mem_get_info()[0] < 2<<30:raise RuntimeError('Device free below 2 GiB')
    routes=read(checked(reg['model_routes']));paths=dict(var_source=routes['author_root'])
    import platform
    actual_runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    assert actual_runtime==routes['native_qualification']['numeric_settings']['runtime']
    for d in routes['author_files']+[routes['dist']]:checked(d)
    for d in routes['files']:
        p=checked(d);paths['vae_checkpoint' if p.name=='vae_ch160v4096z32.pth'else'var_checkpoint' if p.name=='var_d16.pth'else'decoder_checkpoint']=str(p)
    ns=dict(torch=torch,Path=Path,sys=sys,hashlib=hashlib,PATCH_NUMS=SIZES,VOCAB_SIZE=4096)
    definitions(R/'src/var_comm/next_scale_prior.py',{'load_models','state_sha256'},ns)
    dn=dict(torch=torch,nn=torch.nn,copy=copy);definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},dn)
    def load():
        vae,var=ns['load_models'](paths,torch.device('cuda:0'))
        dc=dn['ContinuousDecoder'](vae).cuda();cp=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True);dc.load_state_dict(cp['decoder'],strict=True)
        states={k:ns['state_sha256'](m)for k,m in [('vae',vae),('var',var),('decoder',dc)]}
        assert states==reg['model_states']
        del vae,var,cp;torch.cuda.empty_cache()
        return dc,states
    dc,states=ledger.call('model_load','current_dc_initialization',load)
    teacher=copy.deepcopy(dc).cuda().eval().requires_grad_(False) if arm=='multiprefix_full_anchor' else None
    model=ProfileDecoder(dc,False).cuda().train()
    teacher_sha=ns['state_sha256'](teacher) if teacher is not None else None
    if teacher is not None:assert teacher_sha==states['decoder']
    lp=ledger.call('metric_load','lpips',lambda:load_lpips(reg))
    initial_sha=ns['state_sha256'](model)
    save(out/'identity.json',dict(initial_model_sha256=initial_sha,dc_initial_states=states,model_parameters=sum(p.numel()for p in model.parameters()),affine_parameters=sum(p.numel()for p in model.profile_affine.parameters()),conditional=model.conditional,precision='FP32',TF32=False,device_uuid=str(torch.cuda.get_device_properties(0).uuid),device_name=torch.cuda.get_device_name(0),threads=6,interop=2,affinity=sorted(os.sched_getaffinity(0)),seed=seed,runtime=actual_runtime,registered_resource=job['resource']))
    training=read(checked(reg['train_index']));vc=read(L/'validation_delivery_v1/closure.json');assert vc['status']=='PASS' and vc['registration_sha256']==regsha
    validation=read(checked(vc['index']));sources=training['sources']+validation['sources'];data={}
    save(out/'data_binding.json',dict(train_index=reg['train_index'],validation_closure=pin(L/'validation_delivery_v1/closure.json'),validation_index=vc['index']))
    for split in ['train','validation']:
        subset=[x for x in sources if x['split']==split];targets=[None]*len(subset);legacy={k:[None]*len(subset)for k in ['F','Fq','Fb_TX']};latents=[]
        groups={}
        for i,row in enumerate(subset):groups.setdefault(row['rgb_shard']['path'],[]).append((i,row))
        for group in groups.values():
            boundary();first=group[0][1];rgb=torch.load(checked(first['rgb_shard']),map_location='cpu',weights_only=True)
            old=torch.load(checked(first['old_latent_shard']),map_location='cpu',weights_only=True) if split=='train' else None
            for i,row in group:
                j=row['within_shard'];assert rgb['image_ids'][j]==row['source_id'];targets[i]=rgb['targets_u8'][j].clone()
                assert hashlib.sha256(targets[i].numpy().tobytes()).hexdigest()==row['source_pixels_sha256']
                if old is not None:
                    assert old['image_ids'][j]==row['source_id']
                    for k in legacy:legacy[k][i]=old[k][j].clone()
            del rgb,old
        for row in subset:
            boundary();c=torch.load(checked(row['latents']),map_location='cpu',weights_only=True)
            assert c['source_id']==row['source_id'] and c['endpoints']==[tuple(x)for x in reg['endpoints']]
            assert c['latents'].dtype==torch.float32 and tuple(c['latents'].shape)==(8,32,16,16)
            latents.append(c['latents'])
        data[split]=dict(sources=subset,rgb=torch.stack(targets),latent=torch.stack(latents))
        if split=='train':data[split]['legacy']={k:torch.stack(v)for k,v in legacy.items()}
        boundary()
    train=data['train'];cal=data['validation'];assert len(train['sources'])==19968 and len(cal['sources'])==200
    tn=dict(torch=torch);definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/training.py',{'PairedOrder','draw_mixture','mixed_input'},tn)
    order=tn['PairedOrder'](19968*8,seed);mixture_rng=torch.Generator().manual_seed(seed+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=reg['learning_rate'],weight_decay=reg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0);step=0;samples=0;updates=[];last_save=time.perf_counter();calibration=None
    def checkpoint(reason):
        return write_torch(out/f'step_{step:06d}.pt',dict(model_name='nest_rd_scaleup_'+job_name,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=None,global_step=step,processed_samples=samples,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=dict(name=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),position=np.random.get_state()[2],has_gauss=np.random.get_state()[3],cached_gaussian=np.random.get_state()[4]),python_rng=random.getstate(),data_order=order.state_dict(),mixture_rng=mixture_rng.get_state(),config=reg,registration_sha256=regsha,arm=arm,job=job_name,seed=seed,teacher_sha256=teacher_sha,calibration=calibration,reason=reason),torch)
    initial_checkpoint=checkpoint('common_initial_state');start_training=time.perf_counter()
    try:
        while step<reg['updates']:
            boundary();pair=order.next(reg['batch_size']);ids=pair//8;epids=pair%8
            if arm=='old_recipe_control':
                branches,alpha=tn['draw_mixture'](reg['batch_size'],mixture_rng)
                batch={k:v[ids]for k,v in train['legacy'].items()};z=tn['mixed_input'](batch,branches,alpha)
            else:z=train['latent'][ids,epids]
            target=train['rgb'][ids].float().div(255).cuda();z=z.cuda();epgpu=epids.cuda();nextstep=step+1
            def update():
                torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                pred=model(z,epgpu);mse=(pred-target).square().mean();pl=lp(pred*2-1,target*2-1).mean();full_pred=None;teacher_pred=None
                if teacher is not None:
                    zfull=train['latent'][ids,7].cuda()
                    with torch.no_grad():teacher_pred=ledger.call('anchor_teacher_forward',str(nextstep),lambda:teacher(zfull))
                    full_pred=ledger.call('anchor_full_student_forward',str(nextstep),lambda:model(zfull,torch.full_like(epgpu,7)))
                loss,anchor=objective(mse,pl,full_pred,teacher_pred)
                assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(g);optimizer.step();scheduler.step();torch.cuda.synchronize()
                return dict(step=nextstep,loss=float(loss.detach()),mse=float(mse.detach()),lpips=float(pl.detach()),seconds=time.perf_counter()-t,pair_indices=pair.tolist(),anchor_mse=float(anchor.detach()) if anchor is not None else None,teacher_forwards=1 if teacher is not None else 0,full_student_forwards=1 if teacher is not None else 0)
            result=ledger.call('decoder_update',str(nextstep),update);step=nextstep;samples+=reg['batch_size'];updates.append(result)
            if step%100==0:print(json.dumps(dict(arm=arm,step=step,samples=samples,loss=result['loss'],elapsed=time.perf_counter()-started)),flush=True)
            if time.perf_counter()-last_save>=600 and step<reg['updates']:checkpoint('periodic_update_boundary');last_save=time.perf_counter()
    except (InterruptedError,TimeoutError):
        if not(out/f'step_{step:06d}.pt').exists():checkpoint('safe_boundary_stop')
        save(out/'paused.json',dict(step=step,samples=samples,automatic_resume=False));raise
    training_seconds=time.perf_counter()-start_training
    if teacher is not None:assert ns['state_sha256'](teacher)==teacher_sha and all(p.grad is None for p in teacher.parameters())
    cp=checkpoint('fixed_update_final_no_selection');save(out/'training_updates.json',updates)
    # One final evaluation, no intermediate checkpoint selection or early stopping.
    model.eval();rows=[]
    for i,row in enumerate(cal['sources']):
        target=cal['rgb'][i].float().div(255).cuda()[None]
        for e,(m,k) in enumerate(reg['endpoints']):
            boundary();key=f"{row['source_id']}|m{m}k{k}"
            with torch.no_grad():
                pred=ledger.call('calibration_decoder',key,lambda:model(cal['latent'][i,e][None].cuda(),torch.tensor([e],device='cuda')))
                mse=float((pred.double()-target.double()).square().mean())
                pl=ledger.call('calibration_lpips',key,lambda:float(lp(pred*2-1,target*2-1).item()))
                ms=ledger.call('calibration_ms_ssim',key,lambda:float(ms_ssim(pred,target,data_range=1,size_average=True).item()))
            rows.append(dict(source_id=row['source_id'],source_index=row['source_index'],split='validation',arm=arm,job=job_name,seed=seed,m=m,K=k,terminated_bits=row['endpoint_rows'][e]['terminated_bits'],mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=pl,ms_ssim=ms,utility=mse+.1*pl,image_array_sha256=hashlib.sha256(pred[0].cpu().numpy().tobytes()).hexdigest(),checkpoint_sha256=cp['sha256']))
            if i<2:
                Image.fromarray(np.rint(np.clip(pred[0].cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)).save(out/f'preview_i{i}_m{m}_k{k}.png')
        print(json.dumps(dict(arm=arm,validation_sources=i+1,target=200)),flush=True)
    with(out/'validation.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for d in reg['sources']:checked(d)
    save(out/'completion.json',dict(status='DECODER_SCALEUP_COMPLETE',arm=arm,job=job_name,seed=seed,registration_sha256=regsha,training_updates=step,training_samples=samples,validation_sources=200,validation_endpoint_rows=1600,teacher_training_forwards=step if teacher is not None else 0,full_student_training_forwards=step if teacher is not None else 0,anchor_training_samples=samples if teacher is not None else 0,teacher_sha256=teacher_sha,training_seconds=training_seconds,total_seconds=time.perf_counter()-started,checkpoint=cp,initial_checkpoint=initial_checkpoint,validation=pin(out/'validation.csv'),ledger=pin(out/'ledger.jsonl'),updates=pin(out/'training_updates.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),encoder_updates=0,VAR_updates=0,tokenizer_calls=0,PHY_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration-sha256',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    try:main(a.registration_sha256,a.job)
    except BaseException as e:
        out=L/'training_v1'/a.job;out.mkdir(parents=True,exist_ok=True)
        if not(out/'failure.json').exists():save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),process=process(os.getpid())))
        raise
