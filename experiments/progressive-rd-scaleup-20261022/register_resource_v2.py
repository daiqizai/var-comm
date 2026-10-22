"""Metadata-only new-round registration; restores no models and starts no jobs."""
from rdscale_common import *
from rdscale_policy import *
import subprocess
from rdscale_resource_v2 import admitted

def main():
    assert not(O/'registration.json').exists() and not(L/'dispatch_v1').exists()
    repository=read(O/'repository_preparation.json')
    assert repository['head']=='25d005d2b3896ec698cea4dfd3f0f19da550165f' and repository['tracked_status']==''
    assert 0<=time.time()-repository['observed_unix']<3600
    for d in repository['sources']:checked(d)
    qualification=read(O/'qualification_resource_v2.json');assert qualification['returncode']==0 and qualification['cuda_initialized']is False
    for d in qualification['sources']:checked(d)
    parent=read(OLD_O/'decoder_pilot_registration.json');base=read(OLD_O/'stage0_registration.json')
    recovery=read(OLD_O/'final_delivery_v1/recovery_manifest.json');closure=read(OLD_O/'cache_expansion_delivery_v1/closure_audit.json');assert closure['status']=='PASS'
    seed=read(checked(recovery['seed_index']));expansion=read(checked(recovery['cache_index']));exreg=read(checked(recovery['registration']))
    assert read(OLD_L/'expansion_dispatch_v1/completion.json')['all_exit0']
    old_owner=read(OLD_L/'expansion_initial_control_v1/launch.json')['process'];cur=process(old_owner['pid'])
    assert cur is None or cur['start_ticks']!=old_owner['start_ticks']
    revision=read(O/'resource_admission_revision_v2.json')
    assert revision['prior_registration_actual_wait']['returncode']==1 and revision['scientific_calls_before_revision']==0
    checked(revision['prior_registration_wait']);checked(revision['train_index'])
    train=read(checked(revision['train_index']))['sources']
    assert len(train)==19968 and len({r['source_id']for r in train})==19968
    population=read(checked(base['parent_population_registration']));manifest=read(checked(base['source_manifest']))
    descs={r['destination']:dict(path=str(B/'baseline_strength_20261011/inputs'/r['destination']),sha256=r['sha256'],bytes=r['bytes'])for r in manifest['files']}
    prior=[x['source_id']for x in seed['sources']if x['split']=='calibration'];assert len(prior)==100
    ids=population['calibration_ids'];assert len(ids)==1000
    selected=choose_validation(ids,prior,200);val=[]
    for i in selected:val.append(dict(split='validation',original_split='calibration',source_id=ids[i],source_index=i,within_shard=i%100,rgb_shard=descs[f'rgb/calibration/shard_{i//100:04d}.pt'],latent_shard=descs[f'latent/calibration/shard_{i//100:04d}.pt']))
    assert set(r['source_id']for r in val).isdisjoint(r['source_id']for r in train)
    assert set(r['source_id']for r in val).isdisjoint(prior)
    for d in {r[k]['path']:r[k]for r in val for k in ['rgb_shard','latent_shard']}.values():checked(d)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines();resources=[]
    for line in gpu:
        f=[x.strip()for x in line.split(',')];i=int(f[0]);assert admitted(int(f[3]),int(f[4]))
        first=i*6 if i<4 else 90+(i-4)*6
        resources.append(dict(index=i,uuid=f[1],pci_bus_id=f[2],cpu_affinity=[first,first+1],startup_free_mib=int(f[3]),startup_utilization_percent=int(f[4])))
    assert len(resources)==8
    jobs={}
    for seed_i,seed_value in enumerate(SEEDS):
        for arm_i,arm in enumerate(ARMS):
            name=job_name(arm,seed_value);jobs[name]=dict(arm=arm,seed=seed_value,resource=resources[seed_i*3+arm_i])
    source_paths={d['path']:d for d in parent['sources']}
    for p in sorted(E.glob('*.py')):source_paths[str(p)]=pin(p)
    for p in [OLD_E/'decoder_model.py',OLD_E/'seed_runtime_v2.py',OLD_E/'seed_worker_v2.py',R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/training.py']:
        source_paths[str(p)]=pin(p)
    for d in source_paths.values():checked(d)
    reg=dict(schema='PROGRESSIVE_RD_DECODER_SCALEUP_V1',parent_commit='25d005d2b3896ec698cea4dfd3f0f19da550165f',repository_preparation=pin(O/'repository_preparation.json'),qualification=pin(O/'qualification_resource_v2.json'),resource_admission_revision=pin(O/'resource_admission_revision_v2.json'),admission=dict(free_mib=60000,max_utilization_percent=80,shared_GPU=True,not_timing_evidence=True),user_authorization='2026-10-22 扩大一下试试; bounded independent scale-up of prior matched decoder pilot',train_sources=19968,train_index=pin(L/'train_index.json'),validation_sources=val,validation_selection='First200 SHA256(progressive-rd-scaleup-validation-v1|source_id) among originalcal1000 excluding priorcal100; originalindices retained',prior_calibration100_ids=prior,endpoints=list(ENDPOINTS),jobs=jobs,resources=resources,sources=list(source_paths.values()),seed_index=parent['seed_index'],seed_closure=parent['seed_closure'],model_routes=parent['model_routes'],model_states=parent['model_states'],lpips_linear=parent['lpips_linear'],alexnet=parent['alexnet'],lpips_state_sha256=parent['lpips_state_sha256'],updates=10000,batch_size=4,learning_rate=1e-5,weight_decay=.01,loss='MSE + 0.1 LPIPS-Alex; anchored arm adds1.0 MSE(newfull,frozenDcfull), no scan',precision='FP32 TF32off originalruntime',training_walltime_seconds=7200,validation_walltime_seconds=1800,absolute_deadline_unix=time.time()+10800,original_round_deadline_unchanged=1792672233.1326199,checkpoint_selection='Final fixed step only; no intermediate evaluation',bootstrap=dict(seed=2026102209,resamples=10000,sources=200,unit='source paired; per-seed and mean-across-two-seeds; source CI does not estimate training-seed distribution'),caps=dict(new_training_updates=60000,main_training_samples=240000,anchor_training_samples=80000,new_validation_TX=200,new_validation_RX=1600,new_validation_VAR_completions=1600,new_validation_baseline_lpips=1600,new_validation_baseline_ms_ssim=1600,trained_validation_decoder=9600,trained_validation_lpips=9600,trained_validation_ms_ssim=9600,tokenizer=0,PHY=0,encoder_updates=0,VAR_updates=0),resume_policy='No automatic retry; true boundary checkpoints are retained but original one-shot workers never replayed',parent_recovery_manifest=pin(OLD_O/'final_delivery_v1/recovery_manifest.json'))
    save(O/'registration.json',reg);save(O/'preparation.json',dict(status='REGISTERED_NOT_LAUNCHED',registration=pin(O/'registration.json'),scientific_calls=0,created_unix=time.time()))
    print(json.dumps(dict(registration=pin(O/'registration.json'),train_index=reg['train_index'],validation_source_indices=selected,deadline=reg['absolute_deadline_unix'])))

if __name__=='__main__':main()
