import sys,json,subprocess
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');E=R/'experiments/submission-closure-20261016';sys.path.insert(0,str(E))
from common import *
wait=read(T/'test_m10_control_v1/actual_wait.json');assert wait['returncode']==0 and wait['actual_wait'] and gone(wait['process'])and gone(wait['parent_process']);print('TESTS',Path(wait['stderr']['path']).read_text())
assert not(O/'source_registration.json').exists()and not(O/'metric_registration.json').exists()
oldroot=RT/'qualification/h800_ep_full_source900_v1_attempt1/registered';oldr=read(oldroot/'request.json');done=read(oldroot/'run/worker_completion.json');assert done['request_sha256']==sha(oldroot/'request.json')
oldwait=oldroot.parent/'run_owner_actual_wait.json';assert read(oldwait)['returncode']==0
streams=done['existing100_streams']+done['new900_streams'];assert len(streams)==1000
sources=[]
for src,stream in zip(oldr['records'],streams):
 assert src['source_index']==stream['source_index']and src['source_id']==stream['source_id'];sources.append(dict(src,old_streams=stream['archive'],old_metadata=stream['metadata']))
le=read(R/'results/layered_entropy_feasibility_20261014_v1/registration.json');routes=le['model_routes'];checked(routes)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,pci.bus_id,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);print('GPU',gpu)
gpus={int((x:=line.split(','))[0]):dict(pci=x[1].strip(),uuid=x[2].strip(),free=int(x[3]),util=int(x[4]))for line in gpu.splitlines()}
assert gpus[2]['free']>=15360 and gpus[2]['util']<=25 and gpus[3]['free']>=8192 and gpus[3]['util']<=25
oldfiles=[pin(R/'experiments/entropy-partial-mismatch-20261010/scripts'/n)for n in ['ep_source_codec.py']]+[pin(R/'experiments/wcl-evidence-closure-20261009/scripts'/n)for n in ['t1_codec_runtime.py','t1_entropy_core.py','t1_source_asset_schema.py','t6_source_codec.py']]+[pin(R/p)for p in ['src/var_comm/next_scale_prior.py','experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py','experiments/scale-causal-partial-residual-20261002/partial_receiver.py','experiments/content-real-64qam-20261006/h_source_driver.py']]
reg=dict(schema='SUBMISSION_BATCH1_M10_CAL1000_V1',basis_HEAD='bdb3e3c12d2ea2c18f4371e11cd912d5e8529100',code=[pin(E/n)for n in ['common.py','native.py','m10_codec.py','source_audit.py','control.py']]+oldfiles,old_controls=[pin(oldroot/'request.json'),pin(oldroot/'run/worker_completion.json'),pin(oldwait)],historical_cache_comparison=pin(T/'current_cache_comparison.json'),model_routes=routes,sources=sources,gpu_index=2,gpu_pci=gpus[2]['pci'],gpu_uuid=gpus[2]['uuid'],caps=dict(model_load=1,source_tx=1000,source_rx=8,prior_scale=10080,decoder_forward=0),new_channel=0,new_images=0,automatic_followup=False,synthetic_qualification=pin(T/'test_m10_control_v1/actual_wait.json'))
save(O/'source_registration.json',reg);print('SOURCE_REG',pin(O/'source_registration.json'))
old=R/'results/quality_metrics_supplement_20261014_v1/hifi_fixed16';oreg=read(old/'dependency_recovery_registration.json');raw=read(T/'raw_native_fixed16/manifest.json');pairs=[]
for rawrow in raw['rows']:
 i=rawrow['source_index'];ref=next(s for s in oreg['sources']if s['source_index']==i);assert rawrow['source_id']==ref['source_id']and rawrow['preprocessing_id']==ref['preprocessing_id']
 ar=dict(rawrow['extracted_archive'],path=str(T/'raw_native_fixed16'/rawrow['extracted_archive']['path']));checked(ar)
 pairs.append(dict(method='RAW_PARTIAL',source_index=i,source_id=ref['source_id'],preprocessing_id=ref['preprocessing_id'],reference=ref['archive'],archive=ar,reconstruction_sha256=rawrow['reconstruction_sha256'],noise_seed=6201))
 for method in ['P100000','Swin200000','HiFiSwin200000']:
  row=next(x for x in oreg['original_rows']if x['source_index']==i and x['method']==method);checked(row['archive'])
  pairs.append(dict(method=method,source_index=i,source_id=ref['source_id'],preprocessing_id=ref['preprocessing_id'],reference=ref['archive'],archive=row['archive'],reconstruction_sha256=row['reconstruction_sha256'],noise_seed=2001))
mr=dict(schema='SUBMISSION_FIXED16_RAW_NATIVE_ADDITION_V1',gpu_index=3,gpu_pci=gpus[3]['pci'],gpu_uuid=gpus[3]['uuid'],source_indices=oreg['source_indices'],protocol=oreg['protocol'],pairs=pairs,raw_manifest=pin(T/'raw_native_fixed16/manifest.json'),old_registration=pin(old/'dependency_recovery_registration.json'),old_scores=pin(old/'frame_scores.csv'),metric_assets=oreg['metric_assets'],loader=oreg['loader'],dependency_overlay=oreg['dependency_overlay'],code=[pin(E/n)for n in ['common.py','control.py','fixed16_score.py']],caps=dict(model_load=3,psnr=64,dists=16,dreamsim=16,ms_ssim=16,dreamsim_reference=16),new_scalars_max=112,reuse_old_scalars=144,new_channel=0,new_images=0,new_bootstrap=0,automatic_followup=False)
save(O/'metric_registration.json',mr);print('METRIC_REG',pin(O/'metric_registration.json'))
