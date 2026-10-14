"""Admit exact fixed gray pairs, score only33 missing pairs, never render."""
from common import *
import sys,numpy as np,subprocess,tempfile,platform
def register():
 assert not O.exists() and not T.exists();O.mkdir();T.mkdir();(T/'runtime_cache').mkdir();(T/'runtime_cache/tmp').mkdir()
 src=read(O1/'assets/pilot50.json')['records'];byid={s['source_id']:s for s in src}
 c=read(L/'r1_cache_recovery_v1/score/completion.json');scored=checked(c['scored_rows']);rows=read(scored);gray={}
 for r in rows:
  if r['source_id'] in byid and r['actual_receiver_outcome']['actual_received_kind']=='gray':
   s=byid[r['source_id']];assert r['execution_status']=='COMPLETE' and not r['source_truth_supplied_to_RX'] and r['source_preprocessing_id']==s['preprocessing_id']
   if r['source_id'] in gray:assert gray[r['source_id']]['dinov2_vitl14_cosine']==r['dinov2_vitl14_cosine'];continue
   with np.load(checked(r['image_archive']),allow_pickle=False) as z:
    assert 'image' in z;im=z['image'];assert im.shape==(3,256,256) and im.dtype==np.float32 and np.all(im==np.float32(.5)) and asha(im)==r['reconstruction_sha256']
   gray[r['source_id']]=r
 assert len(gray)==17
 save(O/'gray_reuse.json',dict(records=list(gray.values()),old_scored_rows=pin(scored),old_metric_identity=c['metric_identity'],exact_pair_reuse=17))
 reg=dict(authorized_stages=['LE2_GRAY_ONLY_PREPARATION'],packet_trials=0,wireless_frames=0,source_embedding_cap=33,gray_pair_cap=33,VAR_calls=0,gray_array_sha256=asha(np.full((3,256,256),.5,np.float32)),pilot50=pin(O1/'assets/pilot50.json'),model_routes=pin(O1/'assets/model_routes.json'),reuse=pin(O/'gray_reuse.json'),source_bindings=[pin(E/n) for n in ('common.py','gray_prepare.py')],frozen_metric_source_sha256='dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066',GPU_index=1,resource_min_start_MiB=8192,resource_free_floor_MiB=2048,allocator_GiB=8,math_threads=6,interop=2)
 save(O/'gray_registration.json',reg);print(canonical(pin(O/'gray_registration.json')))
def run(expected):
 reg=read(O/'gray_registration.json');assert sha(O/'gray_registration.json')==expected
 for p in reg['source_bindings']:checked(p)
 out=T/'gray';out.mkdir();save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv,registration=pin(O/'gray_registration.json')))
 ledger=Ledger(out/'calls.sqlite');os.sched_setaffinity(0,{0,1});tempfile.tempdir=str(T/'runtime_cache/tmp')
 def resource(start=False):
  v=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uid,pci,free,util=[s.strip() for s in v.split(',')]
  assert uid=='GPU-11d6ec92-f5d3-e056-0fdf-d3ea2bee02ca' and int(free)>=(8192 if start else 2048) and (not start or int(util)<=50)
  if start:save(out/'resource_admission.json',dict(actual=v))
 resource(True)
 import torch
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((8<<30)/torch.cuda.get_device_properties(0).total_memory)
 routes=read(checked(reg['model_routes']));observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads());assert observed==routes['native_qualification']['numeric_settings']['runtime']
 sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'));import h800_ep_pilot_metrics_v1 as m
 assets=m.bound_assets(routes['metric_config']);module=m.load_module(routes['metric_config']['sources']['metric_models.py'],'_LE2_exact_DINO_gray');assert sha(module.__file__)==reg['frozen_metric_source_sha256']
 with module.no_network():evaluator=ledger.call('dino_load','once',1,lambda:module.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
 save(out/'metric_identity.json',dict(runtime=observed,flags=routes['numeric_flags'],identity=evaluator.identity(),metadata=evaluator.metadata,no_VAR_VAE_Dc=True))
 reused={r['source_id']:r for r in read(checked(reg['reuse']))['records']};rows=[];gray=np.full((3,256,256),.5,np.float32);assert asha(gray)==reg['gray_array_sha256']
 for s in read(checked(reg['pilot50']))['records']:
  resource();sid=s['source_id']
  with np.load(checked(s['archive']),allow_pickle=False) as z:assert asha(z['pixels'])==s['preprocessing_id'];target=z['pixels'].astype(np.float32)/255.
  if sid in reused:dino=float(reused[sid]['dinov2_vitl14_cosine']);origin='EXACT_OLD_R1_GRAY_PAIR'
  else:
   ref=torch.from_numpy(target[None]).to('cuda:0');im=torch.from_numpy(gray[None]).to('cuda:0')
   prepared=ledger.call('source_embedding',sid,33,lambda:evaluator.prepare_reference(ref))
   dino=float(ledger.call('gray_pair',sid,33,lambda:evaluator.score(reference=ref,reconstruction=im,labels=[s['evaluation_class_index']],label_conditioned=False,prepared=prepared))[0]['dinov2_vitl14_cosine']);origin='NEW_MISSING_FIXED_GRAY_PAIR_NO_RENDER'
  rows.append(dict(pilot_slot=s['pilot_slot'],source_id=sid,dinov2_vitl14_cosine=dino,gray_array_sha256=reg['gray_array_sha256'],source_preprocessing_id=s['preprocessing_id'],origin=origin))
 csvsave(O/'gray_quality.csv',rows);assert all(r['status']=='COMPLETE' for r in ledger.summary());save(out/'completion.json',dict(status='COMPLETE_FIXED_GRAY50_REUSE17_NEW33',counts=ledger.summary(),output=pin(O/'gray_quality.csv'),no_reconstruction=True,packet_trials=0));print('GRAY_COMPLETE_REUSE17_NEW33',flush=True)
if __name__=='__main__':
 try:register() if sys.argv[1]=='register' else run(sys.argv[2])
 except BaseException as e:
  if (T/'gray').exists() and not (T/'gray/failure.json').exists():save(T/'gray/failure.json',dict(error=repr(e),no_auto_retry=True))
  raise
