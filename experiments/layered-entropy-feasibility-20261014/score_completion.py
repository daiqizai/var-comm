"""Score only frozen closed clean outputs; no source/reconstruction models."""
from common import *
import sys,tempfile,subprocess,fcntl,ast,platform,traceback
REG=O/'score_completion_v1/registration.json';OUT=T/'score_completion_v1'
def inputs():
 reg=read(REG);old=read(checked(reg['original_registration']))
 for p in old['source_bindings']:checked(p)
 for p in reg['source_bindings']:checked(p)
 w=read(checked(reg['previous_actual_wait']));assert w['returncode']==1 and gone(w['process']) and gone(w['parent_process'])
 checked(reg['ledger_before_score']);checked(reg['all_clean_images'])
 db=sqlite3.connect('file:'+str(T/'science/calls.sqlite')+'?mode=ro',uri=True);failed=list(db.execute("SELECT kind,key,error,status FROM calls WHERE status!='COMPLETE'"));counts={k:n for k,n in db.execute("SELECT kind,count(*) FROM calls WHERE status='COMPLETE' GROUP BY kind")};db.close()
 assert failed==[('source_embedding','n03290653/ILSVRC2012_val_00046068_n03290653','TypeError("MetricEvaluator.prepare_reference() got an unexpected keyword argument \'label_conditioned\'")','FAILED')]
 assert all(counts[k]==n for k,n in dict(source_tx=50,source_rx=24,clean_render=450,decoder_forward=450,prior_scale=5204,dino_model_load=1).items())
 assert 'reconstruction_dino' not in counts and not list((T/'science/clean_images').glob('quality*.json'))
 return reg,old
def run():
 assert sha(REG)==sys.argv[1];reg,old=inputs();assert time.time()<old['deadline_unix']
 OUT.mkdir();save(OUT/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(REG)))
 lock=(T/'science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,{0,1});tempfile.tempdir=os.environ['TMPDIR']
 ledger=Ledger(T/'science/calls.sqlite',reg['effective_caps']);latest=[0]
 def boundary(startup=False):
  assert time.time()<old['deadline_unix'] and not (T/'STOP').exists() and not (O/'STOP').exists()
  if startup or time.monotonic()-latest[0]>10:
   line=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uuid,pci,free,util=[s.strip() for s in line.split(',')]
   assert uuid==reg['gpu_uuid'] and pci.lower()==reg['gpu_pci'].lower() and int(free)>=(8192 if startup else 2048) and (not startup or int(util)<=50);latest[0]=time.monotonic()
   if startup:save(OUT/'admission.json',dict(actual=line))
 boundary(True)
 import numpy as np,torch
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(0)
 from native import definitions
 ns=dict(ctypes=__import__('ctypes'));definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py',{'cuda_driver_pci_bus_id'},ns);pci=ns['cuda_driver_pci_bus_id']();assert pci.lower().lstrip('0')==reg['gpu_pci'].lower().lstrip('0') and torch.cuda.device_count()==1
 props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction((8<<30)/props.total_memory)
 routes=read(checked(old['model_routes']));observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads());assert observed==routes['native_qualification']['numeric_settings']['runtime']
 sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'));import h800_ep_pilot_metrics_v1 as metric
 assets=metric.bound_assets(routes['metric_config']);original=metric.load_module(routes['metric_config']['sources']['metric_models.py'],'_LE01_final_original_dino')
 with original.no_network():evaluator=ledger.call('score_dino_model_reload','once',lambda:original.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
 assert set(evaluator.models)=={'dinov2_vitl14'}
 save(OUT/'actual_metric_identity.json',dict(runtime=observed,numeric_flags=routes['numeric_flags'],GPU_name=props.name,CUDA_UUID=str(props.uuid),PCI=pci,visible_GPU=os.environ['CUDA_VISIBLE_DEVICES'],evaluator_metadata=evaluator.metadata,evaluator_identity=evaluator.identity(),input_order='score(reference=source, reconstruction=fixed_output, labels=evaluation_label, label_conditioned=False, prepared=exact_source_features)',prepare_reference_signature='prepare_reference(reference)',no_VAR_VAE_Dc_loaded=True))
 sources=read(checked(old['pilot50']))['records'];cache=read(checked(reg['all_clean_images']));assert len(cache)==1200 and sum('dino' not in c for c in cache)==450
 byid={s['source_id']:s for s in sources};targets={};prepared={};qrows=[]
 for s in sources:
  with np.load(checked(s['archive']),allow_pickle=False) as z:targets[s['source_id']]=z['pixels'].astype(np.float32)/255.
 for rec in sorted(cache,key=lambda x:(byid[x['source_id']]['pilot_slot'],x['m'],x['K'])):
  boundary();sid=rec['source_id'];s=byid[sid]
  with np.load(checked(rec['image_archive']),allow_pickle=False) as z:im=z[rec['image_key']].copy()
  assert asha(im)==rec['image_array_sha256'];mse=float(np.mean((im.astype(np.float64)-targets[sid].astype(np.float64))**2));psnr=float(-10*np.log10(mse))
  if 'dino' in rec:dino=rec['dino']
  else:
   ref=torch.from_numpy(targets[sid][None]).to('cuda:0');reconstruction=torch.from_numpy(im[None]).to('cuda:0')
   if sid not in prepared:prepared[sid]=ledger.call('score_source_embedding',sid,lambda:evaluator.prepare_reference(ref))
   value=ledger.call('reconstruction_dino',f'{s["pilot_slot"]}:{rec["m"]}:{rec["K"]}',lambda:evaluator.score(reference=ref,reconstruction=reconstruction,labels=[s['evaluation_class_index']],label_conditioned=False,prepared=prepared[sid]))[0];dino=float(value['dinov2_vitl14_cosine'])
  row=dict(source_id=sid,pilot_slot=s['pilot_slot'],m=rec['m'],K=rec['K'],tokens=OFFSETS[rec['m']]+rec['K'],psnr_db=psnr,dinov2_vitl14_cosine=dino,origin=rec['origin'],image_array_sha256=rec['image_array_sha256'],image_path=rec['image_archive']['path'],model_identity=digest(MODELS))
  qrows.append(row);save(T/'science/clean_images'/f'quality_{s["pilot_slot"]:02d}_{rec["m"]}_{rec["K"]}.json',row)
  if len(qrows)%120==0:print(f'CLEAN_QUALITY_COMPLETE {len(qrows)}/1200',flush=True)
 csvsave(O/'clean_quality.csv',qrows)
 db=ledger.db;failed=list(db.execute("SELECT kind,key,status,error FROM calls WHERE status!='COMPLETE'"));assert len(failed)==1 and failed[0][0]=='source_embedding' and failed[0][2]=='FAILED';assert db.execute("SELECT count(*) FROM calls WHERE status='RESERVED'").fetchone()[0]==0
 save(OUT/'completion.json',dict(status='COMPLETE_LE1_SOURCE50_RX24_CLEAN1200_NO_WIRELESS',original_registration=pin(O/'registration.json'),score_registration=pin(REG),TX_sources=50,source_roundtrips=24,clean_quality_rows=1200,reused_clean=750,new_clean=450,source_embedding_actual_successes=50,reconstruction_dino_actual_successes=450,historical_failed_source_embedding_argument_binding_attempts=1,historical_failure_had_zero_embedding_forward=True,historical_failures_preserved=True,active_reserved_calls=0,counts=ledger.summary(),wireless_frames=0,training_updates=0,outputs=[pin(O/n) for n in ('source_lengths.csv','clean_quality.csv','source_roundtrip_report.json')]))
 print('LE1_ALL_REQUIRED_SOURCE_AND_CLEAN_SCORES_COMPLETE',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  if OUT.exists() and not (OUT/'failure.json').exists():save(OUT/'failure.json',dict(status='BLOCKED_PRESERVE_ACTUAL_PREFIX_NO_RETRY',traceback=traceback.format_exc()))
  raise
