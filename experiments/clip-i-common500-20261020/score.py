"""One finite FP32 CLIP image-embedding pass over verified unique native pixels."""
from common import *
import argparse,sqlite3,sys,traceback,tempfile,importlib.util,subprocess

def bind(expected):
 assert sha(O/'registration.json')==expected
 reg=read(O/'registration.json')
 for d in reg['code']+[reg['loader'],reg['inventory']]:checked(d)
 return reg

def main(expected):
 reg=bind(expected);inv=read(checked(reg['inventory']))
 out=T/'score';out.mkdir()
 save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv))
 os.sched_setaffinity(0,set(reg['resource']['cpu_affinity']))
 cache=out/'runtime_cache';(cache/'tmp').mkdir(parents=True)
 tempfile.tempdir=str(cache/'tmp')
 os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
     TRITON_CACHE_DIR=str(cache/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1')
 db=sqlite3.connect(out/'ledger.sqlite')
 db.execute('CREATE TABLE calls(kind TEXT, identity TEXT, status TEXT, PRIMARY KEY(kind,identity))');db.commit()
 def call(kind,identity,fn):
  n=db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]
  assert n<reg['limits'][kind]
  db.execute('INSERT INTO calls VALUES(?,?,?)',(kind,identity,'RESERVED'));db.commit()
  try:value=fn()
  except BaseException:
   db.execute('UPDATE calls SET status=? WHERE kind=? AND identity=?',('FAILED',kind,identity));db.commit();raise
  db.execute('UPDATE calls SET status=? WHERE kind=? AND identity=?',('COMPLETE',kind,identity));db.commit()
  return value
 def gpu(initial=False):
  text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  lines=[v.strip().split(', ')for v in text.strip().splitlines()]
  row=next(x for x in lines if x[1]==reg['resource']['gpu_uuid'])
  assert int(row[0])==reg['resource']['visible_index']
  assert int(row[2])>=reg['resource']['startup_free_mib'if initial else 'runtime_free_mib']
  if initial:assert int(row[3])<=25
  return dict(unix=time.time(),index=int(row[0]),uuid=row[1],free_mib=int(row[2]),utilization=int(row[3]))
 start=time.monotonic();history=[]
 try:
  history.append(gpu(True))
  import torch
  import torch.nn.functional as F
  torch.set_num_threads(6);torch.set_num_interop_threads(2)
  torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
  # Verify visible CUDA device agrees with the UUID/ordinal admitted by NVML.
  import ctypes
  lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int()
  assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
  bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0
  assert bus.value.decode().lower()==reg['resource']['pci_bus_id'].lower()
  torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
  spec=importlib.util.spec_from_file_location('frozen_clip_metric',checked(reg['loader']))
  metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
  model=call('model_load','CLIP_VIT_L14',lambda:metric.load_clip(reg['clip_assets'],torch.device('cuda:0')))
  assert not model.training and all(p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
  identity=dict(model_id=metric.MODEL_IDS['clip'],assets=reg['clip_assets'],loader=reg['loader'],
      preprocessing=reg['metric_definition'],versions=metric.package_versions(),device=torch.cuda.get_device_name(0),
      cuda_version=torch.version.cuda,fp32=True,tf32=False,amp=False,batch_size=1,
      deterministic=True,threads=6,interop_threads=2,cuda_bus=bus.value.decode())
  save(out/'metric_identity.json',identity)
  feature_inputs=read(checked(inv['feature_inputs']))
  assert len(feature_inputs)==reg['limits']['image_embedding']
  (out/'features').mkdir();features={};feature_records=[]
  for n,(ps,locator)in enumerate(sorted(feature_inputs.items()),1):
   assert time.monotonic()-start<reg['max_seconds']
   if n==1 or n%100==0:history.append(gpu())
   image=load_image(locator)
   def embed():
    with torch.inference_mode():
     x=torch.from_numpy(image.copy()).unsqueeze(0).to('cuda:0')
     f=model.encode_image(metric.preprocess_clip(metric.validate_images(x))).float()
     assert f.shape==(1,768)and torch.isfinite(f).all()and f.norm().item()>0
     return f.detach().cpu().numpy()
   value=call('image_embedding',ps,embed)
   p=out/'features'/f'{ps}.npy'
   with p.open('xb')as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
   features[ps]=torch.from_numpy(value).to('cuda:0')
   item=dict(pixel_sha256=ps,feature=desc(p),input=locator);feature_records.append(item)
   with(out/'completed_features.jsonl').open('a')as f:f.write(json.dumps(item)+'\n');f.flush()
   if n%500==0:print('FEATURES',n,'/',len(feature_inputs),'seconds',round(time.monotonic()-start,1),flush=True)
  rows=read(checked(inv['frames_file']));sources=read(checked(inv['sources_file']));grid(rows,sources)
  pair_cache={};results=[]
  for v in rows:
   pair=(v['reference_sha256'],v['pixel_sha256'])
   if pair not in pair_cache:
    def cosine():
     with torch.inference_mode():value=float(F.cosine_similarity(features[pair[0]],features[pair[1]],dim=-1).item())
     assert np.isfinite(value)and -1.000001<=value<=1.000001
     return value
    pair_cache[pair]=call('cosine_pair','/'.join(pair),cosine)
    origin='NEW_EXACT_PIXEL_PAIR'
   else:origin='EXACT_PIXEL_PAIR_REUSE'
   row={k:v[k]for k in ['method','source_index','source_id','N','snr_db','noise_seed','receiver_status','fixed_gray',
       'lineage','reference_sha256','pixel_sha256']}
   row.update(clip_image_cosine=pair_cache[pair],score_origin=origin,endpoint=json.dumps(v['endpoint'],sort_keys=True))
   results.append(row)
  assert len(results)==22500 and len(pair_cache)==reg['limits']['cosine_pair']
  counts=db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall()
  assert all(s=='COMPLETE'for k,s,n in counts)
  csvsave(out/'frame_scores.csv',results);save(out/'feature_index.json',feature_records)
  history.append(gpu());save(out/'resource_history.json',history)
  bind(expected);checked(inv['frames_file']);checked(inv['sources_file'])
  save(out/'completion.json',dict(status='COMPLETE_CLIP_I_22500_LOGICAL_PAIRS',registration_sha256=expected,
      logical_pairs=len(results),unique_pairs=len(pair_cache),unique_features=len(features),counts=counts,unresolved=0,
      seconds=time.monotonic()-start,frame_scores=desc(out/'frame_scores.csv'),features=desc(out/'feature_index.json'),
      metric_identity=desc(out/'metric_identity.json'),new_training=0,new_channel=0,new_reconstruction=0,
      new_caption=0,new_other_metric_scores=0,existing_score_reuse=0,within_run_pair_reuse=22500-len(pair_cache)))
  print('COMPLETE',len(results),len(features),len(pair_cache),flush=True)
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,
       counts=db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall()));raise
 finally:db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration',required=True);main(p.parse_args().registration)
