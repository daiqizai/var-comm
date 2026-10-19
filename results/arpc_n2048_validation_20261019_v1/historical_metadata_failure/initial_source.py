"""Explicit resource-only continuation; never retries an existing model call."""
from shared import *
import argparse,copy,ctypes,fcntl

def gpu_snapshot(index):
 fields=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 return dict(index=int(fields[0]),uuid=fields[1].strip(),name=fields[2].strip(),total_mib=int(fields[3]),free_mib=int(fields[4]),utilization=int(fields[5]))

def mapping(index):
 os.environ['CUDA_VISIBLE_DEVICES']=str(index)
 lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0
 dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
 pci=ctypes.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(pci,64,dev)==0
 uid=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(uid),dev)==0
 return dict(visible_index=index,pci_bus_id=pci.value.decode(),cuda_uuid_hex=bytes(uid).hex(),nvml_uuid=gpu_snapshot(index)['uuid'])

def prepare(phase,expected,index):
 r=registration(expected);old=L/phase/'arpc';new=L/phase/'arpc_resource_v1'
 assert phase in ['qualification','formal'] and not new.exists()
 evidence=[]
 if phase=='qualification':
  c=L/'controls/qualification_arpc';w=read(c/'actual_wait.json')
  assert type(w['returncode'])is int and w['returncode']==1 and not w['timed_out'] and gone(w['process'])
  assert gone(read(c/'supervisor.json')['process'])
  assert {p.name for p in (old/'rx').iterdir()}=={'runtime_cache'}
  err=(c/'stderr').read_text();assert "free>=cfg['resource']['start_free_bytes']" in err
  assert not (old/'rx/calls').exists() and not (old/'rx/completion.json').exists()
  evidence=[pin(c/n) for n in ['supervisor.json','launch.json','actual_wait.json','failure.json','stdout','stderr']]+[pin(old/'rx.cuda_mapping.json')]
 else:
  assert read(O/'qualification_review.json')['status']=='PASS'
  assert not (L/'controls/formal_arpc').exists() and not (old/'rx').exists()
 cfg=read(old/'public.json');before=copy.deepcopy(cfg)
 snap=gpu_snapshot(index);assert snap['free_mib']*1024**2>=cfg['resource']['start_free_bytes'] and snap['utilization']<=25
 actual=mapping(index);cfg.update(run=str(new),schema='N2048_ACTUAL_ARPC_E7_RESOURCE_V1')
 cfg['resource']['gpu']=snap;cfg['cuda_admission']=actual
 # Only location and physical admission change; preserve every numerical value.
 restore=copy.deepcopy(cfg);restore['run']=before['run'];restore['schema']=before['schema'];restore['resource']['gpu']=before['resource']['gpu'];restore['cuda_admission']=before['cuda_admission'];assert restore==before
 jobs=read(old/'wire_inputs/jobs.json');assert len(jobs)==cfg['new_job_count'] and 0<=len(jobs)<=700
 if phase=='qualification':assert len(jobs)==6
 new.mkdir();(new/'wire_inputs').mkdir();newjobs=[]
 for j in jobs:
  data=verify(j['input']).read_bytes();p=new/'wire_inputs'/Path(j['path']).name;p.open('xb').write(data)
  row=dict(j,path=str(p),input=pin(p));assert row['input']['sha256']==j['input']['sha256'];newjobs.append(row)
 put(new/'wire_inputs/jobs.json',newjobs);put(new/'public.json',cfg)
 auth=dict(schema='N2048_ARPC_RESOURCE_CONTINUATION_V1',status='AUTHORIZED_ZERO_MODEL_REPLAY',phase=phase,
  authority='User permits any free card; all original science scope and numerical settings unchanged.',
  registration=pin(REG),source=pin(__file__),original_public=pin(old/'public.json'),public=pin(new/'public.json'),
  old_jobs=pin(old/'wire_inputs/jobs.json'),jobs=pin(new/'wire_inputs/jobs.json'),failure_evidence=evidence,
  prior_failed_model_calls=0,new_source_calls=len(jobs),numerical_configuration_equal_after_only_path_and_device_restore=True,
  admission=snap,mapping=actual,automatic_retry=False)
 put(O/(phase+'_arpc_resource_authorization.json'),auth);print(json.dumps(pin(O/(phase+'_arpc_resource_authorization.json'))))

def run(phase,expected,authsha):
 r=registration(expected);authpath=O/(phase+'_arpc_resource_authorization.json');assert sha(authpath)==authsha
 a=read(authpath);assert a['phase']==phase and a['status']=='AUTHORIZED_ZERO_MODEL_REPLAY'
 for k in ['registration','source','original_public','public','old_jobs','jobs']:verify(a[k])
 for x in a['failure_evidence']:verify(x)
 root=L/'controls'/(phase+'_arpc_resource_v1');assert not root.exists()
 cfg=read(a['public']['path']);index=cfg['cuda_admission']['visible_index'];snap=gpu_snapshot(index)
 assert snap['uuid']==cfg['resource']['gpu']['uuid'] and snap['free_mib']*1024**2>=cfg['resource']['start_free_bytes'] and snap['utilization']<=25
 assert mapping(index)==cfg['cuda_admission']
 if phase=='formal':assert read(O/'qualification_review.json')['status']=='PASS'
 lock=(L/'execution.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 root.mkdir();put(root/'supervisor.json',dict(process=identity(os.getpid()),parent_process=identity(os.getppid()),argv=sys.argv,authorization=pin(authpath),started_unix=time.time()))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=str(index),PYTHONPATH=str(OLD_L/'environment_v1')+':'+str(OLD_E),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 argv=['/usr/bin/python3','-B',str(OLD_E/'a3_rx_v1/native.py'),'rx','--public',a['public']['path']]
 try:
  wait_child(root,argv,env,r['stage_seconds'][phase+'_arpc']);put(root/'completion.json',dict(status='COMPLETE',actual_wait=pin(root/'actual_wait.json'),authorization=pin(authpath)))
 except BaseException:put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('role',choices=['prepare','run']);ap.add_argument('phase',choices=['qualification','formal']);ap.add_argument('--expected',required=True);ap.add_argument('--gpu',type=int);ap.add_argument('--authorization-sha256');s=ap.parse_args()
 if s.role=='prepare':prepare(s.phase,s.expected,s.gpu)
 else:run(s.phase,s.expected,s.authorization_sha256)
