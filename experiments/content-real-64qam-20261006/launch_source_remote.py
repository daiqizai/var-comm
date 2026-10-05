"""Launch the registered exclusive source-only owner exactly once."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path('/home/liulu/projects/VAR_COMM');H=R/'outputs/CONTENT-REAL-64QAM-20261006/H';E=R/'experiments/content-real-64qam-20261006';D=H/'full1000_source_execution_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('source_launch_owner',E/'stage_owner.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
c=read(D/'owner_config.json');r=read(D/'execution_registration.json');seal=read(D/'registration_completion.json')
a.validate_config(c,r,sha(D/'owner_config.json'));assert seal['registration_sha256']==sha(D/'execution_registration.json')
a.verify(seal['outputs']);a.verify(r['source_bindings']);a.verify(r['input_bindings'])
assert r['allowed_stage_ids']==['source'] and r['source_stage_scope']=='FULL1000_SOURCE_CODEC_ONLY'
assert c['stages'][0]['resource']=='gpu' and len(c['stages'][0]['jobs'])==1 and c['stages'][0]['jobs'][0]['id']=='full1000_source'
assert a.budget_snapshot(c['budget_path'],c['budget_registration_sha256'],c['phase_limits'],quiescent=True)==r['budget_before']
assert not any((D/n).exists() for n in ('launch.json','owner_identity.json','failure.json','completion.json')) and not (H/'STOP').exists()
argv=[str(R/'outputs/UNIFIED-METRICS-20261002/environment/bin/python'),'-B',str(E/'stage_owner.py'),'--config',str(D/'owner_config.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(c['gpu_device']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (D/'owner.log').open('xb') as f:
    p=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ident=a.identity(p.pid);assert ident['argv']==argv and ident['uid']==os.getuid()
    value=dict(identity=ident,argv=argv,registration_sha256=sha(D/'execution_registration.json'),owner_config_sha256=sha(D/'owner_config.json'),launched_unix=time.time(),launcher_sha256=sha(Path(__file__).resolve()),GPU_jobs=1,new_packet_decodes=0)
    with (D/'launch.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(value))
