"""Start the registered CPU-only full1000 asset owner once."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path('/home/liulu/projects/VAR_COMM');H=R/'outputs/CONTENT-REAL-64QAM-20261006/H'
D=H/'full1000_assets_execution_v1';E=R/'experiments/content-real-64qam-20261006'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('full_assets_launch_owner',E/'stage_owner.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
cfg=read(D/'owner_config.json');reg=read(D/'execution_registration.json');seal=read(D/'registration_completion.json')
a.validate_config(cfg,reg,sha(D/'owner_config.json'));assert seal['registration_sha256']==sha(D/'execution_registration.json')
a.verify(seal['outputs']);a.verify(reg['source_bindings']);a.verify(reg['input_bindings'])
assert reg['allowed_stage_ids']==['source'] and reg['source_stage_scope']=='FULL1000_CPU_ASSETS_ONLY'
stage=cfg['stages'][0];assert stage['resource']=='cpu' and len(stage['jobs'])==1 and stage['jobs'][0]['id']=='full1000_assets'
assert not any((D/n).exists() for n in ('launch.json','owner_identity.json','failure.json','completion.json'))
assert not (H/'STOP').exists()
argv=[str(R/'outputs/UNIFIED-METRICS-20261002/environment/bin/python'),'-B',str(E/'stage_owner.py'),'--config',str(D/'owner_config.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (D/'owner.log').open('xb') as log:
    p=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ident=a.identity(p.pid);assert ident['argv']==argv and ident['uid']==os.getuid()
    value=dict(identity=ident,argv=argv,registration_sha256=sha(D/'execution_registration.json'),owner_config_sha256=sha(D/'owner_config.json'),
               launched_unix=time.time(),launcher_sha256=sha(Path(__file__).resolve()),GPU_jobs=0,new_packet_decodes=0)
    with (D/'launch.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(value))
