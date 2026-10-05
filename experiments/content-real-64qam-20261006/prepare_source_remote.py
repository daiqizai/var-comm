"""Run CPU qualification and seal the independent source-only request."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/home/liulu/projects/VAR_COMM');O=R/'outputs/CONTENT-REAL-64QAM-20261006';H=O/'H';E=R/'experiments/content-real-64qam-20261006'
Q=O/'prepared_v1/full1000_source_qualification';Q.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def read(p):return json.loads(p.read_text())
files=['full1000_assets.py','test_full1000_assets.py','register_full1000_assets.py','test_register_full1000_assets.py','launch_assets_remote.py','full1000_source_driver.py','test_full1000_source_driver.py','register_full1000_source.py','test_register_full1000_source.py','h_full_payload_cpu.py','h_full_payload_driver.py','test_h_full_payload_cpu.py','test_h_full_payload_driver.py','prepare_source_remote.py','launch_source_remote.py']
subprocess.run(['git','add','--',*[str(E/f) for f in files]],cwd=R,check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',H64_REFERENCE_VAR_COMM=str(R/'src/var_comm'))
python=str(R/'outputs/UNIFIED-METRICS-20261002/environment/bin/python')
results=[]
for name in ('test_full1000_source_driver.py','test_register_full1000_source.py','test_h_full_payload_cpu.py','test_h_full_payload_driver.py'):
    cmd=[python,'-B',str(E/name)];p=Q/(name+'.log')
    with p.open('xb') as f:result=subprocess.run(cmd,cwd=E,env=env,stdout=f,stderr=subprocess.STDOUT)
    results.append(dict(argv=cmd,exit_code=result.returncode,log=str(p),sha256=sha(p)))
    print(name,result.returncode,flush=True)
    if result.returncode:
        save(Q/'failure.json',dict(status='CPU_QUALIFICATION_FAILED_PRESERVE',results=results,GPU_used=False,new_packet_decodes=0));raise SystemExit(1)
bound={str(E/f):sha(E/f) for f in files+['stage_owner.py','wait_then_coarse.py','visual_source_closure.py']}
save(Q/'completion.json',dict(status='H_FULL1000_SOURCE_CPU_QUALIFICATION_PASS',source_bindings=bound,results=results,GPU_used=False,new_packet_decodes=0,created_unix=time.time()))
A=H/'full1000_assets_execution_v1'
paths=dict(assets_registrar_module=E/'register_full1000_assets.py',owner_module=E/'stage_owner.py',wait_module=E/'wait_then_coarse.py',asset_module=E/'full1000_assets.py',source_module=E/'full1000_source_driver.py',static_closure_module=E/'visual_source_closure.py',assets_config=A/'assets_config.json',assets_registration=A/'execution_registration.json',assets_owner_config=A/'owner_config.json',assets_owner_launch=A/'launch.json',assets_owner_completion=A/'completion.json',assets_completion=H/'full1000_assets/completion.json',assets_manifest=H/'full1000_assets/manifest.json',visual_reference_config=H/'render_execution_r1/render_config.json',static_reference=H/'render_execution_r1/prelaunch_static_closure.json',numerical_reference=R/'outputs/M1-N2048-FULL-GRID-20261004/native_qualification.json',prepared_qualification=Q/'completion.json')
request=dict(schema='H_FULL1000_SOURCE_REGISTRATION_REQUEST_V1',execution_dir=str(H/'full1000_source_execution_v1'),source_out=str(H/'full1000_source'),python=python,max_seconds=10800,**{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
save(Q/'request.json',request)
cmd=[python,'-B',str(E/'register_full1000_source.py'),'--request',str(Q/'request.json')]
with (Q/'registration.log').open('xb') as f:result=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
print('registration',result.returncode,flush=True)
print((Q/'registration.log').read_text())
raise SystemExit(result.returncode)
