from pathlib import Path
import base64,hashlib,subprocess,json
H=Path(__file__).resolve().parent;b=(H/'audit_v2.py').read_bytes();compile(b,'audit.py','exec')
R='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM';L='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/n1024_consistency_20261019_v1';py='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
s=f'''from pathlib import Path
import json,hashlib,base64,subprocess,os,time
E=Path({R!r})/'experiments/n1024-consistency-20261019';assert E.is_dir();data=base64.b64decode({base64.b64encode(b).decode()!r});assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(b).hexdigest()!r};(E/'audit_v2.py').write_bytes(data)
D=Path({L!r})/'audit_control_v2';D.mkdir()
def identity(pid):
 raw=Path(f'/proc/{{pid}}/stat').read_text();fs=raw[raw.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(fs[19]))
def save(n,v):
 with (D/n).open('x') as f:json.dump(v,f,indent=2)
argv=[{py!r},'-B',str(E/'audit_v2.py')];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');start=time.monotonic()
with (D/'stdout.txt').open('xb') as so,(D/'stderr.txt').open('xb') as se:
 child=subprocess.Popen(argv,cwd={R!r},env=env,stdout=so,stderr=se);pid=identity(child.pid);parent=identity(os.getpid());save('launch.json',dict(process=pid,parent_process=parent,argv=argv,source_sha256={hashlib.sha256(b).hexdigest()!r},new_scientific_calls=0));rc=child.wait(timeout=1800)
save('actual_wait.json',dict(actual_wait=True,returncode=rc,process=pid,parent_process=parent,argv=argv,seconds=time.monotonic()-start,timed_out=False,terminated=False,killed=False,metadata_error=None));print('ACTUAL_WAIT',rc,time.monotonic()-start)
print((D/'stdout.txt').read_text()[-7000:]);print((D/'stderr.txt').read_text()[-7000:]);raise SystemExit(rc)
'''
(H/'launch_remote_audit_v2.py').write_text(s,encoding='utf8')
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','DevMachineModel-H100-4C-P1660-01',py,'-B','-'],input=s,text=True,encoding='utf8',capture_output=True,timeout=1900)
(H/'audit_v2_launch_stdout.txt').write_text(p.stdout,encoding='utf8');(H/'audit_v2_launch_stderr.txt').write_text(p.stderr,encoding='utf8');print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
