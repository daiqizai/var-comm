from pathlib import Path
import os,json,hashlib,time,base64
C=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1/b2b3_report_root_v1/layout_v1'); D=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/results/arpc_n2048_adaptive_20261019_v1/b2b3_report_v1/figures_layout_v1')
def pin(p):
 p=Path(p); assert p.is_absolute()and all(not q.is_symlink()for q in (p,*p.parents)); a=p.stat();b=p.read_bytes();z=p.stat();assert(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns);return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def gone(d):
 assert d['host']==os.uname().nodename
 p=Path('/proc')/str(d['pid'])/'stat'
 if not p.exists():return True
 s=p.read_text();return int(s[s.rindex(') ')+2:].split()[19])!=d['start_ticks']
assert os.uname().nodename=='aibox-rfbd2ce574fd-5df9766c45-vz6d9'
w=read(C/'actual_wait.json');l=read(C/'launch.json');s=read(C/'supervisor.json')
assert type(w['returncode'])is int and w['returncode']==0 and w['actual_wait'] is True and w['metadata_error'] is None
assert all(w[k]is False for k in ('timed_out','terminated','killed'))and not(C/'failure.json').exists()
assert w['process']==l['process']and w['parent_process']==l['parent_process']and w['argv']==l['argv']
processes=[l['process'],l['parent_process'],s['parent_process']]
assert all(gone(d)for d in processes)
assert pin(D/'manifest.json')['sha256']=='d151340fa3d43ad7a4ffa66f250f4d99fe476d2b3b2ff54c8b2d3b7b513a03c9'
manifest=read(D/'manifest.json')
for d in manifest['files']:assert pin(d['path'])==d
ob=dict(schema='FINAL_PUBLICATION_PROCESS_OBSERVATION_V1',status='CLOSED_ON_EXECUTION_HOST',host=os.uname().nodename,execution_host=os.uname().nodename,observer_host=os.uname().nodename,observer_source=pin(__file__),actual_wait=pin(C/'actual_wait.json'),wait=pin(C/'actual_wait.json'),launch=pin(C/'launch.json'),argv=l['argv'],child=l['process'],parent=l['parent_process'],processes=processes,child_gone=True,parent_gone=True,all_process_generations_gone=True,not_before_unix=l['started_unix'],observed_unix=time.time(),scientific_calls=0)
with(C/'host_observation.json').open('x')as f:json.dump(ob,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
files=manifest['files']+[pin(D/'manifest.json')]+[pin(C/n)for n in ('actual_wait.json','launch.json','supervisor.json','stdout','stderr','host_observation.json','observe.py')]
print(json.dumps(dict(status='PASS',files=[dict(descriptor=d,data=base64.b64encode(Path(d['path']).read_bytes()).decode())for d in files])))
