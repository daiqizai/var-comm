"""Root-authorized exact metadata schema completion; no report/science invocation."""
from pathlib import Path
import hashlib
import json
import os
import time

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
O = R/'results/arpc_n2048_adaptive_20261019_v1'
L = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1')
C = L/'b2b3_report_root_v1/report_runtime_paths_v1'
P = O/'b2b3_report_preparation_v1/runtime_paths_v1'
HISTORY = P/'initial_closure_schema_history'
HOST = 'aibox-rfbd2ce574fd-5df9766c45-vz6d9'


def pin(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts and all(not q.is_symlink() for q in (p,*p.parents))
    a = p.stat(); b = p.read_bytes(); z = p.stat()
    assert p.is_file() and a.st_size < 10000000 and (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def verify(d):
    assert pin(d['path']) == d
    return json.loads(Path(d['path']).read_bytes())


def put(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def canonical(d):
    assert d['host']==HOST and type(d['pid']) is int and type(d['start_ticks']) is int
    return {k:d[k]for k in ('pid','start_ticks','host')}


def gone(d):
    d=canonical(d);p=Path('/proc')/str(d['pid'])/'stat'
    if not p.exists():return True
    raw=p.read_text();f=raw[raw.rindex(') ')+2:].split()
    return int(f[19])!=d['start_ticks']


assert os.uname().nodename==HOST
old_obs=pin(C/'host_observation.json');old_route=pin(P/'success_route.json')
assert old_obs['bytes']==2458 and old_obs['sha256']=='a964508abc1b50969e86ad63acae8b12406d7e30bdbeba127ef8d7fdfe18416e'
assert old_route['bytes']==4376 and old_route['sha256']=='8966fa77df66586b697e0211dc56791883cbb4dfa570780e0f982b1ae12db815'
route=verify(old_route);obs=verify(old_obs)
assert route['schema']=='B2B3_REPORT_RUNTIME_PATHS_ROUTE_V1' and route['status']=='CLOSED'
assert route['host_observation']==old_obs and route['actual_wait']['path']==str(C/'actual_wait.json')
w=verify(route['actual_wait']);launch=verify(route['launch']);sup=json.loads((C/'supervisor.json').read_bytes())
assert type(w['returncode']) is int and w['returncode']==0 and w['metadata_error'] is None
assert all(w[k] is False for k in ('timed_out','terminated','killed'))
assert w['process']==launch['process'] and obs['argv']==launch['argv']
assert all(gone(d)for d in (launch['process'],launch['parent_process'],sup['parent_process']))
assert not(C/'failure.json').exists() and not(O/'b2b3_report_v1/failure.json').exists()
manifest=verify(route['report_manifest']);done=verify(route['report_completion'])
assert done['manifest']==route['report_manifest'] and done['status']=='REPORT_EXPORTED_VISUAL_REVIEW_PENDING'
for d in manifest['files']:assert pin(d['path'])==d
for d in route['original_failure_receipts'].values():assert pin(d['path'])==d
assert not HISTORY.exists();HISTORY.mkdir()
for source,name in [(C/'host_observation.json','host_observation.json'),(P/'success_route.json','success_route.json')]:
    data=source.read_bytes()
    with(HISTORY/name).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
assert pin(HISTORY/'host_observation.json')['sha256']==old_obs['sha256']
assert pin(HISTORY/'success_route.json')['sha256']==old_route['sha256']
new_obs=dict(obs,schema='FINAL_PUBLICATION_PROCESS_OBSERVATION_V1',execution_host=HOST,
    observer_host=HOST,observer_source=pin(__file__),observed_unix=time.time(),
    not_before_unix=launch['started_unix']+w['seconds'],processes=[canonical(d)for d in obs['processes']],
    original_closure_observation=pin(HISTORY/'host_observation.json'),
    metadata_revision_note='Root-authorized schema completion and a new actual same-host process observation; original wait, report and failed attempt unchanged.')
assert new_obs['observed_unix']>=new_obs['not_before_unix']
tmp=C/'host_observation.schema_complete.tmp.json';put(tmp,new_obs);os.replace(tmp,C/'host_observation.json')
new_route=dict(route,host_observation=pin(C/'host_observation.json'),
    initial_closure_schema_history=dict(original_observation=pin(HISTORY/'host_observation.json'),
                                       original_route=pin(HISTORY/'success_route.json')))
tmp=P/'success_route.schema_complete.tmp.json';put(tmp,new_route);os.replace(tmp,P/'success_route.json')
after=dict(observation=pin(C/'host_observation.json'),route=pin(P/'success_route.json'))
put(HISTORY/'revision.json',dict(schema='REPORT_CLOSURE_SCHEMA_COMPLETION_V1',status='COMPLETE',
    before=dict(observation=old_obs,route=old_route),after=after,source=pin(__file__),
    actual_wait=route['actual_wait'],report_manifest=route['report_manifest'],report_completion=route['report_completion'],
    report_calls=0,scientific_calls=0,original_bytes_retained=True,real_same_host_observation=True))
files=manifest['files']+[route['report_manifest'],route['report_completion']]
assert all(Path(d['path']).is_relative_to(O/'b2b3_report_v1')for d in files)
print(json.dumps(dict(status='CLOSED_REPORT_METADATA_SCHEMA_COMPLETE',**after,
    revision=pin(HISTORY/'revision.json'),files=files,actual_wait=route['actual_wait'],launch=route['launch'],
    controls=[pin(C/n)for n in ('supervisor.json','launch.json','actual_wait.json','stdout','stderr','runtime_path_transform.json','host_observation.json')],
    report_calls=0,scientific_calls=0)))
