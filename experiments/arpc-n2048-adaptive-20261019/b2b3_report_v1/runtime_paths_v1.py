"""One exact metadata-path compatibility for the frozen deterministic report."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import stat
import types

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/assets/pilot_metric_code_seed_v1')
O=R/'results/arpc_n2048_adaptive_20261019_v1'
L=BASE/'arpc_n2048_adaptive_20261019_v1'
E=R/'experiments/arpc-n2048-adaptive-20261019/b2b3_report_v1'
REVISION=O/'b2b3_report_preparation_v1/runtime_paths_v1/revision.json'
CONTROL=L/'b2b3_report_root_v1/report_runtime_paths_v1'
HOST='aibox-rfbd2ce574fd-5df9766c45-vz6d9'
SOURCE=dict(path=str(E/'report.py'),bytes=44204,sha256='6e5e90513c77c5fc28fc9b933dd176153920f27f45193b30dee00517dd86fe5f')
AUTH=dict(path=str(O/'b2b3_report_preparation_v1/authorization.json'),bytes=20850,sha256='ab1a9f53f02b32b1fc798e8bd99af50c8931b8950da11856f2f0fc5a18a61fb0')
FIXED={
 'common500_validation_metrics.py':(9529,'3f40eda1ba9d79ba4412a57560f9f5420b92e7e426b9152c7907f5f7eba4e7e3'),
 'metric_models.py':(30518,'dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066'),
 'quality.py':(2789,'589d43a9dede2327c67d363f3b8b879cfb184e40e6a58b1aa2ae9b675d63a1a5'),
 'score_bpg_cached_holdout_r2.py':(24885,'d8e9f85c83f63a288cf0500e53857bc6e81c205cec93897e4d46f9f177c79d8e'),
 'step0_reference_metrics.py':(17381,'2c5b712fcf36e859e73ef6a680989ce9a8ef29884ab63b0c3e14e7720d7dae28')}
EXACT=frozenset((str(RT/name),size,digest)for name,(size,digest)in FIXED.items())
OLD=b"require(p.is_relative_to(R) or p.is_relative_to(BASE), 'input outside R/PFS runtime')"
NEW=b"require(p.is_relative_to(R) or p.is_relative_to(BASE) or _exact_runtime_descriptor(d), 'input outside R/PFS runtime')"

def require(ok,message):
 if not ok:raise RuntimeError(message)

def key(d):return(d['path'],d['bytes'],d['sha256'])

def runtime_allowed(d):
 return isinstance(d,dict)and all(k in d for k in('path','bytes','sha256'))and type(d['bytes'])is int and key(d)in EXACT

def checked(d):
 require(set(d)=={'path','bytes','sha256'}and type(d['bytes'])is int and 0<=d['bytes']<10000000,'Invalid exact metadata descriptor')
 p=Path(d['path']);require(p.is_absolute()and'..'not in p.parts,'Unsafe path')
 require(p.is_relative_to(R)or p.is_relative_to(BASE)or runtime_allowed(d),'Unregistered metadata path')
 for q in(p,*p.parents):require(not q.is_symlink(),'Symlink')
 a=p.stat();require(stat.S_ISREG(a.st_mode),'Not regular');b=p.read_bytes();z=p.stat()
 require((a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns),'Changed metadata')
 require((len(b),hashlib.sha256(b).hexdigest())==(d['bytes'],d['sha256']),'Metadata SHA differs')
 return b

def load(d):return json.loads(checked(d))

def transform(source):
 require(len(source)==SOURCE['bytes']and hashlib.sha256(source).hexdigest()==SOURCE['sha256'],'Frozen report source drift')
 require(source.count(OLD)==1 and NEW not in source,'Expected unique checked gate missing')
 updated=source.replace(OLD,NEW)
 require(updated.replace(NEW,OLD)==source,'Inverse bytes differ')
 before=ast.parse(source);after=ast.parse(updated)
 before_functions={n.name:ast.dump(n,include_attributes=False)for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 after_functions={n.name:ast.dump(n,include_attributes=False)for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 require(before_functions.keys()==after_functions.keys(),'Function set changed')
 require([name for name in before_functions if before_functions[name]!=after_functions[name]]==['checked'],'Unrelated function AST changed')
 require(ast.dump(ast.parse(updated.replace(NEW,OLD)),include_attributes=False)==ast.dump(before,include_attributes=False),'Inverse AST differs')
 return updated

def gone(identity):
 require(identity['host']==os.uname().nodename==HOST,'Wrong process host')
 p=Path('/proc')/str(identity['pid'])/'stat'
 if not p.exists():return True
 s=p.read_text().rsplit(')',1)[1].split()
 return int(s[19])!=identity['start_ticks']

def prepare(revision):
 require(revision['schema']=='N2048_REPORT_EXACT_RUNTIME_PATH_REVISION_V1'and type(revision['scientific_calls'])is int and revision['scientific_calls']==0,'Wrong revision')
 require(revision['original_report']==SOURCE and revision['original_authorization']==AUTH,'Original bindings differ')
 require(revision['wrapper']['path']==str(E/'runtime_paths_v1.py')==str(Path(__file__).resolve()),'Wrapper location differs')
 checked(revision['wrapper']);checked(revision['diagnosis'])
 require({key(d)for d in revision['runtime_descriptors']}==EXACT,'Runtime allowlist differs')
 expected_routes={str(O/'b2b3_revision_v1'/p/'registration.json'):'code'for p in('B2','B3')}
 expected_routes.update({str(O/'b2b3_analysis_v1'/p/'registration.json'):'code'for p in('B2','B3')})
 expected_routes[str(O/'b3_metrics_v1/registration.json')]='original_metric_sources'
 require(len(revision['registered_source_routes'])==5 and {r['record']['path']:r['field']for r in revision['registered_source_routes']}==expected_routes,'Registered source route differs')
 for route in revision['registered_source_routes']:
  value=load(route['record']);outside=[d for d in value[route['field']]if not any(Path(d['path']).is_relative_to(root)for root in(R,BASE))]
  require({key(d)for d in outside}==EXACT,'Unexpected registered external descriptor')
 for d in revision['runtime_descriptors']:checked(d)
 failure=revision['original_failed_report'];require(set(failure)=={'supervisor.json','launch.json','actual_wait.json','stdout','stderr'},'Failure evidence set differs')
 old=L/'b2b3_report_root_v1/report'
 for name,d in failure.items():require(d['path']==str(old/name),'Failure evidence path differs');checked(d)
 require(failure['actual_wait.json']['sha256']=='72156649600cee15a6723d119c5b1c375fd8df9011cab268da406632a7966083','Failure wait differs')
 require(failure['launch.json']['sha256']=='db97f40924210b36bbe42ecdacf0444458e30343b9edaef526721519ee22d704','Failure launch differs')
 wait=load(failure['actual_wait.json']);launch=load(failure['launch.json']);sup=load(failure['supervisor.json'])
 require(type(wait['returncode'])is int and wait['returncode']==1 and wait['process']==launch['process'],'Original failure was relabelled')
 require(all(wait[k]is False for k in('timed_out','terminated','killed'))and wait['metadata_error']is None,'Different failure')
 require(launch['argv']==[str(R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'),'-B',SOURCE['path'],'--authorization',AUTH['path'],'--authorization-sha256',AUTH['sha256'],'--authorization-bytes',str(AUTH['bytes'])],'Original command differs')
 require(all(gone(i)for i in(launch['process'],launch['parent_process'],sup['parent_process'])),'Original process remains live')
 require(not(O/'b2b3_report_v1').exists(),'Report output already exists')
 auth=load(AUTH)
 for d in auth['source_files']:checked(d)
 source=checked(SOURCE);updated=transform(source)
 module=types.ModuleType('frozen_report_exact_runtime_paths');module.__file__=SOURCE['path']
 module.__dict__['_exact_runtime_descriptor']=runtime_allowed
 exec(compile(updated,SOURCE['path'],'exec'),module.__dict__)
 return module,auth,dict(original_source=SOURCE,transformed_sha256=hashlib.sha256(updated).hexdigest(),
                         inverse_bytes_exact=True,only_changed_function='checked',all_other_functions_ast_exact=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True);ap.add_argument('--revision-sha256',required=True);ap.add_argument('--revision-bytes',type=int,required=True)
 a=ap.parse_args();require(os.uname().nodename==HOST and Path(a.revision)==REVISION,'Wrong revision host/path')
 descriptor=dict(path=a.revision,sha256=a.revision_sha256,bytes=a.revision_bytes)
 module,auth,proof=prepare(load(descriptor))
 require(CONTROL.is_dir(),'Unique root control missing')
 with(CONTROL/'runtime_path_transform.json').open('x',encoding='utf8')as stream:
  json.dump(dict(revision=descriptor,transform=proof,scientific_calls=0),stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 module.run(auth)

if __name__=='__main__':main()
