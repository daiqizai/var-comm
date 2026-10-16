from pathlib import Path
import subprocess,json
H=Path(__file__).resolve().parent
old=json.loads((H/'historical_m10_audit.json').read_bytes())
code="""from pathlib import Path
import json,hashlib
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
p=RT/'qualification/h800_ep_full_source900_v1_attempt1/registered'
r=read(p/'request.json');c=read(p/'run/worker_completion.json');assert c['request_sha256']==sha(p/'request.json')
rows=c['existing100_streams']+c['new900_streams'];assert len(rows)==1000
result=[]
for o,current,record in zip(OLD['rows'],rows,r['records']):
 i=o['source_index'];assert i==current['source_index']==record['source_index']
 assert all(o[k]==record[k]for k in ['source_id','preprocessing_id','tokens_sha256'])
 assert sha(current['metadata']['path'])==current['metadata']['sha256']
 m=read(current['metadata']['path']);assert m['archive']==current['archive']and sha(current['archive']['path'])==current['archive']['sha256']
 ep={str(x['m']):x for x in m['endpoints']if x['K']==0}
 diffs=[k for k,v in o['old_whole'].items()if ep[k]['bits_sha256']!=v['sha256']or ep[k]['payload_bits']!=v['bits']]
 result.append(dict(source_index=i,source_id=o['source_id'],m9_exact='9'not in diffs,whole_mismatches=diffs,current_metadata=current['metadata'],current_archive=current['archive']))
print(json.dumps(dict(status='EXACT_CACHE_COMPATIBLE'if not any(x['whole_mismatches']for x in result)else'HISTORICAL_CACHE_DIFFERS_FROM_CURRENT_H800_STREAMS',
 original1000_source_identity_pass=True,request_sha256=sha(p/'request.json'),worker_completion_sha256=sha(p/'run/worker_completion.json'),
 sources_with_whole_mismatch=sum(bool(x['whole_mismatches'])for x in result),m9_mismatch_count=sum(not x['m9_exact']for x in result),rows=result)))
"""
script='OLD='+repr(old)+'\n'+code
PY='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
x=subprocess.run(['ssh','-T','DevMachineModel-H100-4C-P1660-01',PY,'-B','-'],input=script,text=True,encoding='utf8',capture_output=True,timeout=600)
if x.returncode:print(x.stderr);raise SystemExit(x.returncode)
v=json.loads(x.stdout)
with(H/'current_cache_comparison.json').open('x',encoding='utf8')as f:json.dump(v,f,indent=2)
print({k:v for k,v in v.items()if k!='rows'})
