import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_frame_plan_v3 import main
assert read(D/'confirm_D_delivery/decision.json')['confirm_pass'];checked(pin(D/'frozen_final.json'))
for role in ['final_D_source']:
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
assert read(L/'historical_regression_inputs/completion.json')['status']=='HISTORICAL_INPUTS_AUDITED'
common=dict(dataset='common500',sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),catalogues={str(n):pin(D/'grid_catalogues'/f'N{n}.json')for n in(512,1024)},old_frame_indices=[pin(L/'historical_regression_inputs/physical_index.json')],channel_recipe=pin(D/'confirmation_channel_recipe.json'),frozen_candidate=pin(D/'frozen_final.json'))
rp=D/'final_D_plan_registration.json';save(rp,dict(common,grid='token',grid_source_output=str(L/'final_D_source'),methods=['NEST'],variant='D_token',output='final_D_frame_plan'));main(rp)
rp=D/'final_WHOLE7_plan_registration.json';save(rp,dict(common,grid='quarter',methods=['WHOLE'],variant='original',cells=[[1024,7]],output='final_WHOLE7_frame_plan'));main(rp)
a=read(D/'protocol_resolved_p0.json');base=read(D/'fit_phy_1024_registration.json');code=base['code']+[pin(E/x)for x in ['evaluation_frame_plan_v3.py','extended_frame_worker_v3.py']]
# One worker per N deduplicates shared D/WHOLE wire events before any PHY calls.
for N in[512,1024]:
 jobs={}
 for root in ['final_D_frame_plan','final_WHOLE7_frame_plan']:
  for v in read(L/root/f'physical_{N}.json'):
   if v['key'] in jobs:assert jobs[v['key']]==v
   jobs[v['key']]=v
 jp=L/f'final_jobs_{N}.json';save(jp,list(jobs.values()))
 role=f'final_phy_{N}';rp=D/(role+'_registration.json');r=dict(N=N,jobs=pin(jp),catalogue=pin(D/f'grid_catalogues/N{N}.json'),original_catalogue=a['catalogue_N'+str(N)],n512_qualification=pin(B/'nestcom_n512_20261028_v1/phy_qualification/completion.json'),qualification=base['qualification'],code=code,output=str(L/role),cpu_affinity=[24,25]if N==512 else[26,27],seconds=10800,final_freeze=pin(D/'frozen_final.json'))
 save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'extended_frame_worker_v3.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=10830,gpu=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid),'jobs',len(jobs))
