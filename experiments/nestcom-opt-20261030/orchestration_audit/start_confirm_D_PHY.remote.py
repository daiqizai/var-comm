import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_frame_plan_v2 import main
for role in ['confirm_D_source','confirm_base_phy_512','confirm_base_phy_1024']:
 p=L/(role+'_control')/'actual_wait.json'
 if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
rp=D/'confirm_D_plan_registration.json';save(rp,dict(dataset='confirm100',sources=pin(L/'inputs/confirm100.json'),grid='token',grid_source_output=str(L/'confirm_D_source'),methods=['NEST'],variant='D_token',catalogues={str(n):pin(D/'grid_catalogues'/f'N{n}.json')for n in(512,1024)},old_frame_indices=[pin(L/f'confirm_base_phy_{n}/frames.json')for n in(512,1024)],output='confirm_D_frame_plan',channel_recipe=pin(D/'confirmation_channel_recipe.json'),frozen_candidate=pin(D/'D_confirm_frozen.json')));main(rp)
a=read(D/'protocol_resolved_p0.json');base=read(D/'fit_phy_1024_registration.json');code=base['code']+[pin(E/x)for x in ['evaluation_frame_plan_v2.py','extended_frame_worker_v2.py']]
for N in[512,1024]:
 role=f'confirm_D_phy_{N}';rp=D/(role+'_registration.json');r=dict(N=N,jobs=pin(L/f'confirm_D_frame_plan/physical_{N}.json'),catalogue=pin(D/f'grid_catalogues/N{N}.json'),original_catalogue=a['catalogue_N'+str(N)],n512_qualification=pin(B/'nestcom_n512_20261028_v1/phy_qualification/completion.json'),qualification=base['qualification'],code=code,output=str(L/role),cpu_affinity=[24,25]if N==512 else[26,27],seconds=3600,confirmation_freeze=pin(D/'D_confirm_frozen.json'))
 save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'extended_frame_worker_v2.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=3630,gpu=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
