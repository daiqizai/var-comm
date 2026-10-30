import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from grid_frame_plan import main
selection=read(D/'fit_D_delivery/decision.json');assert selection['selected']=='D_token'
save(D/'D_selected_before_guards.json',dict(selected='D_token',grid='token',decision=pin(D/'fit_D_delivery/decision.json'),core_metrics=pin(D/'fit_D_delivery/gate.json'),no_guard_reselection=True))
main(cells=((1024,4),(1024,19),(512,4),(512,10),(512,19)),role='fit_D_guard_frame_plan',grids=('quarter','token'))
a=read(D/'protocol_resolved_p0.json');base=read(D/'fit_phy_1024_registration.json')
code=base['code']+[pin(E/x)for x in ['grid_frame_plan.py','extended_frame_worker_v2.py']]
for N in [512,1024]:
 role=f'fit_D_guard_phy_{N}';rp=D/(role+'_registration.json');r=dict(N=N,jobs=pin(L/f'fit_D_guard_frame_plan/physical_{N}.json'),catalogue=pin(D/f'grid_catalogues/N{N}.json'),original_catalogue=a['catalogue_N'+str(N)],n512_qualification=pin(B/'nestcom_n512_20261028_v1/phy_qualification/completion.json'),qualification=base['qualification'],code=code,output=str(L/role),cpu_affinity=[24,25]if N==512 else[26,27],seconds=3600,selected_before_guards=pin(D/'D_selected_before_guards.json'))
 save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'extended_frame_worker_v2.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=3630,gpu=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
