import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from tx_selected_plan import main
main();role='fit_E_phy_1024';base=read(D/'fit_phy_1024_registration.json');a=read(D/'protocol_resolved_p0.json')
code=base['code']+[pin(E/x)for x in ['tx_selected_plan.py','selected_frame_worker.py']]
r=dict(N=1024,jobs=pin(L/'fit_E_frame_plan/physical_1024.json'),catalogue_by_snr={'7':base['catalogue7'],'10':a['catalogue_N1024']},qualification=base['qualification'],code=code,output=str(L/role),cpu_affinity=[26,27],seconds=1800)
rp=D/(role+'_registration.json');save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'selected_frame_worker.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=1830,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
