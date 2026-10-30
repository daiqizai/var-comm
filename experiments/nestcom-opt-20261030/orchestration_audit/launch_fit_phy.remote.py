import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
a=read(D/'protocol_resolved_p0.json');assert read(L/'p0_control/actual_wait.json')['returncode']==0
cat7=B/'nestcom_7db_20261030_v1/cal_phy_nest/catalogue.json';assert cat7.is_file()
code=[pin(E/p)for p in ['opt_common.py','frame_plan.py','frame_worker.py','control_once.py']]
code +=[pin(R/p)for p in ['experiments/nestcom-n512-20261028/n512_common.py','experiments/nestcom-n512-20261028/n512_phy.py','experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py','experiments/t1b-m10-calibration-20261016/tb_plan.py','experiments/adatoken-response-20261026/common.py','experiments/adatoken-response-20261026/fixed_frame.py','experiments/adatoken-response-20261026/c_predclass_frame.py','experiments/wcl-adatoken-20261027/public_counter_rx.py']]
for N in(512,1024):
 role='fit_phy_'+str(N);rp=D/(role+'_registration.json');assert not rp.exists()
 r=dict(N=N,jobs=pin(L/f'fit_frame_plan/physical_{N}.json'),catalogue=a['catalogue_N'+str(N)],catalogue7=pin(cat7),qualification=read(R/'results/nestcom_7db_20261030_v1/cal_phy_nest_registration.json')['phy_qualification'],code=code,output=str(L/role),cpu_affinity=[24,25]if N==512 else[26,27],seconds=10800)
 save(rp,r);cp=D/(role+'_control_registration.json')
 save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'frame_worker.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=10900,gpu=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
