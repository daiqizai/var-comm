import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for role in ['benchmark_rx','benchmark_tx_identity_v2']:
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
proof=read(D/'benchmark_protocol.json');base=read(D/'final_phy_1024_registration.json');code=base['code']+[pin(E/'benchmark_phy.py')];role='benchmark_phy';rp=D/(role+'_registration.json')
r=dict(code=code,cases=proof['tx_cases'],rx_cases=proof['rx_cases'],catalogue=pin(D/'grid_catalogues/N1024.json'),qualification=base['qualification'],cpu_affinity=proof['cpu_affinity'],benchmark_protocol=pin(D/'benchmark_protocol.json'),output=str(L/role),seconds=1800);save(rp,r)
cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'benchmark_phy.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=1830,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
