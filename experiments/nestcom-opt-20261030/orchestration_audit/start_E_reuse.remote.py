import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from prepare_route_rx import from_frames,baseline_cache
from_frames('fit_E_frame_plan',['fit_E_phy_1024'],'fit_E_rx_plan')
cache=baseline_cache();cache.update({x['key']:x for x in read(L/'fit_E_simulation/outputs.json')})
jobs=read(L/'fit_E_rx_plan/inputs.json');reuse={j['key']:cache[j['key']]for j in jobs if j['projection']['status']=='SOURCE_INPUT'}
assert len(reuse)==len(jobs)
p=L/'inputs/fit_E_actual_reuse.json';save(p,reuse)
role='fit_E_rx';code=[pin(E/x)for x in ['opt_common.py','reuse_receiver.py','control_once.py']];rp=D/(role+'_registration.json')
r=dict(code=code,jobs=pin(L/'fit_E_rx_plan/inputs.json'),reuse=pin(p),output=str(L/role),route='E',successful_receiver_identity=pin(L/'fit_E_simulation/completion.json'),source_model_invocations=0)
save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'reuse_receiver.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=300,gpu=False))
with(L/(role+'_owner.log')).open('xb')as log:
 p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
