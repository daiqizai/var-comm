import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from route_statistics import *
x=np.arange(100)/100;b=np.random.default_rng(1).integers(0,100,(10000,100))
assert interval(x-x,b)==(0.,0.,0.)
base={c:dict(psnr_db=20,lpips_alex=.2,dreamsim=.1)for c in CORE}
assert not fit_gate(base,base)['fit_pass']
better={c:dict(psnr_db=20,lpips_alex=.198,dreamsim=.099)for c in CORE}
assert fit_gate(better,base)['fit_pass']
better[(1024,7)]['psnr_db']=19.8
assert not fit_gate(better,base)['fit_pass']
main('fit_A')
for role in ['fit_D_source','c_cache']:
 print('STATUS',role)
 for p in [L/(role+'_control')/'actual_wait.json',L/role/'failure.json']:
  if p.exists():print(read(p))
 print((L/(role+'_control')/'worker.log').read_text()[-400:])
