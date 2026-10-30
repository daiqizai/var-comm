import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for name in ['cal1000','common500']:
 v=read(B/'nestcom_n512_20261028_v1/qualified_sources'/(name+'.json'));print(name,type(v).__name__,len(v),str(v[0]if isinstance(v,list)else list(v))[:1600])
for path in [R/'results/nestcom_n512_20261028_v1/delivery/statistics_identity.json',B/'nestcom_n512_20261028_v1/qualification/completion.json',B/'nestcom_n512_20261028_v1/qualify/completion.json']:
 if path.exists():print('META',str(path),{k:v for k,v in read(path).items()if len(str(v))<2000})
print('N512QUAL',[p.name for p in(B/'nestcom_n512_20261028_v1').glob('*qual*')])
