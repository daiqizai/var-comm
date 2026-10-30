import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=R/'results/nestcom_7db_common500_20261030_v1/phy_nest_registration.json';v=read(p);print('REG_KEYS',list(v))
for k,x in v.items():
 if k not in ('code','inputs','files'):print(k,str(x)[:1200])
root=B/'nestcom_7db_common500_20261030_v1';c=read(root/'phy_nest/completion.json');print('COMPLETION',c)
for row in read(root/'phy_nest/frames.json'):
 if row['actual_RX']['body'].get('payload')is None:print('REJECT7',{k:v for k,v in row.items()if k not in ('actual_RX','transmission')});break
