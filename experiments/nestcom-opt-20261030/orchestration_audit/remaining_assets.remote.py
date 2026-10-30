import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
a=read(D/'protocol_resolved_p0.json')
for k in ['catalogue_N512','catalogue_N1024','catalogue7']:
 if k in a:
  c=cr(a[k]);print(k,a[k],len(c['profiles']),max(v['profile_id']for v in c['profiles']))
print('KEYS',list(a))
for root in [B/'nestcom_7db_common500_20261030_v1/phy_nest',B/'nestcom_7db_common500_20261030_v1/rx_nest']:
 print('DIR',str(root),[p.name for p in root.iterdir()])
 for name in ['frames.json','outputs.json','completion.json','identity.json','actual_native_identity.json']:
  p=root/name
  if p.exists():
   v=read(p);print(name,str(v[0]if isinstance(v,list)else v)[:2200])
