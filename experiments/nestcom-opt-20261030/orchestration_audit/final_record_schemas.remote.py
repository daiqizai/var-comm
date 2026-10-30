import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
roots=[B/'n1024_consistency_20261019_v1/repair_v1/receiver',B/'n1024_consistency_20261019_v1/repair_v1/delivery',R/'results/n1024_consistency_20261019_v1/repair_v1/delivery',R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1',B/'adatoken_response_20261026_v1/c_predclass_formal_metric_preparation',R/'results/nestcom_7db_common500_20261030_v1/delivery']
for root in roots:
 print('ROOT',str(root),[p.name for p in root.iterdir()][:45]if root.exists()else'MISSING')
 if not root.exists():continue
 for p in root.iterdir():
  if p.suffix=='.csv':
   with p.open()as f:reader=csv.DictReader(f);first=next(reader,None)
   print('CSV',p.name,first)
  elif p.name in ['completion.json','identity.json','actual_native_identity.json','outputs.json','plan.json']:
   v=read(p)
   if isinstance(v,list):v=v[0]
   print('JSON',p.name,str(v)[:4500])
for r in ['repair_v1/receiver_control','repair_root_v1']:
 root=B/'n1024_consistency_20261019_v1'/r
 print('CONTROLS',r,[p.name for p in root.iterdir()][:20]if root.exists()else'MISSING')
