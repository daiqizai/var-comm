import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for rel in ['results/nestcom_7db_common500_20261030_v1','results/nestcom_n512_20261028_v1','results/n1024_consistency_20261019_v1','results/adatoken_response_20261026_v1/c_delivery']:
 p=R/rel;print('DIR',rel,[x.name for x in p.iterdir()][:50]if p.exists()else'MISSING')
for p in [R/'results/nestcom_7db_common500_20261030_v1/delivery/frame_metrics.csv',R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv',R/'paper/manuscripts/nestcom_wcl_adatoken_20261027_v2/input_manifest.json']:
 if p.exists():
  print('FILE',str(p))
  if p.suffix=='.csv':print(next(csv.DictReader(p.open())))
  else:print(str(read(p))[:4500])
for p in [B/'nestcom_7db_common500_20261030_v1',B/'nestcom_n512_20261028_v1',B/'n1024_consistency_20261019_v1',B/'adatoken_response_20261026_v1']:
 print('PFS',str(p),[x.name for x in p.iterdir()if any(t in x.name for t in ('rx','score','deliver','formal','regist','repair'))][:50]if p.exists()else'MISSING')
