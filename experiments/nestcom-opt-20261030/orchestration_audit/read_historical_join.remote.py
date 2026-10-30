import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for p in [R/'results/adatoken_response_20261026_v1/c_predclass_formal_statistics_registration.json',R/'results/adatoken_response_20261026_v1/c_predclass_common500_three_metrics_registration.json',R/'results/adatoken_response_20261026_v1/c_predclass_common500_clip_metrics_registration.json']:
 v=read(p);print('REG',p.name,{k:v for k,v in v.items()if k!='code'})
root=R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/common500'
for n in ['frame_metrics.csv','source_identity_checks.json','paired_source_values.csv']:
 p=root/n
 if p.suffix=='.csv':
  with p.open()as f:print('ROW',n,next(csv.DictReader(f)))
 else:print('IDENT',str(read(p))[:3000])
for p in [B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json',B/'n1024_consistency_20261019_v1/reused_frames.json',B/'n1024_consistency_20261019_v1/repair_v1/receiver/images.json']:
 v=read(p);print('ROW',str(p),str(v[0]if isinstance(v,list)else next(iter(v.items())))[:5000])
