import sys,csv
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
def shape(v):
 if isinstance(v,dict):return {k:(list(x)[:18]if isinstance(x,dict)else('list:'+str(len(x)))if isinstance(x,list)else str(x)[:180])for k,x in v.items()}
 return str(v)[:500]
old=read(B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json')
for v in [old[0],next(x for x in old if not x['physical_reuse'])]:
 print('FRAME_ROW',shape(v));f=cr(v['physical_frame']);print('FRAME',shape(f))
 for k in ['transmission','actual_RX','profile','symbol_generation']: 
  if k in f:print(k,shape(f[k]))
for p in [B/'clip_i_common500_20261020_v1/score/frame_scores.csv',R/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv']:
 with p.open()as f:rows=csv.DictReader(f);print('CSV',str(p),next(rows))
print('METRIC_ID',shape(read(B/'clip_i_common500_20261020_v1/score/completion.json')))
v=read(B/'adatoken_response_20261026_v1/c_predclass_common500_metric_preparation/three_plan.json');print('ADA_PLAN',shape(v))
v=read(B/'nestcom_7db_common500_20261030_v1/phy_nest/frames.json')[0];print('7FRAME',shape(v))
