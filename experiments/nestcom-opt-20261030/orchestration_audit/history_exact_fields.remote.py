import sys,csv,collections
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
p=B/'clip_i_common500_20261020_v1/score/frame_scores.csv'
with p.open()as f:rows=list(csv.DictReader(f))
print('CLIP_METHODS',collections.Counter(v['method']for v in rows))
for method in sorted({v['method']for v in rows}):
 if 'NEST'in method or 'WHOLE'in method or 'PARTIAL'in method:print('CLIP_EXAMPLE',next(v for v in rows if v['method']==method))
for p in [B/'nestcom_n512_20261028_v1/formal_rx_plan/frames.json',B/'nestcom_n512_20261028_v1/formal_rx_plan/reuse.json']:
 v=read(p);print('TOP',p.name,list(v)[:3]);x=next(iter(v.values()));print('SAMPLE',str(x)[:700]);
 if isinstance(x,dict)and'path'in x:
  f=cr(x);print('FRAME_FIELDS',{k:(list(v)if isinstance(v,dict)else str(v)[:100])for k,v in f.items()})
root=B/'nestcom_7db_common500_20261030_v1/phy_nest/frames'
print('7FRAMEFILE',next(root.iterdir())if root.exists()else'MISSING')
for x in [B/'nestcom_7db_common500_20261030_v1/phy_ada',B/'nestcom_7db_common500_20261030_v1/rx_ada']:
 print('7ADADIR',x,[p.name for p in x.iterdir()if p.suffix=='.json']if x.exists()else'MISSING')
