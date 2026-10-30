import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for parent in [R/'experiments',E,R/'results',D]:
 p=parent/'AGENTS.md'
 if p.exists():print('RULES',p,p.read_text())
for root in [B/'n1024_consistency_20261019_v1',B/'n1024_consistency_20261019_v1/repair_v1',B/'adatoken_response_20261026_v1',R/'results/adatoken_response_20261026_v1']:
 print('DIRECTORY',str(root),[p.name for p in root.iterdir()][:90]if root.exists()else'MISSING')
for root in [B/'adatoken_response_20261026_v1/c_predclass_common500_rx_s00',B/'adatoken_response_20261026_v1/c_predclass_formal_statistics',B/'n1024_consistency_20261019_v1/repair_v1']:
 if not root.exists():continue
 for name in ['completion.json','actual_native_identity.json','outputs.json','frame_metrics.csv']:
  p=root/name
  if p.exists():
   if p.suffix=='.json':v=read(p);print(str(p),str(v[0]if isinstance(v,list)else v)[:3600])
print('STORAGE',subprocess.check_output(['du','-sb',str(L)],text=True).strip())
print('FREE',subprocess.check_output(['df','-h',str(L)],text=True).strip())
