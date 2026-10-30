import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from prepare_route_rx import from_frames
from_frames('fit_D_guard_frame_plan',['fit_D_guard_phy_512','fit_D_guard_phy_1024'],'fit_D_guard_rx_plan')
p=L/'fit_D_guard_rx_plan';cache=read(p/'reuse.json');keys={v['key']for v in read(p/'inputs.json')}
w=read(L/'fit_D_rx_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
for v in read(L/'fit_D_rx/outputs.json'):
 if v['key']not in keys:continue
 if v['key']in cache:assert cache[v['key']]['pixel_sha256']==v['pixel_sha256']
 else:cache[v['key']]=v
save(p/'reuse_A_D_core.json',cache)
missing=sum(v['projection']['status']=='SOURCE_INPUT'and v['key']not in cache for v in read(p/'inputs.json'))
save(p/'augmented_completion.json',dict(status='PREPARED',new_decode=missing,reused=len(cache),original_preparation=pin(p/'completion.json'),reuse=pin(p/'reuse_A_D_core.json')))
print(read(p/'augmented_completion.json'))
