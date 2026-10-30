import sys,collections
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
v=read(B/'adatoken_response_20261026_v1/c_predclass_common500_metric_preparation/three_plan.json')
print('ADA_SOURCE',v['sources'][0]);print('ADA_FRAME',v['frames'][0])
v=read(B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json')
print('IMAGE_LAYOUTS',collections.Counter((x['reconstruction'].get('image_key'),str(x['reconstruction'].get('image_slot')))for x in v))
f=v[0]['reconstruction'];p=checked(f['image_archive'])
with np.load(p,allow_pickle=False)as z:print('NPZ',[(k,z[k].shape,str(z[k].dtype),asha(z[k]))for k in z.files])
print('7PATHS',[(k,(B/'nestcom_7db_common500_20261030_v1/phy_nest'/ (k+'.json')).exists())for k in [read(B/'nestcom_7db_common500_20261030_v1/phy_nest/frames.json')[0]['key']]])
print('C_CURRENT',read(L/'fit_C_guard_clip_control/actual_wait.json'))
print('D_PROGRESS', (L/'confirm_D_rx_control/worker.log').read_text()[-300:])
