import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for path in [D/'fit_C_registration.json',D/'fit_A_registration.json']:
 r=read(path);print(path.name,{k:v for k,v in r.items()if k not in ['code','model_routes','static','caps']})
p=B/'nestcom_7db_common500_20261030_v1/rx_nest/actual_native_identity.json';a=read(p);b=read(L/'p0_replay/actual_native_identity.json');print('native identities',list(a),[(k,a.get(k)==b.get(k))for k in ['models','runtime','numeric_flags','source_mathematics','sampling','GPU_name']]);print('wait dirs',[(x.name,[y.name for y in x.glob('*rx_nest*')][:4])for x in (B/'nestcom_7db_common500_20261030_v1').iterdir()if x.is_dir()and x.name in('controls','control','rx_nest_control')])
for role in ['fit_rx_plan','fit_D_rx_plan','fit_E_rx_plan']:
 print(role,[p.name for p in (L/role).glob('*.json')])
