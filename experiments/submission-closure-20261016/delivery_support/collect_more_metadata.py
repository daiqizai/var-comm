from pathlib import Path
import sys,json
E=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/submission-closure-20261016');sys.path.insert(0,str(E));from common import *
p=RT/'qualification/h800_ep_new100_source_v1_attempt1/registered/request.json';x=read(p);print('NEWKEY',list(x))
for k,v in x.items():
 if isinstance(v,list)and v and isinstance(v[0],dict):print('NEWITEM',k,list(v[0]),str(v[0])[:700])
save(O/'facts/new100_source_request_metadata.json',dict(request=pin(p),data=x))
x=read(R/'results/entropy_main_common500_20261013/registration_v4.json');p=Path(x['metric_config']['whole_policy']['path']);print('WHOLE',p,repr(read(p))[:4000]);save(O/'facts/whole_policy_metadata.json',dict(source=pin(p),data=read(p)))
p=R/'results/baseline_strength_20261011/timing/six_method_v6/actual_model_parameter_counts.json';print('PARAM',read(p))
