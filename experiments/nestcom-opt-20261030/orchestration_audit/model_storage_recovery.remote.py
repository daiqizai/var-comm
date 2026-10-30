import sys,os
os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,{26,27})
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
import torch
r=read(D/'p0_registration.json');files=cr(r['model_routes'])['files'];models=[]
def size(v):
 if torch.is_tensor(v):return v.numel()*v.element_size()
 if isinstance(v,dict):return sum(size(x)for x in v.values())
 if isinstance(v,(list,tuple)):return sum(size(x)for x in v)
 return 0
for p in files:
 cp=torch.load(checked(p),weights_only=True,map_location='cpu');key='decoder'if'decoder'in cp else 'raw_state';state=cp['decoder']if key=='decoder'else cp
 models.append(dict(file=p,state_key=key,state_tensor_bytes=size(state),serialized_bytes=p['bytes']));del cp
for variant,p,key in [('A_primary',read(D/'fit_A_registration.json')['anchors'][0],'model'),('A_verification',read(D/'fit_A_registration.json')['anchors'][1],'model')]:
 cp=torch.load(checked(p),weights_only=True,map_location='cpu');models.append(dict(variant=variant,file=p,state_key=key,state_tensor_bytes=size(cp[key]),serialized_bytes=p['bytes'],step=cp['global_step']));del cp
checkpoints=[]
for arm in ['content','no_content']:
 for p in sorted((L/'c_train_primary'/arm).glob('step_*.pt')):
  cp=torch.load(p,weights_only=True,map_location='cpu');checkpoints.append(dict(arm=arm,file=pin(p),step=cp['step'],seed=cp['seed'],state_tensor_bytes=size(cp['model']),checkpoint_keys=list(cp),restart_rule='Do not resume automatically; primary failed downstream guard. Checkpoint preserved for explicit future authorization.'));del cp
assert not torch.cuda.is_initialized()
save(D/'model_storage_and_recovery.json',dict(status='RECOVERY_IDENTITIES_VERIFIED_CPU',models=models,C_checkpoints=checkpoints,C_train_registration=pin(D/'c_train_primary_registration.json'),C_second_seed='NOT_RUN_CONDITIONAL_PRIMARY_GUARD_FAILED',no_old_training_restarted=True,CUDA_initialized=False,weights_stay_PFS=True,state_tensor_bytes_are_uncompressed_named_state_tensors=True))
print('MODEL_BYTES',[(v.get('variant',v['state_key']),v['state_tensor_bytes'])for v in models], 'C_saved',len(checkpoints))
