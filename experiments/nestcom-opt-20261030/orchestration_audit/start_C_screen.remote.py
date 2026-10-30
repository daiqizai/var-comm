import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
w=read(L/'c_train_primary_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
c=read(L/'c_train_primary/completion.json');assert c['status']=='COMPLETE' and c['ledger']['unresolved']==0 and c['updates_per_arm']==3000
p0=read(D/'p0_registration.json');p=L/'fit_C_rx_plan';plan=read(p/'completion.json');count=plan['eligible_for_C']*6
cp=[dict(variant='C_'+arm+'_'+str(step),checkpoint=pin(L/'c_train_primary'/arm/('step_%06d.pt'%step)),arm=arm,step=step,seed=2026103001)for arm in ['content','no_content']for step in [1000,2000,3000]]
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','c_model.py','c_receiver_v2.py','control_once.py','launch_registered.py']]
r=dict(code=code,jobs=pin(p/'inputs.json'),reuse=pin(p/'reuse.json'),checkpoints=cp,checkpoint_selection_rule=pin(D/'C_checkpoint_screen_rule.json'),training_completion=pin(L/'c_train_primary/completion.json'),model_routes=p0['model_routes'],static=p0['static'],cpu_affinity=[20,21],caps=dict(model_load=1,corrector_load=6,prior_scale=count*10,decoder_forward=count,completion_forward=count,render=count))
launch('fit_C',r,E/'c_receiver_v2.py',RT/'envs/frozen_visual_py310/bin/python',3600,'C')
