import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
d=read(D/'fit_C_complete_delivery/decision.json');assert d['status']=='PREPARED'and d['guards_complete']
base=read(D/'c_train_primary_registration.json');r={k:base[k]for k in ['code','cache_completion','sources','arms','steps_per_arm','matched_schedule','preflight']};r.update(seed=2026103002,cpu_affinity=[18,19],verification_only=True,primary_gate=pin(D/'fit_C_complete_delivery/decision.json'),same_selection_rule=pin(D/'C_checkpoint_screen_rule.json'))
launch('c_train_verification',r,E/'c_train_pair_v2.py',RT/'envs/frozen_visual_py310/bin/python',60000,'C')
