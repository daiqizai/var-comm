import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
w=read(L/'benchmark_rx_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
p0=read(D/'p0_registration.json');proof=read(D/'benchmark_protocol.json');count=len(cr(proof['tx_cases']))*5*4;oldpin=read(D/'protocol_resolved_p0.json')['metrics'];old=cr(oldpin)
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[v for v in old['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','grid_codec.py','frame_plan.py','metrics.py','benchmark_transmitter.py','control_once.py','launch_registered.py']]
r=dict(code=code,cases=pin(L/'benchmark_inputs/tx_source_cases.json'),variants=[dict(name='original',grid='quarter'),dict(name='D_eighth',grid='eighth'),dict(name='D_token',grid='token'),dict(name='E',grid='quarter',input_grid='E')],model_routes=p0['model_routes'],static=p0['static'],metric_registration=oldpin,cpu_affinity=proof['cpu_affinity'],benchmark_protocol=pin(D/'benchmark_protocol.json'),caps=dict(model_load=1,load_lpips=1,prior_scale=count*40,decoder_forward=count*3,timing_trial=count))
gpu=read(D/'benchmark_rx_admission.json')['gpu'][1]
launch('benchmark_tx',r,E/'benchmark_transmitter.py',RT/'envs/frozen_visual_py310/bin/python',7200,'final',required_gpu_uuid=gpu)
