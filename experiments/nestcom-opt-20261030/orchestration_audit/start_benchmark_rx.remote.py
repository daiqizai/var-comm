import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
p0=read(D/'p0_registration.json');proof=read(D/'benchmark_protocol.json');variants=cr(proof['variants']);count=len(cr(proof['rx_cases']))*5*len(variants)
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','grid_codec.py','c_model.py','benchmark_receiver.py','control_once.py','launch_registered.py']]
r=dict(code=code,cases=proof['rx_cases'],variants=variants,model_routes=p0['model_routes'],static=p0['static'],cpu_affinity=proof['cpu_affinity'],benchmark_protocol=pin(D/'benchmark_protocol.json'),caps=dict(model_load=1,anchor_load=1,corrector_load=2,prior_scale=count*20,decoder_forward=count,timing_trial=count))
launch('benchmark_rx',r,E/'benchmark_receiver.py',RT/'envs/frozen_visual_py310/bin/python',7200,'final')
