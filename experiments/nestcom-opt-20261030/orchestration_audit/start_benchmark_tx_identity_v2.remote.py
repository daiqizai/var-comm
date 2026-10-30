import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch

import sqlite3,hashlib
failed=read(L/'benchmark_tx_control/actual_wait.json');assert failed['returncode']==1 and ident(failed['child']['pid'])is None
counts=sqlite3.connect(L/'benchmark_tx/calls.sqlite').execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status').fetchall();assert counts==[('model_load','COMPLETE',1)],counts
rows=[]
for case in read(L/'benchmark_inputs/tx_source_cases.json'):
 with np.load(checked(case['archive']),allow_pickle=False)as z:pixels=z['pixels'].copy()
 assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
 raw=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest();assert raw==case['source_pixels_sha256']
 rows.append(dict(source_id=case['source_id'],case_id=case['case_id'],raw_uint8_sha256=raw,dtype_shape_array_sha256=asha(pixels),archive=case['archive']))
assert len(rows)==32
save(D/'benchmark_tx_identity_correction.json',dict(status='EXACT_HASH_CONVENTION_DIAGNOSED',original_registration=pin(D/'benchmark_tx_registration.json'),original_failure=pin(L/'benchmark_tx/failure.json'),actual_wait=pin(L/'benchmark_tx_control/actual_wait.json'),original_ledger_counts=counts,new_encoding_trials_in_failed_attempt=0,model_load_device_seconds_charged=failed['device_seconds'],cause='preprocessing_id is SHA256(raw uint8 CHW bytes); original timing assertion incorrectly applied dtype+shape+bytes hash. All frozen NPZ file identities and raw pixel hashes match.',change='Only assert uses documented raw-byte SHA with explicit uint8 CHW shape. Original source, protocol, output hashes, algorithms, samples and timing repeats unchanged.',verified_cases=rows,old_code=pin(E/'benchmark_transmitter.py'),new_code=pin(E/'benchmark_transmitter_v2.py'),next_stage='First execution of all 640 previously unexecuted timing trials; no successful encoding/quality calls repeated',no_hot_edit=True))
w=read(L/'benchmark_rx_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
p0=read(D/'p0_registration.json');proof=read(D/'benchmark_protocol.json');count=len(cr(proof['tx_cases']))*5*4;oldpin=read(D/'protocol_resolved_p0.json')['metrics'];old=cr(oldpin)
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[v for v in old['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','grid_codec.py','frame_plan.py','metrics.py','benchmark_transmitter_v2.py','control_once.py','launch_registered.py']]
r=dict(code=code,cases=pin(L/'benchmark_inputs/tx_source_cases.json'),variants=[dict(name='original',grid='quarter'),dict(name='D_eighth',grid='eighth'),dict(name='D_token',grid='token'),dict(name='E',grid='quarter',input_grid='E')],model_routes=p0['model_routes'],static=p0['static'],metric_registration=oldpin,cpu_affinity=proof['cpu_affinity'],benchmark_protocol=pin(D/'benchmark_protocol.json'),caps=dict(model_load=1,load_lpips=1,prior_scale=count*40,decoder_forward=count*3,timing_trial=count))
gpu=read(D/'benchmark_rx_admission.json')['gpu'][1]
launch('benchmark_tx_identity_v2',r,E/'benchmark_transmitter_v2.py',RT/'envs/frozen_visual_py310/bin/python',7200,'final',required_gpu_uuid=gpu)
