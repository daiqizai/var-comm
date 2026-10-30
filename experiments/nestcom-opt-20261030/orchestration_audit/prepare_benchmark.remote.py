import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
sources=read(L/'inputs/fit100.json')[:16];ids={s['source_id']for s in sources};mapping={};proof=[]
for role,grids in [('fit_rx_plan',{'original':'quarter'}),('fit_D_rx_plan',{'D_eighth':'eighth','D_token':'token'}),('fit_E_rx_plan',{'E':'E'})]:
 ps={v['key']:v['projection']for v in read(L/role/'inputs.json')}
 for v in read(L/role/'logical.json'):
  variant=v.get('variant','original')
  if v['method']!='NEST'or variant not in grids or v['source_id']not in ids or (v['N'],v['snr_db'])not in [(1024,7),(1024,10)]:continue
  mapping[(v['source_id'],v['snr_db'],grids[variant])]=dict(row=v,projection=ps[v['received_key']])
 proof.extend([pin(L/role/'inputs.json'),pin(L/role/'logical.json')])
physical={}
for role in ['fit_frame_plan','fit_D_frame_plan','fit_E_frame_plan']:
 for j in read(L/role/'physical_1024.json'):physical[j['key']]=j
rx=[];tx=[]
for s in sources:
 for snr in [7,10]:
  entries={g:mapping[(s['source_id'],snr,g)]for g in ['quarter','eighth','token','E']};base=entries['quarter']['row'];case_id=digest([s['source_id'],1024,snr])
  rx.append(dict(case_id=case_id,source_id=s['source_id'],N=1024,snr_db=snr,projections={g:e['projection']for g,e in entries.items()}))
  tx.append(dict(case_id=case_id,source_id=s['source_id'],source_index=s['source_index'],N=1024,snr_db=snr,archive=s['archive'],source_pixels_sha256=s['preprocessing_id'],target=base['target'],capacity_bits=base['capacity_bits'],q=base['q'],rate=base['rate'],expected_payloads={g:physical[e['row']['physical_key']]['payload_sha256']for g,e in entries.items()},physical_cases={g:physical[e['row']['physical_key']]for g,e in entries.items()}))
assert len(rx)==len(tx)==32
out=L/'benchmark_inputs';out.mkdir(exist_ok=False);save(out/'rx_cases.json',rx);save(out/'tx_cases.json',tx)
variants=[dict(name='original',kind='original'),dict(name='A',kind='A',checkpoint=read(D/'fit_A_registration.json')['anchors'][0]),dict(name='B03',kind='B',temperature=.3),dict(name='B07',kind='B',temperature=.7)]
for v in read(D/'fit_C_registration.json')['checkpoints']:
 if v['step']==1000:variants.append(dict(name=v['variant'],kind='C',arm=v['arm'],checkpoint=v['checkpoint']))
variants.extend([dict(name='D_eighth',kind='D',input_grid='eighth'),dict(name='D_token',kind='D',input_grid='token'),dict(name='E',kind='E',input_grid='E')]);save(out/'variants.json',variants)
save(D/'benchmark_protocol.json',dict(status='FROZEN_BEFORE_TIMING',source_selection='First16 in preregistered FIT hash order; no quality-based selection',source_ids=[s['source_id']for s in sources],sources=pin(L/'inputs/fit100.json'),rx_cases=pin(out/'rx_cases.json'),tx_cases=pin(out/'tx_cases.json'),variants=pin(out/'variants.json'),cells=[[1024,7],[1024,10]],warmups=2,measured=3,batch=1,cpu_affinity=[28,29],same_actual_GPU_for_TX_RX=True,shared_workload=True,not_dedicated_hardware_benchmark=True,B_single_output_seed=2026103001,B_quality_evaluation_repeats=3,actual_frame_inputs=proof,purpose='Deployment cost only, repeated calls explicitly timing workload; no new quality or selection outcomes'))
print('PREPARED_TIMING',len(rx),len(variants))
