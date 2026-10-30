import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
d=read(D/'fit_C_delivery/decision.json');assert d['selected']=='C_content_1000'and d['selected_no_content']=='C_no_content_1000'and d['content_increment']
save(D/'C_selected_before_guards.json',dict(selected=d['selected'],no_content=d['selected_no_content'],decision=pin(D/'fit_C_delivery/decision.json'),no_guard_reselection=True))
rows=[v for v in read(L/'fit_rx_plan/logical.json')if not(v['N']==1024 and v['snr_db']in(7,10))];keys={v['received_key']for v in rows};jobs=[v for v in read(L/'fit_rx_plan/inputs.json')if v['key']in keys];cache={v['key']:v for v in read(L/'fit_A/outputs.json')if v['variant']=='original'};reuse={k:cache[k]for k in keys}
p=L/'fit_C_guard_rx_plan';p.mkdir(exist_ok=False);save(p/'logical.json',rows);save(p/'inputs.json',jobs);save(p/'reuse.json',reuse)
eligible=sum(j['projection']['status']=='SOURCE_INPUT'and j['projection']['m']in(7,8)and j['projection']['K']>0 and reuse[j['key']]['status']=='ACCEPTED_SOURCE'for j in jobs);count=eligible*2
save(p/'completion.json',dict(status='PREPARED',logical=len(rows),unique_inputs=len(jobs),eligible_for_C=eligible,variants=2,new_render_cap=count,frozen_rule=pin(D/'C_selected_before_guards.json')))
base=read(D/'fit_C_registration.json');cp=[v for v in base['checkpoints']if v['variant']in[d['selected'],d['selected_no_content']]];assert len(cp)==2
r={k:base[k]for k in ['code','model_routes','static','cpu_affinity','checkpoint_selection_rule','training_completion']};r.update(jobs=pin(p/'inputs.json'),reuse=pin(p/'reuse.json'),checkpoints=cp,caps=dict(model_load=1,corrector_load=2,prior_scale=count*10,decoder_forward=count,completion_forward=count,render=count))
launch('fit_C_guard',r,E/'c_receiver_v2.py',RT/'envs/frozen_visual_py310/bin/python',3600,'C');print('C_GUARD',read(p/'completion.json'))
