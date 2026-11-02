from study_common import *
from launch_extra import launch_extra
import ast
def main():
 T=B/'nestcom_timing_fairness_20261102_v1';reg=read(R/'results/nestcom_timing_fairness_20261102_v1/paired_prefill_child_v4/registration.json');template=reg['worker_template'];ss=cr(template['sources']);assert len(ss)==32
 new=read(T/'paired_prefill_child_v4/rx/rows.json');old=read(T/'clean_gpu6_v1/main_g05_rx/rows.json');symbols=read(T/'paired_prefill_child_v4/tx/received_symbols.json');pool={(x['source_id'],x['pixel_sha256']):x for x in read(FL/'E1_join/frames.json')}
 inputs=[];oldrows=[];donors={}
 for i,s in enumerate(ss):
  a=old[i*5];b=new[i*5];obs=symbols[i*5]
  assert a['source_id']==b['source_id']==obs['source_id']==s['source_id']
  assert all(x['image_sha256']==a['image_sha256']for x in old[i*5:i*5+5])and all(x['output_sha256']==b['output_sha256']for x in new[i*5:i*5+5])
  assert a['received_tokens_sha256']==b['received_tokens_sha256']
  v=pool[s['source_id'],a['image_sha256']];assert v['reference_sha256']==s['reference_sha256'];image=v['image'];image=ast.literal_eval(image)if isinstance(image,str)else image;checked(image)
  root=Path(image['path']).parent.parent
  if str(root)not in donors:donors[str(root)]=read(root/'outputs.json')
  record=next(x for x in donors[str(root)]if x['pixel_sha256']==a['image_sha256']);assert record['received_tokens_sha256']==a['received_tokens_sha256']
  inputs.append(dict(obs,source_index=s['source_index'],expected=b,old_generation_seed=record['generation_seed'],old_received_tokens_sha256=a['received_tokens_sha256']))
  oldrows.append(dict(source_id=s['source_id'],source_index=s['source_index'],arm='original_timed',image=image,pixel_sha256=a['image_sha256'],generation_seed=record['generation_seed'],received_tokens_sha256=a['received_tokens_sha256'],**{m:float(v[m])for m in METRICS}))
 out=L/'compiled_quality_plan';out.mkdir(exist_ok=False);save(out/'inputs.json',inputs);save(out/'old_rows.json',oldrows);save(out/'sources.json',ss)
 save(D/'compiled32_seed_alignment_addendum.json',dict(reason='The source-coded bitstream changes the deterministic tail seed. Preserve exact timed outputs AND compare a fixed-old-seed control on identical actual recovered tokens.',before_any_new_quality=True,sources=32,new_receiver_restore_max=32,additional_tail_render_max=32,max_new_metric_pairs=64,old_metric_pairs_reused=32,new_wireless_events=0,saved_event_PHY_replays=32,new_TX=0,retiming=0,resource_budget_unchanged=True,main_Ada_route_unchanged=True,old_rows=pin(out/'old_rows.json'),inputs=pin(out/'inputs.json')))
 r={k:template[k]for k in ['code','config','group','sources','sionna_package','CUDA_UUID']};r.update(inputs=pin(out/'inputs.json'),cpu_affinity=[38,40],addendum=pin(D/'compiled32_seed_alignment_addendum.json'))
 launch_extra('compiled_quality_restore',r,'compiled_quality_restore.py',VPY,1800,gpu=6,extra_environment={'CPATH':reg['environment']['CPATH']})
if __name__=='__main__':main()
