"""Join only closed role scores; no inference or statistical resampling."""
from common import *
import argparse
def main(expected):
 assert sha(O/'metric_registration.json')==expected;reg=read(O/'metric_registration.json');plan=read(checked(reg['plan']));role_rows={};closures={}
 for role in ('three_metrics','clip_metrics'):
  control=L/(role+'_dispatch_v1');wait=read(control/'actual_wait.json');assert type(wait['returncode']) is int and wait['returncode']==0 and not any(wait[k] for k in ('terminated','killed','timed_out')) and wait['metadata_error'] is None
  outer=L/(role+'_initial_control_v1');ow=read(outer/'actual_wait.json');transport=read(outer/'transport_wait.json');ol=read(outer/'launch.json')
  assert type(ow['returncode']) is int and ow['returncode']==0 and not any(ow[k] for k in ('terminated','killed','timed_out'))
  assert type(transport['ssh_returncode']) is int and transport['ssh_returncode']==0 and transport['remote_command_exit0'] is True and transport['supervisor_process']==ol['process']
  assert not (outer/'failure.json').exists()
  owner=read(control/'owner.json');launch=read(control/'launch.json');assert all(owner['process'][k]==ol['child_process'][k] for k in ('pid','start_ticks'))
  host_closure=read(checked(transport['host_process_closure']));verify_host_closure(host_closure,reg['resources'][role]['hostname'],[ol['process'],owner['process'],launch['process']],expected,role)
  assert not (control/'failure.json').exists() and not (L/role/'failure.json').exists();complete=read(L/role/'completion.json');assert complete['rows']==1296 and complete['unresolved']==0
  role_rows[role]={framekey(r):r for r in csv.DictReader(checked(complete['scores']).open())};assert len(role_rows[role])==1296
  closures[role]=dict(wait=desc(control/'actual_wait.json'),completion=desc(L/role/'completion.json'),owner=desc(control/'owner.json'),outer_wait=desc(outer/'actual_wait.json'),transport_wait=desc(outer/'transport_wait.json'),host_process_closure=transport['host_process_closure'])
 rows=[]
 for f in plan['frames']:
  key=framekey(f);a=role_rows['three_metrics'][key];b=role_rows['clip_metrics'][key]
  for r in (a,b):assert r['source_id']==f['source_id'] and r['pixel_sha256']==f['pixel_sha256'] and r['rx_status']==f['rx_status']
  row={k:f[k] for k in ('method','source_index','source_id','snr_db','noise_seed','rx_status')}
  row.update({k:a[k] for k in ('psnr_db','lpips_alex','dreamsim')});row['clip_image_cosine']=b['clip_image_cosine'];row.update(image_path=f['image']['path'],image_key=f['image_key'],image_file_sha256=f['image']['sha256'],image_array_sha256=f['pixel_sha256'],reference_array_sha256=f['reference_sha256'],array_hash_convention='dtype_shape_cbytes',psnr_reused=f['existing_psnr_db']!='',lpips_reused=f['existing_lpips_alex']!='');rows.append(row)
 assert_grid(rows);csvsave(O/'frame_metrics.csv',rows)
 save(O/'metric_completion.json',dict(status='COMPLETE',registration=desc(O/'metric_registration.json'),rows=1296,metrics=4,logical_scalars=5184,scores=desc(O/'frame_metrics.csv'),role_closures=closures,unique_pairs=plan['unique_pairs'],unique_features=plan['unique_features'],existing_metric_reused_logical=plan['existing_metric_reused_logical'],new_pairs=plan['missing_pairs'],counts={role:read(L/role/'completion.json')['counts'] for role in role_rows},new_channel_events=0,new_reconstructions=0,new_bootstrap_indices=0))
 print(json.dumps(read(O/'metric_completion.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--registration',required=True);a=p.parse_args();main(a.registration)
