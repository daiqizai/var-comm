"""Export the completed bounded study, preserving all denominator and failure evidence."""
from g import *
from dev_dispatch import dependency
import csv,shutil,sqlite3
from PIL import Image

def csvsave(path,rows):
 with Path(path).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 dependency('dev_scores');out=D/'delivery';out.mkdir();(out/'images').mkdir();(out/'receipts').mkdir()
 jobs=read(D/'development_score_jobs.json');rows=list(csv.DictReader((L/'dev_scores/scores.csv').open()));assert len(rows)==14
 for j in jobs:
  row=next(x for x in rows if x['method']==j['method']and int(x['source_index'])==j['source_index'])
  with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image']
  assert asha(y)==row['pixel_sha256'];p=out/'images'/('%04d_%s.png'%(j['source_index'],j['method']));Image.fromarray(np.rint(np.clip(y,0,1)*255).astype(np.uint8).transpose(1,2,0)).save(p)
  row.update(display_image='images/'+p.name,display_png_sha256=sha(p),preprocessing_id=j['preprocessing_id'],repeat='first preregistered development event',noise_identity='Original method-specific noise seed and sampler; new7dB event; not asserted shared across methods')
 for s in read(D/'development_sources.json'):
  with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels']
  assert hashlib.sha256(x.tobytes()).hexdigest()==s['preprocessing_id'];Image.fromarray(x.transpose(1,2,0)).save(out/'images'/('%04d_original.png'%s['source_index']))
 csvsave(out/'figure3_metrics.csv',rows);save(out/'source_receive_reconstruction_map.json',jobs)
 stages=[];total_gpu=0
 for c in sorted(L.glob('*_control')):
  if c.name=='cal_rx_arpc_control':
   assert not(c/'child.json').exists();shutil.copyfile(c/'failure.json',out/'receipts/initial_arpc_environment_failure.json');continue
  v=read(c/'actual_wait.json');assert v['returncode']==0 and gone(v['process'])and gone(read(c/'started.json')['process'])
  role=c.name.removesuffix('_control');r=read(D/(role+'_registration.json'));done=read(L/role/'completion.json')
  assert done['ledger']['unresolved']==0
  if r['gpu']:total_gpu+=v['seconds']
  stages.append(dict(role=role,registration=pin(D/(role+'_registration.json')),actual_wait=v,ledger=done['ledger'],gpu=r['gpu']))
  shutil.copyfile(c/'actual_wait.json',out/'receipts'/(role+'_wait.json'))
 assert total_gpu<=43200
 cal_rx=sum(read(L/role/'completion.json')['count']for role in ['cal_rx_nest','cal_rx_ada','cal_rx_arpc_environment_v2'])
 save(out/'actual_compute_and_reuse.json',dict(status='COMPLETE',new_calibration_frames=2000,calibration_packet_decodes=4000,new_calibration_received_render=cal_rx,calibration_DINO=read(L/'cal_dino/completion.json')['count'],new_development_digital_frames=10,new_development_continuous_trials=4,development_logical_reconstructions=14,new_LPIPS_pairs=14,new_DreamSim_pairs=14,new_NeST_source_probability_passes=2,new_ARPC_source_probability_passes=0,new_tokenizer=0,new_classifier=0,BPG_prior7dB_calibration_reused=True,ARPC_cached_independent_blocks_repacked=True,Ada_cached_predicted_class_streams_reused=True,new_training=0,new_bootstrap=0,common500_expansion=0,GPU_actual_wall_device_seconds=total_gpu,GPU_reserved_seconds=sqlite3.connect(L/'gpu_reservations.sqlite').execute('select sum(maximum_seconds)from reservations').fetchone()[0],GPU_cap_seconds=43200,stages=stages,initial_ARPC_environment_failure_zero_Popen_zero_science=True))
 for name in ['protocol.json','frozen_policies.json','calibration_summary.csv','development_cache_reuse.json','development_sources.json']:shutil.copyfile(D/name,out/name)
 shutil.copyfile(L/'dev_scores/metric_identity.json',out/'receipts/metric_identity.json');save(out/'export_identity.json',dict(native_arrays_only=True,no_enhancement=True,display='clip[0,1],round nearest uint8,CHWtoHWC',sources=[95,21],rows=14,metrics=pin(out/'figure3_metrics.csv'),all_tasks_actual0_gone=True,scope='development comparison, not formal common500'))
 print(json.dumps(dict(status='EXPORTED',rows=14,gpu_seconds=total_gpu)))
if __name__=='__main__':main()
