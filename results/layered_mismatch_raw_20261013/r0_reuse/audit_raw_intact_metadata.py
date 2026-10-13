from pathlib import Path
import json,csv,hashlib,collections,numpy as np
W=Path('C:/Users/11946/Documents/ChatGPT/comm');P=W/'.research/layered_mismatch_raw_20261013/rawcal_selected200_v1';O=W/'results/layered_mismatch_raw_20261013/r0_reuse';c=json.loads((P/'completion.json').read_text());q=json.loads((O/'clean_quality_reuse.json').read_text());data=json.loads((P/'tokens200.json').read_text());cat=json.loads((W/'results/layered_mismatch_raw_20261013/candidates_cpu_v1/catalogue.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k,v in q['frozen_model_states'].items():assert c['frozen_visual_identity']['models'][k]==v
assert c['frozen_visual_identity']['decoder_gate_sha256']==q['decoder_gate_sha256'];assert c['score_identity']['metric_evaluator_identity']==q['metric_evaluator_identity'];assert c['score_identity']['model_metadata']['metrics']['dinov2_vitl14']==q['metric_definition']
assert c['source_count']==1000 and c['frame_count']==90000 and not c['development_used'] and not c['holdout_used'] and c['images_scored']
files=[];good={};counts=collections.Counter();examined=0;variations=[]
for source in data:
 i=source['source_index'];ids=source['source_id'];tokens=np.array(source['tokens'],dtype='<i8');assert hashlib.sha256(b'int64:680\0'+tokens.tobytes()).hexdigest()==source['tokens_sha256'];cpp=P/'source_checkpoints'/f'{i:04d}.json';rp=P/'sources'/f'{i:04d}.json';cp=json.loads(cpp.read_text());rows=json.loads(rp.read_text());source_json_sha=sha(rp);assert len(rows)==90 and cp['frame_count']==90 and cp['source_id']==ids and cp['registration_sha256']==c['registration_sha256']
 originalcp=f'/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007/actualcal_visual_r2/source_checkpoints/{i:04d}.json';originalrows=f'/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007/actualcal_visual_r2/sources/{i:04d}.json';assert c['outputs'][originalcp]==sha(cpp);assert c['outputs'][originalrows]==cp['outputs'][originalrows]==sha(rp)
 files.extend([dict(path=str(cpp),original_path=originalcp,sha256=sha(cpp)),dict(path=str(rp),original_path=originalrows,sha256=sha(rp))])
 for row in rows:
  examined+=1;assert row['source_id']==ids and row['source_index']==i and row['source_tokens_sha256']==source['tokens_sha256'] and row['preprocessing_id']==source['preprocessing_id'];assert row['population_role']=='calibration' and row['order']=='raster' and row['synthetic'] is False
  ev=row['visual_evidence'];assert ev['TX_truth_used_for_generation'] is False and ev['target_used_for_generation'] is False and ev['clean_source_image_substitution'] is False
  st=row['receiver_state'];
  if row['gray'] or not row['header_ok'] or not row['body_crc_accept'] or row['profile_id']!=row['rx_profile_id']:counts['not_accepted_correct_header']+=1;continue
  assert st['order']=='raster' and st['kind']=='tokens'
  m,k=int(st['m']),int(st['K']);expected=sum(x*x for x in [1,2,3,4,5,6,8,10,13,16][:m])+k
  actual=[v for scale in st['prefix'] for v in scale]+st['partial_values'];assert len(actual)==expected
  if actual!=tokens[:expected].tolist():counts['CRC_accept_but_not_source_intact']+=1;continue
  require_fields=('image_reconstruction_complete','visual_quality_scored')
  assert all(row[f] is True for f in require_fields)
  rec=dict(source_index=i,source_id=ids,m=m,K=k,dinov2_vitl14_cosine=row['dinov2_vitl14_cosine'],image_sha256=row['image_sha256'],source_json_sha256=source_json_sha,candidate_id=row['candidate_id'],snr_db=row['snr_db'],noise_seed=row['noise_seed'],actual_state_sha256=row['actual_state_sha256'],actual_received_tokens_sha256=ev['actual_received_tokens_sha256'],model_identity_sha256=ev['model_identity_sha256'],original_source_json=originalrows,actual_token_equality=True,body_crc_accept=True)
  key=(ids,m,k);counts['accepted_token_exact_frames']+=1
  if key in good:
   prior=good[key];assert prior['image_sha256']==rec['image_sha256'],('same clean endpoint different output',key)
   variations.append(abs(prior['dinov2_vitl14_cosine']-rec['dinov2_vitl14_cosine']))
   # Preserve the first row in the original sealed source JSON; never choose by quality.
  else:good[key]=rec
valid=sorted(good.values(),key=lambda r:(r['source_index'],r['m'],r['K']))
p=O/'historical_raw_intact_cal200.csv'
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(valid[0]));w.writeheader();w.writerows(valid)
allstates=sorted({(int(p['m']),int(p['K'])) for p in cat['profiles']});u_states=sorted({(int(p['m']),int(p['K'])) for p in cat['profiles'] if p['G']==2}|{(int(p['j']),0) for p in cat['profiles'] if p['G']==2});coarse={(m,0) for m in range(4,9)};states_count=collections.Counter((r['m'],r['K']) for r in valid);missing=[dict(source_index=s['source_index'],source_id=s['source_id'],m=m,K=k) for s in data for m,k in allstates if (m,k) not in coarse and (s['source_id'],m,k) not in good];umissing=[r for r in missing if (r['m'],r['K']) in u_states]
missingp=O/'clean_quality_missing_pairs.json';missingp.write_text(json.dumps(dict(all_candidate_terminal_states=allstates,missing_pairs=missing,U_EEP_missing_pairs=umissing),indent=2))
audit=dict(status='HISTORICAL_RAW_RASTER_INTACT_CLEAN_Q_METADATA_VERIFIED',population='original_calibration_UEP_MISMATCH_CAL200_V1',sources=200,original_frames_examined=examined,counts=dict(counts),source_endpoint_pairs=len(valid),state_coverage=[dict(m=m,K=k,sources=n) for (m,k),n in sorted(states_count.items())],all_candidate_missing_source_endpoint_pairs=len(missing),U_EEP_missing_source_endpoint_pairs=len(umissing),positiveK_used_only_after_exact_received_token_equality=True,old_complete=dict(path=str(P/'completion.json'),sha256=sha(P/'completion.json')),restoration=dict(path=str(P/'RESTORATION.json'),sha256=sha(P/'RESTORATION.json')),frozen_model_states=q['frozen_model_states'],metric_evaluator_identity=q['metric_evaluator_identity'],original_numerical_runtime=c['numerical_runtime'],selection_of_duplicate='First occurrence in original sealed source JSON row order; never choose by noise quality or score',maximum_same_image_metric_spread=max(variations,default=0),bound_source_files=files,limitation='Historical 4090D actual accepted raw-raster outputs can support disclosed pre-screen only. They are not new H800 clean measurements, new R1 noise frames or a complete all-candidate quality table; missing pairs stay unknown. Original image arrays were not restored/rehashed in this audit; authentic sealed source/score/image hash receipts were verified.',new_model_calls=0,new_channel_calls=0,new_metric_calls=0,outputs=[dict(path=str(p),sha256=sha(p)),dict(path=str(missingp),sha256=sha(missingp))])
(O/'historical_raw_clean_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:audit[k] for k in ['original_frames_examined','counts','source_endpoint_pairs','state_coverage','all_candidate_missing_source_endpoint_pairs','U_EEP_missing_source_endpoint_pairs','maximum_same_image_metric_spread']},indent=2))
