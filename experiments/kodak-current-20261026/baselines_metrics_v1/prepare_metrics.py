"""Bind closed frame manifests and exact native arrays before four-metric scoring."""
from common import *
import argparse,sys
def build(input_descriptor):
 import numpy as np
 from PIL import Image
 inputs=read(checked(input_descriptor))
 assert not (O/'metric_registration.json').exists() and not (O/'metric_plan.json').exists()
 parent=desc(R/'results/kodak_current_20261026_v1/scope_registration.json');assert parent['sha256']=='3b5fe30a7d56acc4564beaa6dfaa75745d691def12df5ce63629875989d93b20'
 population=desc(R/'results/kodak_current_20261026_v1/population.json');assert population['sha256']=='a1d255e7491a8ae7303ccdf481a4abf48ff2133e9b03c56da8b756af657a941f'
 for d in inputs['closure_evidence']:
  value=read(checked(d))
  if 'returncode' in value:assert type(value['returncode']) is int and value['returncode']==0
  if 'expected_status' in d:assert value.get('status')==d['expected_status']
  elif 'status' in value:assert value['status'] in ('COMPLETE','COMPLETED','PASS','SUCCESS')
 for p in inputs['processes']:
  assert p['hostname']==__import__('socket').gethostname(),'Cross-host /proc check forbidden'
  assert gone(p),p
 assert inputs['all_scientific_runs_closed'] is True and inputs['failure_review']=='no_unresolved_new_failures'
 sources=read(checked(population))['sources'];byindex={s['source_index']:s for s in sources};refs={}
 for s in sources:
  x=np.ascontiguousarray(np.asarray(Image.open(checked(s['png'])).convert('RGB'),dtype=np.uint8).transpose(2,0,1))
  assert x.shape==(3,256,256) and rawah(x)==s['pixels_chw_uint8_sha256']
  x=x.astype(np.float32)/np.float32(255);refs[s['source_index']]=ah(x)
 rows=[];verified={};pair_old={}
 for d in inputs['frame_manifests']:
  for f in csv.DictReader(checked(d).open()):
   m,i,snr,n=framekey(f);source=byindex[i];assert f['source_id']==source['source_id'] and f['rx_status']
   image=dict(path=f['image_path'],sha256=f['image_file_sha256'])
   vkey=(image['path'],image['sha256'],f['image_key'])
   if vkey not in verified:
    with np.load(checked(image),allow_pickle=False) as z:y=z[f['image_key']].copy()
    canonical=assert_array(y,f['image_array_sha256'],f['array_hash_convention']);verified[vkey]=canonical
   else:canonical=verified[vkey]
   row=dict(method=m,source_index=i,source_id=source['source_id'],snr_db=snr,noise_seed=n,rx_status=f['rx_status'],image=desc(image['path']),image_key=f['image_key'],pixel_sha256=canonical,reference_sha256=refs[i],existing_psnr_db=f.get('existing_psnr_db',''),existing_lpips_alex=f.get('existing_lpips_alex',''),manifest=d,original_row=f)
   pair=(refs[i],canonical);pair_old.setdefault(pair,{})
   for metric in ('psnr_db','lpips_alex'):
    v=row['existing_'+metric]
    if v!='':
     assert np.isfinite(float(v))
     if metric in pair_old[pair]:assert float(pair_old[pair][metric])==float(v)
     pair_old[pair][metric]=v
   rows.append(row)
 assert_grid(rows)
 for row in rows:
  for metric in ('psnr_db','lpips_alex'):row['existing_'+metric]=pair_old[(row['reference_sha256'],row['pixel_sha256'])].get(metric,'')
 pairs={(r['reference_sha256'],r['pixel_sha256']) for r in rows};features={r['pixel_sha256'] for r in rows}|set(refs.values())
 missing={m:len({(r['reference_sha256'],r['pixel_sha256']) for r in rows if r['existing_'+m]==''}) for m in ('psnr_db','lpips_alex')}
 frozen_desc=desc(R/'results/arpc_decode_sensitivity_20261021_v1/metric_registration.json');assert frozen_desc['sha256']=='09ec1c0c871b3b2126526e4109edda24a218e3f06e2c33c004275dd65911b43f'
 frozen=read(checked(frozen_desc));cfg=frozen['config']
 # Reuse is justified by identical official formula, implementation, preprocessing, weights;
 # historical CPU/GPU execution provenance remains attached, not claimed bitwise identical.
 proof=desc(R/'results/generalization_kodak_final_20261013/metrics_final_v1/metric_math_identity_proof.json');assert proof['sha256']=='d9bc21811de74fd4810ece3fae9cec7500e2a7000d561f9ef1268ecc3bd07c2f'
 for d in cfg['config']['metric_config']['sources'].values():checked(d)
 checked(cfg['clip']['loader']);checked(frozen['original_four'])
 plan=dict(sources=[dict(source_index=s['source_index'],source_id=s['source_id'],png=s['png'],reference_sha256=refs[s['source_index']],pixels_chw_uint8_sha256=s['pixels_chw_uint8_sha256']) for s in sources],frames=sorted(rows,key=lambda r:(r['source_index'],r['method'],r['snr_db'],r['noise_seed'])),logical_frames=len(rows),unique_pairs=len(pairs),unique_features=len(features),missing_pairs=missing,existing_metric_reused_logical={m:sum(r['existing_'+m]!='' for r in rows) for m in missing},native_archive_count=len(verified),no_failure_filter=True)
 save(O/'metric_plan.json',plan)
 resources=inputs['resources']
 assert resources['three_metrics']['torch']=='2.11.0+cu128' and resources['clip_metrics']['torch']=='2.10.0+cu130'
 reg=dict(schema='KODAK_CURRENT_FOUR_METRICS_V1',population=population,parent=parent,inputs=input_descriptor,closure_evidence=inputs['closure_evidence'],plan=desc(O/'metric_plan.json'),original_four=frozen['original_four'],frozen_config=cfg,frozen_config_registration=frozen_desc,reuse_math_proof=proof,source_files=[desc(E/n) for n in ('common.py','prepare_metrics.py','score_metrics.py','owner.py','outer.py','merge_metrics.py','test_metrics.py','test_admission.py','test_host_closure.py')],qualification=inputs['qualification'],additional_qualification=inputs['additional_qualification'],host_qualification=inputs['host_qualification'],resources=resources,limits={'three_metrics':{'load_lpips':1,'load_dreamsim':1,'psnr':missing['psnr_db'],'lpips_pair':missing['lpips_alex'],'image_embedding':len(features),'cosine_pair':len(pairs)},'clip_metrics':{'load_clip':1,'image_embedding':len(features),'cosine_pair':len(pairs)}},max_seconds=7200,automatic_retry=False,new_bootstrap_indices=0,metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine'],pairing='source pairing; historical method noise namespaces retained',zero_calls=['channel','decoder','reconstruction','training','caption','tokenizer','VAR','bootstrap'])
 checked(reg['qualification']);save(O/'metric_registration.json',reg);print(json.dumps(dict(registration=desc(O/'metric_registration.json'),logical_frames=len(rows),unique_pairs=len(pairs),unique_features=len(features),missing_pairs=missing)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',required=True,type=int);a=p.parse_args();build(dict(path=a.input,sha256=a.sha256,bytes=a.bytes))
