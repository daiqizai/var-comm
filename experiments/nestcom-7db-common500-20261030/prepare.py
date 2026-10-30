"""Read-only identity verification and one preregistration, no model/PHY calls."""
from g import *
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='640b359d1c13864184d9d6f0f4a49bb37e622882'
 D.mkdir();L.mkdir()
 oldpolicy=pin(OLD/'frozen_policies.json');assert cr(oldpolicy)['status']=='FROZEN_BEFORE_DEVELOPMENT'
 qualified=B/'nestcom_n512_20261028_v1/qualified_sources/common500.json';sources=read(qualified);assert len(sources)==500
 arpc={};catalogues=[]
 for rel in ['results/arpc_common500_extension_20261019_v1/source_v1/stream_catalogue.json','results/arpc_external_baseline_20261019_v1/a3_source_v1/stream_catalogue.json']:
  p=R/rel;catalogues.append(pin(p))
  for s in read(p)['sources']:
   assert s['source_id']not in arpc;arpc[s['source_id']]=s
 assert len(arpc)==500
 rows=[]
 for s in sorted(sources,key=lambda s:s['source_index']):
  a=arpc[s['source_id']];assert a['source_index']==s['source_index'] and a['source']['sha256']==s['archive']['sha256']
  for d in [s['archive'],s['nest_streams'],s['ada_streams'],*[v['wire']for v in a['streams']]]:checked(d)
  with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels']
  assert x.dtype==np.uint8 and x.shape==(3,256,256) and hashlib.sha256(x.tobytes()).hexdigest()==s['preprocessing_id'] and asha(x.astype(np.float32)/np.float32(255))==s['reference_sha256']
  for method in ['nest','ada']:
   with np.load(checked(s[method+'_streams']),allow_pickle=False)as z:
    for k in z.files:assert asha(z[k])==s[method+'_payload_hashes'][k] and len(z[k])==s[method+'_lengths'][k]
  rows.append(dict(s,arpc_streams=a['streams']))
 assert [s['source_index']for s in rows]==list(range(500))
 cal=read(OLD/'calibration_sources.json');assert not({s['source_id']for s in cal}&{s['source_id']for s in rows})
 save(L/'sources.json',rows)
 manifest=[{k:s[k]for k in ['source_id','source_index','archive','preprocessing_id','reference_sha256','nest_streams','ada_streams','predicted_class']}for s in rows];save(D/'source_manifest.json',manifest)
 bank=pin(B/'baseline_strength_20261011/t1c_m10_test_20261016_v1/delivery/common500_bootstrap_indices.npy')
 assert bank['sha256']=='5ca12fe75af6a09e1d3c18e205d7394bf4cdca53c25f7ca21005e5de30117634'
 ix=np.load(checked(bank),allow_pickle=False);assert ix.shape==(10000,500)and ix.min()==0 and ix.max()==499
 save(D/'cache_identity.json',dict(qualified=pin(qualified),arpc_catalogues=catalogues,sources=pin(L/'sources.json'),all_500_file_and_array_hashes_checked=True,calibration_disjoint=True))
 save(D/'protocol.json',dict(schema='N1024_7DB_COMMON500_SIX_METHODS_V1',base_commit='640b359d1c13864184d9d6f0f4a49bb37e622882',authorization='User confirmed common500 six methods x500 sources x3 repetitions, frozen7dB policies,12 GPU-device-hour ceiling; update Fig2 and manuscript and push.',N=1024,snr_db=7,header_symbols=68,body_symbols=956,Es=2,methods=list(METHODS),source_manifest=pin(D/'source_manifest.json'),sources=pin(L/'sources.json'),frozen_policy=oldpolicy,calibration='Existing100 fixed calibration sources, one realization, five MCS for NEST/ADA/ARPC; BPG historical100x3 PSNR calibration. No additional calibration, no policy or generation tuning.',noise=dict(NEST=[6201,6202,6203],ARPC=[6201,6202,6203],ADA=[2001,2002,2003],BPG=[2001,2002,2003],P100000=[2001,2002,2003],Swin200000=[2001,2002,2003]),noise_scope='Retain method-native generator; no assertion of cross-method identical noise.',public_counter='26000000 +16384*original_source_index +4096*repeat_zero_based; header profile ID added for NEST/ADA',caps=dict(logical_results=9000,digital_frames=6000,continuous_events=3000,packet_decodes=12000,GPU_device_seconds=43200,new_training=0,new_calibration=0,new_tokenizer_for_digital_methods=0,new_source_model_for_digital_methods=0,bpg_source_encodes=500,continuous_source_encodes=1000,unique_metric_pairs_max=9000),metrics=['psnr_db','lpips_alex','dreamsim'],primary_figure_metrics=['lpips_alex','dreamsim'],statistics=dict(bootstrap=bank,replicates=10000,source_order=[s['source_id']for s in rows],aggregation='Average three channel outcomes per source, then equal-source mean; failures kept; direct paired differences; existing index bank only.',interval='pointwise95%source percentile bootstrap'),no_retry=True,large_arrays='PFS only'))
 save(D/'preflight.json',dict(status='PASS',source_count=500,protocol=pin(D/'protocol.json'),new_scientific_calls=0))
 print(json.dumps(read(D/'preflight.json')))
if __name__=='__main__':main()
