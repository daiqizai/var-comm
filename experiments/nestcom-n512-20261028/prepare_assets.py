"""Read and verify existing source caches only: no model or PHY calls."""
from n512_common import *
from n512_phy import ENDS
import subprocess,traceback
def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==BASE
 out=L/'assets';out.mkdir(parents=True,exist_ok=False);O.mkdir(parents=True,exist_ok=True)
 try:
  old={};origins={}
  paths={'nest_cal':'results/t1b_m10_calibration_20261016_v1/registration.json','nest_test':'results/n1024_consistency_20261019_v1/repair_v1/registration.json','ada_cal':'results/adatoken_response_20261026_v1/c_predclass_cal1000_link_registration.json','ada_test':'results/adatoken_response_20261026_v1/c_predclass_common500_link_registration.json'}
  for k,p in paths.items():origins[k]=pin(R/p);old[k]=read(R/p)
  counts={};sources={};audit=[]
  for population,suffix,count in [('cal1000','cal',1000),('common500','test',500)]:
   ns=cr(old['nest_'+suffix]['sources']);ads=cr(old['ada_'+suffix]['sources']);assert len(ns)==len(ads)==count
   amap={s['source_id']:s for s in ads};rows=[]
   for s in ns:
    a=amap[s['source_id']];assert s['source_index']==a['source_index'] and s['archive']==a['archive'] and s['preprocessing_id']==a['preprocessing_id']
    with np.load(checked(s['archive']),allow_pickle=False) as z:x=z['pixels'].copy()
    assert x.dtype==np.uint8 and x.shape==(3,256,256) and hashlib.sha256(x.tobytes()).hexdigest()==s['preprocessing_id']
    with np.load(checked(s['current_streams']),allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
    with np.load(checked(a['streams']),allow_pickle=False) as z:ad={k:z[k].copy() for k in z.files}
    assert set(n)=={f'm{m}_K{K}' for m,K in ENDS} and set(ad)=={f'p{i:03d}' for i in range(1,257)}
    for arr in (*n.values(),*ad.values()):assert arr.dtype==np.uint8 and arr.ndim==1 and len(arr)>=2 and np.isin(arr,(0,1)).all()
    expected=s.get('stream_endpoints',s.get('old_endpoints',[]))
    for e in expected:
     v=n[f'm{e["m"]}_K{e["K"]}'];assert len(v)==e['bits'] and asha(v)==e['payload_sha256']
    if 'm10'in s:assert len(n['m10_K0'])==s['m10']['arithmetic_bits'] and asha(n['m10_K0'])==s['m10']['payload_sha256']
    for e in a['lengths']:assert len(ad[f'p{e["prefix"]:03d}'])==e['bits'] and e['finish_bits']>=2
    checked(a['tokens']);checked(a['cdf_hashes']);cl=cr(a['class_commit']);assert cl['predicted_class']==a['predicted_class']
    row=dict(source_id=s['source_id'],source_index=s['source_index'],population=population,archive=s['archive'],preprocessing_id=s['preprocessing_id'],reference_sha256=asha(x.astype(np.float32)/np.float32(255)),nest_streams=s['current_streams'],ada_streams=a['streams'],predicted_class=a['predicted_class'],ada_tokens=a['tokens'],ada_cdf_hashes=a['cdf_hashes'],ada_class_commit=a['class_commit'],nest_lengths={k:len(v) for k,v in n.items()},ada_lengths={k:len(v) for k,v in ad.items()},nest_payload_hashes={k:asha(v) for k,v in n.items()},ada_payload_hashes={k:asha(v) for k,v in ad.items()})
    rows.append(row)
    if len(rows)%100==0:print('VERIFIED_SOURCE_CACHES',population,len(rows),count,flush=True)
   assert {s['source_index'] for s in rows}==set(range(count));rows.sort(key=lambda s:s['source_index'])
   save(out/(population+'.json'),rows);sources[population]=pin(out/(population+'.json'));counts[population]=len(rows)
  assert not ({s['source_id'] for s in cr(sources['cal1000'])}&{s['source_id'] for s in cr(sources['common500'])})
  inheritance=cr(old['nest_cal']['inheritance']);pilot=cr(inheritance['requests']['pilot_phy'])
  save(out/'original_pilot_structure.json',dict(keys=list(pilot),source_fields={k:v for k,v in pilot.items() if k in ('sources','source_ids','population','source_indices','events','inputs')}))
  save(out/'completion.json',dict(status='COMPLETE_SOURCE_CACHE_INVENTORY',sources=sources,counts=counts,origins=origins,nest_models=old['nest_cal']['model_routes'],nest_static=old['nest_cal']['static'],bootstrap=old['nest_test']['bootstrap'],pilot_request=inheritance['requests']['pilot_phy'],new_tokenizer=0,new_TX_probability=0,old_N1024_packets_reused=0,source_stream_count=1500*(25+256)))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
