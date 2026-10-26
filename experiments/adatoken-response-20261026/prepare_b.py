"""CPU inventory of existing actual receptions, never source decoding or inference."""
from common import *
import numpy as np,collections
def main():
 out=L/'b_preparation';out.mkdir(parents=True,exist_ok=False)
 population=R/'results/arpc_external_baseline_20261019_v1/sources.json'
 chosen=read(population)['external_common100_v1'];assert len(chosen)==100
 rank={s['source_id']:i for i,s in enumerate(chosen)}
 rows_path=B/'n1024_consistency_20261019_v1/repair_v1/delivery/metric_rows_9000.json'
 rows=[v for v in read(rows_path)if v['source_id']in rank and v['strategy_family']=='PARTIAL'and v['snr_db']in(10,19)]
 assert len(rows)==600 and len({(v['source_id'],v['snr_db'],v['noise_seed'])for v in rows})==600
 jobs={};frames=[];files={};counts=collections.Counter()
 def check(d):
  if d['path']not in files:checked(d);files[d['path']]=d
  assert files[d['path']]==d
 for v in sorted(rows,key=lambda z:(rank[z['source_id']],z['snr_db'],z['noise_seed'])):
  pd=v['physical_frame'];check(pd);pkt=read(pd['path']);rx=pkt['actual_RX'];rec=v['reconstruction'];check(rec['image_archive'])
  ev=rec['evidence']
  if 'path'in ev:check(ev);ev=read(ev['path'])['evidence']
  p=rx['rx_profile'];tokens=ev.get('received_tokens');accepted=ev['source_status']=='ARITHMETIC_SOURCE_DECODED'
  if accepted:
   assert ev['source_decode_called']and ev['render_called']
   assert rx['header']['header_ok']and rx['body']['crc_accepted']and rx['body']['parser_accepted']
   assert not ev.get('comparison_truth_used',False)and not ev.get('receiver_truth_used',False)and not ev.get('transmitted_CDF_used',False)
   m,K=int(p['m']),int(p['K']);a=np.array(tokens,dtype=np.int64)
   assert len(a)==OFFSETS[m]+K and np.all((a>=0)&(a<4096))
   inp=dict(m=m,K=K,received_tokens=tokens)
  else:
   m=K=None;inp=dict(status='ACTUAL_FIXED_GRAY',source_status=ev['source_status'])
  key=digest(inp);counts[(v['snr_db'],'positive_K'if K else 'K0_or_rejected')]+=1
  row=dict(source_id=v['source_id'],source_index=v['source_index'],rank=rank[v['source_id']],snr_db=v['snr_db'],noise_seed=v['noise_seed'],m=m,K=K,key=key,accepted=accepted,physical_frame=pd,C0=rec,old_metrics=v['metrics'])
  frames.append(row)
  if accepted and K:
   if key in jobs:assert jobs[key]==dict(key=key,**inp)
   else:jobs[key]=dict(key=key,**inp)
 assert len(jobs)<=200
 save(out/'actual_received_only.json',list(jobs.values()));save(out/'frame_lineage.json',frames)
 conformance=[];references=[]
 for snr in(10,19):
  row=next(v for v in frames if v['snr_db']==snr and v['K'])
  job=jobs[row['key']];conformance.append(job);references.append(row)
 save(out/'conformance_received_only.json',conformance);save(out/'conformance_references.json',references)
 save(out/'audit.json',dict(population=pin(population),old_rows=pin(rows_path),frames=600,unique_positive_received=len(jobs),maximum_new_A0_B0_renders=2*len(jobs),conformance_renders=2,counts=[dict(snr=k[0],group=k[1],frames=n)for k,n in counts.items()],inputs=list(files.values()),new_source=0,new_PHY=0,new_metrics=0))
 old=read(R/'results/t1c_m10_test_20261016_v1/link_registration.json')
 pinned={x['path']:x for x in old['code']}
 for f in['kn_common.py','kn_native.py']:
  p=R/'experiments/kodak-current-20261026/nest_v1'/f;pinned[str(p)]=pin(p)
 for f in['common.py','control.py','b_receiver.py','b_ablation.py','test_contract.py']:
  p=E/f;pinned[str(p)]=pin(p)
 for x in pinned.values():checked(x)
 for role in('b_conformance','b_receiver'):
  inp=out/('conformance_received_only.json'if role=='b_conformance'else'actual_received_only.json')
  n=2 if role=='b_conformance'else len(jobs)
  reg=dict(schema='ACTUAL_RECEIVED_BOUNDARY_ABLATION_V1',role=role,scope=pin(O/'scope_registration.json'),code=list(pinned.values()),model_routes=old['model_routes'],inputs=pin(inp),caps=dict(model_load=1,var_render=n,prior_scale=10*n,decoder_forward=n if role=='b_conformance'else 2*n),max_seconds=7200,stages={role:dict(argv=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'b_receiver.py'),role],environment={'CUDA_VISIBLE_DEVICES':'6'},gpu=dict(index=6,pci='00000000:A3:00.0'),max_seconds=7200)},qualification_required='b_contract_qualification',no_source_or_TX_read=True,automatic_retry=False)
  if role=='b_receiver':reg['conformance_review_required']=str(O/'b_conformance_review.json')
  save(O/(role+'_registration.json'),reg)
 print(json.dumps(dict(audit=pin(out/'audit.json'),conformance=pin(O/'b_conformance_registration.json'),receiver=pin(O/'b_receiver_registration.json'))))
if __name__=='__main__':main()
