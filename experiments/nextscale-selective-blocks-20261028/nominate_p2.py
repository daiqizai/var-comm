"""Frozen proxy nomination from completed probes; no PHY/model/bootstrap calls."""
from block_common import *
from protocol import choose,CONFIGS,RATES
from audit_prepare_p1 import closed
import csv,collections,traceback
METRICS=('dino_cosine','psnr_db','lpips_alex','dreamsim','clip_image_cosine')

def lengths(s,t):
 v={'single':s['nest_lengths'],'C':{str(m):t['lengths']['C'+str(m)]['bits']for m in (4,5)}}
 for m,K in CONFIGS:v[f'm{m}_K{K}']={g:t['lengths'][f'm{m}_K{K}_{g}']['bits']for g in ('A','B')}
 return v

def probabilities(p,ph,body):
 if p['null']:return {'GRAY':1.0}
 ps=[body[g['N'],g['rate']]for g in p['layouts']]
 if p['kind']=='SINGLE':return {'SINGLE':ph*ps[0],'GRAY':1-ph*ps[0]}
 pc=ph*ps[0]
 if p['K']==0:return {'C':pc,'GRAY':1-pc}
 a,b=ps[1:]
 return {'CAB':pc*a*b,'CA':pc*a*(1-b),'CB':pc*(1-a)*b,'C':pc*(1-a)*(1-b),'GRAY':1-pc}

def main():
 out=L/'p2_nomination';out.mkdir(exist_ok=False)
 try:
  scope=read(O/'scope.json');cat=cr(scope['candidate_catalog']);src=cr(scope['source_manifest'])
  assert read(O/'p1_gate.json')['status']=='PASS_FOR_P2'
  body={};headers={};receipts=[]
  for i in range(4):
   role=f'p2_probe_{i:02d}';c=closed(L,role,L/(role+'_control'));receipts.append(pin(L/role/'completion.json'))
   for row in cr(c['body_cells']):
    assert row['trials']==128 and row['undetected']==0
    key=(row['snr_db'],row['N'],row['rate']);assert key not in body;body[key]=row['accepted']/128
   for row in cr(c['header_trials']):
    assert not row['undetected'];headers.setdefault(row['snr_db'],[]).append(row['header_ok'])
  assert len(body)==84 and all(len(v)==128 for v in headers.values())
  tx={r['source_id']:r for r in read(O/'p1_source_delivery.json')['records']}
  scores=read(L/'p1_summary/pair_scores.json');ev=read(O/'clean_metric_events.json');lookup={}
  for e in ev:
   key=(e['source_id'],e['state'],e.get('m'),e.get('K'));assert key not in lookup;lookup[key]=scores[e['pair_key']]
  rows=[];fits=[]
  for candidate in cat['candidates']:
   for s in src:
    p=choose(candidate,cat['profiles'],lengths(s,tx[s['source_id']]))
    fits.append(dict(candidate_id=candidate['candidate_id'],source_id=s['source_id'],source_index=s['source_index'],profile_id=p['profile_id'],m=p['m'],K=p['K'],null=p['null'],fallback=p['profile_id']!=candidate['profiles'][0]))
    for snr in scope['actual_snr_db']:
     probs=probabilities(p,float(np.mean(headers[snr])),{(N,r):v for (sn,N,r),v in body.items()if sn==snr});assert abs(sum(probs.values())-1)<1e-12
     value={k:0.0 for k in METRICS}
     for state,prob in probs.items():
      if state=='GRAY':key=(s['source_id'],'GRAY',None,None)
      elif state=='C'and p['K']==0:key=(s['source_id'],'SINGLE',p['m'],0)
      else:key=(s['source_id'],state,p['m'],p['K'])
      for k in METRICS:value[k]+=prob*lookup[key][k]
     rows.append(dict(candidate_id=candidate['candidate_id'],source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,profile_id=p['profile_id'],**value))
  summary=[]
  for c in cat['candidates']:
   rr=[r for r in rows if r['candidate_id']==c['candidate_id']];ff=[f for f in fits if f['candidate_id']==c['candidate_id']]
   assert len(rr)==96 and len(ff)==32
   summary.append(dict(candidate_id=c['candidate_id'],kind=c['kind'],m=c['target_m'],K=c['target_K'],uniform=c.get('uniform'),rate=c.get('rate'),J_proxy=float(np.mean([r['dino_cosine']for r in rr])),J_0dB=float(np.mean([r['dino_cosine']for r in rr if r['snr_db']==0])),null_sources=sum(f['null']for f in ff),fallback_sources=sum(f['fallback']for f in ff)))
  selected={}
  def best(options,metric,reason):
   row=sorted(options,key=lambda r:(-r[metric],r['candidate_id']))[0]
   selected.setdefault(row['candidate_id'],dict(**row,reasons=[]))['reasons'].append(reason)
  for m,K in CONFIGS:
   subset=[r for r in summary if r['kind']=='THREE'and(r['m'],r['K'])==(m,K)]
   best([r for r in subset if r['uniform']],'J_proxy','best_uniform_J')
   best([r for r in subset if not r['uniform']],'J_proxy','best_strict_J')
   best([r for r in subset if not r['uniform']],'J_0dB','strict_0dB_pressure_anchor')
  for rate in RATES:best([r for r in summary if r['kind']=='SINGLE'and r['rate']==rate],'J_proxy','best_single_per_rate')
  chosen=list(selected.values());nt=sum(r['kind']=='THREE'for r in chosen);ns=sum(r['kind']=='SINGLE'for r in chosen);assert nt<=12 and ns==4
  csvsave(O/'p2_proxy_all_candidates.csv',summary);csvsave(O/'p2_proxy_source_snr.csv',rows);csvsave(O/'p2_actual_length_fallback.csv',fits)
  nomination=dict(status='FROZEN_FOR_P3',scope=pin(O/'scope.json'),catalog=pin(O/'candidate_catalog.json'),source_manifest=pin(O/'source_manifest.json'),probe_receipts=receipts,metric_scores=pin(L/'p1_summary/pair_scores.json'),selected=chosen,three_candidates=nt,single_candidates=ns,physical_frames=(nt+ns)*32*9,tx_body_blocks=(3*nt+ns)*32*9,header_trials=(nt+ns)*32*9,selection='All220; mean DINO over32sources and0/2/4dB; fixed lexical ties; no resampling',proxy_assumption='Independent packet acceptance, empirical128 trials/cell; used only for nomination',excluded_layouts=[],unselected=[dict(candidate_id=r['candidate_id'],reason='Not a fixed-stratum maximum; retain all proxy results')for r in summary if r['candidate_id']not in selected],P3_not_run=True)
  save(O/'nomination.json',nomination);save(out/'completion.json',dict(status='COMPLETE_PROXY_NOMINATION',nomination=pin(O/'nomination.json'),new_phy=0,new_inference=0,new_bootstrap=0))
  print(json.dumps({k:nomination[k]for k in ('three_candidates','single_candidates','physical_frames','tx_body_blocks','selected')}))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':main()
