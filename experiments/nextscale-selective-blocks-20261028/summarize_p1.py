"""All predeclared P1 configurations, exact native pairs; no bootstrap or inference."""
from block_common import *
from protocol import CONFIGS
from audit_prepare_p1 import closed

def main():
 out=L/'p1_summary';out.mkdir(exist_ok=False)
 scores={};receipts=[]
 for mode in ('dino','three','clip'):
  role='p1_metrics_'+mode;c=closed(L,role,L/(role+'_control'));assert c['count']==960
  v=cr(c['scores']);assert len(v)==960
  receipts.append(dict(mode=mode,completion=pin(L/role/'completion.json'),wait=pin(L/(role+'_control')/'actual_wait.json'),identity=pin(L/role/'identity.json')))
  for key,row in v.items():
   if key not in scores:scores[key]=dict(pair_key=key,source_id=row['source_id'],reference_sha256=row['reference_sha256'],pixel_sha256=row['pixel_sha256'])
   assert all(scores[key][k]==row[k]for k in('source_id','reference_sha256','pixel_sha256'))
   fields={'dino':['value'],'three':['psnr_db','lpips_alex','dreamsim'],'clip':['clip_image_cosine']}[mode]
   for k in fields:scores[key]['dino_cosine'if k=='value'else k]=row[k]
 events=read(O/'clean_metric_events.json');metrics=['dino_cosine','psnr_db','lpips_alex','dreamsim','clip_image_cosine']
 rows=[dict(**e,**{k:scores[e['pair_key']][k]for k in metrics})for e in events]
 save(out/'pair_scores.json',scores)
 # Heterogeneous event columns are deliberately made explicit for CSV.
 fields=['source_id','source_index','state','m','K','state_key','pair_key']+metrics
 csvsave(O/'p1_clean_metrics.csv',[{k:r.get(k,'')for k in fields}for r in rows])
 summary=[];delta=[]
 for m,K in CONFIGS:
  subset=[r for r in rows if r.get('m')==m and r.get('K')==K and r['state']in('C','CA','CB','CAB')]
  lookup={(r['source_id'],r['state']):r for r in subset};assert len(lookup)==128
  for sid in sorted({r['source_id']for r in subset}):
   c,b=lookup[sid,'C'],lookup[sid,'CB']
   delta.append(dict(source_id=sid,source_index=c['source_index'],m=m,K=K,**{k:b[k]-c[k]for k in metrics}))
  for k in metrics:
   ds=np.asarray([r[k]for r in delta if r['m']==m and r['K']==K],np.float64);assert len(ds)==32
   summary.append(dict(m=m,K=K,metric=k,n_sources=32,C_mean=float(np.mean([r[k]for r in subset if r['state']=='C'])),CB_mean=float(np.mean([r[k]for r in subset if r['state']=='CB'])),CB_minus_C=float(ds.mean()),positive=int((ds>0).sum()),negative=int((ds<0).sum()),zero=int((ds==0).sum()),direction='lower'if k in('lpips_alex','dreamsim')else'higher'))
 csvsave(O/'p1_cb_minus_c_per_source.csv',delta);csvsave(O/'p1_cb_minus_c_summary.csv',summary)
 dino=[r for r in summary if r['metric']=='dino_cosine'];passed=any(r['CB_minus_C']>0 for r in dino)
 proof=read(O/'render_alignment_proof.json');assert proof['status']=='PASS'
 save(O/'p1_quality_gate.json',dict(status='PASS'if passed else'NO_GO_STOP',criterion='At least one predeclared mean DINO(C+B)-DINO(C)>0',configurations=dino,technical_source_proof=pin(O/'codec_source_proof.json'),pixel_proof=pin(O/'render_alignment_proof.json'),all_report_metrics=metrics,source_pairs=128,statistical_claim='32 calibration sources; descriptive, no new bootstrap or formal significance',score_receipts=receipts,protocol_integration_review_pending=True,visual_review_pending=True))
 save(out/'completion.json',dict(status='COMPLETE_P1_METRIC_SUMMARY',pair_scores=pin(out/'pair_scores.json'),quality_gate=pin(O/'p1_quality_gate.json'),new_inference=0,new_bootstrap=0))
 print(json.dumps(dict(quality_gate='PASS'if passed else'NO_GO_STOP',dino=dino)))
if __name__=='__main__':main()
