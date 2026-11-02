"""Check the data contract before writing claims or rendering updated figures."""
from pathlib import Path
import csv,json,hashlib,math
P=Path(__file__).resolve().parents[1];U=P/'inputs/unified_calibration_20261103'
MET=('psnr_db','lpips_alex','dreamsim','clip_image_cosine','dists')

def rows(p):
 with p.open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))
def one(rr,**kw):
 q=[r for r in rr if all(str(r[k])==str(v)for k,v in kw.items())];assert len(q)==1,(kw,len(q));return q[0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 assert json.loads((U/'science_closure.json').read_text())['status']=='COMPLETE_SCIENCE'
 summary=rows(U/'external_join/summary.csv');paired=rows(U/'external_join/paired_deltas.csv');cap=rows(U/'formal/capacity_summary.csv')
 assert len(summary)==len(paired)==140 and len(cap)==12
 for rr in [summary,paired]:
  for r in rr:
   assert int(r['N'])==1024 and int(r['snr_db'])in[4,7,10,19]and r['metric']in MET
   a,b,c=[float(r[k])for k in ['ci_low','mean','ci_high']];assert all(math.isfinite(v)for v in [a,b,c])and a<=b<=c
 for r in cap:assert int(r['sources'])==500 and int(r['logical'])==1500
 inventory={}
 for comparison in sorted({r['comparison']for r in paired}):
  vals=[]
  for snr in [4,7,10,19]:
   row=dict(snr_db=snr)
   for m in MET:
    r=one(paired,comparison=comparison,snr_db=snr,metric=m);lo=float(r['ci_low']);hi=float(r['ci_high'])
    favorable=(lo>0)if m in ['psnr_db','clip_image_cosine']else(hi<0)
    adverse=(hi<0)if m in ['psnr_db','clip_image_cosine']else(lo>0)
    row[m]=dict(mean=float(r['mean']),ci=[lo,hi],direction='candidate'if favorable else'reference'if adverse else'crosses_zero')
   vals.append(row)
  inventory[comparison]=vals
 w={m:float(one(summary,method='WHOLE',snr_db=10,metric=m)['mean'])for m in MET}
 n={m:float(one(summary,method='NeST',snr_db=10,metric=m)['mean'])for m in MET}
 ten=dict(WHOLE=w,NeST=n,relative_distance_reductions={m:100*(w[m]-n[m])/w[m]for m in ['dists','dreamsim']},utilization={m:float(one(cap,method=m,snr_db=10)['utilization'])for m in ['WHOLE','NeST']})
 f2=rows(P/'figures/figure2_data.csv');assert len(f2)==92
 aliases={'final_NeST':'NeST','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada'}
 for r in f2:
  if r['experiment']=='fixed_symbols'and r['method']in aliases:
   z=one(summary,method=aliases[r['method']],snr_db=int(float(r['x'])),metric=r['metric'])
   assert all(float(r[k])==float(z[v])for k,v in [('mean','mean'),('ci95_low','ci_low'),('ci95_high','ci_high')])
 mapping=json.loads((P/'inputs/fig3_unified_20261103/native_mapping.json').read_text());assert len(mapping)==10
 visual=[]
 for sid in [135,201]:
  rr=[v for v in mapping if v['source_index']==sid];assert len(rr)==5
  n=next(v for v in rr if v['method']=='final_NeST')
  visual.append(dict(source_index=sid,NeST_vs={v['display_method']:{m:n[m]-v[m]for m in MET}for v in rr if v['method']not in ['Source','final_NeST']}))
 for v in mapping:
  assert sha(P/v['display']['path'])==v['display']['sha256']
 report=dict(status='DATA_CHECKS_PASS',science_closed=True,summary_rows=140,paired_rows=140,formal_logical=18000,formal_source_means=6000,fig2_main_means=84,fig2_complete_means=92,comparison_directions=inventory,N1024_10_example=ten,figure3_comparisons=visual,inputs={str(p.relative_to(P)):sha(p)for p in [U/'external_join/summary.csv',U/'external_join/paired_deltas.csv',U/'formal/capacity_summary.csv',P/'figures/figure2_data.csv',P/'inputs/fig3_unified_20261103/native_mapping.json']},human_text_review='PENDING',artifact_visual_review='PENDING')
 (P/'validation/unified_data_checks.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(ten=ten,comparison_directions=inventory,figure3_comparisons=visual),indent=2))

if __name__=='__main__':main()
