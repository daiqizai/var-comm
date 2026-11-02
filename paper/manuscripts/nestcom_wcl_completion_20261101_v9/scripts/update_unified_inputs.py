"""Derive manuscript tables and Fig.2 input from closed source-paired results."""
from pathlib import Path
import csv,json,re,hashlib
P=Path(__file__).resolve().parents[1]
U=P/'inputs/unified_calibration_20261103'

def rows(p):
 with Path(p).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))
def write(p,rr):
 with Path(p).open('w',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def one(rr,**kw):
 a=[r for r in rr if all(str(r[k])==str(v)for k,v in kw.items())];assert len(a)==1,(kw,len(a));return a[0]
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 closure=json.loads((U/'science_closure.json').read_text());assert closure['status']=='COMPLETE_SCIENCE'
 summary=rows(U/'external_join/summary.csv');pair=rows(U/'external_join/paired_deltas.csv');cap=rows(U/'formal/capacity_summary.csv')
 assert len(summary)==140 and len(pair)==140 and len(cap)==12
 aliases={'final_NeST':'NeST','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada'}
 f2=rows(P/'figures/figure2_data.csv');assert len(f2)==92;changes=[]
 for r in f2:
  if r['experiment']!='fixed_symbols'or r['method']not in aliases:continue
  v=one(summary,method=aliases[r['method']],snr_db=int(float(r['x'])),metric=r['metric'])
  changes.append(dict(key={k:r[k]for k in ['panel','method','metric','x']},before={k:r[k]for k in ['mean','ci95_low','ci95_high']},after={k:v[vv]for k,vv in [('mean','mean'),('ci95_low','ci_low'),('ci95_high','ci_high')]}))
  r.update(mean=v['mean'],ci95_low=v['ci_low'],ci95_high=v['ci_high'])
 assert len(changes)==24;write(P/'figures/figure2_data.csv',f2)
 ti=rows(P/'tables/table_i.csv');assert len(ti)==10
 for r in ti:
  if r['block']!='A':continue
  snr=int(r['snr_db']);r['whole_utilization']=one(cap,method='WHOLE',snr_db=snr)['utilization'];r['nest_utilization']=one(cap,method='NeST',snr_db=snr)['utilization']
  for metric in ['dists','dreamsim']:
   v=one(pair,comparison='NeST-WHOLE',snr_db=snr,metric=metric)
   r.update({metric+'_'+k:v[vv]for k,vv in [('delta','mean'),('low','ci_low'),('high','ci_high')]})
 write(P/'tables/table_i.csv',ti)
 text=(P/'tables/table_i.tex').read_text()
 text=text.replace('family-specific policies','uniformly calibrated policies')
 for snr in [4,7,10,19]:
  r=one(ti,block='A',snr_db=snr)
  cells=[str(snr),f"{100*float(r['whole_utilization']):.2f}"+r'$\to$'+f"{100*float(r['nest_utilization']):.2f}"]
  for m in ['dists','dreamsim']:cells.append(r'\shortstack{'+f"{float(r[m+'_delta']):+.4f}"+r'\\{}['+f"{float(r[m+'_low']):.4f}, {float(r[m+'_high']):.4f}"+']}')
  line=' & '.join(cells)+r'\\[2pt]'
  pattern=r'^'+str(snr)+r' & .*?\\\\\[2pt\]$'
  text,n=re.subn(pattern,lambda _:line,text,count=1,flags=re.M);assert n==1
 (P/'tables/table_i.tex').write_text(text,encoding='utf8')
 labels={'WHOLE_final_receiver':'WHOLE','final_NeST':'NeST-Com','Ada_frozen':'Ada-TokenCom','Latent_JSCC':'Latent JSCC','SwinJSCC':'SwinJSCC','BPG_LDPC':'BPG + LDPC','ARPC':'ARPC + LDPC'}
 tii=[]
 for method in labels:
  a=aliases.get(method,method);rr=dict(method=method,N=1024,snr_db=10)
  rr.update({m:one(summary,method=a,snr_db=10,metric=m)['mean']for m in ['psnr_db','lpips_alex','clip_image_cosine']});tii.append(rr)
 write(P/'tables/table_ii.csv',tii)
 text=r'''\begin{table}[t]
\caption{Auxiliary quality on common500 at $N=1024$, 10 dB. Means include three realizations per source and gray failures. Policy-specific timing is reported separately in S7.}
\label{tab:external}\centering\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrr@{}}\toprule
Method & PSNR (dB)$\uparrow$ & LPIPS$\downarrow$ & CLIP-I$\uparrow$\\\midrule
'''
 for r in tii:
  if r['method']=='ARPC':continue
  text+=' & '.join([labels[r['method']],f"{float(r['psnr_db']):.2f}",f"{float(r['lpips_alex']):.3f}",f"{float(r['clip_image_cosine']):.3f}"])+r'\\'+'\n'
 text+=r'\bottomrule\end{tabular}\end{table}'+'\n';(P/'tables/table_ii.tex').write_text(text,encoding='utf8')
 new=json.loads((U/'frozen_policies.json').read_text())['policies'];old=json.loads((U/'historical_policy_metadata.json').read_text())['policies'];policy=[]
 old512=[dict(v,method='NeST'if v['method']=='NEST'else v['method'])for v in old if v['N']==512]
 ada512=json.loads((U/'historical_n512_policies.json').read_text())
 for v in ada512['selection']:
  if v['method']!='ADA':continue
  z=re.fullmatch(r'ADA_m(\d+)_K(\d+)_q(\d+)_r(\d+)-(\d+)',v['selected']['candidate_id']);assert z
  old512.append(dict(method='Ada',snr_db=v['snr_db'],target=[int(z[1]),int(z[2])],q=int(z[3]),rate=z[4]+'/'+z[5]))
 assert len(old512)==9
 for n,snrs in [(1024,[4,7,10,19]),(512,[4,10,19])]:
  for snr in snrs:
   rr=dict(N=n,snr_db=snr)
   for method,key in [('WHOLE','whole'),('NeST','nest'),('Ada','ada')]:
    v=new[str(snr)][method]if n==1024 else one(old512,method=method,snr_db=snr)
    target=v['target'];rr.update({key+'_q':v['q'],key+'_rate':v['rate'],key+'_m':target[0],key+'_K':target[1]})
   policy.append(rr)
 write(P/'tables/table_iv.csv',policy)
 text=r'''\begin{table}[t]
\caption{Frozen policies: MCS $(q,r)$ (bits/symbol, code rate) and target ceiling $e$, with feasible fallback. Ada's ceiling is 256 tokens. Budgets are calibrated independently.}
\label{tab:policies}\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccc@{}}\toprule
$N$/SNR & $a_{\rm W}$ & $e_{\rm W}$ & $a_{\rm N}$ & $e_{\rm N}$ & $a_{\rm Ada}$\\\midrule
'''
 for r in policy:
  a=lambda k:'$'+str(r[k+'_q'])+','+str(r[k+'_rate'])+'$'
  e=lambda k:'$('+str(r[k+'_m'])+','+str(r[k+'_K'])+')$'
  text+=' & '.join([str(r['N'])+'/'+str(r['snr_db']),a('whole'),e('whole'),a('nest'),e('nest'),a('ada')])+r'\\'+'\n'
 text+=r'\bottomrule\end{tabular}\end{table}'+'\n';(P/'tables/table_iv.tex').write_text(text,encoding='utf8')
 (P/'validation/unified_table_data.json').write_text(json.dumps(dict(fig2_updated_records=changes,retained_equal_budget_records=36,table_i_B_C_unchanged=True,table_ii_timing_removed_for_configuration_separation=True,policy_rows=policy,inputs={str(p.relative_to(P)):h(p)for p in [U/'external_join/summary.csv',U/'external_join/paired_deltas.csv',U/'formal/capacity_summary.csv',U/'frozen_policies.json']}),indent=2))
 print(dict(fig2_points=92,updated_points=24,policy_rows=len(policy),table_ii_quality_only=True))

if __name__=='__main__':main()
