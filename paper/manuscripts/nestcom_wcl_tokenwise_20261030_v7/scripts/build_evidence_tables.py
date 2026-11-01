"""Generate manuscript tables from exact delivered CSVs, without recomputing CI.

Pending timing is explicit in the draft. Publication validation must reject it.
"""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parents[1]
def rows(name):
 with (P/'inputs'/name).open(newline='',encoding='utf8')as f:return list(csv.DictReader(f))
def pick(data,**keys):
 found=[v for v in data if all(str(v.get(k))==str(x)for k,x in keys.items())]
 assert len(found)==1,(keys,len(found));return found[0]
def fmt(x,n=3):return f'{float(x):.{n}f}'
def ci(v,n=4):return f"{float(v['mean_delta']):+.{n}f} [{float(v['ci95_low']):.{n}f}, {float(v['ci95_high']):.{n}f}]"
def texci(v,n=4):
 return r'\shortstack{'+f"{float(v['mean_delta']):+.{n}f}"+r'\\{}'+f"[{float(v['ci95_low']):.{n}f}, {float(v['ci95_high']):.{n}f}]"+'}'
def savecsv(name,data):
 fields=list(dict.fromkeys(k for v in data for k in v))
 with (P/'tables'/name).open('w',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def savetex(name,text):(P/'tables'/name).write_text(text+'\n',encoding='utf8',newline='\n')
S=rows('e1/summary.csv');Delta=rows('e1/paired_deltas.csv');Cap=rows('e1/capacity_summary.csv')
S3=rows('e3/summary.csv');D3=rows('e3/paired_deltas.csv');Cap3=rows('e3/efficiency.csv');D4=rows('e4/paired_deltas.csv')
def val(method,N,snr,metric):return pick(S,method=method,N=N,snr_db=snr,metric=metric)
def diff(ref,N,snr,metric,cand='final_NeST'):return pick(Delta,candidate=cand,reference=ref,N=N,snr_db=snr,metric=metric)
def cap(method,N,snr):return pick(Cap,method=method,N=N,snr_db=snr)
labels={'final_NeST':'NeST-Com','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada (adapted)','Latent_JSCC':'Latent JSCC','SwinJSCC':'SwinJSCC','BPG_LDPC':'BPG + LDPC','ARPC':'ARPC + LDPC'}
I=[];body=[]
body += [r'\multicolumn{4}{l}{\textit{A. common500, $N=1024$, family-specific policies}}\\',r'SNR & Util. W$\to$N (\%) & $\Delta$DISTS & $\Delta$DreamSim\\',r'\midrule']
for snr in [4,7,10,19]:
 c1=cap('WHOLE_final_receiver',1024,snr);c2=cap('final_NeST',1024,snr);d=diff('WHOLE_final_receiver',1024,snr,'dists');ds=diff('WHOLE_final_receiver',1024,snr,'dreamsim')
 body.append(f"{snr} & {100*float(c1['utilization_mean']):.2f}$\\to${100*float(c2['utilization_mean']):.2f} & {texci(d)} & {texci(ds)}"+r'\\[2pt]')
 I.append(dict(block='A',N=1024,snr_db=snr,n_sources=500,whole_utilization=float(c1['utilization_mean']),nest_utilization=float(c2['utilization_mean']),dists_delta=d['mean_delta'],dists_low=d['ci95_low'],dists_high=d['ci95_high'],dreamsim_delta=ds['mean_delta'],dreamsim_low=ds['ci95_low'],dreamsim_high=ds['ci95_high']))
body += [r'\midrule',r'\multicolumn{4}{l}{\textit{B. control100, 10 dB, common ceiling $(8,126)$}}\\',r'Grid (ends) & Utilization (\%) & DISTS & DreamSim\\',r'\midrule']
for m in ['whole','quarter','eighth','token']:
 c=pick(Cap3,method=m);d=pick(S3,method=m,metric='dists');ds=pick(S3,method=m,metric='dreamsim')
 body.append(f"{m.title()} ({int(float(c['candidate_count']))}) & {float(c['utilization'])*100:.2f} & {fmt(d['mean'],4)} & {fmt(ds['mean'],4)}"+r'\\')
 I.append(dict(block='B',N=1024,snr_db=10,n_sources=100,grid=m,candidate_count=c['candidate_count'],utilization=c['utilization'],dists=d['mean'],dreamsim=ds['mean'],length_comparisons=c['actual_length_comparisons']))
body += [r'\midrule',r'\multicolumn{4}{l}{\textit{C. control100, 10 dB, same received prefixes}}\\',r'Intervention & $\Delta$PSNR (dB) & $\Delta$DISTS & $\Delta$DreamSim\\',r'\midrule']
for a,b in [('B0','A0'),('C0','B0')]:
 vs={m:pick(D4,candidate=a,reference=b,snr_db=10,metric=m)for m in ['psnr_db','dists','dreamsim']}
 body.append(f"{a}--{b} & {texci(vs['psnr_db'],3)} & {texci(vs['dists'])} & {texci(vs['dreamsim'])}"+r'\\[2pt]')
 I.append(dict(block='C',N=1024,snr_db=10,n_sources=100,candidate=a,reference=b,**{m+'_'+k:v[k]for m,v in vs.items()for k in ['mean_delta','ci95_low','ci95_high']}))
capI='Within-scale transmission evidence. (A) NeST minus WHOLE under the frozen family policies. (B) Only endpoint granularity changes under matched model, MCS, ceiling, and noise. (C) Ordered receiver interventions on the actual NeST prefixes. Brackets are source-paired 95\\% intervals; each block uses its stated population.'
savetex('table_i.tex',r'\begin{table}[t]'+'\n'+r'\caption{'+capI+'}\n'+r'\label{tab:internal}\centering\footnotesize'+'\n'+r'\setlength{\tabcolsep}{2.1pt}\renewcommand{\arraystretch}{1.05}'+'\n'+r'\begin{tabular}{@{}lccc@{}}\toprule'+'\n'+'\n'.join(body)+'\n'+r'\bottomrule\end{tabular}\end{table}')
savecsv('table_i.csv',I)
II=[];body=[r'Method & PSNR & LPIPS & CLIP-I & TX & RX\\',r' & (dB)$\uparrow$ & $\downarrow$ & $\uparrow$ & (ms) & (ms)\\',r'\midrule']
timing=rows('e5/timing_summary.csv')if(P/'inputs/e5/timing_summary.csv').exists()else None
timing_groups={'WHOLE_final_receiver':'WHOLE_N1024_S10','final_NeST':'Token_NeST_N1024_S10',
 'Ada_frozen':'Ada_N1024_S10','Latent_JSCC':'Latent_JSCC_N1024_S10','SwinJSCC':'SwinJSCC_N1024_S10',
 'BPG_LDPC':'BPG_LDPC_N1024_S10','ARPC':'ARPC_N1024_S10'}
if timing is not None:
 assert len(timing)==48 and len({(v['group'],v['direction'])for v in timing})==48
 closure=json.loads((P/'inputs/e5/execution_closure.json').read_text())
 assert closure['actual_wait_returncode']==0 and closure['owner_gone'] and closure['child_gone']
 assert closure['summary_sha256']==hashlib.sha256((P/'inputs/e5/timing_summary.csv').read_bytes()).hexdigest()
for method in ['WHOLE_final_receiver','final_NeST','Ada_frozen','Latent_JSCC','SwinJSCC','BPG_LDPC','ARPC']:
 v={m:val(method,1024,10,m)for m in ['psnr_db','lpips_alex','clip_image_cosine']}
 costs={}
 if timing is not None:
  for direction in ['TX','RX']:
   t=pick(timing,group=timing_groups[method],direction=direction)
   assert (int(t['sources']),int(t['measured']),int(t['warmup']))==(32,96,64)
   costs[direction]=1000*float(t['equal_source_mean_seconds'])
 tail=' & '.join(f'{costs[d]:.1f}'if costs else '--' for d in ['TX','RX'])
 body.append(labels[method]+' & '+fmt(v['psnr_db']['mean'],2)+' & '+fmt(v['lpips_alex']['mean'])+' & '+fmt(v['clip_image_cosine']['mean'])+' & '+tail+r'\\')
 II.append(dict(method=method,N=1024,snr_db=10,**{m:x['mean']for m,x in v.items()},timing_status='ACTUAL_COMPLETE'if costs else 'PENDING_ACTUAL_E5',tx_ms=costs.get('TX',''),rx_ms=costs.get('RX','')))
draft_note='' if timing else r' \textbf{Draft: timing not yet delivered.}'
savetex('table_ii.tex',r'\begin{table}[t]'+'\n'+r'\caption{Auxiliary quality and processing costs at $N=1024$, 10 dB. Quality uses common500; timing uses 32 sources, one RTX PRO 6000, two CPU cores, batch one, and complete TX/RX processing boundaries. BPG uses CPU only.}'+draft_note+'\n'+r'\label{tab:external}\centering\footnotesize\setlength{\tabcolsep}{2.5pt}'+'\n'+r'\begin{tabular}{@{}lrrrrr@{}}\toprule'+'\n'+'\n'.join(body)+'\n'+r'\bottomrule\end{tabular}\end{table}')
savecsv('table_ii.csv',II)
III=[];body=[r'SNR / method & PSNR$\uparrow$ & DISTS$\downarrow$ & DreamSim$\downarrow$\\',r'\midrule']
for snr in [4,10,19]:
 for method in ['WHOLE_final_receiver','final_NeST','Ada_frozen']:
  vals={m:val(method,512,snr,m)for m in ['psnr_db','lpips_alex','dreamsim','clip_image_cosine','dists']}
  body.append(f"{snr} / {labels[method]} & {fmt(vals['psnr_db']['mean'],2)} & {fmt(vals['dists']['mean'],4)} & {fmt(vals['dreamsim']['mean'],4)}"+r'\\')
  III.append(dict(method=method,N=512,snr_db=snr,**{m:x['mean']for m,x in vals.items()}))
 d=diff('WHOLE_final_receiver',512,snr,'psnr_db')
 body += [r'\multicolumn{4}{l}{NeST--WHOLE $\Delta$PSNR: '+ci(d,3)+r' dB}\\',r'\midrule']
savetex('table_iii.tex',r'\begin{table}[t]'+'\n'+r'\caption{Reduced-budget reconstruction on common500 at $N=512$. Means include all failures; PSNR differences use direct source-paired 95\% intervals. All five metrics and paired comparisons are in Supplement~S2.}'+'\n'+r'\label{tab:reduced}\centering\footnotesize\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabular}{@{}lrrr@{}}\toprule'+'\n'+'\n'.join(body[:-1])+'\n'+r'\bottomrule\end{tabular}\end{table}')
savecsv('table_iii.csv',III)
record=dict(source_hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest()for p in (P/'inputs').rglob('*.csv')},table_rows={'I':len(I),'II':len(II),'III':len(III)},new_statistics=0,full_precision_csv=True,timing_status='ACTUAL_COMPLETE'if timing else 'PENDING',publication_ready=timing is not None)
(P/'validation/table_data_checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record['table_rows']))
