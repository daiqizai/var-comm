"""Deterministic display formatting only; no recomputed statistics."""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parents[1]
def rows(name):return list(csv.DictReader((P/'inputs'/name).open()))
def one(data,**kw):
 v=[r for r in data if all(r[k]==str(x) for k,x in kw.items())];assert len(v)==1,(kw,len(v));return v[0]
def write_csv(name,rs):
 with(P/'tables'/name).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
data=rows('common500_summary.csv')
for r in data:r['method']=aliases.get(r['method'],r['method'])
assert len(data)==72 and len({(r['method'],r['snr_db'],r['metric'])for r in data})==72
style=json.loads((P/'style.json').read_text())['methods']
table2=[];tex=[r'\begin{table}[t]',r'\caption{Perceptual distance and image-feature similarity on common500. Each cell is LPIPS-Alex $\downarrow$ / CLIP-I $\uparrow$.}',r'\label{tab:external}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lccc}',r'\toprule Method & 4 dB & 10 dB & 19 dB \\',r'\midrule']
for method,s in style.items():
 cells=[]
 for snr in (4,10,19):
  lp=one(data,method=method,snr_db=snr,metric='lpips_alex');clip=one(data,method=method,snr_db=snr,metric='clip_image_cosine')
  table2.append(dict(method=method,snr_db=snr,lpips_alex=lp['mean'],lpips_ci_low=lp['ci_low'],lpips_ci_high=lp['ci_high'],clip_image_cosine=clip['mean'],clip_ci_low=clip['ci_low'],clip_ci_high=clip['ci_high']))
  cells.append(f"{float(lp['mean']):.3f} / {float(clip['mean']):.3f}")
 label='Ada (adapted)' if method=='ADA_PREDCLASS_FIXEDFRAME'else s['label']
 tex.append(label+' & '+' & '.join(cells)+r' \\')
tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'tables/table_ii.tex').write_text('\n'.join(tex)+'\n');write_csv('table_ii.csv',table2)
tex=[r'\begin{table}[t]',r'\caption{Within-scale endpoint evidence. (A) Main family-specific policies; $\Delta=\mathrm{PARTIAL}-\mathrm{WHOLE}$ with source-paired 95\% CIs. (B) Controlled 100-source diagnostic at 10 dB.}',r'\label{tab:internal}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{ccc}',r'\toprule',r'\multicolumn{3}{l}{(A) Main policies: utilization and paired PSNR gain} \\',r'SNR & Utilization W$\to$P (\%) & $\Delta$PSNR (dB) [95\% CI] \\',r'\midrule']
records=[]
for pop,n in (('common500',500),('kodak',24)):
 util=rows('internal_'+pop+'_utilization.csv');paired=rows('internal_'+pop+'_paired.csv')
 tex.append(r'\multicolumn{3}{l}{'+('common500'if n==500 else'Kodak24')+r'} \\')
 for snr in(4,10,19):
  u=[one(util,family=f,snr_db=snr)['mean_utilization']for f in('WHOLE','PARTIAL')]
  delta=one(paired,snr_db=snr,metric='psnr_db',**({'comparison':'PARTIAL_minus_WHOLE'}if n==500 else{'method':'NEST_PARTIAL','reference':'NEST_WHOLE'}))
  records.append(dict(block='A',population=pop,source_count=n,snr_db=snr,grid='PARTIAL_minus_WHOLE',candidate_count='',whole_utilization=u[0],partial_utilization=u[1],psnr_mean=delta['mean'],ci_low=delta['ci_low'],ci_high=delta['ci_high']))
  tex.append(f"{snr} & {100*float(u[0]):.2f} $\\to$ {100*float(u[1]):.2f} & $+{float(delta['mean']):.3f}\\;[{float(delta['ci_low']):.3f},{float(delta['ci_high']):.3f}]$"+r' \\')
tex +=[r'\midrule',r'\multicolumn{3}{l}{(B) Fixed model, MCS, maximum endpoint and fitting} \\',r'Grid (candidates) & Utilization (\%) & PSNR (dB) \\']
ef=rows('a_efficiency.csv');su=rows('a_summary.csv')
for arm in('WHOLE','HALF','QUARTER','EIGHTH'):
 a=one(ef,snr_db=10,arm=arm);s=one(su,snr_db=10,arm=arm,metric='psnr_db')
 records.append(dict(block='B',population='controlled100',source_count=100,snr_db=10,grid=arm,candidate_count=a['candidates'],whole_utilization='',partial_utilization=a['mean_arithmetic_utilization'],psnr_mean=s['mean'],ci_low=s['ci_low'],ci_high=s['ci_high']))
 tex.append(f"{arm.title()} ({a['candidates']}) & {100*float(a['mean_arithmetic_utilization']):.2f} & {float(s['mean']):.3f}"+r' \\')
tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'tables/table_i.tex').write_text('\n'.join(tex)+'\n');write_csv('table_i.csv',records)
print('Tables: 72 frozen metric rows validated; 18 two-metric cells; 6 main paired rows + 4 diagnostic rows.')
