"""Generate all three tables from frozen, fully identified CSV inputs."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((P/'inputs'/n).open(encoding='utf8')))
def one(rows,**keys):
 a=[r for r in rows if all(str(r[k])==str(v)for k,v in keys.items())];assert len(a)==1,(keys,len(a));return a[0]
def ci(r,n=3,sign=True):
 m=float(r['mean']);l=float(r['ci_low']);h=float(r['ci_high'])
 return f'{m:+.{n}f}'if sign else f'{m:.{n}f}',f'[{l:.{n}f},{h:.{n}f}]'
def writecsv(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with(P/'tables'/name).open('w',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
rows=[];a=[];kodak_rows=[];kodak_tex=[]
for pop,pairfile,utilfile in [('common500','internal_common500_paired.csv','internal_common500_utilization.csv'),('Kodak24','internal_kodak_paired.csv','internal_kodak_utilization.csv')]:
 pairs=read(pairfile);util=read(utilfile)
 target_tex,target_rows=(a,rows) if pop=='common500' else (kodak_tex,kodak_rows)
 for snr in (4,10,19):
  r=one(pairs,snr_db=snr,metric='lpips_alex',**({'comparison':'PARTIAL_minus_WHOLE'}if pop=='common500'else{'method':'NEST_PARTIAL','reference':'NEST_WHOLE'}));w=float(one(util,family='WHOLE',snr_db=snr)['mean_utilization'])*100;p=float(one(util,family='PARTIAL',snr_db=snr)['mean_utilization'])*100
  m,b=ci(r,4);target_tex.append(f'{snr} & {w:.2f} $\\to$ {p:.2f} & ${m}\\;{b}$ '+r'\\');target_rows.append(dict(block='A',population=pop,snr_db=snr,comparison='PARTIAL minus WHOLE',metric='lpips_alex',mean=float(r['mean']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high']),utilization_W_percent=w,utilization_P_percent=p,sources=500 if pop=='common500' else 24))
eff=read('a_efficiency.csv');summ=read('a_summary.csv');b=[]
for arm in ['WHOLE','HALF','QUARTER','EIGHTH']:
 e=one(eff,arm=arm,snr_db=10);r=one(summ,arm=arm,snr_db=10,metric='lpips_alex');u=100*float(e['mean_arithmetic_utilization']);m=float(r['mean']);b.append(f"{arm.title()} ({e['candidates']}) & {u:.2f} & {m:.4f} "+r'\\');rows.append(dict(block='B',population='external_common100_v1',snr_db=10,arm=arm,metric='lpips_alex',mean=m,ci_low=float(r['ci_low']),ci_high=float(r['ci_high']),utilization_percent=u,candidates=int(e['candidates']),mean_length_checks=float(e['mean_descending_length_checks']),sources=100))
inter=read('b_paired.csv');c=[]
for snr in (10,19):
 vals=[];bounds=[]
 for comp in ['B0_minus_A0','C0_minus_B0']:
  r=one(inter,snr_db=snr,group='all',comparison=comp,metric='lpips_alex');m,br=ci(r,4);vals.append('$'+m+'$');bounds.append('$'+br+'$');rows.append(dict(block='C',population='external_common100_v1',snr_db=snr,comparison=comp,metric='lpips_alex',mean=float(r['mean']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high']),sources=100))
 c.append(f'{snr} & '+ ' & '.join(vals)+r' \\')
 c.append(' & '+ ' & '.join(bounds)+r' \\')
caption=r'Within-scale endpoint and receiver evidence. (A) Main common500 policies, PARTIAL minus WHOLE. (B) Fixed 100-source model/MCS/maximum-endpoint control. (C) Ordered interventions on the same 100 sources and actual prefixes. Brackets give source-paired 95\% CIs.'
t=r'''\begin{table}[t]
\caption{CAPTION}
\label{tab:internal}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{ccc}
\toprule
\multicolumn{3}{l}{(A) Main policies at $N=1024$} \\
SNR & Util. W$\to$P (\%) & $\Delta$LPIPS [95\% CI] \\
\midrule
ROWSA
\midrule
\multicolumn{3}{l}{(B) Fixed-granularity diagnostic, 10 dB} \\
Grid (candidates) & Utilization (\%) & LPIPS \\
ROWSB
\midrule
\multicolumn{3}{l}{(C) Receiver interventions: $\Delta$LPIPS [95\% CI]} \\
SNR & B0$-$A0 & C0$-$B0 \\
ROWSC
\bottomrule
\end{tabular}
\par\smallskip\parbox{\columnwidth}{\footnotesize Boundary tokens: A0, unused; B0, residual correction only; C0, residual correction and conditioning update. All three use the received complete coarse scales.}
\end{table}
'''.replace('CAPTION',caption).replace('ROWSA','\n'.join(a)).replace('ROWSB','\n'.join(b)).replace('ROWSC','\n'.join(c))
(P/'tables/table_i.tex').write_text(t,encoding='utf8');writecsv('table_i.csv',rows)

k=r'''\clearpage\subsection*{S2.5. Transfer to Kodak24}
\label{sec:kodak_transfer}
ImageNet-calibrated policies are applied unchanged to the 24 Kodak sources at $N=1024$. Differences are PARTIAL minus WHOLE. Each source is averaged over three channel outcomes before the direct source-paired 95\% confidence interval is computed. Utilization is the source-bit capacity fraction for WHOLE (W) and PARTIAL (P).
\begin{center}
\small
\begin{tabular}{ccc}
\toprule
SNR (dB) & Util. W$\to$P (\%) & $\Delta$LPIPS [95\% CI] \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{center}
'''.replace('ROWS','\n'.join(kodak_tex))
(P/'sections/supplement_kodak_v4.tex').write_text(k,encoding='utf8')
writecsv('supplement_kodak_v4.csv',kodak_rows)
data=read('common500_summary.csv');aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
for r in data:r['method']=aliases.get(r['method'],r['method'])
style=json.loads((P/'style.json').read_text());rows=[];tex=[]
for method,s in style['methods'].items():
 label=s['label'].replace('Ada-TokenCom','Ada');cells=[]
 for snr in (4,10,19):
  v={m:float(one(data,method=method,snr_db=snr,metric=m)['mean'])for m in ['psnr_db','clip_image_cosine']};cells.append(f"{v['psnr_db']:.3f} / {v['clip_image_cosine']:.3f}");rows.append(dict(method=method,label=label,N=1024,snr_db=snr,**v))
 tex.append(label+' & '+' & '.join(cells)+r' \\')
t=r'''\begin{table}[t]
\caption{Pixel fidelity and image-feature similarity on common500 at $N=1024$. Each cell is PSNR (dB)$\uparrow$ / CLIP-I$\uparrow$. Ada denotes the fixed-frame adaptation. Values are source means.}
\label{tab:external}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule Method & 4 dB & 10 dB & 19 dB \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
'''.replace('ROWS','\n'.join(tex));(P/'tables/table_ii.tex').write_text(t,encoding='utf8');writecsv('table_ii.csv',rows)
data=read('paired_deltas.csv');rows=[];tex=[]
for block,comparison,label in [('A','NEST_minus_WHOLE','NeST-Com minus WHOLE'),('B','NEST_minus_ADA','NeST-Com minus Ada (adapted)')]:
 if block=='B':tex.append(r'\midrule')
 tex.append(r'\multicolumn{3}{l}{('+block+') '+label+r'} \\')
 for snr in (4,10,19):
  r=one(data,comparison=comparison,snr_db=snr,metric='lpips_alex');s=one(data,comparison=comparison,snr_db=snr,metric='dreamsim');a,ac=ci(r,4);b,bc=ci(s,4)
  tex+=[f'{snr} & ${a}$ & ${b}$ '+r'\\',f' & ${ac}$ & ${bc}$ '+r'\\'];rows.extend([dict(N=512,**r),dict(N=512,**s)])
t=r'''\begin{table}[t]
\caption{Reduced-budget perceptual differences on common500 at $N=512$. (A) NeST-Com minus WHOLE with shared models. (B) NeST-Com minus the fixed-frame Ada-TokenCom adaptation. Brackets give direct source-paired 95\% CIs; negative values favor NeST-Com.}
\label{tab:reduced}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ccc}
\toprule SNR & $\Delta$LPIPS & $\Delta$DreamSim \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
'''.replace('ROWS','\n'.join(tex)).replace(r'\end{table}',r'''\par\smallskip\begin{minipage}{\columnwidth}\footnotesize
Successful receptions (4/10/19~dB), each out of 1500: NeST 1496/1499/1500; WHOLE 1493/1500/1500; Ada 1495/1500/1500. All failures are body-CRC rejections retained as gray.
\end{minipage}
\end{table}''');(P/'tables/table_iii.tex').write_text(t,encoding='utf8');writecsv('table_iii.csv',rows)
print('Generated Table I/II/III and Kodak24 supplement from identified CSV data')

