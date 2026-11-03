"""Build the concise S1--S8 reading edition from immutable, closed result files.

This is presentation only: no model, channel, metric or bootstrap execution.
Every rendered numerical cell carries a CSV/row/field pointer in the companion.
"""
from pathlib import Path
import csv,hashlib,json,re

METRICS=['psnr_db','dists','dreamsim','lpips_alex','clip_image_cosine']
HEADERS=[r'PSNR $\uparrow$',r'DISTS $\downarrow$',r'DreamSim $\downarrow$',r'LPIPS $\downarrow$',r'CLIP-I $\uparrow$']
NAMES={'final_NeST':'NeST','original_NeST':'Quarter','WHOLE_final_receiver':'WHOLE',
       'Ada_frozen':'Ada','Token_NeST':'NeST','Token':'NeST','BPG_LDPC':'BPG',
       'Latent_JSCC':'Latent','SwinJSCC':'Swin','whole':'Whole','quarter':'Quarter',
       'eighth':'Eighth','token':'Token'}

def apply_reader_revision(full,P):
 P=Path(P);out=P/'reproduction';out.mkdir(exist_ok=True)
 archived=out/'technical_record_20261103_full.tex'
 # The prior published edition is retained as a technical record, not regenerated.
 assert archived.exists(), 'Create the immutable previous-edition archive before authoring.'
 s=archived.read_text(encoding='utf8')
 ledger=[];sources={}
 def rows(rel):
  p=P/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
  rr=list(csv.DictReader(p.open(encoding='utf8')))
  return [dict(r,_file=rel,_row=i+2)for i,r in enumerate(rr)]
 def plain(x):return str(x).replace('_',r'\_').replace('%',r'\%')
 def name(x):return NAMES.get(x,x)
 def fmt(x,metric='',signed=False):
  n=3 if metric=='psnr_db'else 4
  z=float(x)
  while z!=0 and round(z,n)==0 and n<9:n+=1
  return f'{z:+.{n}f}'if signed else f'{z:.{n}f}'
 def cell(r,ci=False,signed=False):
  mf='mean_delta'if'mean_delta'in r else'mean'
  lf='ci95_low'if'ci95_low'in r else'ci_low';hf='ci95_high'if'ci95_high'in r else'ci_high'
  a=fmt(r[mf],r['metric'],signed)
  val=(r'\cival{'+a+'}{'+fmt(r[lf],r['metric'])+'}{'+fmt(r[hf],r['metric'])+'}')if ci else a
  ledger.append(dict(file=r['_file'],csv_row=r['_row'],metric=r['metric'],fields={k:r[k]for k in([mf,lf,hf]if ci else[mf])},rendered=val))
  return val
 def table(headers,rr,spec=None,title='',groups=None):
  spec=spec or ('l'+'r'*(len(headers)-1))
  return ('\n'+(r'\textbf{'+title+r'}\par'+'\n'if title else'')+
   r'{\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}'+'\n'+
   r'\begin{longtable}{@{}'+spec+r'@{}}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+
   '\n'.join(' & '.join(map(str,r))+(r'\\*'if groups and i+1<len(rr)and groups[i]==groups[i+1]else r'\\')for i,r in enumerate(rr))+'\n'+r'\bottomrule\end{longtable}}'+'\n')
 def pivot(rr,keys,headers,title='',ci=False,signed=False,metrics=None):
  metrics=metrics or METRICS;groups={}
  for r in rr:
   if r['metric']not in metrics:continue
   k=tuple(r[z]for z in keys);assert r['metric']not in groups.setdefault(k,{}),(k,r['metric'])
   groups[k][r['metric']]=r
  def order(t):return tuple(float(z)if re.fullmatch(r'[0-9.]+',z)else z for z in t)
  data=[];blocks=[]
  for k,g in sorted(groups.items(),key=lambda t:order(t[0])):
   assert set(g)==set(metrics),(title,k,set(g))
   data.append([plain(name(x))for x in k]+[cell(g[m],ci,signed)for m in metrics])
   blocks.append(k[:-1]if keys[-1]in ['method','comparison']else k)
  cols=('l'*len(keys))+('r'*len(metrics))
  return table(headers+[HEADERS[METRICS.index(m)]for m in metrics],data,cols,title,blocks)
 def section(n):return s[s.index(r'\section{',s.index(r'\label{sup:'+str(n)+'}')-70):s.index(r'\section{',s.index(r'\label{sup:'+str(n+1)+'}')-70)]if n<8 else s[s.index(r'\section{Transfer to Kodak24}'):s.index(r'\end{document}')]
 def old_tables(block):return re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}',block,re.S)
 def para(block,start):
  a=block.index(start);b=block.find('\n\n',a);return block[a:b if b!=-1 else len(block)]

 pre=s[:s.index(r'\section{Reproduction configuration}')]
 pre=pre.replace('Implementation recipe, complete results, paired statistics and source examples. Numerical tables use the same frozen CSV sources as the main manuscript.',
   'Algorithms, matched controls and additional results. Full numerical precision, all intervals and technical records accompany this reading edition in the reproduction files.')
 pre=pre.replace('S2 Complete quality results','S2 System-level results')
 pre=pre.replace(r'\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,url}',r'\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,url,array,needspace,placeins}')
 pre=r'\RequirePackage{fix-cm}'+'\n'+pre
 pre=pre.replace(r'\begin{document}',r'\newcommand{\cival}[3]{\shortstack[r]{#1\\{\fontsize{8.5}{10}\selectfont[#2, #3]}}}'+'\n'+r'\begin{document}')
 s1=section(1)
 s1=s1.replace(r'\section{Reproduction configuration}',r'\section{Algorithms and protocol}')
 a=s1.index(r'\textbf{Reading map.}');b=s1.index(r'\subsection{Representation',a)
 s1=s1[:a]+r'''\textbf{Reading map.} S1.1--S1.4 specify the model, coding, frame and calibration; S1.5 gives the fitting algorithm and length proof. S1.6 states verification and statistical assumptions. The companion \path{REPRODUCTION_CHECKLIST.md} links complete configurations, identities, commands and numerical tables. It separates publicly available assets from checkpoints retained in project storage.

'''+s1[b:]
 s1=s1.replace(para(s1,'Public VAR/LlamaGen/ConvNeXt/ARPC sources,'),'')
 s1=s1.replace('The adapted decoder checkpoint is update0038000. It minimizes','The image decoder minimizes')
 s1=s1.replace('The full recipe and its repository path are indexed in the companion checklist.','The selected decoder is frozen for evaluation.')
 s1=s1.replace('Exact software versions, device identities and distinct file/tensor-state hashes are provided in the companion records. ','')
 s1=s1.replace('The public LlamaGen weight URLs and revision are included in the manifest. ','')
 s1=s1.replace('Directory qualification performs two real complete-frame checks per directory (source and NULL frames), totaling 72 frames and 144 header/body decodes. ','')
 old=para(s1,'Profile CRC/catalogue, body CRC, invalid length,')
 new=r'''Profile/header CRC, body CRC, invalid length, padding or arithmetic-canonical failures produce gray RGB 0.5. A valid NULL frame also produces gray and retains the full statistical denominator. If no source prefix fits, TX sends a NULL frame with a two-bit zero payload using the full $N$-symbol budget; its directory entry is the NULL profile. Current directories have seven WHOLE, 651 NeST or 256 Ada source profiles, plus one NULL profile each. Every ID fits in 12 bits without changing the header. Each method/MCS directory is fixed before calibration and shared across all images and target ceilings. Both terminals know the MCS and directory before transmission; the received Profile ID selects only the endpoint. Changing policy requires configuring both terminals before data transmission. Earlier shared catalogues also stored MCS parameters in each profile entry; their identities and experiment associations are distinguished below.'''
 s1=s1.replace(old,new)
 s1=s1.replace('Its seed is derived from the first 16 SHA256 bytes, little-endian, of the canonical protocol/"scramble"/session/counter tuple. Exact mask and mapping definitions are in \\texttt{ep\\_phy.py}; interleaver hashes and constructor metadata are in \\texttt{phy\\_layouts.csv}.','The public seed and exact mapping are specified in the companion protocol.')
 a=s1.index('U denotes the public per-method/MCS directory');b=s1.index(r'\begin{longtable}',a)
 s1=s1[:a]+r'''U denotes the current per-method/MCS directory. HW/HT/HA, H7 and LW/LT identify the earlier endpoint/MCS catalogues; KT adds NULL entries without changing source entries. Exact directory identities are in the companion policy index.
'''+s1[b:]
 s1=s1.replace('The complete 1584 pilot candidate/SNR scores and all confirmation scores, with the public directory and policy hashes, are in \\path{inputs/unified_calibration_20261103/}. ','')
 s1=s1.replace(', yielding 18000 logical events','')
 s1=s1.replace(para(s1,'The pilot and confirmation contribute 158400'),'')
 s1=s1.replace('Supplementary 7-dB/10-dB examples (S6)','Additional image examples')
 s1=s1.replace('Historical outputs; main Fig.3 uses U10','S6/companion; main Fig.3 uses U10')
 s1=s1.replace('Historical timing (S7)','Historical timing')
 s1=s1.replace('Earlier resource protocol, labeled in S7','Companion only; separate SMT protocol')
 s1=s1.replace('Actual-length fitting and scan accounting','Actual-length fitting and length property')
 s1=s1.replace(r'\subsection{Actual-length fitting',r'\Needspace{20\baselineskip}\subsection{Actual-length fitting')
 a=s1.index('We checked the exact original integer coder');b=s1.index(r'\subsection{Reference vectors',a)
 s1=s1[:a]+r'''Integer reference checks and the existing source streams verify this coder-specific identity and non-mutating finalization. The measured implementation still constructs and scans all allowed candidates; candidate comparisons do not add prior forward passes. The reported timings include this full scan.

'''+s1[b:]
 a=s1.index(r'\subsection{Reference vectors');b=s1.index('Common500 uses',a)
 vectors=old_tables(s1[a:b])[0]
 s1=s1[:a]+r'''\subsection{Verification and statistical units}
Integer reference vectors verify positive frequencies, frequency totals, deterministic ties, independent termination/decoding and rejection of noncanonical appended bits. Two real-model vectors check independently reconstructed CDFs and recovered tokens:
'''+vectors+r'''
RX uses no TX probability table, unsent token or hidden source state. These checks establish correct decoding under the specified numerical route; heterogeneous-device CDF interchangeability is not claimed. Commands and complete verification records are in the companion.

'''+s1[b:]
 # Compact the two policy/index tables without dropping configurations.
 s1=s1.replace(r'\begin{longtable}',r'{\small\setlength{\tabcolsep}{3pt}\begin{longtable}').replace(r'\end{longtable}',r'\end{longtable}}')

 ext=rows('inputs/unified_calibration_20261103/external_join/summary.csv')
 old512=[r for r in rows('inputs/e1/summary.csv')if r['N']=='512']
 s2=r'''\section{System-level results}\label{sup:2}
The $N=1024$ WHOLE/NeST/Ada results use unified calibration. Latent JSCC, SwinJSCC, BPG and the ARPC digital adaptation retain their original outputs and configurations. All results use 500 test images and three channel realizations per image. The three calibrated token methods share noise; external methods retain their original noise identities. Pairing across systems is by source. $N=512$ is independently calibrated; Quarter denotes the earlier quarter-grid system, while NeST uses token-level fallback.

ARPC is a progressive source codec coupled here to the specified protected digital link, using its public 0.06M preset, CFG 3, temperature 0.5 and paid caption/container. This comparison concerns that adaptation. Lower DISTS, DreamSim and LPIPS, and higher PSNR and CLIP-I, are favorable. Means below retain every operating point; complete pointwise intervals and every external paired comparison remain in the companion CSVs.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/figure2_complete_with_arpc.pdf}
\caption{System-level and equal-information-budget comparisons, including the ARPC digital adaptation. All 92 points and pointwise 95\% source-bootstrap intervals are retained. Ada denotes the predicted-class, fixed-frame adaptation.}\label{fig:complete_comparison}\end{figure}
'''
 s2+=pivot(ext,['snr_db','method'],['SNR','Method'],'$N=1024$ source means (PSNR in dB).')
 s2+=r'\Needspace{18\baselineskip}'+pivot(old512,['snr_db','method'],['SNR','Method'],'$N=512$ source means (PSNR in dB).')
 s2+=r'''At $N=1024$, 10 dB, Ada has lower mean DISTS and DreamSim than NeST, while SwinJSCC and BPG have higher PSNR. NeST's gains over WHOLE and its metric-dependent differences from Ada are quantified below. ARPC's values describe the specified digital adaptation rather than its compression performance independent of the link.
'''
 pair=rows('inputs/unified_calibration_20261103/external_join/paired_deltas.csv')
 pair=[r for r in pair if r['comparison']in['NeST-WHOLE','NeST-Ada','NeST-ARPC','WHOLE-Ada']]
 for r in rows('inputs/e1/paired_deltas.csv'):
  if r['N']=='512'and name(r['reference'])!='Quarter':
   r['comparison']=name(r['candidate'])+'-'+name(r['reference']);pair.append(r)
 s2+=r'''\textbf{Direct paired comparisons.} Every cell below is the first method minus the second, with its 95\% source-paired interval. These contrasts retain both favorable and unfavorable results; differences of marginal intervals are not used.
'''+pivot(pair,['N','snr_db','comparison'],['$N$','SNR','Difference'],ci=True,signed=True)
 cap=rows('inputs/unified_calibration_20261103/formal/capacity_summary.csv')
 cr=[]
 for snr in ['4','7','10','19']:
  q={r['method']:r for r in cap if r['snr_db']==snr}
  cr.append([snr]+[f"{100*float(q[m]['utilization']):.2f} / {q[m]['failed_or_null']}"for m in['WHOLE','NeST','Ada']])
 s2+=table(['SNR','WHOLE','NeST','Ada'],cr,title=r'Source-bit utilization (\%) / gray outputs out of 1500, $N=1024$.')
 s2+=r'''No valid NULL frame occurs in these cells; the listed gray outputs are rejected receptions. BPG has 11 body-CRC rejections across the four SNRs, while the other reused external methods have none. All failures stay in the denominator. Source-level better/tie/worse counts and their full distributions are supplied as CSVs; aggregate gains do not imply improvement for every image.
'''

 s3=r'''\section{Matched Endpoint and Receiver Controls}\label{sup:3}
These controls use the same fixed 100 source images and three shared noise realizations per image. They isolate termination granularity or receiver use of a fixed received prefix; they do not decompose the independently calibrated 500-image system gains. All differences are source-paired, with 95\% intervals.
\subsection{Matched termination grids}
At $N=1024$, 10 dB, all four grids share 16-QAM rate 2/3, ceiling $(8,126)$, model and receiver. At $N=512$, 10 dB, the same factors are held fixed with ceiling $(7,0)$. Only allowed termination points change. The complete-scale control has five/four endpoints at the respective budgets; it is not replaced by the separately calibrated WHOLE policy.
'''
 grid=rows('inputs/e3/summary.csv')
 lo=rows('inputs/unified_calibration_20261103/low_budget/summary.csv')
 for r in lo:r['N']='512'
 eff={r['method']:r for r in rows('inputs/e3/efficiency.csv')}
 means=[]
 for N,rr in [('1024',grid),('512',lo)]:
  for m in ['whole','quarter','eighth','token']:
   d={r['metric']:r for r in rr if r['method']==m}
   util=float(eff[m]['utilization'])if N=='1024'else float(d['utilization']['mean'])
   means.append([N,name(m),f'{100*util:.2f}']+[cell(d[z])for z in METRICS])
 s3+=table(['$N$','Grid',r'Util. (\%)']+HEADERS,means)
 gp=rows('inputs/e3/paired_deltas.csv')
 for r in gp:r['comparison']=name(r['candidate'])+'-'+name(r['reference'])
 lp=rows('inputs/unified_calibration_20261103/low_budget/paired_deltas.csv')
 for r in lp:r['N']='512';r['comparison']='-'.join(name(x)for x in r['comparison'].split('-'))
 s3+=pivot(gp+lp,['N','comparison'],['$N$','Difference'],ci=True,signed=True)
 s3+=r'''Eighth retains about 92\% and 87\% of the Whole-to-Token mean PSNR gains at $N=1024$ and $N=512$, respectively. At $N=1024$, Token improves all five metrics over Eighth in the paired intervals; at $N=512$, the additional DISTS and CLIP-I intervals cross zero. Finer grids therefore provide a quality--processing-cost choice, not a universally optimal endpoint spacing. S7 gives the measured cost of the same $N=1024$ grids; comparisons of cached code lengths are not model-forward counts.
\subsection{Ordered boundary-token interventions}
A0 completes from the prior without the received boundary correction. B0 inserts that correction into the residual while retaining A0's future-condition path. C0 updates both the residual and future conditioning. The actual received prefix, generation settings and failures are held fixed. A0 is not a no-completion baseline, and these ordered differences are not independent additive contributions.
'''
 ip=rows('inputs/e4/paired_deltas.csv')
 for r in ip:r['comparison']=r['candidate']+'-'+r['reference']
 s3+=pivot(ip,['snr_db','comparison'],['SNR','Difference'],ci=True,signed=True)
 s3+=r'''At $N=1024$, 19 dB, the received boundary lies on the last scale (or the representation is complete). B0 and C0 then use the identical final residual map: no finer-scale conditioning contribution remains.
\subsection{Ada completion sensitivity}
Under the auxiliary configurations in S1.3, only the unreceived-tail choice changes from sampling to argmax. The same 100 sources, paid classes, received prefixes, three original channel outcomes, weights, CFG and precision are retained. Greedy generation rebuilds and updates its own context; it does not reuse the sampled tail. The entropy CDF is unchanged. Full-prefix and gray outputs retain their original treatment. The table gives argmax minus sampling; DISTS was not part of this four-metric diagnostic.
'''
 sens=[r for r in rows('inputs/v9_package/inputs/ada_generation_sensitivity.csv')if r['comparison']=='Ada-argmax minus Ada-sampling']
 s3+=pivot(sens,['N','snr_db'],['$N$','SNR'],ci=True,signed=True,metrics=['psnr_db','dreamsim','lpips_alex','clip_image_cosine'])
 s3+=r'''Argmax increases both LPIPS and DreamSim at all three points. The main curves retain sampling; the sensitivity result is tied to these auxiliary policies and does not replace the current calibration.
'''

 s4=r'''\section{Equal information budgets}\label{sup:4}
Total information budgets are 620, 1212, 1302, 1940, 2248 and 2577 bits. They include 57 common profile/CRC/length bits and another 10 class bits for Ada; arithmetic-code capacities are $B-57$ and $B-67$, respectively. WHOLE/NeST share ceiling $(10,0)$ and Ada permits 1--256 tokens. Independent source reception is error-free, with no PHY or MCS. This removes channel effects but retains tokenizer, prior, class-conditioning and decoder differences.

Each of 500 sources contributes one deterministic WHOLE/NeST reconstruction and three fixed Ada generation repetitions; metric values are averaged within source. These are generation repeats, not channel repeats. All outputs are accepted, with no NULL output. Figure and paired table retain all six budgets.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/supp_figure_s1_information_budget.pdf}
\caption{Equal-source means and pointwise 95\% intervals at error-free information budgets, including paid metadata. The numerical spacing of the six budgets is retained.}\label{fig:information-budget}\end{figure}
'''
 eb=rows('inputs/e2/paired_deltas.csv')
 eb=[r for r in eb if r['candidate']=='Token_NeST']
 for r in eb:r['comparison']='NeST-'+name(r['reference'])
 s4+=pivot(eb,['B_bits','comparison'],['Bits','Difference'],ci=True,signed=True)
 s4+=r'''All five NeST--WHOLE intervals favor NeST. Against Ada, NeST has higher PSNR throughout, but DISTS/DreamSim favor Ada at 2248/2577 bits; DreamSim crosses zero at 1940 bits. CLIP-I favors Ada at 620/1940/2577 bits and crosses zero elsewhere. LPIPS favors NeST through 2248 bits and crosses zero at 2577 bits.

Between 1212 and 1302 bits, NeST uses 89.904 of the extra 90 bits on average, versus 3.096 for WHOLE. At 1940/2248/2577 bits, Ada transmits all 256 tokens on 4/10/75 of 500 sources; WHOLE and NeST do not reach their complete representations. Unused capacity after a complete representation is saturation, not a fitting failure. Full means, intervals and source/padding-bit counts remain in the companion.
'''

 s5=r'''\section{Fixed-policy channel threshold}\label{sup:5}
The earlier 10-dB policies are fixed while actual SNR varies over 8, 8.5, 9, 9.5 and 10 dB. Each method/SNR uses 100 images with 20 AWGN realizations per image. The original source prefix, waveform and generation identity remain fixed; RX knows the true noise variance. The policies have the same MCS and ceilings as current 10-dB calibration, but use the earlier directories (S1.3).
\begin{figure}[htbp]\centering\includegraphics[width=0.92\linewidth]{figures/supp_figure_s2_threshold.pdf}
\caption{Fixed 10-dB transmit policies under downward SNR mismatch. Means and source-bootstrap intervals include gray failures. Dotted quality curves condition on successful receptions and have changing denominators; no such point exists at 8 dB.}\label{fig:threshold}\end{figure}
'''
 th=rows('inputs/e6/summary.csv');tr=[]
 for snr in ['8','8.5','9','9.5','10']:
  d={r['method']:r for r in th if r['snr_db']==snr}
  tr.append([snr]+[f"{2000-int(d[m]['total_success'])} / 2000"for m in['WHOLE','Token_NeST','Ada']])
 s5+=table(['Actual SNR','WHOLE rejects','NeST rejects','Ada rejects'],tr,groups=['threshold']*len(tr))
 s5+=r'''All rejections arise from body/profile parsing; no header rejection, source-decoding rejection or valid NULL frame occurs. All 2000 receptions per method are rejected at 8 dB, whereas none is rejected at 10 dB. The source-cluster intervals retain all 20 realizations of each sampled image. This scan characterizes the tested fixed policies near their threshold, rather than establishing general robustness or low-error-rate operation.
'''

 s6=r'''\section{Qualitative examples}\label{sup:6}
The main Fig.3 shows sources 135 and 201 (C10-016/C10-017), selected from the earlier gallery for favorable NeST perceptual reconstructions and retained through unified calibration. WHOLE/NeST/Ada use current 10-dB policies and the first channel repetition, seed 6201; SwinJSCC retains its original first repetition, seed 2001. Full $256\times256$ fields and metrics from the displayed native outputs are retained. These selected cases illustrate possible gains; population comparisons are established by S2, where Ada has lower mean DISTS and DreamSim at 10 dB.

The complementary images below retain their earlier 10-dB configurations. Sources were ranked by mean DISTS gain of Token over WHOLE, with source-index tie breaking and no success filter. Source 209 is the fixed median-rank choice and source 39 the minimum-gain choice, both displayed with the first original repetition. Source 39 worsens in DISTS but improves in DreamSim. Complete source IDs, all five display metrics, and the selected 7-dB advantage examples remain in the companion assets; no sample is reselected in this revision.
\begin{figure}[htbp]\centering\includegraphics[width=5.8in]{figures/supp_examples_distribution.pdf}
\caption{Median-rank (top) and minimum DISTS-gain (bottom) sources under the earlier $N=1024$, 10-dB policies. Columns are source, WHOLE and NeST. Values are DISTS / DreamSim for the first displayed repetition; lower is favorable.}\label{fig:counterexamples}\end{figure}
'''

 s7=r'''\FloatBarrier\section{Processing costs and paired quality}\label{sup:7}
Main timing comparisons use two distinct physical CPU cores on the same RTX PRO 6000 host. Historical measurements used two SMT siblings of one physical core and are retained separately in the companion; the groups are not pooled.
\subsection{Complete TX/RX processing}\label{sup:timing-main}
The first 32 test images use batch one, two warmups and three measurements per image and direction. GPU implementations run serially; BPG uses the same CPU allocation. Timed spans use two intra-op and two inter-op threads. At $N=1024$, 10 dB, token systems use 16-QAM rate 2/3 and WHOLE/NeST/Ada ceilings $(9,0)/(8,126)/256$. These match the current quality experiment's MCS and ceilings, while the directories are earlier versions.

GPU synchronization brackets TX from in-memory RGB to transmitted symbols and RX from received symbols to RGB. Tokenization, condition extraction, CDF/arithmetic coding, independent termination, complete candidate scanning, link processing and reconstruction are included. Model/input loading, final hashing, logging and metric scoring are excluded; BPG retains its source search, codec invocation and RAM serialization. No cached tokens or reconstruction replace timed work. Total processing time is TX plus RX, not radio propagation latency. Means and source-bootstrap intervals (10\,000 resamples) describe this session's image-to-image variation.
'''
 old7=section(7);ot=old_tables(old7)
 s7+=ot[0]+'\n'
 s7+=r'''\subsection{Matched granularity}\label{sup:timing-grid}
The four grids fix ceiling $(8,126)$, model, MCS and fitting rule. Token shares the identical NeST TX cell; the main WHOLE policy, with its different ceiling, is not substituted for matched Whole. Times include shared scale-wise probability computation, independent termination and full candidate scanning.
'''+ot[1]+'\n'
 s7+=r'''Token costs about 14.1\% more TX time than Eighth and adds 0.066 dB mean PSNR at this matched point (S3). Eighth and Token offer different quality--computation choices. All listed implementations successfully complete their measured receptions.
\subsection{Compiled Ada and its matched output quality}\label{sup:timing-compiled}\label{sup:timing-compiled-quality}
The compiled route uses conditional causal prefill and \texttt{torch.compile}; independent RX reconstructs CDFs from received tokens with the same tensor shape and strict causal mask, using public zeros for unknown future inputs. CFG 4, sampling rules, weights, classifier, tokenizer, decoder, MCS and precision are unchanged. Cold compilation and model loading are excluded; timed spans use the same two physical cores. Compilation itself uses eight cores.
'''+ot[2]+'\n'+r'''
Prefill/compilation can change floating-point CDFs and bitstreams. Although independent decoding recovers the same source tokens, payload-derived tail seeds can consequently change. The 32-source quality comparison below uses the actual timed outputs and matching original events. All 32 restored outputs match their timed hashes. Fixing the original generation seeds makes all 32 native images identical to the original route; the actual timed outputs differ on 30 sources. Intervals crossing zero do not establish implementation equivalence.
'''
 cq=[r for r in rows('inputs/unified_calibration_20261103/compiled32/paired_deltas.csv')if r['comparison']=='compiled_timed-original_timed']
 s7+=table(['Metric',r'Compiled minus original [paired 95\% CI]'],[[HEADERS[METRICS.index(r['metric'])],cell(r,True,True)]for r in cq])
 s7+=r'''The compiled outputs have mean PSNR 19.36069, DISTS 0.13098, DreamSim 0.09640, LPIPS 0.16355 and CLIP-I 0.87857. Their paired processing cost is 3064.30 ms (114.55 TX + 2949.76 RX); no new latency is assigned to the fixed-seed control. The 500-image main quality results keep the original implementation. Historical SMT timings, two-source prefill diagnostics, counters and execution history are available in the companion rather than pooled into this comparison.
'''

 s8=r'''\section{Transfer to Kodak24}\label{sup:8}
The 24 images use the original RGB center $256\times256$ crop without prior resizing. Models and earlier ImageNet-calibrated policies remain frozen at 4/10/19 dB, with three channel repetitions per image. Quarter and NeST share the target ceilings but use different fallback grids. At 19 dB, historical Ada uses 64-QAM rate 5/6 rather than the current 16-QAM rate 5/6; this tests transfer of the historical configurations, not the current Table IV policy. All receptions are accepted. Statistics average repetitions within each of 24 sources and use the fixed 10\,000-by-24 source bank.
'''
 ks=rows('inputs/kodak_v9/summary.csv');s8+=pivot(ks,['snr_db','method'],['SNR','Method'],'All source means.')
 kp=[r for r in rows('inputs/kodak_v9/paired_deltas.csv')if r['comparison'].startswith('Token minus')]
 for r in kp:r['comparison']=r['comparison'].replace('Token minus ','NeST-')
 s8+=pivot(kp,['snr_db','comparison'],['SNR','Difference'],r'Direct paired differences [95\% CI].',True,True)
 s8+=r'''NeST improves PSNR, DISTS and DreamSim over WHOLE at all three SNRs. Against Quarter, DISTS/DreamSim intervals exclude zero at 10/19 dB and cross zero at 4 dB. Against Ada, both distances favor NeST at 4 dB; DISTS favors Ada at 10/19 dB and DreamSim intervals cross zero. The five-metric tables retain these mixed outcomes; complete marginal intervals, other contrasts and source-level distributions are in the companion CSVs.
'''
 final=pre+s1+s2+s3+s4+s5+s6+s7+s8+'\n'+r'\end{document}'+'\n'
 # Consistent local table typography; no shrink-to-fit transformation.
 final=final.replace(r'\begin{longtable}',r'\begin{longtable}')
 final='\n'.join(x.rstrip()for x in final.splitlines()).rstrip()+'\n'
 (out/'reader_table_cells.json').write_text(json.dumps(dict(scope='Formatting and selection only; no statistics recomputed.',source_files=sources,cells=ledger),indent=2)+'\n',encoding='utf8')
 (P/'supplementary.tex').write_text(final,encoding='utf8')
 return final

if __name__=='__main__':
 p=Path(__file__).resolve().parents[1]
 apply_reader_revision('',p)
 print('Built concise reader edition from unchanged result CSVs.')
