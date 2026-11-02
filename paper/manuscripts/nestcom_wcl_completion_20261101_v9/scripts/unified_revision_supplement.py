"""Replace current N1024 evidence while retaining separately identified controls."""
import csv,json

MET={'psnr_db':'PSNR','lpips_alex':'LPIPS','dreamsim':'DreamSim','clip_image_cosine':'CLIP-I','dists':'DISTS'}
LABEL={'NeST':'NeST','WHOLE':'WHOLE','Ada':'Ada','ARPC':'ARPC','Latent_JSCC':'Latent','SwinJSCC':'Swin','BPG_LDPC':'BPG','final_NeST':'NeST','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada','original_NeST':'Quarter'}

def apply_unified_revision(s,P):
 U=P/'inputs/unified_calibration_20261103'
 def rows(path):
  with path.open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))
 def table(headers,rr,spec):
  return '\n'+r'\begin{longtable}{@{}'+spec+r'@{}}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\'for r in rr)+'\n'+r'\bottomrule\end{longtable}'+'\n'
 def pair(r):return f"{float(r['mean']):+.5f} [{float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f}]"
 def between(start,end,new):
  nonlocal s
  a=s.index(start);b=s.index(end,a);s=s[:a]+new+'\n'+s[b:]
 assert json.loads((U/'science_closure.json').read_text())['status']=='COMPLETE_SCIENCE'
 s=s.replace('The complete machine-readable configuration, candidate pools, PHY layouts, model manifest and reference vectors are supplied in', 'The historical frame layouts, model manifest and reference vectors are supplied in')
 # The unchanged protected-header design now selects an explicitly public fixed directory.
 a=s.index('The current N1024 directory retains');b=s.index(r'\subsection{Candidate pools and calibration}',a)
 s=s[:a]+r'''For the current N1024 calibration, each method/MCS uses a public directory fixed before the pilot and shared across all target ceilings and source images. WHOLE has seven source profiles, NeST 651 and Ada 256, plus one paid-NULL profile in each directory. Thus every ID fits in 12 bits without changing the 68-symbol header. The method and frozen policy directory are common protocol context; neither depends on the source. The receiver obtains the actual endpoint from the received profile, not from a transmitter witness. Directory qualification performs two real complete-frame checks per directory (source and NULL), totaling 72 frames and 144 header/body decodes. The original 3378-profile directory and its added NULL entries remain the identified context of historical controls.

'''+s[b:]
 pilot=rows(U/'pilot_calibration/candidate_scores.csv');full=rows(U/'full_calibration/candidate_scores.csv')
 calibration=r'''\subsection{Candidate pools and calibration}
At N1024, all three token families use the same 12 MCSs, $q\in\{2,4,6\}$ and $r\in\{1/2,2/3,3/4,5/6\}$. WHOLE has seven complete-scale target ceilings. NeST searches the 25 Quarter target ceilings and permits every integer endpoint up to each target. Ada has the maximum target of 256 and permits every shorter positive prefix. This gives 84/300/12 candidates for WHOLE/NeST/Ada, respectively. Equal MCS and selection rules do not mean identical endpoint grammars. NeST's search is finite over 25 ceilings, not an exhaustive search over all 651 possible ceilings.

Each candidate uses the same fixed 100 calibration sources with one channel realization (seed 4101) at 4/7/10/19 dB. Per family/SNR, the top three plus the predetermined maximum-ceiling QPSK rate-1/2 anchor advance, with duplicate candidate IDs removed. Final confirmation uses all 1000 calibration sources and three realizations (4101/4102/4103). Source-level means of DINOv2-L cosine include every gray failure and paid NULL; their equal-weight mean is maximized. Ties use a fixed lexical candidate-ID order. Identical actual streams or receiver inputs may be reused after full identity checks while all candidate logical positions remain in the denominator. The original serial Ada quality implementation, FP32/TF32-off configuration, CFG 4, T1, top-k 2000 and top-p 1 are fixed before evaluation.

The complete 1584 pilot candidate/SNR scores and all confirmation scores, with the public directory and policy hashes, are in \path{inputs/unified_calibration_20261103/}. No common500 result participates in policy selection. Frozen policies are then evaluated on 500 sources, three actual channel realizations per source (6201/6202/6203), yielding 18000 logical events. The original N512 calibration and its inherited Token refinement remain unchanged; the new matched N512 diagnostic below instead holds every factor except the termination grid fixed. Kodak, threshold and historical receiver diagnostics retain their specifically identified earlier policies.

'''
 calibration+=f"The pilot contributes {sum(int(r['logical_count'])for r in pilot)} logical positions; confirmation contributes {sum(int(r['logical_count'])for r in full)}. The confirmation candidate scores are:\n"
 # Candidate IDs are machine-readable, long strings; human table uses explicit tuple components.
 catalog=json.loads((U/'public_directories.json').read_text());cm={v['candidate_id']:v for v in catalog['candidates']}
 calibration+=table(['Method','SNR','$(q,r)$','Ceiling','DINO cosine','Reject/NULL'],[[r['method'],r['snr_db'],str(cm[r['candidate_id']]['q'])+','+cm[r['candidate_id']]['rate'],str(tuple(cm[r['candidate_id']]['target'])),f"{float(r['mean_dino_cosine']):.6f}",r['failed_or_null']]for r in full],'lrrlrr')
 calibration+=r'''DINOv2 is the policy objective. DISTS, DreamSim and CLIP-I are feature-based source-reference measures; their agreement is not independent human perceptual validation. Source-capacity utilization excludes paid body fields and is not total air-interface efficiency. Ada may transmit all 256 tokens and leave capacity unused. N512 and N1024 are separately calibrated systems, not a budget-only causal intervention.

\subsection{Actual-length fitting and scan accounting}
For each complete scale, the prior evaluates the probability map from known coarser maps. Raster tokens enter one shared incremental arithmetic encoder. At every allowed endpoint, a non-mutating finish produces a separate terminated stream. The current measured path is:
\begin{enumerate}
\item Fix MCS, public directory, legal endpoint grid and calibrated ceiling. Order endpoints by retained token count.
\item Traverse source tokens once, sharing probability evaluation and incremental coding. At each allowed endpoint copy its independently terminated bits without changing encoder state.
\item Compare every allowed candidate's actual length with the source capacity. For NeST/WHOLE require $b+29\le k$; Ada additionally pays its 10-bit class.
\item Select the most-token feasible endpoint below the ceiling. Transmit its entire terminated stream and paid profile; if none fits, transmit the paid NULL frame.
\item Recover the profile and then independently decode the corresponding body. Rejected/NULL images remain gray in every metric denominator.
\end{enumerate}
For Quarter/Eighth, boundary counts are $\lfloor jL/d\rfloor$, $j=1,\ldots,d-1$, $d=4/8$, with duplicates removed and complete boundaries included. Token includes every integer boundary count. No partial stream is obtained by cutting a longer finalized stream.

Let $E$ be emitted bits and $P$ pending underflow bits. The original coder's finalization yields $b=E+P+2$. Every E1/E2 renormalization emits $P+1$ bits and clears $P$, and every E3 increments $P$; each raises $E+P$ by one. A token's interval update alone leaves it unchanged. Thus independently terminated lengths along this shared path are nondecreasing, with possible ties. We checked the exact original integer coder on 512 synthetic prefixes, and 976500 existing streams from cal1000/common500 with all 975000 adjacent length relations. Array hashes, non-mutating finish and canonical roundtrips pass; no image or PHY was rerun for this audit.

The timed code still generates and scans all allowed candidates. Its historical \texttt{length\_checks} field is the first feasible rank, not the list comprehension's full comparison count. Candidate comparisons are not extra prior forwards. Monotonic lookup or early stopping could reduce work but is not implemented or retimed here; no such speedup is attributed to the reported timings.

'''
 between(r'\subsection{Candidate pools and calibration}',r'\subsection{Reference vectors and verification}',calibration)
 q=r'''\section{Complete quality results}
The N1024 WHOLE/NeST/Ada rows below use the new uniformly calibrated policies. Latent JSCC, SwinJSCC, BPG and the supplementary ARPC digital adaptation reuse their exact original native outputs after source-ID, preprocessing, pixel-reference and metric checks. The three newly calibrated token methods share the source/repetition noise vector. Reused external methods retain their original independent noise identities; cross-method pairing is by source, not a claim of shared noise across all systems. N512 rows retain the independently calibrated study. Every source averages three events before equal-source means or direct paired differences. Intervals use the same fixed 10000-by-500 source bank, with no simultaneous-coverage claim.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/figure2_complete_with_arpc.pdf}
\caption{Complete comparison including our digital-link adaptation of the ARPC source codec. The 92 means and pointwise source-bootstrap intervals retain all operating points. ARPC is the closest progressive source-coding precedent; this supplementary curve is its specified digital adaptation.}\end{figure}
'''
 summ=rows(U/'external_join/summary.csv');pairs=rows(U/'external_join/paired_deltas.csv')
 for r in rows(P/'inputs/e1/summary.csv'):
  if r['N']=='512':summ.append(dict(method=r['method'],N=r['N'],snr_db=r['snr_db'],metric=r['metric'],mean=r['mean'],ci_low=r['ci95_low'],ci_high=r['ci95_high']))
 for r in rows(P/'inputs/e1/paired_deltas.csv'):
  if r['N']=='512':pairs.append(dict(comparison=LABEL[r['candidate']]+'-'+LABEL[r['reference']],N=r['N'],snr_db=r['snr_db'],metric=r['metric'],mean=r['mean_delta'],ci_low=r['ci95_low'],ci_high=r['ci95_high']))
 q+=table(['$N$','SNR','Method','Metric',r'Mean [pointwise 95\% CI]'],[[r['N'],r['snr_db'],LABEL[r['method']],MET[r['metric']],pair(r)]for r in summ],'rrlll')
 q+=table(['$N$','SNR','Comparison','Metric',r'Difference [paired 95\% CI]'],[[r['N'],r['snr_db'],r['comparison'].replace('Latent_JSCC','Latent').replace('SwinJSCC','Swin').replace('BPG_LDPC','BPG'),MET[r['metric']],pair(r)]for r in pairs],'rrlll')
 cap=rows(U/'formal/capacity_summary.csv')
 q+=r'''\subsection{Capacity and rejected receptions}
The current N1024 counts retain all 1500 logical events per method/SNR. Valid paid NULL is listed within gray outputs and separately; it is not a lost denominator. Unchanged BPG has 11 body-CRC rejections across the four SNRs, while the other reused external methods have no rejected reconstruction. Complete event counts and native identities are retained in the result files.
'''
 q+=table(['Method','SNR',r'Util. (\%)','Gray / 1500','Paid NULL'],[[r['method'],r['snr_db'],f"{100*float(r['utilization']):.3f}",r['failed_or_null'],r['paid_NULL']]for r in cap],'lrrrr')
 q+=r'''\subsection{Source-level variation}
Positive gain is favorable: reference minus NeST for distances and NeST minus reference for PSNR/CLIP-I. Better/tie/worse uses tolerance $10^{-10}$, not a perceptual visibility threshold. The distance distributions below include all sources and failures; complete five-metric distributions are in the companion CSV.
'''
 rr=rows(U/'external_join/per_source_improvement.csv')
 q+=table(['SNR','Reference','Metric','Better/tie/worse','Median gain'],[[r['snr_db'],LABEL[r['reference']],MET[r['metric']],r['better']+'/'+r['equal']+'/'+r['worse'],f"{float(r['median_gain']):+.5f}"]for r in rr if r['metric']in ['dists','dreamsim']],'rlllr')
 between(r'\section{Complete quality results}',r'\section{Matched controls and generation sensitivity}',q)
 low=r'''\subsection{Matched low-budget termination grid}
At N512/10 dB, the same control100 sources, three original noise realizations, fixed 16-QAM rate 2/3, ceiling $(7,0)$, frozen model and receiver are shared by Whole/Quarter/Eighth/Token. Only allowed termination positions change. There are 1200 logical positions, 963 unique physical frames (846 exact reuses, 117 new) and 321 unique receiver inputs (282 reuses, 39 new). Thirty-nine new native source/output pairs receive all five metrics. The actual all-candidate length comparisons per image are 4/13/25/126, respectively; these are not model-forward or latency counts.
'''
 lr=rows(U/'low_budget/summary.csv');ls=rows(U/'low_budget/source_means.csv')
 ll=[]
 for grid in ['whole','quarter','eighth','token']:
  v={r['metric']:r for r in lr if r['method']==grid};rr=[r for r in ls if r['method']==grid];assert len(rr)==100
  ll.append([grid.title(),f"{100*sum(float(r['utilization'])for r in rr)/100:.3f}"]+[f"{float(v[m]['mean']):.5f}"for m in MET])
 low+=table(['Grid',r'Util. (\%)',*MET.values()],ll,'lrrrrrr')
 low+=table(['Comparison','Metric',r'Difference [paired 95\% CI]'],[[r['comparison'],MET[r['metric']],pair(r)]for r in rows(U/'low_budget/paired_deltas.csv')],'lll')
 low+=r'''Eighth retains approximately 87\% of the Whole-to-Token mean PSNR gain at this point. Token--Eighth improves PSNR, LPIPS and DreamSim in the paired intervals, while DISTS and CLIP-I intervals cross zero. This diagnostic isolates granularity; it does not decompose the separately calibrated common500 system gain.

'''
 marker=r'\section{Equal information budgets}';s=s.replace(marker,low+marker)
 # Historical diagnostic images retain their old frozen policies, explicitly separated.
 a=s.index(r'\section{Qualitative examples}');b=s.index('The following panel shows',a)
 figure=r'''\section{Qualitative examples}
Fig.3 retains the two source IDs selected by the user from the prior complete gallery: source135 (C10-016, truck) and source201 (C10-017). WHOLE/NeST/Ada now use the current frozen N1024/10 policies and the first registered channel repetition, seed6201; SwinJSCC retains its original first repetition, seed2001. Sources were not reselected after the new calibration. Images retain the full 256-by-256 field. The displayed DISTS/DreamSim values are computed from the corresponding native float output, not a PNG or a three-repeat mean. These remain selected illustrative cases, not random or representative samples.
'''
 mapping=json.loads((P/'inputs/fig3_unified_20261103/native_mapping.json').read_text())
 figure+=table(['Index','Method',*MET.values()],[[v['source_index'],v['display_method']]+[f"{float(v[m]):.5f}"for m in MET]for v in mapping if v['method']!='Source'],'rlrrrrr')
 figure+=r'''The following supplemental image panels retain their historical policies and their previously fixed selections. Their labels and numbers are not substituted for the newly calibrated main Fig.3. '''
 s=s[:a]+figure+s[b:]
 s=s.replace('With 10-dB policies fixed','With the historical 10-dB policies fixed')
 s=s.replace('Kodak uses the original RGB center', 'Kodak retains the earlier ImageNet-calibrated policies, without repeating the new N1024 calibration on Kodak. It uses the original RGB center')
 # Original resource sessions are valid measurements, but not attached to new main-quality cells.
 marker=r'\section{Processing costs and implementation diagnostics}'
 if marker not in s:marker=r'\section{Unified processing costs}'
 assert marker in s
 s=s.replace(marker,marker+'\nThe timing sessions below retain their original registered ceilings, MCSs and numerical routes. The new calibration does not remeasure latency; these measurements are not attached as the timing of a different newly selected policy.\n',1)
 quality=r'''\subsection{Quality paired with compiled Ada timing}
We restore the 32 saved timed received waveforms and compare native outputs on the exact same sources, received prefixes/classes and channel events. The restored compiled outputs match all 32 timed output hashes. No TX, new channel event or new timing run is performed. The original and compiled coders recover identical source tokens, but their bitstreams differ; the existing seed derivation from received payload bytes can therefore change the tail-generation seed.

Before successful new quality evaluation, an additional fixed-original-seed arm is specified. It needs 30 extra tail renders and reuses two identical cases. Holding that seed fixed makes all 32 native reconstructed images byte-identical to the original timed route, and all five paired quality differences are exactly zero. The actual timed outputs differ on 30 images. Their paired mean differences below all have intervals crossing zero; this 32-source diagnostic does not establish population equivalence or replace common500. Each source has one distinct timed event; five timing repetitions are not five quality samples. The original serial quality route was selected for unified calibration before this diagnostic's outcome.
'''
 cp=rows(U/'compiled32/paired_deltas.csv')
 quality+=table(['Arm difference','Metric',r'Mean [paired 95\% CI]'],[[r['comparison'].replace('compiled_timed','Compiled').replace('original_timed','Original').replace('compiled_fixed_old_seed','Fixed-seed'),MET[r['metric']],pair(r)]for r in cp],'lll')
 tt=rows(U/'compiled32/quality_latency.csv')[0]
 quality+='The actual compiled timed outputs have source-mean '+', '.join(MET[m]+' '+f"{float(tt[m]):.5f}"for m in MET)+f". Their corresponding TX/RX/total means are {float(tt['tx_ms']):.2f}/{float(tt['rx_ms']):.2f}/{float(tt['total_ms']):.2f} ms. The fixed-seed diagnostic is not assigned a newly measured latency.\n\n"
 s=s.replace(r'\subsection{Historical E5 resource protocol}',quality+r'\subsection{Historical E5 resource protocol}')
 s=s.replace('No main image-quality experiment or metric evaluation is rerun.', 'That timing study did not evaluate image quality; the subsequent same-source quality diagnostic below supplies it.')
 s=s.replace('Current all-candidate scanning','Measured all-candidate scanning')
 return s
