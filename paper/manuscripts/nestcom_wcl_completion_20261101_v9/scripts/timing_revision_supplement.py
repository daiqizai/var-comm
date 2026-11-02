"""Current corrected-resource timing, with historical E5 tables explicitly retained."""
import csv,json

def apply_timing_revision(s,P):
 D=P/'inputs/timing_fairness_20261102'
 assert json.loads((D/'closure.json').read_text())['status']=='COMPLETE'
 def rows(name):return list(csv.DictReader((D/name).open()))
 def tab(headers,rr,spec):
  return '\n'+r'\begin{longtable}{@{}'+spec+r'@{}}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\'for r in rr)+'\n'+r'\bottomrule\end{longtable}'+'\n'
 t={(r['method'],r['direction']):r for r in rows('timing_summary.csv')}
 label={'WHOLE':'WHOLE','Token_NeST':'NeST','Ada':'Ada (audit only)','Latent_JSCC':'Latent','SwinJSCC':'Swin','BPG_LDPC':'BPG'}
 new=r'''\section{Unified processing costs}
\subsection{Corrected resource protocol for the main comparisons}
The resource audit uses the same first 32 common500 IDs at $N=1024$, 10 dB, batch one, with two warmups and three measurements per source and direction. All five GPU implementations run serially on the same physical RTX PRO 6000. Every method uses CPU affinity [38,40], which selects distinct physical cores 19 and 20 of socket 0, with two intra-op and two inter-op threads. BPG uses these CPU cores without a GPU. The numerical routes, weights, policies, source order and observed RX symbols remain frozen. Every emitted waveform and reconstructed image matches its historical event hash exactly. Ada's measurements below document our adaptation only and are withheld from the main timing comparison for the implementation reasons given below.

TX starts with canonical RGB in memory and ends with actual symbols; RX starts with received symbols in memory and ends with RGB. GPU synchronization brackets each span. Tokenization, condition extraction, CDF construction, arithmetic coding, frame fitting, link processing and generation are included. Model loading, input loading, final-output hashing, saved logs and quality scoring are outside the timer. BPG retains source search, codec invocation and RAM serialization. No cached source tokens or reconstructions replace work inside a timed span. TX+RX is the sum of processing spans.

The previous affinity [38,39] selected two SMT siblings of one physical core, although the earlier manuscript called them two CPU cores; its intra-op pool had six threads. This was the same allocation for all methods. The corrected run replaces the available main timing cells; Ada's cells remain unreported rather than presenting its serial adaptation as an author-aligned implementation. The initial GPU0 session was interrupted by a foreign GPU diagnostic process; all GPU0 timings are excluded. GPU6 supplies every final GPU cell, and only the completed, exactly matching CPU-only BPG cells are reused. Prior resource use and interrupted-call ledgers are retained. No foreign process or GPU clock setting was changed.

The table gives equal-source means in milliseconds. Brackets for total time are pointwise 95\% source-bootstrap intervals (10\,000 resamples, fixed seed 2026110201). They describe source variation during this session, not all possible system load. The six main methods each complete 96 measured RX calls successfully. Full call times, source means, median/p95 and resource telemetry are in \path{inputs/timing_fairness_20261102/} and the accompanying result directory.
'''
 rr=[]
 for m in ['WHOLE','Token_NeST','Ada','Latent_JSCC','SwinJSCC','BPG_LDPC']:
  z=t[m,'total'];rr.append([label[m],f"{float(t[m,'tx']['mean_ms']):.2f}",f"{float(t[m,'rx']['mean_ms']):.2f}",f"{float(z['mean_ms']):.2f} [{float(z['ci_low_ms']):.2f}, {float(z['ci_high_ms']):.2f}]"])
 new+=tab(['System','TX (ms)','RX (ms)',r'Total (ms) [95\% CI]'],rr,'lrrl')
 rr=[]
 for r in rows('granularity_summary.csv'):
  rr.append([r['group'].replace('granularity_','').title(),f"{float(r['mean_ms']):.2f}",f"[{float(r['ci_low_ms']):.2f}, {float(r['ci_high_ms']):.2f}]"])
 new+=r'''\subsection{Matched granularity and audit sensitivity}
The matched grid fixes ceiling $(8,126)$, model, MCS and fitting rule. Whole/Quarter/Eighth TX are measured in the same resource session; Token reuses the identical main NeST TX cell. The main WHOLE policy has a different ceiling and is not substituted for the matched Whole cell. Quality values and all received inputs are unchanged.
'''+tab(['Grid','TX (ms)',r'95\% source interval'],rr,'lrl')
 new+=r'''The frozen Ada adaptation evaluates probabilities serially for all 256 source positions at TX. It executes conditional and unconditional branches, although only the conditional branch supplies entropy CDFs, and retains its FP32 math-attention route. RX decodes the prefix and continues generation along the same context. NeST predicts maps scale by scale. These model and implementation choices remain part of the measured costs.

Ada-TokenCom's published Section IV-E instead describes a single encoder prefill and \texttt{torch.compile}. Its Table IV reports 19 ms encoding and 1012 ms decoding on an RTX 5090 and Xeon Platinum 8260. The reported boundary comprises tokenizer, AR probability estimation, softmax and arithmetic coding, whereas our spans additionally include class prediction and digital-link processing. The paper also describes repeated full-prefix evaluation at RX to preserve entropy-model numerical consistency. Thus, neither its published numbers nor our serial-adaptation numbers establish a matched end-to-end speed comparison. In particular, the serial TX loop and unused TX branch are implementation overheads, not requirements of Ada's method. The published latency section does not specify enough numerical detail to assume that a different precision is responsible for the discrepancy.

An auxiliary diagnostic uses only the first two registered sources. It defers Ada's redundant per-token finite-logit audit until after timing and removes unused CDF audit hashes; for NeST it removes an unused completed-token hash. CRC, length/parser checks, arithmetic canonical re-encoding, actual generation and final output identity remain. The NeST TX path is unchanged and serves as a negative control. The following means compare this diagnostic with the original-audit measurements on the same two sources. Their session variation and tiny sample preclude replacing the 32-source results or subtracting these differences from them.
'''
 rr=[];dd=rows('audit_diagnostics.csv')
 for m in ['Token_NeST','Ada']:
  for direction in ['tx','rx']:
   v=[x for x in dd if x['method']==m and x['direction']==direction];assert len(v)==2
   mean=lambda key:sum(float(x[key])for x in v)/2
   rr.append([label[m],direction.upper(),f"{mean('original_audits_ms'):.2f}",f"{mean('lean_ms'):.2f}",f"{mean('delta_ms'):+.2f}"])
 new+=tab(['System','Span','Original audits','Diagnostic','Difference (ms)'],rr,'llrrr')
 new+=r'''The measurements characterize the frozen implementations on a common resource allocation. They do not measure equally optimized kernels or an architecture-independent speed limit. Ada's raw measurements remain audit evidence only, with no NeST/Ada speedup claim. All diagnostic outcomes, including the unchanged-path control, are retained.

'''
 qualification=json.loads((D/'ada_prefill_identity.json').read_text())['diagnostic']
 checks=qualification['rows']
 assert len(checks)==4
 new+=r'''\subsection{Prefill compatibility diagnostic}
Two fixed sources from the preceding diagnostic are reused for four full-prefix probability calls: conditional-only and the original two-branch calculation for each source. We preserve FP32, math attention, weights and the exact integer-CDF conversion. Existing source tokens, predicted classes, 256 CDF hashes and 256 independently terminated streams per source provide the reference; no classifier, tokenizer, image decoder, PHY or quality evaluation is rerun. The table counts exact matches to the frozen incremental route.
'''
 new+=tab(['Source index','Branches','CDF matches','Stream matches'],[[str(x['source_index']),str(x['branches']),str(x['conditional_CDF_matches'])+'/256',str(x['independently_terminated_stream_matches'])+'/256']for x in checks],'rrrr')
 if qualification['identity_qualification']=='PASS':
  new+=r'''All registered checks match. This permits further implementation validation; it is not yet a compiled, complete-TX/RX timing result and does not fill Ada's main timing cells.'''
 else:
  new+=r'''The full-prefix route is not an exact replacement for the frozen incremental probability path on these inputs. It requires a separately validated transmitter/receiver numerical route rather than substituting new probabilities into old bitstreams. The following study validates such a paired route; the preceding diagnostic and original streams remain unchanged.'''
 F=P/'inputs/ada_compiled_prefill_20261103'
 if F.exists():
  proof=json.loads((F/'actual_compute_and_reuse.json').read_text())
  assert proof['status']=='COMPLETE_PAIRED_NUMERICAL_IMPLEMENTATION'
  tt={r['direction']:r for r in csv.DictReader((F/'summary.csv').open())}
  new+=r'''

\subsection{Separately qualified compiled Ada implementation}
We replace the serial encoder with a single conditional causal prefill and compile the forward operation with \texttt{torch.compile}. At RX, the conditional CDF is reconstructed from the received prefix alone using the same tensor shape and strict causal mask; unknown future inputs are public zeros. The generated tail follows the actual recovered context, using the original CFG=4, temperature, filtering and sampling rules. Entropy CDFs use only the conditional branch. We retain FP32, math attention, disabled TF32, the same weights, classifier, tokenizer, image decoder, fixed MCS and candidate set. The compiler may change floating-point execution, so this is a new numerical implementation, not a bit-exact substitution for the original quality-evaluation route.

Two fixed sources first pass exact recovery of all 512 tokens and 512 integer CDFs, plus two fixed-prefix tail checks. That successful qualification is reused, without repeating its model calls. The formal study then performs fresh RGB-to-symbol TX and independent symbol-to-RGB RX on the same 32 sources, with two warmups and three measurements per direction. RX has no access to source pixels or TX witness files. All 160 received prefixes, decoded classes and endpoints match the external TX witness after RX has exited; all 96 measured receptions succeed. Repeated outputs are deterministic within the new implementation.

The following timings include class prediction and actual digital-link processing at the same synchronized boundaries as the preceding resource audit. Model loading and cold compilation are excluded from latency, but remain charged to the resource ledger. Compilation uses eight CPU cores; each timed span returns to the same two distinct cores and 2/2 thread pools. No main image-quality experiment or metric evaluation is rerun. The 10\,000-by-32 bootstrap index bank and source order are reused exactly.
'''
  new+=tab(['Implementation span','Mean (ms)',r'95\% source interval'],[[d.upper(),f"{float(tt[d]['mean_ms']):.2f}",f"[{float(tt[d]['ci_low_ms']):.2f}, {float(tt[d]['ci_high_ms']):.2f}]"]for d in ['tx','rx','total']],'lrl')
  match=proof['comparison_to_frozen_outputs']
  new+=(f"The new route matches {match['tx']['exact_old_output_matches']}/160 historical emitted waveforms and {match['rx']['exact_old_output_matches']}/160 historical reconstructed images exactly. "
        "Its latency is therefore reported separately from the original-quality Ada row; it does not replace those quality arrays or establish a NeST/Ada algorithmic speedup. The serial audit cost is not an intrinsic Ada requirement. The authors' published timing also uses different hardware and a different processing boundary, so its 19/1012 ms values are not substituted into the local comparison.\n\n")
  new+=r'''Three unsuccessful compilation attempts (missing development headers, parent and compiler-child module-name collisions) precede the formal run. They produced no timed TX/RX results and are retained with their actual nonzero exits and resource use. Private matching Python headers and the existing child-namespace binding repair the environment without changing the qualified model calculation. These failures are not included in per-image processing times. All study attempts remain charged against the original absolute deadline and the separately authorized resource extension.
'''
 new+=r'''

\subsection{Historical E5 resource protocol}
The historical 7-dB, N512, ARPC and grid-RX groups below were not rerun. For completeness, the original N1024/10 groups are also retained and labeled historical. They use affinity [38,39] (two SMT siblings of one physical core), six intra-op and two inter-op threads. These values are not combined with the corrected-resource main tables.
'''
 start=s.index(r'\section{Unified processing costs}')
 end=s.index(r'\section{Transfer to Kodak24}',start)
 old=s[start:end]
 old=old[old.index('Timing uses the first 32'):]
 old=old.replace('one actual RTX PRO 6000 and two CPU cores','one actual RTX PRO 6000 and two SMT siblings of one physical CPU core')
 old=old.replace('within the same two-core quota','within the same SMT-pair affinity')
 old=old.replace(r'\subsection{All measured groups}',r'\subsection{Historical measured groups}')
 old=old.replace(r'\subsection{Computation and interpretation}',r'\subsection{Shared computation accounting}')
 return s[:start]+new+'\n'+old+s[end:]
