"""Editorial-only protocol wording and supplementary navigation (2026-11-03)."""
import re
from pathlib import Path
MARK='% Signal and supplement organization 20261103'
def apply_signal_supplement_revision(s,P):
 if MARK in s:return s
 def replace(old,new):
  nonlocal s
  assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
 replace('S3 Matched controls and generation','S3 Matched endpoint and receiver controls')
 opening=s[s.index('NeST-Com transmits source-token prefixes'):s.index(r'\subsection{Representation, models and numerical route}')]
 replace(opening,r'''% Signal and supplement organization 20261103
\textbf{Algorithm overview.} Both terminals configure the frozen MCS and profile directory before transmission. TX obtains scale-wise probabilities, independently terminates each legal token prefix, and fits its arithmetic-code length to the protected body. It sends the selected prefix bitstream with its Profile ID, or a full-budget NULL frame when no source prefix fits. RX recovers the endpoint, independently decodes the prefix, and uses VAR's known-token overwrite and recursive state update for completion. Valid NULL reception and rejected frames both retain their gray outputs in the statistical denominator.

\textbf{Reading map.} S1.1 gives the representation and numerical rules; S1.2 specifies CDFs and independent termination; S1.3 defines fields, rejection and experiment configurations; S1.4 describes calibration; S1.5 gives the algorithm and coder-specific length property; S1.6 describes reference checks. The companion \path{REPRODUCTION_CHECKLIST.md} links the complete hash, environment/device and file-path records, protocol files, verification commands and all calibration scores. These machine-readable files accompany the manuscript in \path{reproduction/}; model and dataset access are identified separately from the downloadable metadata.

''')
 replace(r'The recipe is \path{experiments/var-latent-enhancement-20260917/configs/experiment.json}.','The full recipe and its repository path are indexed in the companion checklist.')
 replace('The NeST reference route is Python 3.10.12, NumPy 2.2.6, Torch 2.11.0+cu128, CUDA 12.8 and cuDNN 91900. Inference is FP32, batch 1, eval/no-grad, TF32 off, matmul precision highest, deterministic algorithms and cuDNN deterministic on, benchmark off. The revised timing uses two intra-op and two inter-op threads on two distinct physical CPU cores; S7 separates it from the historical SMT-pair run. Device PCI/UUID, tensor-state hashes, checkpoint-file hashes and source-file hashes are separate manifest fields. Reproducing a floating-point probability stream requires this numerical route as well as the weights.',
 'NeST uses FP32, batch one, eval/no-grad, TF32 off, highest matmul precision, deterministic algorithms and cuDNN deterministic on, with benchmark off. Exact software versions, device identities and distinct file/tensor-state hashes are provided in the companion records. Reproducing the floating-point probability stream requires the recorded numerical route as well as the weights. S7 distinguishes the two physical-core timing protocol from the historical SMT-pair protocol.')
 replace(r'Ada retains LlamaGen-L/VQ16 (upstream commit \texttt{ce98ec41803a74a90ce6}\allowbreak\texttt{8c40ababa9eaeffeb4ec}),','Ada retains the pinned LlamaGen-L/VQ16 implementation,')
 replace('The paid $(m,K)$ specifies the token count.','The paid Profile ID identifies the endpoint $(m,K)$ in the configured directory and hence the token count.')
 replace('A valid paid-NULL also produces gray, while preserving its full transmitted header/body and statistical denominator.',r'A valid NULL frame also produces gray, while preserving its full transmitted header/body and statistical denominator. A NULL profile is the directory entry used to identify this frame. If no source prefix fits, TX sends a NULL frame with a two-bit zero payload using the full $N$-symbol budget.')
 s=s.replace('paid-NULL profile','NULL profile').replace('paid NULL;','valid NULL frames;')
 s=s.replace('KT adds paid-NULL entries','KT adds NULL profile entries')
 # Preserve the large numerical tables as verbatim companion sources.
 out=P/'reproduction';out.mkdir(exist_ok=True)
 table_re=r'\\begin\{longtable\}.*?\\end\{longtable\}'
 phy=next(t for t in re.findall(table_re,s,re.S)if 'Filler & Punctured' in t)
 (out/'ldpc_layouts_table.tex').write_text(phy+'\n',encoding='utf-8')
 replace(phy,r'The complete 24 constructor layouts, including base graphs, lifting factors, filler and puncturing counts, are supplied in the companion \path{phy_layouts.csv}.')
 confirmation=next(t for t in re.findall(table_re,s,re.S)if 'DINO cosine & Reject/NULL' in t)
 (out/'calibration_confirmation_table.tex').write_text(confirmation+'\n',encoding='utf-8')
 replace('The pilot contributes 158400 logical positions; confirmation contributes 141000. The confirmation candidate scores are:',r'The pilot and confirmation contribute 158400 and 141000 logical positions, respectively. The companion \path{pilot_candidate_scores.csv} and \path{confirmation_candidate_scores.csv} retain every candidate score, rank and failure count; the frozen selection is summarized in the policy index above.')
 replace(confirmation,'')
 replace(r'\section{Matched controls and generation sensitivity}',r'\section{Matched Endpoint and Receiver Controls}')
 replace(r'\subsection{Actual source-code lengths and endpoint saturation}',r'\subsection{Arithmetic-code lengths and endpoint saturation}')
 s=s.replace('Source-capacity utilization','Source-bit capacity utilization')
 replace('Compare every allowed candidate\'s actual length with the source capacity.',"Compare every allowed candidate's terminated arithmetic-code length with the source-bit capacity.")
 replace('Transmit its entire terminated stream and paid profile; if none fits, transmit the paid NULL frame.','Transmit its entire prefix bitstream and Profile ID; if none fits, transmit the full-budget NULL frame.')
 # Keep S1.3 unchanged as a locator, and use labels for the reordered timing sections.
 start=s.index(r'\section{Unified processing costs}');end=s.index(r'\section{Transfer to Kodak24}',start)
 block=s[start:end];matches=list(re.finditer(r'\\subsection\{([^}]+)\}',block));intro=block[:matches[0].start()]
 parts={m.group(1):block[m.start():(matches[i+1].start()if i+1<len(matches)else len(block))]for i,m in enumerate(matches)}
 order=[('Corrected resource protocol for the main comparisons','Main physical-core timing protocol','sup:timing-main'),
 ('Matched granularity and audit sensitivity','Matched granularity','sup:timing-grid'),
 ('Separately qualified compiled Ada implementation','Compiled Ada implementation','sup:timing-compiled'),
 ('Quality paired with compiled Ada timing','Quality paired with compiled Ada timing','sup:timing-compiled-quality'),
 ('Historical E5 resource protocol','Historical SMT-pair resource protocol','sup:timing-historical-protocol'),
 ('Historical measured groups','Historical measured groups','sup:timing-historical-groups'),
 ('Prefill compatibility diagnostic','Prefill compatibility diagnostic','sup:timing-prefill-diagnostic'),
 ('Shared computation accounting','Shared computation accounting','sup:timing-accounting')]
 assert set(parts)=={v[0]for v in order}
 protocol='Main timing comparisons use two distinct physical CPU cores. Historical measurements use two SMT siblings of one physical core and are reported separately; the two groups are not pooled.\n\n'
 intro=intro.replace(r'\label{sup:7}',r'\label{sup:7}'+'\n'+protocol)
 chunks=[]
 for old,new,label in order:
  chunk=parts[old].replace(r'\subsection{'+old+'}',r'\subsection{'+new+r'}\label{'+label+'}',1)
  chunks.append(chunk)
 revised=intro+''.join(chunks)
 revised=revised.replace('Every method uses CPU affinity [38,40], which selects distinct physical cores 19 and 20 of socket 0, with two intra-op and two inter-op threads.','Every method uses the same two distinct physical CPU cores, with two intra-op and two inter-op threads. Exact affinity and device identities are in the companion records.')
 revised=revised.replace('They use affinity [38,39] (two SMT siblings of one physical core), six intra-op and two inter-op threads.','They use two SMT siblings of one physical core, six intra-op and two inter-op threads; exact affinity is recorded in the companion files.')
 revised=revised.replace('The following study therefore validates a matched prefill transmitter/receiver, including independent recovery of its new bitstreams.',r'The matched prefill transmitter/receiver in S\ref{sup:timing-compiled} independently validates recovery of its new bitstreams.')
 # These labels include the leading S in their counter representation.
 revised=revised.replace(r'S\ref{sup:timing-compiled}',r'\ref{sup:timing-compiled}')
 revised=revised.replace('S7.4',r'\ref{sup:timing-compiled}').replace('S7.5',r'\ref{sup:timing-compiled-quality}').replace('S7.1',r'\ref{sup:timing-main}')
 s=s[:start]+revised+s[end:]
 for old,new in {'not receiver successes or NULLs.': 'not receiver successes or valid NULL frames.', 'every gray failure and valid NULL frames': 'gray failures and valid NULL frames', '(source and NULL), totaling': '(source and NULL frames), totaling', 'Rejected/NULL images remain gray in every metric denominator.': 'Rejected frames and valid NULL frames both retain gray outputs in every metric denominator.', 'Valid paid NULL is listed within gray outputs and separately; it is not a lost denominator.': 'Valid NULL frames are counted within gray outputs and separately; both valid NULL frames and decoding failures retain their denominators.', 'Paid NULL\\\\': 'NULL frames\\\\', 'no NULL or gray output occurs in this error-free diagnostic.': 'no gray output, including a NULL-profile output, occurs in this error-free diagnostic.', 'complete-system comparison': 'system-level comparison', 'NULL / 500': 'NULL outputs / 500', 'source-decoding rejection or NULL.': 'source-decoding rejection or valid NULL frame.'}.items():
  s=s.replace(old,new)
 return s
