"""Readable supplementary tables from the same exact CSVs as the main paper."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parents[1]
def rows(name):
 with (P/'inputs'/name).open(newline='',encoding='utf8')as f:return list(csv.DictReader(f))
def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def unique(vs,**keys):
 a=[v for v in vs if all(str(v[k])==str(x)for k,x in keys.items())];assert len(a)==1;return a[0]
def ci(v):return f"{float(v['mean_delta']):+.5f} [{float(v['ci95_low']):.5f}, {float(v['ci95_high']):.5f}]"
def table(head,body,widths=None):
 cols=widths or ('l'+'r'*(len(head)-1))
 return r'\begin{longtable}{@{}'+cols+r'@{}}\toprule'+'\n'+' & '.join(head)+r'\\\midrule\endhead'+'\n'+'\n'.join(' & '.join(map(str,v))+r'\\'for v in body)+'\n'+r'\bottomrule\end{longtable}'+'\n'
names={'final_NeST':'NeST','original_NeST':'Quarter','WHOLE_final_receiver':'WHOLE','Ada_frozen':'Ada','BPG_LDPC':'BPG','Latent_JSCC':'Latent','SwinJSCC':'Swin','ARPC':'ARPC'}
metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine','dists']
metricnames=dict(zip(metrics,['PSNR','LPIPS','DreamSim','CLIP-I','DISTS']))
S=rows('e1/summary.csv');D=rows('e1/paired_deltas.csv');C=rows('e1/capacity_summary.csv')
policies=json.loads((P/'inputs/e1/frozen_policy_metadata.json').read_text())['policies']
sections=[]
sections.append(r'''\section{System settings and interpretation}
This V8 supplement accompanies the token-wise configuration of NeST-Com, evaluated for source-referenced perceptual reconstruction. DISTS and DreamSim are the primary display metrics; PSNR, LPIPS and CLIP-I retain the complete frozen-system results. The editorial focus does not change training, calibration or any scientific output. The main method is the original frozen tokenizer, VAR-d16, adapted $D_c$, and argmax receiver with a finer legal endpoint set. Historical Quarter, Kodak, coarse-only and Ada-argmax results remain in their original versioned deliveries; they are not relabelled as new token-wise tests. No model training or parameter scan was performed for this evaluation.

\subsection{Policies and actual frame accounting}
The header contains a 12-bit profile and CRC16, convolutionally protected in 68 symbols. The body pays a 13-bit length and CRC16; Ada additionally pays its 10-bit predicted class. Independent termination produces actual arithmetic streams. Main NeST and WHOLE targets and MCSs are shown below. Both budgets use $E_s=2$, and the smaller frame has 444 body symbols rather than 956. NeST retains the existing targets and MCSs while using integer-token endpoints. Family-specific calibration and target choices are therefore distinct from the matched-ceiling controls.
''')
body=[]
for N,snr in [(1024,s)for s in [4,7,10,19]]+[(512,s)for s in[4,10,19]]:
 vals={m:unique(policies,method=m,N=N,snr_db=snr)for m in['NEST','WHOLE']}
 def target(v):return '('+','.join(map(str,v['target']))+')'
 n,w=vals['NEST'],vals['WHOLE'];assert(n['q'],n['rate'])==(w['q'],w['rate'])
 capacity=unique(C,method='final_NeST',N=N,snr_db=snr)['capacity_bits_mean']
 body.append([N,snr,target(n),target(w),f"{2**int(n['q'])}-QAM",n['rate'],f"{float(capacity):.0f}"])
sections.append(table(['$N$','SNR','NeST target','WHOLE target','Modulation','Rate','Source cap.'],body))
sections.append(r'''QPSK is listed as 4-QAM. Source capacity is the body information length minus its 29-bit length/CRC overhead. A full 256-token Ada representation can leave unused capacity; that is not a fitting failure. The $N=1024$ no-fit branch raises an error (none observed), whereas $N=512$ has a paid NULL branch. Receiver failures and NULL retain the fixed gray image and the complete statistical denominator.

The original 4/10/19-dB policies use an initial 100-image screening stage with one channel realization per image and final cal1000 evaluation with three realizations per image. The 7-dB extension uses the previously frozen finite 100-image, five-MCS calibration. No test-set score reselects a policy here. DINOv2 cosine was the calibration objective. The DISTS, DreamSim and CLIP feature criteria are related to that objective and do not constitute independent human perceptual judgments.

\subsection{Models, metrics, and source identity}
The decoder adaptation uses 20\,000 ImageNet training images, MSE plus $0.1$ LPIPS, and a 50/25/25 mixture of continuous, fully quantized, and interpolated prefix-completed latents. The final category interpolates continuous latents with prior completions conditioned on eight received source scales. Tokenizer, VAR prior and codebook are frozen. ARPC also uses next-scale representations, and must not be treated as a raster-order control. Ada is a predicted-class fixed-frame adaptation, with ConvNeXt category prediction, paid category bits, original conditional entropy CDFs, and CFG4/T1/top-$k$2000/top-$p$1 tail generation. It does not reproduce the long-term queue or ARQ controller.

DISTS uses the official revision \texttt{1267d8cb626c98706db3697422701c56a85ebf2e}, learned alpha/beta weights and VGG16 ImageNet weights, in evaluation FP32 without AMP. Its input is the native $256\times256$ float32 RGB array without extra cropping, resizing, alignment or PNG quantization. DreamSim uses the frozen ensemble and $224\times224$ floating-point bicubic-antialias preprocessing; LPIPS-Alex, PSNR and CLIP-I retain the established routes. The previously missing external 7-dB CLIP values were scored separately in the matching Torch2.10 route, with no cross-version feature-cache mixing.

Full source IDs, native pixel hashes, preprocessing, actual receiver inputs, model hashes and numerical routes control every reuse decision. Common500 means first average three channel events per source. The original $10\,000\times500$ source-bootstrap bank supplies pointwise percentile intervals and direct paired differences. The control100 bank has its own fixed source order and $10\,000\times100$ indices; it is not a slice of the 500-source bank. No multiplicity-adjusted simultaneous claim is made. The exact hashes and orders accompany the CSVs.
''')
sections.append(r'\section{Complete quality results}'+'\n'+r'\subsection{Five-metric source means}'+'\n'+r'All means retain failures. Pointwise intervals and direct paired intervals are provided in the full-precision CSVs; the compact table reports means.'+'\n')
body=[]
for N,snr,method in sorted(set((int(v['N']),int(v['snr_db']),v['method'])for v in S)):
 vals={m:unique(S,method=method,N=N,snr_db=snr,metric=m)for m in metrics}
 body.append([N,snr,names[method]]+[f"{float(vals[m]['mean']):.{3 if m=='psnr_db'else 4}f}"for m in metrics])
sections.append(table(['$N$','SNR','Method',r'PSNR$\uparrow$',r'LPIPS$\downarrow$',r'DreamSim$\downarrow$',r'CLIP-I$\uparrow$',r'DISTS$\downarrow$'],body))
sections.append(r'\subsection{Direct paired differences}'+'\n'+r'NeST minus the stated reference. Positive PSNR/CLIP-I and negative distance differences favor NeST. A confidence interval spanning zero does not establish a direction.'+'\n')
body=[]
for v in D:
 if v['candidate']=='final_NeST'and v['reference']in['WHOLE_final_receiver','original_NeST','Ada_frozen']:
  body.append([v['N'],v['snr_db'],names[v['reference']],metricnames[v['metric']],ci(v)])
sections.append(table(['$N$','SNR','Reference','Metric',r'Difference [95\% CI]'],body,'rrlll'))
sections.append(r'\subsection{Failures and denominator}'+'\n'+r'The following nonzero gray-image counts are verified from native pixel hashes and receiver statuses. Other cells have zero observed gray outputs. Zero observed failures is not zero underlying error probability. Historical noise namespaces differ between some methods and budgets, so count differences alone do not establish a causal granularity effect on frame error rate.'+'\n')
F=rows('e1/failure_counts_native_v2.csv');body=[[v['N'],v['snr_db'],names[v['method']],v['native_gray_pixel_count'],v['logical']]for v in F if int(v['native_gray_pixel_count'])]
sections.append(table(['$N$','SNR','Method','Gray count','Denominator'],body,'rrlrr'))
sections.append(r'''\section{Matched granularity and receiver interventions}
\subsection{Endpoint control}
The four grids use control100, $N=1024$, 10 dB, 16-QAM rate 2/3, a common maximum token position $(8,126)$, and the actual shared noise vectors for repeats 6201--6203. Only legal termination points change. This WHOLE control has five permitted complete-scale endpoints; the main WHOLE system has a different target and must not be substituted. There are 1200 logical outcomes, 891 unique physical inputs, 591 newly executed physical events, and 300 exact physical reuses. Independent RX reconstruction uses 100 new and 197 reused receiver inputs.
''')
S3=rows('e3/summary.csv');C3=rows('e3/efficiency.csv');body=[]
for method in ['whole','quarter','eighth','token']:
 c=unique(C3,method=method)
 body.append([method.title(),int(float(c['candidate_count'])),f"{100*float(c['utilization']):.3f}",f"{float(c['actual_length_comparisons']):.2f}"]+[f"{float(unique(S3,method=method,metric=m)['mean']):.4f}"for m in ['dists','dreamsim']])
sections.append(table(['Grid','Ends',r'Util. (\%)','Length checks','DISTS','DreamSim'],body))
sections.append(r'The quality experiment reuses previously finalized streams, so its new source traversals and independent finalizations are zero. Candidate-set cardinality and descending comparisons must not be described as new network forwards. Fresh deployment computation and measured costs belong to S7. The legacy main-evaluation field named length\_checks sometimes records the entire evaluated endpoint set; it is not substituted for the matched-control stopping count shown here.'+'\n')
body=[[v['candidate'].title()+'--'+v['reference'].title(),metricnames[v['metric']],ci(v)]for v in rows('e3/paired_deltas.csv')]
sections.append(table(['Comparison','Metric',r'Difference [95\% CI]'],body,'lll'))
sections.append(r'''\subsection{Ordered boundary-token interventions}
A0 retains prior completion but omits the extra boundary-token correction and updated conditioning. B0 inserts the direct residual correction while keeping the A0 future-condition path. C0 applies both. All three arms use the same actual received prefix and retain failures. This is an ordered intervention on the full received boundary, not an additive decomposition of the tokens beyond Quarter. The complete set contains 1800 logical outcomes. It requires 394 new A0/B0 renders for 197 unique prefixes; two qualification C0 replays are pixel-identical to the official output. No new PHY events are generated.
''')
body=[[v['snr_db'],v['candidate']+'--'+v['reference'],metricnames[v['metric']],ci(v)]for v in rows('e4/paired_deltas.csv')]
sections.append(table(['SNR','Comparison','Metric',r'Difference [95\% CI]'],body,'rlll'))
sections.append(r'At $N=1024$, 19 dB, 1482 of 1500 NeST events have a partial final-scale boundary and 18 transmit the full representation. There is no finer scale to condition, hence B0 and C0 apply the identical final residual map. This structural check introduces no new renders, and historical Quarter A0/B0 means remain labelled Quarter.'+'\n')
sections.append(r'''\section{Equal information budgets}
The information budget includes 12 profile bits, header CRC16, length13 and body CRC16: 57 shared bits, plus 10 class bits for Ada. The arithmetic-code limits are $B-57$ for WHOLE/NeST and $B-67$ for Ada. The fixed budgets are $B=620,1212,1302,1940,2248,2577$ bits. WHOLE/NeST share maximum $(10,0)$; Ada permits 1--256 tokens. Actual information fields and arithmetic streams are independently parsed at RX without PHY noise. The receiver cannot inspect unsent source tokens. This comparison removes wireless and MCS effects, but retains differences among tokenizer, prior, conditioning and decoder.

WHOLE and NeST have one deterministic output per source/budget. Ada uses the three fixed generation anchors 2026103001--2026103003, with the original per-source seed mechanism; metrics, not RGBs, are averaged. There are at most 15\,000 logical outputs and 9000 source-level units. These are generation repeats, not new channel realizations.

\textbf{Draft: the Ada reconstruction stage and resulting complete five-metric statistics remain pending. No quality result is filled from partial outputs.}

\section{Fixed-policy channel threshold}
The 10-dB policies of WHOLE, NeST and Ada remain frozen at actual SNRs 8, 8.5, 9, 9.5 and 10 dB. Each cell has 100 sources and 20 new AWGN vectors, hence 2000 actual complete frames and 30\,000 frames in total. The source's original repeat-zero payload, waveform, session, counter and scrambler identity are retained; only noise changes. RX uses the true noise variance. Ada's original repeat-zero generation identity remains fixed across noise and SNR.

\textbf{Draft: the 30\,000 physical events have exited successfully, but the complete distance/conditional-quality summary awaits the final missing gray-image scores.} Successful payload identities are authenticated before image reuse; no unexpected accepted source input occurred. These points do not establish low-error-rate operation or general channel robustness.

\section{Selected reconstructions}
The main figure uses source index 84 (television) and 143 (patterned leather), both at $N=1024$, 7 dB. They were selected from the complete common500 contact sheets after inspection, using DISTS/DreamSim improvements and visible source fidelity. They are illustrative advantages, not random or representative samples. Original method-specific repeat-zero noise/generation identities remain fixed. Full source IDs, native array hashes, actual prefixes and all five displayed-output metrics are in the figure-source CSVs. The standalone all-method comparison PDF retains the full $256\times256$ field, including the old Quarter and four additional systems.

\section{Unified processing costs}
Timing uses the first 32 common500 IDs, batch one, two warmups and three measurements per source, one actual RTX PRO 6000 and two CPU cores. Original numerical routes are retained; six library compute threads are confined to those two logical CPUs. Model loading, disk I/O, logging and quality scoring are excluded. TX starts with in-memory canonical RGB and ends with actual transmitted symbols. RX starts with in-memory received symbols and ends with canonical RGB. GPU boundary synchronization surrounds each measured span; TX+RX is a processing sum, not radio propagation latency.

The measured operations include tokenizer, condition extraction, CDF/arithmetic coding, finalization/fitting, paid packaging, LDPC/QAM and real receiver PHY, independent source decoding, completion and image decoding. Ada includes ConvNeXt prediction; ARPC includes caption generation; BPG includes its actual search and codec invocation. BPG is CPU-only on the same host and CPU quota; necessary codec serialization uses RAM files. Fresh timing does not substitute cached source tokens or reconstructions. Successful-RX conditional time is reported separately from fast rejection. Means first average within source; median/p95 use measured calls with linear quantiles. The old six-core and component-sum results are not inserted here.

\textbf{Draft: the timing queue waits for this round's reconstruction/scoring jobs to finish before taking exclusive use of the designated GPU. All 16 groups and 32 TX/RX stages remain pending.}
''')
text=r'''\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.65in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,url}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\begin{center}\Large Supplement to NeST-Com: Fixed-Budget Next-Scale Token Communication\\[4pt]\normalsize Anonymous Authors\end{center}
\textbf{Working draft.} This version contains complete E1/E3/E4 data and explicit gates for pending E2/E5/E6 deliveries. It is not a completed publication artifact.
'''+ '\n'.join(sections)+'\n'+r'\end{document}'+'\n'
from supplement_closed_sections import enrich
text = enrich(text, P, table)
from supplement_timing import enrich_timing
text = enrich_timing(text, P, table)
(P/'supplementary.tex').write_text(text,encoding='utf8',newline='\n')
print('supplement sources generated from available hash-bound closures')
