"""Typeset frozen CSV values; no metric evaluation or resampling."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parents[1]
def read(name):return list(csv.DictReader((P/'inputs'/name).open(encoding='utf8')))
metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine']
labels=['PSNR (dB) $\\uparrow$','LPIPS $\\downarrow$','DreamSim $\\downarrow$','CLIP-I $\\uparrow$']
def cell(r):
 d=3 if r['metric']=='psnr_db' else 4
 return r'\shortstack{'+f"{float(r['mean']):.{d}f}"+r'\\{}'+f"[{float(r['ci_low']):.{d}f}, {float(r['ci_high']):.{d}f}]"+'}'
def table(rows,groups,groupkey,headers):
 out=[r'\begin{center}\small\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.55}',r'\begin{tabular}{l'+'c'*len(groups)+'}',r'\toprule','Metric & '+' & '.join(headers)+r'\\\midrule']
 for metric,label in zip(metrics,labels):
  out.append(label+' & '+' & '.join(cell(next(r for r in rows if r['metric']==metric and r[groupkey]==g)) for g in groups)+r'\\')
 return '\n'.join(out+[r'\bottomrule\end{tabular}\end{center}'])
base='https://github.com/daiqizai/var-comm/blob/2c303b615f73edd0a7a971308c0df8bd5bb885d0/'
def link(path,label):return r'\href{'+base+path+'}{'+label+'}'
parts=[r'''
\documentclass[11pt,letterpaper]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,url}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={NeST-Com: Supplementary Configurations and Results},pdfauthor={Anonymous Authors}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\emergencystretch=2em
\newcommand{\Dc}{D_c}
\begin{document}
\begin{center}\Large\bfseries NeST-Com: Fixed-Budget Next-Scale Token Communication\\[4pt]\large Supplementary Configurations and Results\end{center}
\section*{S1. Configuration and calibration}
This supplement collects the completed experiments supporting the accompanying letter. It reformats existing results; it adds no model inference, wireless trial, metric evaluation, training, or statistical resampling. All differences below retain the original comparison direction and population.

\textbf{Models and training.} NeST and WHOLE share the VAR tokenizer, null-class prior, entropy coder, and adapted image decoder $\Dc$. Its adaptation uses 20\,000 ImageNet training images and continuous, fully quantized, and prefix-completion-interpolated latents. The convolutional Latent JSCC uses MSE, LPIPS, and latent-consistency losses and the same decoder; its final checkpoint is at 100k updates. SwinJSCC uses the same 20\,000 images, $256\times256$ input, paid side information, and a final 200k-update checkpoint. The Swin selection record reports \texttt{BUDGET\_CAP\_WHILE\_IMPROVING}; no claim of full convergence is made. The selected checkpoint, calibration history, stopping checks, and original training plots are available through the linked index.

\textbf{Ada adaptation.} Ada retains LlamaGen-L/VQ16 and genuine autoregressive prefix arithmetic coding. A fixed ConvNeXt predicts the class at TX; its 10-bit ID travels in the protected body, giving $b+39\leq k_a$. RX uses the recovered class and prefix. All 1--256 independently terminated prefixes are eligible. Main-curve tail generation uses CFG=4, temperature 1, top-$k=2000$, top-$p=1$, FP32 with TF32 off. The fixed single-frame adaptation does not reproduce the original long-term average-budget controller or ARQ. At N1024/19~dB all 256 tokens are sent; remaining source capacity therefore does not indicate fitting failure.

\textbf{Calibration.} The 100-image pilot uses one channel realization per source. Final evaluation uses all 1000 calibration images with three realizations per source, including fixed-gray failures. Selection maximizes DINOv2 cosine, first averaging realizations and then equally weighting sources. Exact ties use the frozen candidate order. DINO-based selection is related to feature-based evaluation; the controlled interventions separately examine the mechanism.

N1024 target screening starts at scales 7--9 and includes full-scale $(10,0)$ candidates in final evaluation. N512 screens all 7 WHOLE and 25 Quarter targets against 12 MCSs: $\log_2 M\in\{2,4,6\}$ and rates $\{1/2,2/3,3/4,5/6\}$. Each method/SNR advances the top three and a maximum-endpoint QPSK rate-1/2 anchor. Ada uses the same N512 MCS pool and up to 256 tokens. Each budget is independently calibrated; its results describe separately selected systems, rather than an isolated causal change in budget or representation order.
''']
parts += [link('results/baseline_strength_20261011/Swin_1to19/selected_checkpoint.json','Swin checkpoint and training status')+'; '+link('results/baseline_strength_20261011/training_figures_v1/training_calibration_curves.pdf','training and calibration curves')+'; '+link('results/nestcom_n512_20261028_v1/protocol.json','N512 frozen protocol')+'.']
parts += [r'''
\clearpage\section*{S2. Statistical populations and reception accounting}
The main population is common500; Kodak has 24 sources. The endpoint, receiver, and E1/E2 controls use the existing fixed 100-source subset, not a new independent blind test. Full source IDs, preprocessing identities, and native pixel hashes join the records. Three original outcomes are averaged within a source before equal-source means and paired differences. Percentile 95\% intervals use 10\,000 source resamples; they are pointwise, not simultaneous intervals. Marginal intervals are never subtracted.

N512 methods share the registered source/repeat AWGN identities. N1024 preserves historical noise identities and is not presented as sharing noise across every method or budget. E1/E2 reuse those actual reception states; receiver counterfactuals are not new wireless frames. The 100-source bank is matched by its complete source order, not sliced from the 500-source bank. Both registered banks use seed 2026100701. Full hashes and source orders remain in the statistical identity files.

\textbf{Main matched-SNR failures.} N1024 has no rejected or failed reconstructions in 1500 trials per method/SNR; continuous Latent JSCC has no discrete CRC rejection. N512 failures below are body-CRC rejections, retained as fixed gray in all means. Header, parser, and source-decoder rejection counts are zero. No evaluated N512 source requires a NULL frame.
\begin{center}\begin{tabular}{lrrrr}\toprule
Method & 4 dB & 10 dB & 19 dB & Trials per SNR\\\midrule
NeST & 4 & 1 & 0 & 1500\\
WHOLE & 7 & 0 & 0 & 1500\\
Ada & 5 & 0 & 0 & 1500\\\bottomrule
\end{tabular}\end{center}
\textbf{No-feasible-endpoint behavior.} The actual implementations differ. N512 sends a paid NULL profile and two zero payload bits through the complete 68+444-symbol frame, returning gray. N1024 raises an execution error when no independently terminated legal stream fits; this case did not occur in its evaluated sources. The revision preserves both frozen implementations. Seven synthetic CPU checks of the original source-selection/RX-projection functions verify no-fit and ordinary-fit branches, with no PHY or model calls. Source hashes and line ranges are in the implementation audit; these checks do not constitute new wireless qualification.

\textbf{Granularity and receiver controls.} Mean terminated-length checks are 3.55/5.55/9.49/17.58 for Whole/Half/Quarter/Eighth. They compare already terminated streams, not extra model forwards or measured latency. Quarter is a predefined configuration, not a demonstrated optimum. The A0/B0/C0 differences are ordered interventions on the same prefixes; they are not a universal percentage decomposition. A0 still performs prior completion. At the final scale C0 and B0 coincide because no finer context is used.
''']
for N,snr in [(512,4),(512,10),(1024,10)]:
 rs=[r for r in read('e1_summary.csv') if int(r['N'])==N and int(r['snr_db'])==snr]
 ds=[r for r in read('e1_paired_deltas.csv') if int(r['N'])==N and int(r['snr_db'])==snr]
 parts += [rf'\clearpage\section*{{S3. Ada tail sensitivity: $N={N}$, {snr} dB}}',r'All entries use the same 100 sources and three original repetitions (300 logical results per arm). Each cell is a mean or direct paired difference with its 95\% source-bootstrap interval. Higher PSNR/CLIP-I and lower LPIPS/DreamSim are favorable.']
 parts += [r'\subsection*{Absolute metrics}',table(rs,['Ada-sampling','Ada-argmax','NeST'],'arm',['Ada sampling','Ada argmax','NeST']),r'\subsection*{Direct source-paired differences}',table(ds,['Ada-argmax minus Ada-sampling','NeST minus Ada-argmax','NeST minus Ada-sampling'],'comparison',[r'\shortstack{Argmax minus\\sampling}',r'\shortstack{NeST minus\\argmax}',r'\shortstack{NeST minus\\sampling}'])]
 failures={a:next(r['fixed_gray_count'] for r in rs if r['arm']==a) for a in ['Ada-sampling','Ada-argmax','NeST']}
 parts.append('Fixed-gray counts per 300: '+', '.join(f'{a}: {n}' for a,n in failures.items())+'.')
 if N==512:parts.append('The deterministic tail lowers Ada PSNR and worsens its three feature metrics relative to sampling. NeST retains higher PSNR and lower LPIPS/DreamSim than argmax; the CLIP-I difference interval crosses zero. The sampling configuration remains the main baseline.')
 else:parts.append('The argmax PSNR difference from sampling includes zero, while DreamSim becomes worse. NeST retains higher PSNR than argmax; Ada argmax retains better DreamSim and CLIP-I, and the LPIPS interval crosses zero. Decoding sensitivity does not remove other complete-system differences.')
 parts.append(r'Only unreceived-tail token selection changes. Context/KV is rebuilt from the actual received prefix and follows each new greedy token. The paid class, CFG composition, entropy-decoding conditional CDF, model, decoder, and numerical route remain fixed. No sampling-tail KV is reused. Full-token receptions reuse the existing output exactly; failed/NULL inputs remain gray.')
parts += [r'''
\clearpage\section*{S4. Completion versus direct coarse decoding}
R0 decodes $\Dc(\sum_{s=1}^{m}C_s(q_s))$ from the actual received WHOLE prefix. R1 is the existing VAR-completed reconstruction of that same prefix. Both retain the original ten-scale indexing, scale-dependent residual transforms, full latent grid, frozen adapted $\Dc$, clipping, and metric preprocessing. Unreceived scales contribute nothing to R0; codebook ID zero is not used as zero contribution. Neither arm uses unsent source tokens or a continuous source latent. This is distinct from A0/B0/C0, which intervene on received boundary-token use.
''']
for snr in [4,10]:
 rs=[r for r in read('e2_summary.csv') if int(r['snr_db'])==snr];ds=[r for r in read('e2_paired_deltas.csv') if int(r['snr_db'])==snr]
 combined=[]
 for r in rs:combined.append(dict(r,column=r['arm']))
 for r in ds:combined.append(dict(r,column='delta'))
 parts += [rf'\subsection*{{$N=512$, {snr} dB: 100 sources, 300 results per arm}}',table(combined,['R0-coarse-only','R1-VAR-completed','delta'],'column',['R0 coarse','R1 completed','R1 minus R0'])]
 parts.append('Fixed-gray count per arm: '+next(r['fixed_gray_count'] for r in rs)+'.')
parts += [r'Completion improves all three feature metrics at both points; PSNR improves clearly at 10~dB, while its 4~dB interval includes zero. These results concern completion in the existing adapted receiver, including its decoder training.']
parts += [r'''
\clearpage\section*{S5. Additional results and processing cost}
\textbf{N2048.} On 100 sources, the existing experiment uses two paid 1024-symbol subframes. Complete representations at 19~dB give NeST $+1.806$~dB PSNR over ARPC, but worse DISTS\footnote{K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, ``Image quality assessment: Unifying structure and texture similarity,'' \emph{IEEE Trans. Pattern Anal. Mach. Intell.}, vol. 44, no. 5, pp. 2567--2581, May 2022.}: $+0.0054\,[0.0024,0.0086]$. DISTS is an additional perceptual measure in this historical experiment, not a replacement for the four main metrics.

\textbf{SNR mismatch.} With both WHOLE/PARTIAL transmit policies fixed at 10~dB and RX using the true noise variance, all 300 receptions per method on 100 sources are rejected at 8~dB. All succeed at 10 and 12~dB; for each source and method the recovered prefix and reconstruction are identical at those two points. PARTIAL exceeds WHOLE by $0.660\,[0.515,0.814]$~dB PSNR. These tested points do not locate the rejection threshold or measure the steepness of the 8--10~dB transition.

\textbf{Processing cost.} The following software times are from an RTX PRO~6000 with six CPU cores, batch one, 16 development images, and three measured repetitions after warm-up. They are measured processing times, distinct from the count of length checks. Ada was not timed under this protocol.
\begin{center}\small\begin{tabular}{llrrr}\toprule
Method & SNR (dB) & TX mean (ms) & RX mean (ms) & Total mean (ms)\\\midrule
''']
for r in read('timing_summary.csv'):
 label={'NEST':'NeST','P100000':'Latent JSCC','Swin200000':'SwinJSCC','BPG':'BPG + LDPC','ARPC_ADAPTED':'ARPC + LDPC'}.get(r['method'],r['display_name'].split('(')[0].strip())
 parts.append(label+' & '+r['snr_db']+' & '+' & '.join(f"{float(r[k]):.2f}" for k in ['tx_ms_mean','rx_ms_mean','total_ms_mean'])+r'\\')
parts += [r'\bottomrule\end{tabular}\end{center}',r'All underlying means, medians, and 95th percentiles remain in \texttt{inputs/timing\_summary.csv}.',r'''
\clearpage\section*{S6. Preserved qualitative cases and source records}
The previous two-case N512 figure is retained below, with full-field images and LPIPS/DreamSim per displayed realization. Its source-ID selection history is unchanged. The main letter now uses the already displayed lamp as an editorial example, with one identical ROI across all methods; this is not a new preregistered or random selection. The crop is $x\in[80,180)$, $y\in[0,100)$ in the original 256-pixel grid. Main-figure PSNR/LPIPS values are for the full image, not the crop. All displayed images use N512, 10~dB, original repetition seed 6201.
\begin{center}\includegraphics[width=6.0in]{supplemental/N512_previous_figure3.pdf}\end{center}
The earlier N1024 figure remains in the supplemental asset directory. The native-array identifiers, source pixel hashes, wire inputs, display-image hashes, and full-precision metrics are retained in the source map and prior selection record. No image content was enhanced or repaired.
\subsection*{Machine-readable evidence}
''']
for path,label in [('results/nestcom_evidence_20261029_v1/e1/REPORT_zh.md','E1 report and complete frame/source/paired data'),('results/nestcom_evidence_20261029_v1/e2/REPORT_zh.md','E2 report and complete frame/source/paired data'),('results/nestcom_evidence_20261029_v1/statistics_identity.json','100-source statistics identity'),('results/nestcom_n512_20261028_v1/delivery/REPORT.md','N512 formal results and policies'),('results/wcl_revision_20261027_v1/p1_bootstrap_clarification_v1.md','Historical mismatch: index-only bootstrap reuse clarification'),('results/five_method_timing_20261020_v1/delivery/summary_ms_per_image.csv','Timing data')]:
 parts.append(link(path,label)+r'\par')
parts += [r'\end{document}']
(P/'supplementary.tex').write_text('\n'.join(parts),encoding='utf8',newline='\n')
print('Supplement generated from frozen inputs; no statistics recomputed.')
