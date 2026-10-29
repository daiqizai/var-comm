"""Typeset frozen CSV values; no metric evaluation or resampling."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parents[1]
REVIEW=P.parents[2]/'results/nestcom_focused_review_20261029_v2'
def review_rows(name):return list(csv.DictReader((REVIEW/name).open(encoding='utf8')))
failures=review_rows('failure_summary.csv')
assert len(failures)==15 and all(r['logical_frames']=='1500' for r in failures)
assert all(int(r[k])==0 for r in failures for k in ['TX_NO_FIT','NULL_received','header_reject','body_parser_reject','source_decoder_reject'])
assert all(r['payload_parsed']=='1500' for r in failures if r['N']=='1024')
for method,counts in [('NEST',[4,1,0]),('WHOLE',[7,0,0]),('ADA',[5,0,0])]:
 assert [int(next(r['body_crc_reject'] for r in failures if r['N']=='512' and r['method']==method and int(r['snr_db'])==s)) for s in [4,10,19]]==counts
def read(name):return list(csv.DictReader((P/'inputs'/name).open(encoding='utf8')))
metrics=['psnr_db','lpips_alex','dreamsim','clip_image_cosine']
labels=['PSNR (dB) $\\uparrow$','LPIPS $\\downarrow$','DreamSim $\\downarrow$','CLIP-I $\\uparrow$']
def cell(r):
 d=3 if r['metric']=='psnr_db' else 4
 return r'\shortstack{'+f"{float(r['mean']):.{d}f}"+r'\\{}'+f"[{float(r['ci_low']):.{d}f}, {float(r['ci_high']):.{d}f}]"+'}'
def table(rows,groups,groupkey,headers,tabcolsep=7):
 out=[r'\begin{center}\small\setlength{\tabcolsep}{'+str(tabcolsep)+r'pt}\renewcommand{\arraystretch}{1.55}',r'\begin{tabular}{l'+'c'*len(groups)+'}',r'\toprule','Metric & '+' & '.join(headers)+r'\\\midrule']
 for metric,label in zip(metrics,labels):
  out.append(label+' & '+' & '.join(cell(next(r for r in rows if r['metric']==metric and r[groupkey]==g)) for g in groups)+r'\\')
 return '\n'.join(out+[r'\bottomrule\end{tabular}\end{center}'])
base='https://github.com/daiqizai/var-comm/blob/93205a2e007ac587e28246cd399a531347a4f3f7/'
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
This supplement collects the experiments and implementation records supporting the accompanying letter. The image-quality results and bootstrap intervals retain their original populations and numerical values. The current revision adds read-only accounting of the corresponding formal events and separately identifies the bounded protocol checks; it adds no model inference, training, metric evaluation, or statistical resampling.

\textbf{Models and training.} NeST and WHOLE share the VAR tokenizer, VAR-d16 null-class prior, entropy coder, and adapted image decoder $\Dc$. Its adaptation uses 20\,000 ImageNet training images and continuous, fully quantized, and prefix-completion-interpolated latents. The convolutional Latent JSCC uses MSE, LPIPS, and latent-consistency losses and the same decoder; its final checkpoint is at 100k updates. SwinJSCC uses the same 20\,000 images, $256\times256$ input, paid side information, and a final 200k-update checkpoint. The selected Swin checkpoint, calibration history, training-budget stopping rule, and original training plots are available through the linked index.

\textbf{Ada adaptation.} Ada retains LlamaGen-L/VQ16 and genuine autoregressive prefix arithmetic coding. A fixed ConvNeXt predicts the class at TX; its 10-bit ID travels in the protected body, giving $b+39\leq k_a$. RX uses the recovered class and prefix. All 1--256 independently terminated prefixes are eligible. Main-curve tail generation uses CFG=4, temperature 1, top-$k=2000$, top-$p=1$, FP32 with TF32 off. The fixed single-frame adaptation does not reproduce the original long-term average-budget controller or ARQ. At N1024/19~dB all 256 tokens are sent; remaining source capacity therefore does not indicate fitting failure.

\textbf{Calibration.} The 100-image pilot uses one channel realization per source. Final evaluation uses all 1000 calibration images with three realizations per source, including fixed-gray failures. Selection maximizes DINOv2 cosine, first averaging realizations and then equally weighting sources. Exact ties use the frozen candidate order. DINO-based selection is related to feature-based evaluation; the controlled interventions separately examine the mechanism.

N1024 target screening starts at scales 7--9 and includes full-scale $(10,0)$ candidates in final evaluation. N512 screens all 7 WHOLE and 25 Quarter targets against 12 MCSs: $\log_2 M\in\{2,4,6\}$ and rates $\{1/2,2/3,3/4,5/6\}$. Each method/SNR advances the top three and a maximum-endpoint QPSK rate-1/2 anchor. Ada uses the same N512 MCS pool and up to 256 tokens. Each budget is independently calibrated; its results describe separately selected systems, rather than an isolated causal change in budget or representation order.
''']
parts += [link('results/baseline_strength_20261011/Swin_1to19/selected_checkpoint.json','Swin checkpoint and training status')+'; '+link('results/baseline_strength_20261011/training_figures_v1/training_calibration_curves.pdf','training and calibration curves')+'; '+link('results/nestcom_n512_20261028_v1/protocol.json','N512 frozen protocol')+'.']
configuration=json.loads((REVIEW/'configuration_audit.json').read_text(encoding='utf8'))
assert configuration['model']['name']=='VAR-d16' and configuration['CDF']['total']==2**24
assert configuration['CDF']['minimum_frequency']==1 and configuration['CDF']['integer_interval_bits']==32
state=configuration['model']['state_hashes']
parts += [r'''
\clearpage\subsection*{S1.1. Frozen model, coder and policies}
The following configuration is verified against the actual NeST/WHOLE loader and formal receiver receipts. State SHA-256 prefixes identify the weights below; full hashes, file hashes, numerical flags and exact implementation paths are supplied in \path{results/nestcom_focused_review_20261029_v2/configuration_audit.json}.
\begin{center}\small\renewcommand{\arraystretch}{1.2}
\begin{tabular}{p{1.45in}p{4.75in}}\toprule
Component & Frozen setting\\\midrule
Prior & VAR-d16; null class; greedy completion without CFG\\
Tokenizer & 4096 entries; 32 latent channels; ten scales\\
Numerical route & FP32 model logits; TF32 off; deterministic kernels; flash/fused attention off\\
CDF & CPU float64 softmax; total $2^{24}$; minimum frequency one\\
Integer frequencies & $\lfloor p_i(2^{24}-4096)\rfloor+1$; all remaining mass goes to the largest-probability token (first index for a tie)\\
Arithmetic coder & 32-bit interval; independent termination per candidate\\
''',f"VAE state & \\texttt{{{state['vae'][:16]}}}\\\\",f"VAR state & \\texttt{{{state['var'][:16]}}}\\\\",f"$D_c$ state & \\texttt{{{state['decoder'][:16]}}}\\\\",r'''
Runtime & NeST/WHOLE receipt: Python 3.10.12, PyTorch 2.11.0+cu128, CUDA 12.8\\\bottomrule
\end{tabular}\end{center}
The entropy-decoding CDF does not use generation-stage CFG. These identities specify the numerical implementation; FP32 alone is not a guarantee of cross-hardware bitwise agreement.
\begin{center}\small\setlength{\tabcolsep}{8pt}
\begin{tabular}{rlrlrr}\toprule
$N$ & Method & SNR & Target & $\log_2 M$ & Rate\\\midrule
''']
policies=review_rows('frozen_policies_compact.csv')
assert len(policies)==18
for r in policies:
 label={'NEST':'NeST','WHOLE':'WHOLE','ADA':'Ada'}[r['method']]
 target='256 tokens' if r['target']=='p256' else '$'+r['target']+'$'
 parts.append(f"{r['N']} & {label} & {r['snr_db']} & {target} & {r['q']} & {r['rate']}"+r'\\')
parts += [r'\bottomrule\end{tabular}\end{center}',r'These are calibrated targets. Online fitting can select a shorter legal endpoint under the same MCS. SNR is in dB. Full candidate identifiers are retained in \texttt{frozen\_policies\_compact.csv}.']
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
\textbf{Historical no-feasible-endpoint behavior.} The evaluated implementations differ. N512 sends a paid NULL profile and two zero payload bits through the complete 68+444-symbol frame, returning gray. The frozen N1024 implementation raises an execution error when no independently terminated legal stream fits; no evaluated N1024 source encountered this case. The historical implementations and results remain unchanged. The separately versioned N1024 protocol extension and its bounded checks are documented in S7.

\textbf{Granularity and receiver controls.} Mean terminated-length checks are 3.55/5.55/9.49/17.58 for Whole/Half/Quarter/Eighth. They compare already terminated streams, not extra model forwards or measured latency. Quarter is a predefined configuration, not a demonstrated optimum. The A0/B0/C0 differences are ordered interventions on the same prefixes; they are not a universal percentage decomposition. A0 still performs prior completion. At the final scale C0 and B0 coincide because no finer context is used.
''']
energies=review_rows('energy_summary.csv')
logical=[r for r in energies if r['weighting']=='logical']
assert len(logical)==15 and all(int(r['rows'])==1500 and int(r['sources'])==500 for r in logical)
parts += [r'''
\clearpage\subsection*{S2.1. Actual formal-frame energy}
The transmitted body follows packing, LDPC/RV0 rate matching, public mask XOR, and QAM modulation. The receiver reverses the mask by changing the LLR signs before LDPC decoding. The deterministic mask uses public protocol/session/frame-counter identities; it is not secret side information. Its implementation identifier is
\path{WCL_T1_public_PCG64_SHA256_protocol_session_counter_after_RV0_v1}.

The following table joins the current formal source/policy/repetition records to their original physical-event files and stored energies. It covers 22\,500 logical frames and 20\,613 distinct physical-event paths: N1024 WHOLE/PARTIAL and N512 WHOLE/NeST/Ada at 4/10/19~dB. Each row equally weights 500 sources with three repetitions each. Every method/SNR row has 1500 distinct event paths; some paths are shared between arms, explaining the smaller cross-arm total. Method-group logical and physical-unique summaries are both provided in the CSV. No older-policy energy table substitutes for these events.

Let $\rho=E_{\mathrm{frame}}/(2N)$. The table gives its mean, population standard deviation (SD, divisor 1500), minimum and maximum, together with mean $E_{\mathrm{frame}}/N$. The corresponding SD and extremes of $E_{\mathrm{frame}}/N$ are twice those of $\rho$; all values are retained in the CSV.
\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{rlrrrrrr}\toprule
$N$ & Method & SNR & Mean $E/N$ & Mean $\rho$ & SD $\rho$ & Min $\rho$ & Max $\rho$\\\midrule
''']
for r in logical:
 label={'NEST':'NeST','PARTIAL':'PARTIAL','WHOLE':'WHOLE','ADA':'Ada'}[r['method']]
 parts.append(f"{r['N']} & {label} & {r['snr_db']} & "+' & '.join(f"{float(r[k]):.6f}"for k in ['E_per_symbol_mean','rho_mean','rho_std','rho_min','rho_max'])+r'\\')
parts += [r'\bottomrule\end{tabular}\end{center}',r'''
All 4500 N512/4~dB logical frames use QPSK and have exactly $E_{\mathrm{frame}}=1024$ and $\rho=1$; the 68 header symbols each have energy two. Higher-order QAM exhibits finite-frame energy variation. The largest absolute paired-arm ratio of mean energies below is 0.0063~dB; the means are close without asserting equality of individual frame energies.
\clearpage
\begin{center}\small\setlength{\tabcolsep}{6pt}
\begin{tabular}{rrrrr}\toprule
$N$ & SNR & $10\log_{10}(\bar E_P/\bar E_W)$ (dB) & Mean $\Delta\rho$ & $r(\Delta E,\Delta p)$\\\midrule
''']
for r in review_rows('paired_energy.csv'):
 cor='---' if not r['delta_energy_delta_padding_Pearson_r'] else f"{float(r['delta_energy_delta_padding_Pearson_r']):+.4f}"
 parts.append(f"{r['N']} & {r['snr_db']} & {float(r['ratio_of_means_dB']):+.6f} & {float(r['delta_rho_mean']):+.6f} & {cor}"+r'\\')
parts += [r'\bottomrule\end{tabular}\end{center}',r'''
Here $P$ is PARTIAL/NeST, $W$ is WHOLE, and $p$ is the zero-padding fraction of body information bits. The paired quantities use the same source/repetition identities; $r$ is a descriptive Pearson correlation across the 1500 pairs. A dash denotes zero energy variance. Across the nonconstant method/SNR groups, the absolute correlation between $\rho$ and padding fraction is at most 0.042. These summaries characterize the recorded transmissions; no per-frame normalization or new power-matched experiment was introduced.
\subsection*{S2.2. Actual endpoints and retained source content}
At N512/10~dB, WHOLE and NeST share target $(7,0)$ and 16-QAM rate 2/3. Direct checks of all three repetitions confirm that every source has the same TX endpoint across its repeats. The following counts are therefore source counts, not independent frame observations.
\begin{center}\begin{tabular}{rrr}\toprule
Actual endpoint & WHOLE sources & NeST sources\\\midrule
''']
endpoints=[r for r in review_rows('endpoint_distribution.csv')if r['N']=='512' and r['snr_db']=='10' and r['method'] in ('WHOLE','NEST')]
assert all(r['three_repeats_endpoint_identical']=='True' and int(r['logical_frames'])==3*int(r['sources'])for r in endpoints)
for m,k in [(6,0),(6,16),(6,32),(6,48),(7,0)]:
 counts={method:sum(int(r['sources'])for r in endpoints if r['method']==method and int(r['m'])==m and int(r['K'])==k)for method in ('WHOLE','NEST')}
 parts.append(f"$({m},{k})$ & {counts['WHOLE']} & {counts['NEST']}"+r'\\')
parts += [r'\midrule Total & 500 & 500\\\bottomrule\end{tabular}\end{center}',r'''
Both methods fall back from the target on 498/500 sources (99.6\%). NeST retains a within-scale endpoint on 445/500 (89.0\%). The first six scales contain 91 tokens and the seventh contains 64. Mean retained-token counts are 91.256 for WHOLE and 107.896 for NeST, a difference of 16.640. Source-bit utilization rises from 79.30\% to 93.46\%. The finer fallback retains additional information even when a full next scale does not fit; it does not reduce the target-fallback rate at this point.

The N1024 matched-SNR gain occurs with zero rejections in both families at every tested SNR. At N512/19~dB, both have zero rejections and NeST still gains 0.792~dB. At N512/10~dB, NeST has one additional body-CRC rejection yet gains 0.399~dB in the unconditional mean. These observations retain all fixed-gray failures and do not replace the primary population with a success-only subset.

Full-precision energies, paired energy differences, padding correlations, failure counts, source/endpoint counts, original-file hashes and the logical-to-physical mapping are in
\path{results/nestcom_focused_review_20261029_v2/}.
''']
for N,snr in [(512,4),(512,10),(1024,10)]:
 rs=[r for r in read('e1_summary.csv') if int(r['N'])==N and int(r['snr_db'])==snr]
 ds=[r for r in read('e1_paired_deltas.csv') if int(r['N'])==N and int(r['snr_db'])==snr]
 parts += [rf'\clearpage\section*{{S3. Ada tail sensitivity: $N={N}$, {snr} dB}}',r'All entries use the same 100 sources and three original repetitions (300 logical results per arm). Each cell is a mean or direct paired difference with its 95\% source-bootstrap interval. Higher PSNR/CLIP-I and lower LPIPS/DreamSim are favorable.']
 parts += [r'\subsection*{Absolute metrics}',table(rs,['Ada-sampling','Ada-argmax','NeST'],'arm',['Ada sampling','Ada argmax','NeST'],tabcolsep=12),r'\subsection*{Direct source-paired differences}',table(ds,['Ada-argmax minus Ada-sampling','NeST minus Ada-argmax','NeST minus Ada-sampling'],'comparison',[r'\shortstack{Argmax minus\\sampling}',r'\shortstack{NeST minus\\argmax}',r'\shortstack{NeST minus\\sampling}'])]
 failures={a:next(r['fixed_gray_count'] for r in rs if r['arm']==a) for a in ['Ada-sampling','Ada-argmax','NeST']}
 parts.append('Fixed-gray counts per 300: '+', '.join(f'{a}: {n}' for a,n in failures.items())+'.')
 if N==512:parts.append('The deterministic tail lowers Ada PSNR and worsens its three feature metrics relative to sampling. NeST retains higher PSNR and lower LPIPS/DreamSim than argmax; the CLIP-I difference interval crosses zero. The sampling configuration remains the main baseline.')
 else:parts.append('The argmax PSNR difference from sampling includes zero, while DreamSim becomes worse. NeST retains higher PSNR than argmax; Ada argmax retains better DreamSim and CLIP-I, and the LPIPS interval crosses zero. Decoding sensitivity does not remove other complete-system differences.')
 parts.append(r'Only unreceived-tail token selection changes. Context/KV is rebuilt from the actual received prefix and follows each new greedy token. The paid class, CFG composition, entropy-decoding conditional CDF, model, decoder, and numerical route remain fixed. No sampling-tail KV is reused. Full-token receptions reuse the existing output exactly; failed/NULL inputs remain gray.')
parts += [r'''
\clearpage\section*{S4. Completion versus direct coarse decoding}
R0 decodes $\Dc(\sum_{s=1}^{m}C_s(q_s))$ from the actual received WHOLE prefix. R1 is the existing VAR-completed reconstruction of that same prefix. Both retain the original ten-scale indexing, scale-dependent residual transforms, full latent grid, frozen adapted $\Dc$, clipping, and metric preprocessing. Unreceived scales contribute nothing to R0; codebook ID zero is not used as zero contribution. Neither arm uses unsent source tokens or a continuous source latent. This is distinct from A0/B0/C0, which intervene on received boundary-token use.

\textbf{Decoder training coverage.} The fixed decoder was adapted with an exact 50\% continuous-latent, 25\% fully quantized-latent, and 25\% interpolation mixture. The interpolation starts from $F_{b,\mathrm{TX}}$, a complete latent formed by \texttt{complete\_latent} from the first eight true source scales and generated subsequent scales, and moves toward the continuous latent. That training-cache completion uses the training source's class label, whereas the formal NeST receiver uses null-class greedy completion without CFG. The interpolation does not start from a coarse-only residual sum. The training code has no explicit coarse-only input branch. Sharing $\Dc$ therefore fixes the parameters but does not equalize adaptation to the two input distributions.
''']
for snr in [4,10]:
 rs=[r for r in read('e2_summary.csv') if int(r['snr_db'])==snr];ds=[r for r in read('e2_paired_deltas.csv') if int(r['snr_db'])==snr]
 combined=[]
 for r in rs:combined.append(dict(r,column=r['arm']))
 for r in ds:combined.append(dict(r,column='delta'))
 parts += [rf'\subsection*{{$N=512$, {snr} dB: 100 sources, 300 results per arm}}',table(combined,['R0-coarse-only','R1-VAR-completed','delta'],'column',['R0 coarse','R1 completed','R1 minus R0'])]
 parts.append('Fixed-gray count per arm: '+next(r['fixed_gray_count'] for r in rs)+'.')
parts += [r'Within this fixed adapted receiver, completion improves all three feature metrics at both points; PSNR improves clearly at 10~dB, while its 4~dB interval includes zero. This controlled switch does not establish superiority over a coarse-only decoder trained with comparable input coverage. No new decoder training was performed for this revision.']
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
The existing NeST timing logs also record actual sequential length-fit attempts. Each SNR below has 16 development sources and three measured repetitions (48 calls), with 48 warm-up calls excluded. This is the timing population, not common500. The 95th percentile uses linear interpolation at position $0.95(n-1)$.
\begin{center}\small\begin{tabular}{rrrrr}\toprule
SNR & Mean attempts & Median & 95th percentile & Range\\\midrule
''']
for r in review_rows('timing_attempt_summary.csv'):
 assert r['sources']=='16' and r['measured_calls']=='48' and r['warmups_excluded']=='48'
 parts.append(f"{r['snr_db']} & {float(r['attempt_mean']):.4f} & {float(r['attempt_median']):.1f} & {float(r['attempt_p95']):.1f} & {r['attempt_min']}--{r['attempt_max']}"+r'\\')
parts += [r'\bottomrule\end{tabular}\end{center}',r'These are observed candidate-length comparisons after the shared source traversal; their count is neither additional prior forward calls nor the size of the candidate catalogue.',r'''
\clearpage\section*{S6. Preserved qualitative cases and source records}
The main letter restores two existing N1024/10~dB development examples, Pastries and Dog, with full source IDs:
\begin{itemize}
\item Pastries: \path{n07860988/ILSVRC2012_val_00010848_n07860988}
\item Dog: \path{n02105641/ILSVRC2012_val_00018762_n02105641}
\end{itemize}
They are not newly sampled common500 test examples. NeST/ARPC retain historical noise identity 6201; Ada/Latent JSCC retain 2001, with Ada from its independent development-two-image event. No claim of identical AWGN vectors across these methods is made. Displayed LPIPS-Alex/DreamSim values correspond to each original native-array realization.

The N512 lamp full-field/ROI view is preserved below as a local illustration. The crop is $x\in[80,180)$, $y\in[0,100)$ in the original 256-pixel grid and is identical across methods. PSNR/LPIPS values are for each full image, not the crop. It uses N512, 10~dB, original repetition seed 6201; the already displayed lamp was selected editorially, not by a new preregistered or random rule.
\begin{center}\includegraphics[width=6.0in]{supplemental/N512_lamp_roi.pdf}\end{center}
\clearpage
The earlier two-case N512 figure is also retained, with full-field images and LPIPS-Alex/DreamSim per displayed realization. Its source-ID selection history is unchanged. Both cases use N512, 10~dB, original repetition seed 6201.
\begin{center}\includegraphics[width=6.0in]{supplemental/N512_previous_figure3.pdf}\end{center}
The native-array identifiers, source pixel hashes, wire inputs, display-image hashes, and full-precision metrics are retained in the source map and original selection record. No image content was enhanced or repaired.
\subsection*{Machine-readable evidence}
''']
for path,label in [('results/nestcom_evidence_20261029_v1/e1/REPORT_zh.md','E1 report and complete frame/source/paired data'),('results/nestcom_evidence_20261029_v1/e2/REPORT_zh.md','E2 report and complete frame/source/paired data'),('results/nestcom_evidence_20261029_v1/statistics_identity.json','100-source statistics identity'),('results/nestcom_n512_20261028_v1/delivery/REPORT.md','N512 formal results and policies'),('results/wcl_revision_20261027_v1/p1_bootstrap_clarification_v1.md','Historical mismatch: index-only bootstrap reuse clarification'),('results/five_method_timing_20261020_v1/delivery/summary_ms_per_image.csv','Timing data')]:
 parts.append(link(path,label)+r'\par')
nofit=json.loads((P.parents[2]/'results/nestcom_focused_review_20261029_v2/nofit/completion.json').read_text(encoding='utf8'))
assert nofit['status']=='PASS_TARGETED_PROTOCOL_CHECKS' and nofit['historical_protocol_files_unchanged']
assert nofit['normal_old_new_waveform_and_rx_identical'] and not nofit['formal_quality_claim']
parts += [r'''
\clearpage\section*{S7. Versioned N1024 no-fit extension}
The historical N1024 experiments used the error-on-no-fit implementation and contained no no-fit event. The independent protocol version \texttt{NESTCOM\_N1024\_PAID\_NULL\_20261029\_V2} adds paid NULL handling without changing those frozen sources or results. It preserves the original 372 profile records, header waveforms, public scrambling convention, and normal source path. Twelve appended IDs cover the original twelve MCS layouts. Missing source evidence remains an error and is not converted to NULL.

The bounded CPU checks use three historical payloads, each through the original and extended catalogues, plus twelve constructed no-fit cases covering all MCS layouts. The ordinary-path full waveforms, profiles, information/codeword hashes, and RX outputs are identical. Each no-fit case sends the full 1024-symbol frame: the paid 68-symbol header and 956-symbol body, including the 13-bit length and CRC16. The receiver recovers the explicit two-zero-bit NULL payload and returns the fixed 0.5 float gray image. Additional pure checks cover full-scale WHOLE/PARTIAL fitting and rejection guards.
''',f"Actual checks: {nofit['actual_full_frame_encodes']} full-frame encodes, {nofit['actual_full_frame_receives']} full-frame receptions and {nofit['actual_packet_decodes']} packet decodes; {nofit['source_model_calls']} source-model calls, {nofit['metric_calls']} metric calls and {nofit['neural_replays']} neural replays. The process exited successfully after {nofit['elapsed_seconds']:.2f} seconds.",r'''
Three existing image archives are hash-verified and reused because their RX inputs are identical; no new neural output equivalence is claimed from inference. These are finite noiseless-input checks with a 40~dB demapper setting. They establish targeted protocol behavior, not noisy-channel reliability or new image-quality performance. No training or resampling was performed.

The reproducible implementation and input identities are provided in:
\begin{itemize}
\item \path{experiments/nestcom-focused-review-20261029/}
\item \path{results/nestcom_focused_review_20261029_v2/nofit/}
\end{itemize}
''']
parts += [r'\end{document}']
(P/'supplementary.tex').write_text('\n'.join(parts),encoding='utf8',newline='\n')
print('Supplement generated from frozen inputs; no statistics recomputed.')
