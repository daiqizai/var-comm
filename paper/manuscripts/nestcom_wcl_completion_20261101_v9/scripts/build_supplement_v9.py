"""Build V9 supplement from preserved closed V8 tables and new pinned V9 evidence."""
from pathlib import Path
import csv,json,re
P=Path(__file__).resolve().parents[1]
def rows(rel):return list(csv.DictReader((P/rel).open()))
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(headers,rr,spec):
 return '\n'+r'\begin{longtable}{@{}'+spec+r'@{}}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\'for r in rr)+'\n'+r'\bottomrule\end{longtable}'+'\n'
metric={'psnr_db':'PSNR','dists':'DISTS','lpips_alex':'LPIPS','dreamsim':'DreamSim','clip_image_cosine':'CLIP-I'}
base=(P/'inputs/supplement_v8_template.tex').read_text()
s1=r'''
\section{Reproduction configuration}
NeST-Com transmits source-token prefixes from a shared, frozen next-scale representation. The complete machine-readable configuration, candidate pools, PHY layouts, model manifest and reference vectors are supplied in \texttt{inputs/reproduction/}. The repository entry \texttt{experiments/nestcom-v9-completion-20261101/verify\_cdf\_ac.py} replays the integer coding checks on CPU; \texttt{verify\_assets.py} resolves an external asset directory and checks file identities. No private server path is required for those verification commands.

\subsection{Representation, models and numerical route}
The tokenizer has a 4096-entry codebook and ten maps of side lengths $(1,2,3,4,5,6,8,10,13,16)$. The VAR-d16 prior uses null class 1000 and greedy prediction, without CFG. Complete coarse maps determine every position's probability in the next map. Both terminals use those known maps; no unreceived same-scale token enters the receiver's entropy context.

The adapted decoder checkpoint is update0038000. It minimizes MSE+$0.1$ LPIPS on 20\,000 ImageNet training images with a 50/25/25 mixture of continuous latents, fully quantized latents, and interpolated prefix-completed latents. The last input is $F_b+\alpha(F-F_b)$, $\alpha\sim U[0,1]$, with $F_b$ completed from eight source scales. Selection minimizes full-calibration mixture image loss. Tokenizer, prior and codebook remain frozen. The recipe is \texttt{experiments/var-latent-enhancement-20260917/configs/experiment.json}.

The NeST reference route is Python 3.10.12, NumPy 2.2.6, Torch 2.11.0+cu128, CUDA 12.8 and cuDNN 91900. Inference is FP32, batch 1, eval/no-grad, TF32 off, matmul precision highest, deterministic algorithms and cuDNN deterministic on, benchmark off. The revised timing uses two intra-op and two inter-op threads on two distinct physical CPU cores; S7 separates it from the historical SMT-pair run. Device PCI/UUID, tensor-state hashes, checkpoint-file hashes and source-file hashes are separate manifest fields. Reproducing a floating-point probability stream requires this numerical route as well as the weights.

Ada retains LlamaGen-L/VQ16 (upstream commit \texttt{ce98ec41803a74a90ce68c40ababa9eaeffeb4ec}), predicted ConvNeXt-tiny classes, conditional-branch entropy CDFs and CFG 4/T1/top-$k$2000/top-$p$1 sampled completion. Its 10-bit class is paid. The public LlamaGen weight URLs and revision are included in the manifest. ARPC uses its public 0.06M preset, CFG 3, temperature 0.5 and seed 0; its paid caption and source container are preserved. Latent JSCC uses MSE+$0.1$ LPIPS+$0.01$ latent consistency and shared $D_c$, with 100k updates; SwinJSCC uses RGB MSE, 200k updates and minimum calibration-MSE selection. Both use the fixed 20k training subset. External models retain their own numerical routes rather than inheriting NeST's environment.

Public VAR/LlamaGen/ConvNeXt/ARPC sources, exact file pins, available tensor-state pins and architecture configurations are listed in \texttt{model\_manifest.json}. Project-trained $D_c$ and JSCC checkpoints are retained in the project storage; no public download URL is asserted for them. Their recipes and exact checkpoint identities are supplied. Set \texttt{NESTCOM\_ASSET\_ROOT}, or provide \texttt{NESTCOM\_ASSETS\_JSON} mapping checkpoint names to local paths. A file SHA is never presented as a tensor-state SHA.

\subsection{Integer probabilities and independently terminated streams}
FP32 logits pass through \texttt{torch.log\_softmax}; NumPy float64 then performs stable normalization. For vocabulary $V=4096$ and total frequency $M=2^{24}$,
\begin{align*}
p_j&=\frac{\exp(\ell_j-\max_k\ell_k)}{\sum_k\exp(\ell_k-\max_k\ell_k)},&
f_j&=\lfloor p_j(M-V)\rfloor+1,\\
d&=M-\sum_j f_j,&f_{j^*}&\leftarrow f_{j^*}+d,\quad j^*=\arg\max_jp_j.
\end{align*}
Exact ties use the smallest codebook index. The int64 CDF is $[0,\operatorname{cumsum}(f)]$, with strictly positive frequencies. No CDF table is transmitted. The implementation is \texttt{src/var\_comm/entropy.py}.

The arithmetic interval starts at $[0,2^{32}-1]$. An incremental encoder evaluates successive prefixes; each is finalized independently, without changing the incremental state. Finalization emits $b=0$ if low$<2^{30}$, otherwise $b=1$, followed by pending+1 inverse bits. RX initializes a 32-bit state, uses bounded zero extension, and requires exactly 30 terminal zero reads. It re-encodes recovered tokens and requires exact equality to the received payload. The paid $(m,K)$ specifies the token count. Prefix streams are not slices of a longer codeword. Missing files and invalid model/CDF inputs are execution errors, not receiver successes or NULLs.

\subsection{Digital frame and rejection rules}
The 68-symbol header carries a 12-bit profile, CRC16 and six zero termination bits. Its convolutional encoder has constraint length 7 and octal generators $(171,133)$, rate 1/2; its 68 coded bits are repeated to 136 bits and mapped to 68 rotated-BPSK symbols. Terminated Viterbi decoding requires final state zero. CRC uses polynomial 0x1021, initial 0xFFFF, MSB-first integer fields. The body CRC covers the 13-bit length, source bits and known zero padding.

For $q=\log_2M_{\rm QAM}$, $n=(N-68)q$, $k=\lfloor rn\rfloor$ and source capacity $k-29$. All 12 declared MCSs are constructor-admitted at both budgets. Sionna2.2.0 uses CPU FP32, one 5G-NR LDPC block and RV0: filler shortening, first-$2Z$ and tail puncturing, no repetition. Decoder settings are boxplus-phi/check, sum/variable, flooding, 20 iterations, LLR clipping 20, hard output and information bits returned, PCM pruning on, HARQ and early stopping off. Gray-mapped QAM has nominal $E_s=2$ without per-frame renormalization. Public PCG64 scrambling follows rate matching; RX reverses its LLR signs. Its seed is derived from the first 16 SHA256 bytes, little-endian, of the canonical protocol/"scramble"/session/counter tuple. Exact mask and mapping definitions are in \texttt{ep\_phy.py}; interleaver hashes and constructor metadata are in \texttt{phy\_layouts.csv}.
'''
layout=rows('inputs/reproduction/phy_layouts.csv')
s1+=table(['$N$','$q$','$r$','$k$','$n$','BG','$Z$','Filler','Punctured'],[[r['N'],r['q'],r['rate'],r['k'],r['n'],r['bg'],r['z'],r['k_filler'],r['puncturing_bits']]for r in layout],'rrrrrlrrr')
s1+=r'''
Profile CRC/catalogue, body CRC, invalid length, known padding or arithmetic canonical failures produce fixed gray RGB 0.5. A valid paid-NULL also produces gray, while preserving its full transmitted header/body and statistical denominator. The current N1024 directory retains all 3378 original profiles and reuses five existing NULL IDs; seven missing MCS-specific NULLs occupy free IDs 3390--3396. Each carries payload [0,0]. The 33-check CPU regression performed 21 real header decodes and 20 body decodes, including every MCS no-fit path, exact capacity, one-bit excess, legal fallbacks, normal-path bit/wave equality and malformed packets. Older directories remain identifiable in the original result provenance; their quality outputs had no no-fit events.

\subsection{Candidate pools and calibration}
The requested MCS grid is $q\in\{2,4,6\}$, $r\in\{1/2,2/3,3/4,5/6\}$. The \texttt{candidate\_pool.csv} records the NeST-family pools, all N512 families and the N1024 7-dB candidates, including target, MCS, capacity, admitted status, screening/final qualification, winner, objective, tie rule and registration source. N512 screens 396 candidates: 25 Quarter targets$\times12$, seven WHOLE targets$\times12$ and Ada's maximum 256$\times12$. Each candidate uses 100 calibration sources and one channel event. Per family/SNR, the top three plus the predefined maximum-endpoint $q2,r1/2$ anchor enter cal1000 with three events per source. Selection maximizes equal-source DINOv2-L cosine including gray failures; ties use lexical candidate ID.

The NeST-family N1024 policies at 4/10/19 dB use the actual inherited finalists plus the prescribed full-$m=10$ candidates; ties prefer legacy candidates and then lexical ID. At 7 dB the five MCSs are $(2,2/3),\allowbreak(2,3/4),\allowbreak(2,5/6),\allowbreak(4,1/2),\allowbreak(4,2/3)$, on 100 calibration sources with one event; ties follow declaration order. Token retains the selected family target/MCS and enlarges fallback endpoints only. Equal-bit controls instead share ceiling $(10,0)$; the matched-granularity control shares $(8,126)$. These are separate comparisons.

For Ada at $N=1024$, the two $(q,r)$ candidates at 4/10/19 dB are $\{(2,1/2),(2,2/3)\}$, $\{(4,2/3),(4,3/4)\}$ and $\{(4,2/3),(6,5/6)\}$, respectively. Selection on cal1000 with three events per source freezes $(2,2/3)$, $(4,2/3)$ and $(6,5/6)$; it maximizes equal-source DINOv2-L cosine with lexical candidate-ID ties. The six scores and frozen policies are copied unchanged to \path{inputs/reproduction/ada_n1024_calibration/} as \texttt{candidate\_summary.csv} and \texttt{frozen\_policies.json}.
'''
pol=base[base.index(r'\begin{longtable}'):base.index(r'\bottomrule\end{longtable}')+len(r'\bottomrule\end{longtable}')];s1+=pol
s1+=r'''
Source capacity excludes the body length/CRC but is not total air-interface efficiency. Ada can saturate its full 256-token representation and leave capacity unused. Budgets have independently calibrated pools. DINOv2 is the policy objective; DISTS/DreamSim are source-reference evaluation measures, not human judgments. DISTS uses the official revision \texttt{1267d8cb626c98706db3697422701c56a85ebf2e}, learned alpha/beta and VGG16 weights, FP32 evaluation on native 256 RGB. DreamSim/CLIP use 224 bicubic-antialias; metric identities pin exact weights and dependencies. CLIP2.10 features are not mixed with 2.11 caches.

\subsection{Reference vectors and verification}
Six deterministic CPU probability cases and six independently terminated prefixes verify positive integer frequencies, total frequency, first-index ties, non-mutating finalization, independent decoding, exact 30-bit extension and rejection of noncanonical appended bits. Run from the repository root:
\begin{quote}\ttfamily\small
python experiments/nestcom-v9-completion-20261101/verify\_cdf\_ac.py\\
\quad --repo . --output /tmp/nest-cdf-check
\end{quote}
The output directory must be fresh. Two real-model vectors below record complete TX/RX scale-CDF hashes, exact bits/tokens, canonical checks, model/numerical identity and source provenance. RX did not consume TX probability tables, source tokens or hidden source states. Comparisons were made only after independent reception completed.
'''
vv=[json.loads((P/f'inputs/reproduction/reference_vectors/{name}.json').read_text())for name in ['whole','partial']]
s1+=table(['Vector','$(m,K)$','Tokens','Source bits','CDF scales','Exact TX/RX'],[[label,str(tuple(v['endpoint'])),str(sum(x*x for x in [1,2,3,4,5,6,8,10,13,16][:v['endpoint'][0]])+v['endpoint'][1]),v['payload_bits'],len(v['TX_CDF']),'yes']for label,v in zip(['Whole','Boundary'],vv)],'llrrrr')
s1+=r'''
The whole reference uses one explicitly counted receiver diagnostic without a new PHY event; the boundary reference reuses a completed Kodak input. Integer vectors replay without weights. Independent model/CDF regeneration requires the pinned assets and numerical environment; reference hashes characterize that tested route.

Common500 uses its original 10\,000$\times500$ source-bootstrap bank; controls use a separately ordered 10\,000$\times100$ bank. Kodak uses the existing 10\,000$\times24$ bank. Each source first averages its three events, then receives equal weight; paired differences precede resampling. Full source IDs, pixel hashes and native receiver identities control reuse. No simultaneous multiplicity-adjusted interval claim is made.
'''
start=base.index(r'\section{System settings');end=base.index(r'\section{Complete quality results}');s=base[:start]+s1+'\n'+base[end:]
# Descriptive distribution tables retain all metrics and all comparisons.
desc=r'''\subsection{Per-source improvement distributions}
Gain is target minus reference for PSNR/CLIP-I and reference minus target for distances. Better/tie/worse uses numerical tolerance$10^{-10}$; this is not a perceptual visibility threshold. Quantiles use the linear rule. Rows retain all sources and failures. N1024/10 Token versus WHOLE improves DISTS for 83.4\%, DreamSim for 83.0\%, and both for 72.2\%; each distance worsens on 16.2\%. These are descriptive proportions, alongside the direct paired mean intervals above.
'''
labels={'WHOLE_final_receiver':'WHOLE','original_NeST':'Quarter','Ada_frozen':'Ada','ARPC':'ARPC','SwinJSCC':'Swin','Latent_JSCC':'Latent','BPG_LDPC':'BPG','WHOLE':'WHOLE','Ada':'Ada'}
for kind in ['e1','e2']:
 rr=rows(f'inputs/v9_package/derived/{kind}_per_source_improvement.csv');data=[]
 for r in rr:
  point=(r.get('N','')+'/'+r.get('snr_db',''))if kind=='e1'else r['B_bits']
  data.append([point,labels.get(r['reference'],tex(r['reference'])),metric[r['metric']],r['improved']+'/'+r['tied']+'/'+r['worsened'],f"{float(r['p05_gain']):+.4f}",f"{float(r['median_gain']):+.4f}",f"{float(r['p95_gain']):+.4f}"])
 desc+=table(['$N$/SNR'if kind=='e1'else'Bits','Reference','Metric','Better/tie/worse','P05','Median','P95'],data,'llllrrr')
s=s.replace(r'\section{Matched granularity and receiver interventions}',desc+'\n'+r'\section{Matched controls and generation sensitivity}')
ada=[r for r in rows('inputs/v9_package/inputs/ada_generation_sensitivity.csv')if r['comparison']=='Ada-argmax minus Ada-sampling']
control=r'''\subsection{Ada sampled versus argmax completion}
The same 100 source IDs, paid predicted classes, actual received prefixes, three original channel outcomes, models, CFG 4 and numerical route are fixed. Only unreceived-tail selection changes. The greedy context/KV is rebuilt and updated along its own generated path; conditional entropy CDFs do not use CFG. Completed full prefixes and gray outputs are reused. This is a receiver sensitivity control, not additional wireless frames. Table values are argmax minus sampling. At the three tested points, argmax increases both LPIPS and DreamSim. The main comparison retains the original sampling configuration.
'''
control+=table(['$N$','SNR','Metric',r'Difference [paired 95\% CI]'],[[r['N'],r['snr_db'],metric[r['metric']],f"{float(r['mean']):+.5f} [{float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f}]"]for r in ada],'rrll')
s=s.replace(r'\section{Equal information budgets}',control+'\n'+r'\section{Equal information budgets}')
beg=s.index(r'\section{Selected reconstructions}');end=s.index(r'\section{Unified processing costs}',beg)
qual=r'''\section{Qualitative examples}
Fig.3 uses sources 135 and 201 at $N=1024$, 10 dB (catalog C10-016/C10-017), chosen after inspection of the complete candidate gallery to illustrate perceptual advantages. These are selected examples, not random or representative samples. The full IDs are \path{n04461696/ILSVRC2012_val_00026517_n04461696} and \path{n06596364/ILSVRC2012_val_00001019_n06596364}. NeST has lower DISTS and DreamSim than WHOLE, SwinJSCC and Ada in both displayed realizations; this does not imply superiority in every metric or over the population. SwinJSCC/NeST PSNR is 21.96/19.99 dB and 19.75/16.82 dB, respectively. Ada has lower LPIPS on source 135 (0.1955 versus 0.2112) and higher CLIP-I on source 201 (0.6926 versus 0.6596). Full-precision scores and image identities are in \texttt{inputs/fig3\_selected\_20261102/}.

All images retain the full $256\times256$ field and each method's original first channel repetition: seed 6201 for WHOLE, NeST and Ada; seed 2001 for SwinJSCC, preserving historical method-specific noise. No failure or repetition was substituted. Metrics use the native float outputs; PNGs serve display only. The following panel shows the selected 7-dB television (index 84) and leather (index 143) advantage examples with complete method comparisons.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/figure3_all_methods_supplement.pdf}\caption{Complete method comparisons for the television and leather advantage examples. Full256$\times$256 fields are retained.}\end{figure}
For a complementary view, the next two sources were fixed before their images were opened: common500 N1024/10dB, ranked by each source's mean DISTS gain of Token over WHOLE, with stable source-index tie breaking. We use the package's median-rank source 209 and minimum-gain source 39, without a success filter. Display is the first original repetition (noise 6201 for both methods). Source209 is \texttt{n04509417/ILSVRC2012\_val\_00035733\_n04509417}; source 39 is \texttt{n02097130/ILSVRC2012\_val\_00048087\_n02097130}. This is a fixed median-rank and DISTS-degradation comparison. Source39 worsens in DISTS but improves in DreamSim, illustrating disagreement between distances.
\begin{figure}[htbp]\centering\includegraphics[width=5.8in]{figures/supp_examples_distribution.pdf}\caption{Median-rank (top) and minimum DISTS-gain (bottom) sources at N1024/10dB. Columns are source, WHOLE and NeST. Values are DISTS / DreamSim for the displayed first repetition, lower favorable.}\end{figure}
'''
s=s[:beg]+qual+'\n'+s[end:]
s=re.sub(r'All 16 groups and 32 TX/RX stages completed.*?\\subsection\{All measured groups\}',r'All 16 groups contribute 3072 measured boundary spans after 2048 warmups. ARPC includes its original model runtime and CPU-PHY interprocess transfer within the same two-core quota. BPG includes codec invocation and source search.\n\n\\subsection{All measured groups}',s,flags=re.S)
s=s.replace('The old six-core and component-sum results are not inserted here.','')
kodak=r'''\section{Transfer to Kodak24}
Kodak uses the original RGB center 256$\times$256 crop without resizing before cropping. ImageNet-calibrated policies, model components and receiver configuration remain fixed. WHOLE and Quarter are the prior complete-scale and quarter-grid systems, respectively; Token uses integer endpoints. Ada remains the paid predicted-class fixed-frame adaptation. SNR 4/10/19 each use three original events per image (seeds2001/2002/2003), giving 216 logical results per method. Native outputs and existing four metrics are reused under exact identities; DISTS is evaluated on all 269 unique source/output pairs. Token needs 213 new physical frames plus 3 exact reused frames, and 71 new unique formal reconstructions plus 1 reused reconstruction. All 864 four-method events are accepted; the independent source population is 24, not 72.

Token improves PSNR, DISTS and DreamSim over WHOLE at all three SNRs. Compared with Quarter, Token's DISTS/DreamSim intervals exclude zero at 10/19 dB and cross zero at 4 dB. Against Ada, Token improves DISTS/DreamSim at 4 dB; Ada improves DISTS at 10/19 dB, while DreamSim differences cross zero. The complete five-metric results follow; all intervals use the existing 10\,000$\times24$ source bank after source-level event averaging.
'''
summ=rows('inputs/kodak_v9/summary.csv');pair=rows('inputs/kodak_v9/paired_deltas.csv')
kodak+=table(['SNR','Method','Metric',r'Mean [pointwise 95\% CI]'],[[r['snr_db'],r['method'],metric[r['metric']],f"{float(r['mean']):.5f} [{float(r['ci_low']):.5f}, {float(r['ci_high']):.5f}]"]for r in summ],'rlll')
kodak+=table(['SNR','Comparison','Metric',r'Difference [paired 95\% CI]'],[[r['snr_db'],r['comparison'],metric[r['metric']],f"{float(r['mean']):+.5f} [{float(r['ci_low']):+.5f}, {float(r['ci_high']):+.5f}]"]for r in pair],'rlll')
dist=rows('inputs/kodak_v9/per_source_improvement.csv')
kodak+=table(['SNR','Comparison','Metric','Better/tie/worse','Median gain'],[[r['snr_db'],r['comparison'],metric[r['metric']],r['better']+'/'+r['equal']+'/'+r['worse'],f"{float(r['median_gain']):+.5f}"]for r in dist if r['comparison'].startswith('Token')],'rlllr')
s=s.replace(r'\end{document}',kodak+'\n'+r'\end{document}')
s=s.replace('Complete experimental results and implementation details. Numerical tables use the same frozen CSV sources as the main manuscript.','Implementation recipe, complete results, paired statistics and source examples. Numerical tables use the same frozen CSV sources as the main manuscript.')
s=re.sub(r'\\texttt\{([^{}]*[/][^{}]*)\}',lambda m:r'\path{'+m[1].replace(r'\_', '_')+'}',s)
s=s.replace(r'\texttt{ce98ec41803a74a90ce68c40ababa9eaeffeb4ec}',r'\texttt{ce98ec41803a74a90ce6}\allowbreak\texttt{8c40ababa9eaeffeb4ec}')
s=s.replace(r'\setlength{\parindent}{0pt}',r'\setlength{\emergencystretch}{2em}\setlength{\parindent}{0pt}')
s=s.replace('Zero observed failures is not zero underlying error probability. Historical noise namespaces differ between some methods and budgets, so count differences alone do not establish a causal granularity effect on frame error rate.', 'Main systems use their prescribed independent noise streams; the matched-granularity control uses shared noise.')
s=s.replace('numerical tolerance$10^{-10}$','numerical tolerance $10^{-10}$')
s=s.replace('Older directories remain identifiable in the original result provenance; their quality outputs had no no-fit events.', 'No no-fit event occurred in the reported quality evaluations.')
s=s.replace('Zero observed failures in a timing cell is not a reliability guarantee. No confidence interval is fabricated from the timing repetitions.', 'The table reports the observed success counts and descriptive timing statistics.')

# Editorial scope: retain all ARPC adaptation results and expose the complete plot.
complete_plot=r"""\section{Complete quality results}
ARPC + LDPC applies our common digital frame to the ARPC progressive source
codec. Its transmitted caption and other required fields are charged to the
frame budget. Fig.~\ref{fig:complete_comparison} reports this additional comparison
alongside the methods in the main Fig.~2. This section gives the complete
quality results; S7 reports processing costs for the same adaptation.
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/figure2_complete_with_arpc.pdf}
\caption{Complete comparison, including the digital-link adaptation of the ARPC
source codec. Panels (a,b) retain all seven methods at $N=1024$ and 4/7/10/19 dB;
panels (c,d) retain all six equal-information budgets for NeST-Com, WHOLE and Ada.
The 92 points and error bars are equal-source means and pointwise 95\%
source-bootstrap intervals after within-source averaging. Ada is the
predicted-class, fixed-frame adaptation.}
\label{fig:complete_comparison}
\end{figure}
"""
assert s.count(r'\section{Complete quality results}')==1
s=s.replace(r'\section{Complete quality results}',complete_plot)

from timing_revision_supplement import apply_timing_revision
s=apply_timing_revision(s,P)
from unified_revision_supplement import apply_unified_revision
s=apply_unified_revision(s,P)
from wording_policy_revision import apply_wording_policy_revision
s=apply_wording_policy_revision(s,P)

(P/'supplementary.tex').write_text('\n'.join(line.rstrip() for line in s.splitlines()).rstrip()+'\n')
print('Built supplement with S1--S8, 165 descriptive rows, 12 Ada-control rows, 60 Kodak means and 90 paired intervals')
