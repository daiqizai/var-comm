"""Readable V3 supplement from frozen score files and closed execution receipts."""
from pathlib import Path
import csv,json,shutil,hashlib
P=Path(__file__).resolve().parents[1];W=P.parents[2];D=W/'results/nestcom_focused_review_20261029_v3'
def rows(p):return list(csv.DictReader(p.open(encoding='utf8')))
def write(n,t):(P/'sections'/n).write_text(t.replace(r'\,[',r'\,\allowbreak['),encoding='utf8',newline='\n')
def one(a,**q):
    x=[r for r in a if all(str(r[k])==str(v)for k,v in q.items())];assert len(x)==1,(q,len(x));return x[0]
def signed(x):return f'{float(x):+.5f}'
for folder,names in [('reception',['success_failure_quality.csv','joint_success_paired_quality.csv','reception_decomposition_identity.json']),('timing',['partial_timing_summary.csv','partial_paired_timing_summary.csv','operation_counts.csv','actual_compute_and_reuse.json'])]:
    dst=P/'inputs'/folder;dst.mkdir(exist_ok=True)
    for n in names:shutil.copy2(D/folder/n,dst/n)
shutil.copy2(D/'configuration_evidence.json',P/'inputs/configuration_evidence.json')
config=[
dict(system='NeST / WHOLE',model='VAR-d16 and residual VQ tokenizer; shared adapted Dc',adaptation='Dc: 20k ImageNet training images; continuous/full-quantized/prefix-completed interpolated latents; all components frozen at evaluation',candidates='N512: 25 Quarter / 7 WHOLE; N1024: family-specific targets and full m=10 revision, S1.1',mapping='Discrete LDPC/QAM; N512 12 MCSs; N1024 frozen pool and targets in S1.1',selection='DINOv2 cosine; cal1000, three channel outcomes/source; 100-source pilot one outcome',paid_condition='Profile endpoint/MCS; body 13-bit length +16-bit CRC',receiver='Null class, greedy next-scale completion; received-token overwrite and shared Dc'),
dict(system='Latent JSCC',model='Frozen VAR tokenizer continuous 32x16x16 latent; convolutional channel encoder/decoder; shared Dc',adaptation='495208 trainable communication parameters; width64,3 residual blocks; 20k training images; final100k updates; MSE+0.1LPIPS+0.01 standardized-latent MSE',candidates='No discrete token-prefix candidates',mapping='Continuous1024 complex symbols, per-image Es=2 normalization; no MCS',selection='Checkpoint calibration on1000 images,5 training-SNR cells,3 noises; frozen composite loss',paid_condition='No predicted class or text',receiver='Learned latent recovery and frozen Dc; no autoregressive tail'),
dict(system='SwinJSCC',model='Author SA+RA architecture adapted to256x256, initialized without pretrained communication weights',adaptation='20k training images; C6/C13 alternating budgets; 200k total updates; RGB MSE; budget-capped stopping while calibration still improved',candidates='No discrete token-prefix candidates',mapping='N1024:768 continuous data symbols plus256 charged mask/power side-information symbols; no MCS',selection='1000 calibration images across2 budgets and6 SNRs, one noise per cell; mean RGB MSE',paid_condition='Mask and power side information charged within N',receiver='Learned image decoder; no token completion'),
dict(system='BPG + LDPC',model='libbpg0.9.8 / x265; 8-bit YCbCr4:2:0, compression level8',adaptation='No neural training; image-adaptive downsampling to256/128/64/32 and QP search0..51',candidates='Coarse QP0,8,16,24,32,40,48,51 plus integer refinement at capacity crossings; select legal minimum-MSE stream',mapping='15 MCS catalogue: QPSK/16-/64-QAM x rates1/3,1/2,2/3,3/4,5/6; three screened per SNR',selection='100 calibration images,3 noises; maximum source-mean PSNR; initial32-source proxy screen',paid_condition='Complete BPG file plus protected framing; no oracle RX size or source',receiver='Actual BPG decoding and bicubic resize to256x256; rejected receptions gray'),
dict(system='ARPC adapted',model='Official ARPC adapted Infinity2B/BSQ-VAE16 weights; BLIP caption and Flan-T5-XL text conditioning',adaptation='Frozen public communication weights; fixed-frame container and actual LDPC/QAM interface; no new training',candidates='7 complete-scale endpoints; one probability pass;16-bit group trial arithmetic coding vs raw',mapping='12 MCSs: QPSK/16-/64-QAM x rates1/2,2/3,3/4,5/6; 2 shortlisted/SNR',selection='100-source84-strategy proxy screen, then cal1000 x3 actual noises; maximum DINOv2 cosine',paid_condition='Caption and token boundaries included in lossless source container',receiver='Actual prefix plus paid caption; CFG3,temperature0.5,top-k900,top-p0.97,seed0; full scale schedule'),
dict(system='Ada adapted',model='LlamaGen-L / VQ16; fixed ConvNeXt predicts ImageNet class',adaptation='Frozen pretrained representation and prior; fixed per-image frame, actual prefix arithmetic coding and paid class; no ARQ/controller reproduction',candidates='All1..256 independently terminated raster prefixes',mapping='N1024:6 preselected MCS/SNR combinations; N512:12 MCS pool in S1.1',selection='cal1000 x3 actual noises; maximum DINOv2 cosine, then freeze',paid_condition='Recovered10-bit predicted class; body overhead39bits including length/CRC',receiver='CFG4,temperature1,top-k2000,top-p1 sampled tail; FP32/TF32off')]
readability = {'Dc':'D_c', '20k':'20,000', '495208':'495,208', 'width64,3':'width 64, 3', 'final100k':'final 100,000', '200k total':'200,000 total', '100k updates':'100,000 updates', 'to256x256':'to 256 x 256', '32x16x16':'32 x 16 x 16', 'C6/C13':'C6 / C13', 'libbpg0.9.8':'libbpg 0.9.8', 'YCbCr4:2:0':'YCbCr 4:2:0', 'level8':'level 8', 'to256/128/64/32':'to 256 / 128 / 64 / 32', 'search0..51':'search 0--51', 'Infinity2B':'Infinity 2B', 'BSQ-VAE16':'BSQ-VAE 16', 'continuous1024':'continuous 1024', 'Es=2':'Es = 2', 'on1000':'on 1000', 'images,5':'images, 5', 'cells,3':'cells, 3', '3 noises':'3 noises', 'across2 budgets and6':'across 2 budgets and 6', 'N1024:768':'N1024: 768', 'plus256':'plus 256', 'QP0,8,16,24,32,40,48,51':'QP 0, 8, 16, 24, 32, 40, 48, 51', 'rates1/3,1/2,2/3,3/4,5/6':'rates 1/3, 1/2, 2/3, 3/4, 5/6', 'images,3':'images, 3', 'initial32-source':'initial 32-source', 'one probability pass;16-bit':'one probability pass; 16-bit', '100-source84-strategy':'100-source, 84-strategy', 'cal1000 x3':'cal1000 x 3', 'CFG3,temperature0.5,top-k900,top-p0.97,seed0':'CFG 3, temperature 0.5, top-k 900, top-p 0.97, seed 0', 'rates1/2,2/3,3/4,5/6':'rates 1/2, 2/3, 3/4, 5/6', 'All1..256':'All 1--256', 'N1024:6':'N1024: 6', 'N512:12':'N512: 12', 'Recovered10-bit':'Recovered 10-bit', 'overhead39bits':'overhead 39 bits', 'CFG4,temperature1,top-k2000,top-p1':'CFG 4, temperature 1, top-k 2000, top-p 1', 'FP32/TF32off':'FP32 / TF32 off', 'full m=10 revision':'full-scale m = 10 candidates', 'MSE+0.1LPIPS+0.01':'MSE + 0.1 LPIPS + 0.01'}
for c in config:
    for key in c:
        for before, after in readability.items():
            c[key] = c[key].replace(before, after)
        c[key] = c[key].replace('Continuous1024', 'Continuous 1024')
with(D/'six_system_configuration.csv').open('w',newline='',encoding='utf8')as f:
    w=csv.DictWriter(f,fieldnames=list(config[0]));w.writeheader();w.writerows(config)
shutil.copy2(D/'six_system_configuration.csv',P/'inputs/six_system_configuration.csv')
def esc(t):return t.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace(r'D\_c', r'$D_c$')
t=r'''\clearpage\subsection*{S1.2. Six-system adaptations and selection}
The main comparison evaluates complete systems with different pretraining and adaptation. The tables distinguish these resources from the shared-model WHOLE/NeST comparison. No configuration is changed by this revision.
\begin{center}\small\renewcommand{\arraystretch}{1.15}
\begin{tabular}{p{0.95in}p{5.2in}}\toprule
System & Representation and adaptation\\\midrule
'''
for c in config:t+=esc(c['system'])+' & '+esc(c['model']+'. '+c['adaptation'])+r'\\[5pt]'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
The Swin run stops at its training-budget cap, not a demonstrated convergence criterion. Its original training and calibration curves remain linked in S1. The decoder adaptation for NeST/WHOLE does not define a separately trained coarse-only branch.
\clearpage\subsection*{S1.2. Transmission and calibration (continued)}
\small\renewcommand{\arraystretch}{1.1}
\begin{longtable}{p{0.9in}p{2.5in}p{2.75in}}\toprule
System & Candidates and channel mapping & Selection and receiver\\\midrule\endhead
'''
for c in config:t+=esc(c['system'])+' & '+esc(c['candidates']+'. '+c['mapping'])+' & '+esc(c['selection']+'. '+c['paid_condition']+'. '+c['receiver'])+r'\\[6pt]'+'\n'
t+=r'''\bottomrule\end{longtable}\normalsize
BPG's final 100-source PSNR calibration differs from the 1000-source DINO calibration of the generative systems. Continuous JSCC systems do not have a token-prefix/MCS search. None of these differences is treated as an isolated representation-order effect. Full settings and source SHA-256 identities are in \texttt{six\_system\_configuration.csv} and \texttt{configuration\_evidence.json} in the V3 result directory.
''';write('supplement_config_v3.tex',t)
quality=rows(D/'reception/success_failure_quality.csv');joint=rows(D/'reception/joint_success_paired_quality.csv')
t=r'''\clearpage\subsection*{S2.3. Quality decomposition by reception status}
These calculations use only the existing N512 native-array metric scores; no decoder, metric model, channel, or bootstrap is run. Every method/SNR has 500 sources and three repetitions. With frame-equal success and failure means,
\[Q_{\rm all}=p_{\rm success}Q_{\rm success}+(1-p_{\rm success})Q_{\rm failure}.\]
Because each source has three outcomes, the original equal-source overall mean equals this frame-equal mean. The maximum numerical residual of the identity over all 36 cells is $1.12\times10^{-16}$. Gray-image scores are taken from each actual failed event; they need not be equal across sources. A dash means no failed frame, with zero weighted failure term.
'''
for ix,metric in enumerate(['lpips_alex','dreamsim','psnr_db','clip_image_cosine']):
    if ix==2:t+=r'\clearpage\subsection*{S2.3. Quality decomposition (continued)}'+'\n'
    label={'lpips_alex':'LPIPS-Alex','dreamsim':'DreamSim','psnr_db':'PSNR (dB)','clip_image_cosine':'CLIP-I'}[metric]
    t+=r'\textbf{'+label+r'}\begin{center}\small\begin{tabular}{llrrrr}\toprule Method & SNR & Success/1500 & Overall & Success frames & Failed frames\\\midrule'+'\n'
    for r in quality:
        if r['metric']!=metric:continue
        vals=[f"{float(r[k]):.6f}" if r[k] else '--' for k in ['all_frame_mean','successful_frame_mean','failed_frame_mean']]
        t+=f"{r['method']} & {r['snr_db']} & {r['success_frames']} & "+' & '.join(vals)+r'\\'+'\n'
    t+=r'\bottomrule\end{tabular}\end{center}'+'\n'
t+=r'''\clearpage\subsection*{S2.3. Joint-success source-paired comparisons}
For each comparison, events are joined by complete source ID and original repetition identity. Only events where both methods succeed are retained. Differences are averaged over the available joint-success repetitions within each source, then equally over sources. These are descriptive conditional comparisons with no new confidence interval; they do not replace the failure-inclusive primary estimates.
\begin{center}\small\begin{tabular}{llrrrr}\toprule Comparison & SNR & Joint frames & Sources & Two repeats & Three repeats\\\midrule
'''
for r in joint:
    if r['metric']=='psnr_db':t+=f"{r['comparison']} & {r['snr_db']} & {r['joint_success_frames']} & {r['retained_sources']} & {r['sources_with_2_repeats']} & {r['sources_with_3_repeats']}"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
Every source retains at least two jointly successful repeats; no source is discarded.
\begin{center}\small\begin{tabular}{llrrrr}\toprule Comparison & SNR & $\Delta$LPIPS & $\Delta$DreamSim & $\Delta$PSNR & $\Delta$CLIP-I\\\midrule
'''
for comp in ['NEST-WHOLE','NEST-ADA']:
    for snr in [4,10,19]:
        values=[]
        for metric in ['lpips_alex','dreamsim','psnr_db','clip_image_cosine']:
            r=one(joint,comparison=comp,snr_db=snr,metric=metric);assert float(r['source_equal_paired_delta'])*float(r['primary_all_event_delta'])>0
            values.append(signed(r['source_equal_paired_delta']))
        t+=f'{comp} & {snr} & '+' & '.join(values)+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
The LPIPS and DreamSim directions agree with the original all-event contrasts at all reported cells. Thus the observed quality differences persist on joint successes; the comparison still includes the differing models and training of each complete system. Files: \texttt{success\_failure\_quality.csv}, \texttt{joint\_success\_paired\_quality.csv}, and the source-level differences and identity JSON in the V3 \texttt{reception/} directory.
''';write('supplement_reception_v3.tex',t)
t=r'''\clearpage\subsection*{S2.4. Perceptual endpoint and receiver evidence}
The following are existing source-paired bootstrap results, reorganized for the perceptual-reconstruction narrative. No statistic is recomputed. All common500 LPIPS and DreamSim intervals favor NeST/within-scale transmission at both budgets and all three SNRs. This claim does not extend to every dataset or metric; the N1024 common500 pair file does not provide CLIP-I.
\begin{center}\small\begin{tabular}{rrll}\toprule $N$ & SNR & $\Delta$LPIPS [95\% CI] & $\Delta$DreamSim [95\% CI]\\\midrule
'''
for budget,filename,comp in [(1024,'internal_common500_paired.csv','PARTIAL_minus_WHOLE'),(512,'paired_deltas.csv','NEST_minus_WHOLE')]:
    data=rows(P/'inputs'/filename)
    for snr in [4,10,19]:
        cells=[]
        for metric in ['lpips_alex','dreamsim']:
            r=one(data,comparison=comp,snr_db=snr,metric=metric);assert float(r['ci_high'])<0
            cells.append(f"{float(r['mean']):+.5f} [{float(r['ci_low']):+.5f},{float(r['ci_high']):+.5f}]")
        t+=f'{budget} & {snr} & '+' & '.join(cells)+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
Differences are PARTIAL/NeST minus WHOLE; populations are 500 sources, each with three original channel outcomes. The 13.7\% LPIPS and 18.3\% DreamSim reductions at N1024/10~dB describe relative changes of metric means, not percentages of human perceptual improvement.

\textbf{Controlled granularity at N1024/10~dB, 100 sources.}
\begin{center}\small\begin{tabular}{lrrrrr}\toprule Grid & Util. (\%) & LPIPS & DreamSim & PSNR & Length checks\\\midrule
'''
ss=rows(P/'inputs/a_summary.csv');ee=rows(P/'inputs/a_efficiency.csv')
for arm in ['WHOLE','HALF','QUARTER','EIGHTH']:
    e=one(ee,arm=arm,snr_db=10);vals=[f"{float(one(ss,arm=arm,snr_db=10,metric=m)['mean']):.6f}"for m in ['lpips_alex','dreamsim','psnr_db']]
    t+=f"{arm.title()} & {100*float(e['mean_arithmetic_utilization']):.2f} & "+' & '.join(vals)+f" & {float(e['mean_descending_length_checks']):.2f}"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
Eighth--Quarter differences are LPIPS $-0.002231\,[-0.003228,-0.001310]$, DreamSim $-0.000730\,[-0.001956,0.000477]$, and PSNR $+0.122946\,[0.078557,0.169317]$~dB. A DreamSim interval crossing zero establishes neither equivalence nor saturation. Quarter remains the predefined main configuration.

\textbf{Ordered receiver intervention at N1024/10~dB.}
\begin{center}\small\begin{tabular}{lll}\toprule
Intervention & $\Delta$LPIPS [95\% CI] & $\Delta$DreamSim [95\% CI]\\\midrule
B0--A0 & $-0.010030\,[-0.012026,-0.008026]$ & $-0.008454\,[-0.011357,-0.005643]$\\
C0--B0 & $-0.016620\,[-0.020494,-0.012862]$ & $-0.015372\,[-0.019880,-0.011113]$\\
\bottomrule\end{tabular}\end{center}
The 100-source intervention order is fixed; these are not order-independent causal percentages. The final-scale C0--B0 effect at 19~dB is zero.
''';write('supplement_perceptual_v3.tex',t)
timing=rows(D/'timing/partial_timing_summary.csv');pair=rows(D/'timing/partial_paired_timing_summary.csv')
t=r'''\clearpage\subsection*{S5.1. Finite Quarter/Eighth timing: incomplete identity validation}
\textbf{Status: the planned 100-source, three-repeat benchmark is not complete.} The fixed experiment uses N1024, 10~dB, maximum endpoint $(10,0)$, 16-QAM rate 2/3, batch one, fixed CPU affinity/threads, and one shared PRO6000. The two grids are interleaved. Its cap is 4 qualification, 6 warmup, and 600 measured TX calls, with a two-hour absolute wall limit including load.

The four real-weight qualification calls matched the historical tokens, all candidate streams and CDF hashes, selected endpoint, and full waveform. The first 88 sources then passed both grids in the first measured repetition. At source position 88 (common500 index 114), fresh tokenizer tokens differed from the historical cache; CDF, stream, endpoint/payload, and waveform comparisons consequently failed. The run stopped immediately. The model weights, source archive, and numerical-route identities had passed their checks. These facts do not establish the root cause of the token discrepancy. No numerical tolerance was used to relabel the event as a pass, and no cached-token substitution or retry was made.

The run performed 187 tokenizer calls, 1870 prior-scale calls and 187 complete TX waveform calls: 4 qualification, 6 warmup, 176 validated measurement calls and one invalid measurement call. The latter remains in the ledger and raw timing CSV; 423 planned TX calls were never attempted. Both workers exited, and no RX, quality metric, noise, training, or bootstrap call was made.

\textbf{Partial diagnostic only: 88 sources, one repetition per grid.} The table excludes the invalid call and all warmup/qualification calls. It must not be combined with the full 100-source quality experiment to infer a Quarter/Eighth quality--cost frontier.
\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{3}{c}{Quarter (ms)} & \multicolumn{3}{c}{Eighth (ms)}\\
Component & Mean & Median & p95 & Mean & Median & p95\\\midrule
'''
labels={'tokenizer_ms':'Tokenizer','prior_cdf_context_ms':'Prior/CDF/context','ac_encode_ms':'Incremental AC','endpoint_finish_ms':'Endpoint finalization','ac_and_finish_ms':'AC + finalization','length_lookup_ms':'Length lookup','packing_ldpc_qam_ms':'Packing/LDPC/QAM','phy_with_ipc_ms':'PHY + IPC','total_tx_ms':'Complete TX'}
for metric,label in labels.items():
    cells=[]
    for grid in ['Quarter','Eighth']:
        r=one(timing,grid=grid,metric=metric);cells += [f'{float(r[k]):.3f}'for k in ['mean_ms','median_ms','p95_ms']]
    t+=label+' & '+' & '.join(cells)+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{center}
Each validated call uses ten prior-scale calls and encodes 680 tokens once. Quarter/Eighth finalize 25/49 streams; mean length checks in this partial subset are 9.534/17.659. The full 100-source quality diagnostic retains its original 9.49/17.58 means. The partial source-paired complete-TX difference (Eighth minus Quarter) is $-1.924$~ms in mean and $+1.683$~ms in median; these shared-server, incomplete measurements support no speed-ranking claim.

Complete TX runs from resident uint8 RGB to the complete host waveform, including preprocessing/H2D, tokenizer, prior/CDF, incremental AC, non-mutating finalization, length lookup, packing/LDPC/QAM, and persistent CPU-worker IPC. Model loading, file I/O, and durable audit writes are outside the timed interval. CUDA is synchronized at the required boundaries; prior/CDF/context is wall-time attribution, not isolated kernel time. Raw calls, means/medians/p95, per-source differences, resource samples, and original failure/exit receipts are preserved in the V3 \texttt{timing/} directory. Earlier resource-gate and moved-checkpoint-path failures occurred before any scientific call and remain separately recorded.
''';write('supplement_timing_v3.tex',t)
p=P/'supplementary.tex';s=p.read_text();s=s.replace('graphicx,booktabs,array,url','graphicx,booktabs,array,url,longtable')
s=s.replace('The current revision adds read-only accounting of the corresponding formal events and separately identifies the bounded protocol checks; it adds no model inference, training, metric evaluation, or statistical resampling.','The current revision adds reception-conditioned summaries from existing scores, consolidates configuration evidence, and records the finite TX timing attempt in S5.1. Historical quality results, protocol checks, and their populations are preserved; no new receiver generation, training, metric evaluation, or statistical resampling is performed.')
s=s.replace(r'\textbf{Calibration.}',r'\textbf{NeST/WHOLE calibration.}')
for marker,insert in [(r'\clearpage\section*{S2.',r'\input{sections/supplement_config_v3}'+'\n'),(r'\clearpage\section*{S3. Ada tail sensitivity: $N=512$, 4 dB}',r'\input{sections/supplement_reception_v3}'+'\n'+r'\input{sections/supplement_perceptual_v3}'+'\n'),(r'\clearpage\section*{S6.',r'\input{sections/supplement_timing_v3}'+'\n')]:
    assert marker in s
    if insert.strip()not in s:s=s.replace(marker,insert+marker,1)
p.write_text(s,encoding='utf8')
p=P/'sections/results.tex';s=p.read_text().replace('TIMING_SENTENCE','The grids share ten prior-scale calls and encode 680 tokens once; complete TX timing remains unvalidated because a source-token identity check failed (Supplement~S5.1).');p.write_text(s)
print('Generated configuration, reception, perceptual evidence, and explicitly incomplete timing supplement.')
