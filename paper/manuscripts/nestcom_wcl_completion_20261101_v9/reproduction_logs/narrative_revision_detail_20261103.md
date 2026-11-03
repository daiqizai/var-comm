# Timing implementation detail retained from the previous manuscript

Baseline: `62f3e21edddd320efac25dc348398a1402bf1959`. The previous S7.3–S7.5 text is retained below for reproduction. The revised supplement keeps the valid configuration, numerical compatibility findings, timing, all five quality metrics and the original tables. This archive does not authorize rerunning any task.

```latex
\subsection{Prefill compatibility diagnostic}
Two fixed sources from the preceding diagnostic are reused for four full-prefix probability calls: conditional-only and the original two-branch calculation for each source. We preserve FP32, math attention, weights and the exact integer-CDF conversion. Existing source tokens, predicted classes, 256 CDF hashes and 256 independently terminated streams per source provide the reference; no classifier, tokenizer, image decoder, PHY or quality evaluation is rerun. The table counts exact matches to the frozen incremental route.

\begin{longtable}{@{}rrrr@{}}\toprule
Source index & Branches & CDF matches & Stream matches\\\midrule\endhead
0 & 1 & 0/256 & 4/256\\
0 & 2 & 0/256 & 4/256\\
1 & 1 & 0/256 & 2/256\\
1 & 2 & 0/256 & 2/256\\
\bottomrule\end{longtable}
The full-prefix route is not an exact replacement for the frozen incremental probability path on these inputs. It requires a separately validated transmitter/receiver numerical route rather than substituting new probabilities into old bitstreams. The following study validates such a paired route; the preceding diagnostic and original streams remain unchanged.

\subsection{Separately qualified compiled Ada implementation}
We replace the serial encoder with a single conditional causal prefill and compile the forward operation with \texttt{torch.compile}. At RX, the conditional CDF is reconstructed from the received prefix alone using the same tensor shape and strict causal mask; unknown future inputs are public zeros. The generated tail follows the actual recovered context, using the original CFG=4, temperature, filtering and sampling rules. Entropy CDFs use only the conditional branch. We retain FP32, math attention, disabled TF32, the same weights, classifier, tokenizer, image decoder, fixed MCS and candidate set. The compiler may change floating-point execution, so this is a new numerical implementation, not a bit-exact substitution for the original quality-evaluation route.

Two fixed sources first pass exact recovery of all 512 tokens and 512 integer CDFs, plus two fixed-prefix tail checks. That successful qualification is reused, without repeating its model calls. The formal study then performs fresh RGB-to-symbol TX and independent symbol-to-RGB RX on the same 32 sources, with two warmups and three measurements per direction. RX has no access to source pixels or TX witness files. All 160 received prefixes, decoded classes and endpoints match the external TX witness after RX has exited; all 96 measured receptions succeed. Repeated outputs are deterministic within the new implementation.

The following timings include class prediction and actual digital-link processing at the same synchronized boundaries as the preceding resource audit. Model loading and cold compilation are excluded from latency, but remain charged to the resource ledger. Compilation uses eight CPU cores; each timed span returns to the same two distinct cores and 2/2 thread pools. That timing study did not evaluate image quality; the subsequent same-source quality diagnostic below supplies it. The 10\,000-by-32 bootstrap index bank and source order are reused exactly.

\begin{longtable}{@{}lrl@{}}\toprule
Implementation span & Mean (ms) & 95\% source interval\\\midrule\endhead
TX & 114.55 & [114.15, 114.93]\\
RX & 2949.76 & [2945.89, 2953.91]\\
TOTAL & 3064.30 & [3060.46, 3068.37]\\
\bottomrule\end{longtable}
The new route matches 0/160 historical emitted waveforms and 10/160 historical reconstructed images exactly. Its latency is therefore reported separately from the original-quality Ada row; it does not replace those quality arrays or establish a NeST/Ada algorithmic speedup. The serial audit cost is not an intrinsic Ada requirement. The authors' published timing also uses different hardware and a different processing boundary, so its 19/1012 ms values are not substituted into the local comparison.


\subsection{Quality paired with compiled Ada timing}
We restore the 32 saved timed received waveforms and compare native outputs on the exact same sources, received prefixes/classes and channel events. The restored compiled outputs match all 32 timed output hashes. No TX, new channel event or new timing run is performed. The original and compiled coders recover identical source tokens, but their bitstreams differ; the existing seed derivation from received payload bytes can therefore change the tail-generation seed.

A preregistered fixed-original-seed arm isolates the effect of payload-derived generation seeds, using 30 additional tail renders and two exact reuses. Holding that seed fixed makes all 32 native reconstructed images byte-identical to the original timed route, and all five paired quality differences are exactly zero. The actual timed outputs differ on 30 images. Their paired mean differences below all have intervals crossing zero; this 32-source diagnostic does not establish population equivalence or replace common500. Each source has one distinct timed event; five timing repetitions are not five quality samples. The original common500 quality route remains fixed.

\begin{longtable}{@{}lll@{}}\toprule
Arm difference & Metric & Mean [paired 95\% CI]\\\midrule\endhead
Compiled-Original & PSNR & -0.03839 [-0.47062, +0.39219]\\
Compiled-Original & LPIPS & -0.00168 [-0.00524, +0.00177]\\
Compiled-Original & DreamSim & -0.00349 [-0.00837, +0.00082]\\
Compiled-Original & CLIP-I & -0.00103 [-0.00614, +0.00379]\\
Compiled-Original & DISTS & -0.00175 [-0.00444, +0.00090]\\
Fixed-seed-Original & PSNR & +0.00000 [+0.00000, +0.00000]\\
Fixed-seed-Original & LPIPS & +0.00000 [+0.00000, +0.00000]\\
Fixed-seed-Original & DreamSim & +0.00000 [+0.00000, +0.00000]\\
Fixed-seed-Original & CLIP-I & +0.00000 [+0.00000, +0.00000]\\
Fixed-seed-Original & DISTS & +0.00000 [+0.00000, +0.00000]\\
Compiled-Fixed-seed & PSNR & -0.03839 [-0.47062, +0.39219]\\
Compiled-Fixed-seed & LPIPS & -0.00168 [-0.00524, +0.00177]\\
Compiled-Fixed-seed & DreamSim & -0.00349 [-0.00837, +0.00082]\\
Compiled-Fixed-seed & CLIP-I & -0.00103 [-0.00614, +0.00379]\\
Compiled-Fixed-seed & DISTS & -0.00175 [-0.00444, +0.00090]\\
\bottomrule\end{longtable}
The actual compiled timed outputs have source-mean PSNR 19.36069, LPIPS 0.16355, DreamSim 0.09640, CLIP-I 0.87857, DISTS 0.13098. Their corresponding TX/RX/total means are 114.55/2949.76/3064.30 ms. The fixed-seed diagnostic is not assigned a newly measured latency.


```
