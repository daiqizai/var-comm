# Current captions and exact LaTeX sources — signaling and supplement revision 2026-11-03

## sections/method.tex

```latex
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figure1_final.pdf}
\caption{NeST-Com. (a) The shared policy fixes the MCS and profile directory; the protected Profile ID selects the endpoint. VAR and known coarse tokens provide coding probabilities. (b) Independently terminated prefix bitstreams up to the target endpoint fit the body capacity; bars are schematic. (c) Received tokens overwrite boundary predictions. Recursive state updates accumulate residuals and condition each finer scale before decoding. Unreceived boundary positions retain their coarse-conditioned predictions.}
\label{fig:system}
\end{figure*}
```

## sections/figure2_float.tex

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figure2_1x4.pdf}
\caption{Source-referenced reconstruction on 500 ImageNet images. (a,b) Six methods at $N=1024$; WHOLE, NeST and Ada share the 12-MCS calibration procedure. Three outcomes per source include gray failures. (c,d) Equal, error-free information budgets pay 57 shared protocol bits and Ada's additional 10-bit class; WHOLE/NeST share ceiling $(10,0)$. Points and bars are equal-source means and pointwise 95\% source-bootstrap intervals after within-source averaging. Ada-TokenCom is the predicted-class, fixed-frame adaptation. Lower values are favorable.}
\label{fig:performance}
\end{figure*}
```

## sections/figure3_float.tex

```latex
\begin{figure*}[t]
\centering\includegraphics[width=6.0in]{figure3.pdf}
\caption{Selected reconstructions at $N=1024$ and 10 dB. WHOLE, NeST and the predicted-class fixed-frame Ada-TokenCom adaptation use the policies in Table~\ref{tab:policies}. Values are full-image DISTS$\downarrow$ / DreamSim$\downarrow$ for each method's first prescribed realization. The sources were selected for favorable perceptual examples before recalibration.}
\label{fig:qualitative}\end{figure*}
```

## tables/table_i.tex

```latex
\begin{table}[t]
\caption{Within-scale transmission. (A) Shared-model NeST--WHOLE differences. (B) Matched endpoint granularity. (C) Ordered receiver interventions. Brackets are source-paired 95\% intervals; sample sizes are specified for each block.}
\label{tab:internal}\centering\footnotesize
\setlength{\tabcolsep}{2.1pt}\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{}lccc@{}}\toprule
\multicolumn{4}{l}{\textit{A. 500 sources, $N=1024$, uniformly calibrated policies}}\\
SNR & Util. W$\to$N (\%) & $\Delta$DISTS & $\Delta$DreamSim\\
\midrule
4 & 73.81$\to$99.56 & \shortstack{-0.0142\\{}[-0.0155, -0.0127]} & \shortstack{-0.0297\\{}[-0.0330, -0.0264]}\\[2pt]
7 & 83.19$\to$99.71 & \shortstack{-0.0103\\{}[-0.0139, -0.0068]} & \shortstack{-0.0159\\{}[-0.0207, -0.0112]}\\[2pt]
10 & 77.70$\to$99.79 & \shortstack{-0.0141\\{}[-0.0153, -0.0128]} & \shortstack{-0.0270\\{}[-0.0299, -0.0241]}\\[2pt]
19 & 85.24$\to$99.83 & \shortstack{-0.0048\\{}[-0.0053, -0.0043]} & \shortstack{-0.0043\\{}[-0.0051, -0.0036]}\\[2pt]
\bottomrule\end{tabular}
\par\smallskip
\begin{tabular}{@{}lrrrr@{}}\toprule
\multicolumn{5}{l}{\textit{B. 100 quality sources; 32 timing sources; ceiling $(8,126)$}}\\
Grid & Util. (\%) & DISTS & DreamSim & TX (ms)\\\midrule
Whole & 78.13 & 0.1650 & 0.1387 & 185.2\\
Quarter & 95.19 & 0.1535 & 0.1143 & 215.8\\
Eighth & 97.44 & 0.1516 & 0.1137 & 217.8\\
Token & 99.85 & 0.1505 & 0.1115 & 248.4\\
\bottomrule\end{tabular}
\par\smallskip
\begin{tabular}{@{}lccc@{}}\toprule
\multicolumn{4}{l}{\textit{C. 100 sources, 10 dB, same received prefixes}}\\
Intervention & $\Delta$PSNR (dB) & $\Delta$DISTS & $\Delta$DreamSim\\
\midrule
B0--A0 & \shortstack{+0.287\\{}[0.240, 0.335]} & \shortstack{-0.0062\\{}[-0.0077, -0.0048]} & \shortstack{-0.0113\\{}[-0.0147, -0.0083]}\\[2pt]
C0--B0 & \shortstack{+0.537\\{}[0.404, 0.682]} & \shortstack{-0.0083\\{}[-0.0102, -0.0065]} & \shortstack{-0.0159\\{}[-0.0205, -0.0116]}\\[2pt]
\bottomrule\end{tabular}
\par\smallskip
\begin{minipage}{\columnwidth}\scriptsize
A0: prior completion without received boundary tokens. B0: A0 plus boundary residual correction. C0: B0 plus updated finer-scale conditioning.
\end{minipage}
\end{table}
```

## tables/table_ii.tex

```latex
\begin{table}[t]
\caption{Auxiliary quality on 500 test images at $N=1024$, 10 dB. Means include three realizations per source and gray failures.}
\label{tab:external}\centering\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrr@{}}\toprule
Method & PSNR (dB)$\uparrow$ & LPIPS$\downarrow$ & CLIP-I$\uparrow$\\\midrule
WHOLE & 19.31 & 0.199 & 0.858\\
NeST-Com & 20.19 & 0.166 & 0.877\\
Ada-TokenCom & 19.56 & 0.167 & 0.884\\
Latent JSCC & 20.80 & 0.173 & 0.789\\
SwinJSCC & 23.03 & 0.469 & 0.669\\
BPG + LDPC & 22.54 & 0.589 & 0.635\\
\bottomrule\end{tabular}\end{table}
```

## tables/table_iii.tex

```latex
\begin{table}[t]
\caption{Reduced-budget reconstruction on 500 sources at $N=512$. Means include all failures; PSNR differences use direct source-paired 95\% intervals.}
\label{tab:reduced}\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrr@{}}\toprule
SNR / method & PSNR$\uparrow$ & DISTS$\downarrow$ & DreamSim$\downarrow$\\
\midrule
4 / WHOLE & 15.11 & 0.2580 & 0.3247\\
4 / NeST-Com & 15.61 & 0.2363 & 0.2740\\
4 / Ada-TokenCom & 11.32 & 0.2996 & 0.3847\\
\multicolumn{4}{l}{NeST--WHOLE $\Delta$PSNR: +0.504 [0.440, 0.573] dB}\\
\midrule
10 / WHOLE & 16.82 & 0.2030 & 0.2049\\
10 / NeST-Com & 17.41 & 0.1922 & 0.1837\\
10 / Ada-TokenCom & 13.16 & 0.2367 & 0.2582\\
\multicolumn{4}{l}{NeST--WHOLE $\Delta$PSNR: +0.592 [0.544, 0.643] dB}\\
\midrule
19 / WHOLE & 18.64 & 0.1718 & 0.1458\\
19 / NeST-Com & 19.63 & 0.1554 & 0.1150\\
19 / Ada-TokenCom & 17.69 & 0.1496 & 0.1078\\
\multicolumn{4}{l}{NeST--WHOLE $\Delta$PSNR: +0.994 [0.940, 1.048] dB}\\
\bottomrule\end{tabular}\end{table}
```

## tables/table_iv.tex

```latex
\begin{table}[t]
\caption{Frozen policies: MCS entries are (bits/symbol, code rate); $e$ is the target ceiling, with feasible fallback. Ada's ceiling is 256 tokens. Budgets are calibrated independently.}
\label{tab:policies}\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccc@{}}\toprule
$N$/SNR & $a_{\rm W}$ & $e_{\rm W}$ & $a_{\rm N}$ & $e_{\rm N}$ & $a_{\rm Ada}$\\\midrule
1024/4 & $2,2/3$ & $(7,0)$ & $2,2/3$ & $(7,25)$ & $2,2/3$\\
1024/7 & $4,1/2$ & $(8,0)$ & $4,1/2$ & $(8,0)$ & $4,1/2$\\
1024/10 & $4,2/3$ & $(9,0)$ & $4,2/3$ & $(8,126)$ & $4,2/3$\\
1024/19 & $6,5/6$ & $(10,0)$ & $6,5/6$ & $(10,0)$ & $4,5/6$\\
512/4 & $2,2/3$ & $(6,0)$ & $2,2/3$ & $(5,27)$ & $2,2/3$\\
512/10 & $4,2/3$ & $(7,0)$ & $4,2/3$ & $(7,0)$ & $4,2/3$\\
512/19 & $6,5/6$ & $(8,0)$ & $6,5/6$ & $(8,84)$ & $6,5/6$\\
\bottomrule\end{tabular}\end{table}
```
