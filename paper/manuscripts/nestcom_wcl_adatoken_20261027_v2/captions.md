# English captions synchronized from the revised manuscript

**Fig. 1** NeST-Com. (a) A paid profile specifies endpoint $e$ and MCS $a$; TX/RX construct CDFs locally using the shared frozen VAR and their known coarse tokens. (b) Independently terminated prefixes, a 13-bit length field, and CRC16 fit the information capacity $k_a$. Bars show the common500 mean of per-image entropy-stream utilization at 10~dB (77.70\%/95.21\% for WHOLE/PARTIAL). (c) Received positions overwrite the predicted boundary map before residual accumulation and advancement to finer scales. $H_m$ denotes the prior state after scale $m$.

**Fig. 2** Six-system performance on common500 at $N=1024$. Points and bars are source means and pointwise 95\% source-bootstrap CIs after averaging three channel realizations. Ada denotes the predicted-class, fixed-frame adaptation. Higher PSNR and lower DreamSim are favorable.

**Fig. 3** Selected methods on two fixed development examples at $N=1024$ and 10~dB: (a) pastries and (b) a dog. Values below each reconstruction are LPIPS-Alex$\downarrow$ / DreamSim$\downarrow$ for that displayed realization.

**Table I** Within-scale endpoint evidence. (A) Main family-specific policies; $\Delta=\mathrm{PARTIAL}-\mathrm{WHOLE}$ with source-paired 95\% CIs. (B) Controlled 100-source diagnostic at 10 dB.

**Table II** Perceptual distance and image-feature similarity on common500. Each cell is LPIPS-Alex $\downarrow$ / CLIP-I $\uparrow$.
