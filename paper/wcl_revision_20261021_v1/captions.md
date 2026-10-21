# English captions

## Fig. 1

NeST-Com under a fixed complex-symbol budget. (a) A budget/SNR policy specifies the target endpoint and MCS; actual terminated entropy-code length determines per-image fallback. The protected header and payload pass through LDPC–QAM transmission and AWGN before actual channel and entropy decoding. (b) The receiver predicts the boundary scale, overwrites received positions, and generates finer scales from the merged state before latent reconstruction and image decoding. The 75-received/25-predicted grid illustrates one within-scale endpoint; the bird thumbnails are schematic. Quantitative entropy-bitstream capacity utilization is reported in Table I.

## Fig. 2

Five-method comparison on common500 over AWGN at N = 1024 complex symbols per image. Each source is first averaged over its three original noise realizations; markers show means over 500 sources and error bars show pointwise 95% source-bootstrap confidence intervals. LPIPS uses AlexNet. CLIP-I is OpenAI CLIP ViT-L/14 image–image feature cosine similarity, evaluated in FP32 with the frozen 224-pixel preprocessing. Higher PSNR and CLIP-I and lower LPIPS and DreamSim are favorable. The entropy-coded NeST-Com results include the T1-C m10 correction at 19 dB. Comparisons are paired by source, not by identical cross-method noise realizations.

## Fig. 3

Previously selected development examples at N = 1024 and 10 dB: Pastries (source index 95) and Shopping cart (75). Each row preserves the full field of view and shows Original, Latent JSCC, SwinJSCC, BPG + LDPC, ARPC + LDPC, and entropy-coded NeST-Com. Values below reconstructions are whole-image PSNR (dB) / LPIPS-Alex for the displayed single realization, read from the saved native-float scores. These are selected qualitative examples, not random common500 samples.

## Table I

Effect of within-scale endpoints at N = 1024 on common500. Utilization is terminated entropy-bitstream length divided by usable source capacity after the length and CRC fields. Differences are PARTIAL − WHOLE with direct source-paired 95% bootstrap intervals. Positive ΔPSNR and negative ΔDreamSim favor PARTIAL.

## Table II

Larger-budget comparison at N = 2048 on matched100. Differences are NeST − ARPC with direct source-paired 95% bootstrap intervals after averaging three noise realizations per source. The 4 and 10 dB rows use the frozen adaptive settings; 19 dB uses full representations. Each method delivered 300/300 events per condition. These rows are distinct configurations. At 19 dB, the retained DISTS result is ΔDISTS = +0.0054 [ +0.0024, +0.0086 ], favoring ARPC, despite NeST's higher PSNR.

## Table III

Mean software processing latency at 10 dB, in ms/image, for the frozen N = 1024 implementations on one NVIDIA RTX PRO 6000 Blackwell GPU and the same six allocated CPU cores; batch size is one. Each of 16 fixed development sources has three warmups followed by three measurements, giving 48 measured calls per method. TX and RX are measured separately; Total = TX + RX. Model loading, simulated AWGN generation, process handoff, and audit/result persistence are excluded; BPG codec subprocess and necessary file-conversion costs are included. The original per-method framework and precision settings and sampled shared load are retained. These are software processing times, not radio airtime, and are not merged with the historical H800 measurements.
