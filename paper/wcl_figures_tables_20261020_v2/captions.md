# English captions and definitions

## Fig. 1 — existing confirmed system diagram

**NeST-Com image communication under a fixed complex-symbol budget.** The transmitter tokenizes the image, selects a prefix using actual terminated source-code lengths, and constructs the paid frame for CRC protection, LDPC coding and QAM transmission over an AWGN channel. The receiver decodes and parses the received stream to recover transmitted tokens, completes the unsent representation with the shared VAR prior, and reconstructs the image with \(D_c\). Entropy coding is the main source interface; raw token coding is an optional alternative.

Editorial binding: retain the user-confirmed `system_diagram_v3_entropy.pptx`; do not replace it with an older repository diagram. That exact file was not found in the current checkout or local workspace/Downloads inventory. No Fig. 1 artwork was changed or recreated in this delivery. The author-held final asset must be supplied at manuscript assembly.

## Fig. 2 — main performance comparison

**Image communication performance over AWGN at \(N=1024\) complex symbols on common500.** Panels show (a) PSNR, (b) LPIPS with AlexNet, (c) DreamSim and (d) CLIP-I, the image–image cosine similarity of OpenAI CLIP ViT-L/14 features. Higher PSNR and CLIP-I and lower LPIPS and DreamSim are favorable. Each source is averaged over its three original noise realizations; points are equal-source means and error bars are pointwise 95% source-bootstrap confidence intervals over 500 sources. All methods retain their frozen versions and actual-length fallback rules, including the corrected T1-C partial-scale result for NeST-Com at 19 dB.

## Fig. 3 — selected reconstruction examples

**Previously selected development examples at \(N=1024\) and SNR = 10 dB.** Pastries (top, source index 95) and Shopping cart (bottom, source index 75) are shown in the fixed order Original, Latent JSCC, SwinJSCC, BPG + LDPC, ARPC + LDPC and NeST-Com. Numbers below reconstructions are PSNR (dB) / LPIPS with AlexNet for that displayed reception. The original field of view, saved display processing and method-specific noise realization are retained. These two selected examples are qualitative illustrations, not randomly sampled representatives of common500.

## Table I — budget extension and method boundary

**Source-paired NeST–ARPC differences at \(N=2048\).** \(\Delta=\mathrm{NeST}-\mathrm{ARPC}\), using matched100 with three noise realizations per source; each method delivered 300/300 frames correctly in every displayed condition. The 4 and 10 dB rows use adaptive representations, whereas the 19 dB row uses full representations. Intervals are direct source-paired 95% confidence intervals. Positive \(\Delta\mathrm{PSNR}\) favors NeST; positive \(\Delta\mathrm{DISTS}\) favors ARPC. At 19 dB, NeST's higher PSNR coexists with ARPC's better DISTS: \(\Delta\mathrm{DISTS}=+0.0054\,[+0.0024,+0.0086]\).

## Timing table — software processing latency

**Software processing latency in ms/image at \(N=1024\).** TX, RX and Total are existing mean measurements at 4, 10 and 19 dB, using the fixed 16 development images, batch size 1, three fresh warmups and three measured calls per source and setting. Total is the sum of TX and RX software processing time. The same host, CPU affinity and PRO6000 GPU are used, with original method-specific software environments and recorded shared load. Model setup, AWGN synthesis, cross-process handoff and audit/result persistence are excluded; required BPG codec subprocess and file-conversion costs are included. These measurements exclude air-interface transmission time and are separate from the historical H800 timing study.

## Metric and statistical usage

- CLIP-I is \(f_I(x)^\top f_I(\hat{x})/(\|f_I(x)\|_2\|f_I(\hat{x})\|_2)\), with both inputs being images. The frozen 224-input floating-point preprocessing and FP32 CLIP inference were used in the existing supplement. Values remain raw cosines, without softmax or percentage conversion.
- CLIP-I is not caption-oriented CLIPScore, classification accuracy or an image–text score.
- Source-paired comparisons use the same source IDs; historical methods retain different noise namespaces. No identical cross-method noise claim is made.
- Compare methods using the saved direct paired contrasts and their intervals, not overlap of separate mean intervals. Pointwise intervals do not establish simultaneous significance across all comparisons.
- DINOv2-L cosine, ConvNeXt prediction agreement, MS-SSIM and N1024 DISTS remain available in the complete eight-metric results; they are not additional panels in Fig. 2.
