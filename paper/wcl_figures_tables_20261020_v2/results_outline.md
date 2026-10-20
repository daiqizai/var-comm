# Results-section outline

Scope: paragraph points only, using frozen results at `a08e2517389f4df00ea4162c98417df3f52560a8`. The central question is the tradeoff among pixel fidelity, perceptual quality, semantic representation consistency and software processing cost under a fixed complex-symbol budget and specified channel conditions.

## 1. Experimental settings and fair comparison

- Introduce the unchanged system in Fig. 1. Define \(N=1024\) complex symbols and AWGN at 4/10/19 dB; include paid protocol overhead and actual terminated-length fallback in the budget.
- Identify the frozen five methods: entropy-coded NeST-Com (current PARTIAL, corrected T1-C m10 at 19 dB), Latent JSCC final 100k, SwinJSCC adapted final 200k, adaptive BPG + LDPC and adapted ARPC + LDPC. Calibration choices remain fixed.
- Use common500 for Fig. 2. Each source contributes the average of its three original noise realizations. The 500 sources, rather than 1500 individual receptions, are the statistical units at each working point.
- Describe the saved pointwise 95% source-bootstrap intervals and direct source-paired differences. The existing index bank was reused in the completed experiments; this figure revision neither estimates intervals nor draws bootstrap indices.
- Pair by source identity. Do not imply identical cross-method noise samples. Keep selected development examples, matched100 N2048 results and fixed16 timing separate from common500.

## 2. Main performance comparison — Fig. 2

- **Pixel fidelity:** NeST is not the best-PSNR method at every condition. Its PSNR means are 17.448, 20.014 and 22.164 dB. SwinJSCC has the highest mean at 4/10 dB (21.671/23.026 dB), and BPG + LDPC at 19 dB (24.230 dB). These comparisons describe the saved means; do not infer significance from mean-CI overlap.
- **Perceptual similarity:** at 4 dB, NeST's LPIPS (0.2633) is worse than Latent JSCC (0.2188), while its DreamSim (0.1839) is better than Latent JSCC (0.2317). This disagreement must remain in the discussion. At 10 dB their LPIPS means are close (0.1715/0.1733); avoid a significance claim without the corresponding direct paired interval. At 19 dB NeST's LPIPS and DreamSim means are 0.1182 and 0.0785.
- **Semantic representation consistency:** CLIP-I measures similarity of image features, not label accuracy or image–text alignment. NeST means are 0.8206/0.8731/0.9005. The saved direct NeST-minus-baseline CLIP contrasts have positive pointwise 95% intervals for all 12 comparisons; they are not simultaneous multiple-comparison guarantees.
- Against ARPC, saved NeST-minus-ARPC CLIP-I differences are 0.17647 [0.16836, 0.18461], 0.15977 [0.15249, 0.16709] and 0.11250 [0.10588, 0.11901]. Exact unrounded values remain in `inputs/clip_nest_minus_baselines.csv`.
- Treat the four metrics as complementary, potentially correlated measurements. Do not present them as four independent proofs or claim universal superiority. Without a caption ablation, do not attribute CLIP differences to ARPC's caption conditioning.
- Keep DINOv2-L, ConvNeXt agreement, MS-SSIM and full eight-metric results in supplementary result files. The former agreement panel is replaced with actual CLIP-I data.

## 3. Qualitative comparison — Fig. 3

- State that Pastries and Shopping cart were previously selected development examples. All six columns retain the same complete field of view; scores refer to the displayed reception, not a three-noise mean.
- **Pastries:** Latent JSCC and NeST retain recognizable pastry layers, plates and the jar. SwinJSCC and BPG show smoother/coarser pastry surfaces and boundaries; ARPC renders different pastry textures and shapes. NeST has LPIPS 0.167 versus Latent JSCC 0.169, but lower PSNR (18.51 versus 19.57 dB). Do not equate plausible detail with exact recovery.
- **Shopping cart:** NeST retains visible basket mesh and surrounding vegetation, while SwinJSCC and BPG appear smoother; ARPC alters cart details and the distant skyline. Latent JSCC has slightly lower LPIPS than NeST (0.150 versus 0.153) and higher PSNR (18.95 versus 18.19 dB). The source cart geometry is not exactly reproduced by every generative output.
- Restrict these observations to the displayed images. Do not use two selected cases to establish population-wide superiority.
- The old Fig. 3 containing PSNR paired gain and source-capacity utilization remains archived in v1. Refer to that material as a supplementary same-prior WHOLE/PARTIAL ablation if retained; every current main-text “Fig. 3” reference should mean the reconstruction grid.

## 4. Larger budget and method boundaries — Table I

- Keep the matched100 N2048 comparison separate from the common500 main result. All shown conditions have three original noises per source and 300/300 correctly delivered frames per method.
- At 4/10 dB, adaptive NeST–ARPC differences favor NeST in PSNR and DISTS: PSNR gains 3.779/3.877 dB and DISTS differences −0.1128/−0.0568, with the direct paired intervals retained in Table I.
- At 19 dB with full representations, NeST has higher PSNR (+1.806 dB [1.742, 1.871]) while ARPC has better DISTS: NeST−ARPC = +0.0054 [+0.0024, +0.0086]. Report both directions as a fidelity–perception tradeoff.
- These rows change representation policy between adaptive and full conditions; they are not a single fixed-configuration SNR curve. Do not insert N1024 CLIP-I values into this table.
- Preserve the same-prior WHOLE/PARTIAL ablation and source-capacity utilization as archived supporting material, without turning clean/source utilization into wireless quality evidence.

## 5. Computational cost — existing timing table

- Use the phrase **software processing latency**, with TX/RX/Total in ms/image. Total = TX + RX; it excludes air-interface transmission time.
- Retain all three SNRs, the original 16 development sources, batch 1, three warmups and three measured calls per source/setting. These are 48 measured calls per method/SNR, rather than 48 independent source images.
- The table preserves the current same-host PRO6000/CPU measurements. Document six CPU cores (affinity 0–5), intra/inter-op threads 6/2, TF32 disabled, original method-specific precision/frameworks and sampled shared load. Do not describe the allocation as exclusive or the software stacks as identical.
- Existing NeST Total means are 544.21/619.71/774.62 ms/image, versus approximately 40.65–44.69 ms/image for the two learned JSCC baselines, 2304.43/2293.01/2296.50 for adaptive BPG, and 1435.49/1520.30/1586.89 for ARPC. Discuss these observed costs alongside Fig. 2's mixed quality outcomes.
- BPG's full adaptive search and candidate decoding are included. Preserve the disclosed child-only CPU-count adapter and ARPC's mixed text-encoder precision; do not imply equal implementation overhead.
- Keep historical H800 timings separate. No extra timing curve is needed.

## Figure/table correspondence

| Item | Prior arrangement | Current main-paper reference |
|---|---|---|
| Fig. 1 | Confirmed system diagram | Unchanged asset; aligned caption only |
| Fig. 2 | PSNR / LPIPS / DreamSim / agreement | PSNR / LPIPS / DreamSim / CLIP-I, five-method common500 |
| Fig. 3 | Paired PSNR gain + source-capacity utilization | Two selected reconstruction examples, 2×6 |
| Table I | N2048 paired PSNR and DISTS | Unchanged values, signs, intervals and condition names |
| Timing | Current five-method TX/RX/Total | Software processing latency table; no new curve |

The captions and outline are updated here. Historical manuscripts and archived figures are preserved rather than silently rewritten into a new full paper.
