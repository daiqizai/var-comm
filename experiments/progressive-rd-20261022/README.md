# Progressive token rate-distortion research — stage 0

Independent follow-up to frozen WCL commit be122fd60033360010a9ea783c87a827a534838a.
The attached request is design/authorization, not evidence of completed work.
No WCL model, paper, test population, channel policy or existing receipt is edited.

## Current bounded execution

Resource inventory and exact restored-cache audit precede execution. The first
benchmark uses 32 **training** sources chosen by ascending SHA256 of
`progressive-rd-stage0-v1|<source_id>` over the original train20k list. Original
source indices are preserved. The two endpoints are (7,25) and (9,128).
These exercise an early and a late within-scale boundary; they are not selected
by quality. One TX probability pass per source generates both independently
terminated endpoint streams with the existing `PartialSourceCodec`.

For every stream the existing independent arithmetic RX decodes the actual
bits/profile, then the original `render_received` predicts, overwrites received
positions and advances. A decoder wrapper copies its actual input latent without
changing mathematics. The class condition is the original unconditional1000;
no source class, source image, TX CDF or unsent tokens enter RX calls. Only after
the output is fixed does the audit compare tokens and score against the source.
The old class-conditioned m8 `Fb_TX` cache is not reused as current output.

Exact old RGB and full tokens are reused after file SHA, source ID and original
index checks. This is a token-cache benchmark: no tokenizer calls are needed.
Cached multi-endpoint latents are FP32 in stage0. Source bits include arithmetic
finish; no channel transmission or packet reliability is measured here.
Metrics are FP32 LPIPS-Alex, float64 pixel MSE/PSNR and MS-SSIM as an auxiliary.

Then a separately named **throughput-only** unconditional decoder starts from
current Dc for exactly50 AdamW updates, batch4, FP32/TF32off, gradientclip1,
learning rate1e-5, weight decay.01, MSE+.1LPIPS. Encoder/codebook/scale transforms/
VAR remain frozen. First5 updates are included in counts and excluded only from
the steady-throughput estimate. No validation, model selection or gain claim.
The initial and final resumable states retain model/optimizer/scheduler/RNG/data
order, registration and sample counts; periodic saving is at600 seconds and an
owned stop saves at an update boundary. No AMP scaler exists in this FP32 run.

Exact caps:32 TX passes,64 source decodes,64 receiver completions,64 LPIPS and64
MS-SSIM distances,50 updates/200 training samples,zero tokenizer/PHY/held-out
calls. Job wall time1800s; actual parent wait and stderr are retained. Unknown
failure stops without automatic retries. Main/conditional/old-recipe matched
training and DDP qualification remain NOT_RUN until later explicit registration
after the throughput report and available resource window are known.

## Later gates, not started by this entry

The proposed eight endpoints are (7,0),(7,25),(8,0),(8,84),(8,126),(9,0),
(9,128),(10,0). All partial K values must match the existing floored-quarter
catalog; m10 uses the separately qualified complete ten-scale codec. Seed cache
1000–2000 sources x8 endpoints precedes any20k expansion. Single-GPU inference
shards are disjoint. No assumption that eight GPU memories pool or that DDP is
efficient. A1/A2 matched controls precede optional A3; encoder/prior changes
invalidate current latent/length caches. No learned selection, new channel grid,
layered radio experiment, A4 tokenizer/prior or WCL replacement is authorized by
this stage0 registration.

Code/light results are in this repository. Arrays, checkpoint, logs and private
runtime cache are under shared PFS `liberai-tmp/liulu/liulu-var-comm/progressive_rd_20261022_v1`.
