## 2026-10-27 A/B/C response study: formal Ada comparison completed

## 2026-10-27 — WCL reproducibility revision and frozen-policy SNR diagnostic

The revised anonymous [manuscript v2](paper/manuscripts/nestcom_wcl_adatoken_20261027_v2/README.md) includes verified candidate/MCS/fallback rules, A0/B0/C0 definitions, actual power/SNR assumptions and final receiver counts. The five-page IEEE 10pt PDF, 13-page double-spaced single-column PDF and editable 14-page Word were compiled/rendered and inspected. All three figures, the 72 frozen main means/CIs and historical v1 remain unchanged.

The [bounded P1 diagnostic](results/wcl_revision_20261027_v1/p1_delivery/REPORT.md) completed 1800 logical receptions on the existing100 sources: frozen10dB WHOLE/PARTIAL policies, true8/10/12dB, shared actual AWGN and true RX noise variance. Both reject300/300 at8dB and accept300/300 at10/12dB, where their received prefixes/reconstructions match. PARTIAL gains0.660dB [0.515,0.814] and reduces DreamSim by0.02384 [0.01797,0.02999]. All failed gray outputs remain included. Actual new work:1200 physical frames,2400 packet decodes,461 metric scalars;600 verified old10dB events reused,zero new source-model/VAR reconstruction and no new bootstrap indices.

The original link worker exited1 after its completed loop at an invalid global CUDA-context assertion; its failure is preserved. Independent metadata-only ledger/output checks closed all saved scientific results without repeating a successful call; metrics/statistics exited0. Complete evidence and final main-frame failure counts are in [the result directory](results/wcl_revision_20261027_v1/README.md). P2 remains unstarted. Old automation stays paused; no training or main-grid rerun.

## 2026-10-27 — WCL three-figure/two-table manuscript and two Ada examples

The complete anonymous manuscript is in [paper/manuscripts/nestcom_wcl_adatoken_20261027_v1](paper/manuscripts/nestcom_wcl_adatoken_20261027_v1/README.md): 5-page IEEEtran 10pt PDF, double-spaced single-column PDF, editable Word, native Fig.1 PPTX, three figures and two tables. The 72 frozen six-system mean/CI rows are unchanged and verified; main common500/Kodak and the 100-source diagnostics remain distinct. Perceptual tradeoffs favoring Ada at some operating points are reported.

The separately authorized Pastries95/Dog21 supplement completed exactly two Ada source traversals, two physical frames/four packet decodes, two independent RX reconstructions, and two LPIPS/DreamSim pairs, all with actual exit 0 and processes gone. See [the report](results/wcl_adatoken_20261027_v1/REPORT.md). The public-counter adapter uses the actually decoded paid profile; the historical 22,716-frame audit found no affected frame. Old science, scores and failed preparations are preserved. No new training, main-grid run, bootstrap or timing was performed. Further calibration sensitivity, SNR mismatch, Swin training and Ada timing remain unperformed; old automation remains paused.

All formal Ada predicted-class fixed-frame results are closed: common500500sources
/4500events and Kodak24sources/216events, N1024 and AWGN4/10/19dB, three noises.
The independently frozen calibration policy, paid10-bit predicted class, original
source identities and receiver-only inputs are preserved.1496 unique formal RX
outputs and1496 unique pairs per metric cover every logical result. Source-paired
statistics reuse the original10000x500 and10000x24 banks with no new random draws.
Old five-method scores/intervals remain unchanged; new six-system curves are in
paper/adatoken_response_20261026_v1/figures_v1/ and were actually visually reviewed.

Common500 NeST-minus-Ada PSNR is+4.116/+0.455/+1.226dB at4/10/19dB.
Ada has higherCLIP-I at4/10dB and lowerDreamSim at10/19dB; at10dB itsLPIPS is
also lower. These direct paired intervals do not cross zero. Kodak4dB favors
NeST in allfour; at10dB onlyPSNR has a nonzero interval, and at19dB PSNR/LPIPS
favorNeST while DreamSim/CLIP spanzero. No claim of universal superiority.
The new result invalidates unqualified all-SNR perceptual/semantic leadership
claims in earlier five-method drafts; exact replacement wording is delivered.

A/B complete evidence remains unchanged, including uncertain eighth-grid gains
and zero final-scale propagation. Full report: results/adatoken_response_20261026_v1/
FINAL_REPORT_zh.md. ActualC compute/evidence: c_final_execution_evidence_v1/.
Cumulative22740PHY/45480packets/1532sourcepasses/5702decoderoutputs are within
the frozen scope. All metrics/statistics realexit0/gone/unresolved0; initialasset
and metadata-preparation failures retained. No new training or old-study restart.
This bounded round is scientifically complete; large caches remain on PFS and
backups are deferred. Normal publication receipts are recorded separately.

---

## 2026-10-26 Controlled endpoints completed; predicted-class Ada adaptation in calibration

A and C now have equal first priority. A fixes models, MCS and maximum endpoint
across WHOLE/HALF/QUARTER/EIGHTH on the existing100-source subset. All3600
logical results are closed:2214 physical events,4428 packet decodes,738 unique
receiver outputs. Four metrics give48 mean intervals and72 direct source-paired
intervals using the existing bank. At10dB quarter-minus-whole PSNR is+0.660149dB;
eighth-minus-quarter is+0.122946dB, but DreamSim and CLIP-I intervals for that
increment cross zero. The frozen main configuration is unchanged.

B is also closed: direct correction and subsequent propagation are reported
separately; at19dB the propagation increment is exactly zero at the final
scale. Original numerical/import failures and explicit amendments are retained.
Reports: results/adatoken_response_20261026_v1/{a_delivery,b_delivery}/REPORT.md.
Reviewed vector/PNG curves: results/adatoken_response_20261026_v1/figures_a_v2/.

C is Ada-TokenCom (predicted-class, fixed-frame adaptation): official LlamaGen-L
and VQ16, fixed sender classifier, paid10-bit predicted class, actual terminated
prefix fitting, N1024=68+956 and Es2. Null-class qualification remains separately
named Ada-inspired. Independent conditional source decoding and24 real-frame
tests closed;22 were accepted and2 retained gray output. All qualification
pages were actually inspected. The remaining992 calibration source encodes
are registered and running after8 exact prior conditional encodes were reused.
Calibration strategy selection and formal common500/Kodak results are pending;
no Ada main-curve or comparative superiority result is claimed yet.

Read experiments/adatoken-response-20261026/C_PREDICTED_CLASS_PLAN.md for finite
caps and MANUSCRIPT_REVISION_NOTES.md for attribution and result-placement tasks.
No training or old-study restart. Arrays/weights/ledgers stay on PFS; backups
remain deferred. This publication is a completed A/B and qualified C-entry
milestone, not final delivery of the complete external comparison.

---

## 2026-10-26 Current-model Kodak evaluation and WCL revision

All24 fixed original RGB256 center crops are evaluated at N1024, Es2,
AWGN4/10/19dB with3 noise realizations. Six configurations retain1296
logical method frames:648 new NeST PARTIAL/WHOLE and ARPC frames,648
exactly reused P100k/Swin200k/adaptiveBPG frames. The frozen main Stage-A
update38000 decoder is used; progressive-RD models remain separate.

New science:597 actual physical events,1194 header/body decodes,199 unique
native RX reconstructions. Four metrics cover5184 logical scores. The one
original10000x24 bank produced72mean CIs and60direct source-paired CIs.
All18method/SNR denominators are72/72 received; engineering failures and
additive CUDA/resource/metadata routes remain preserved. In particular,
ARPC's100 blocked native.py read attempts were reviewed as exact denied
reads before data return; do not claim an attempt-free original gate.

Kodak NeST has lowest DreamSim and highest CLIP-I means among five methods
at all3SNRs, while Swin/BPG retain higher PSNR and Latent lower4dB LPIPS.
At10dB, PARTIAL-WHOLE PSNR +0.673559dB and utilization78.1052%->94.9917%.
All3PSNR internal paired CIs are positive;19dB DreamSim CI crosses zero.
Full results: results/kodak_current_20261026_v1/delivery/REPORT_zh.md.

Anonymous WCL manuscript: paper/manuscripts/nestcom_wcl_kodak_20261026_v1/.
C1-C9 and actual Kodak prose are integrated; Table I has separate datasets.
Fig.1-3 remain unchanged. Normal IEEEtran double-column PDF is5pages;
single-column double-spaced PDF and editable Word are14pages each, all
pages actually inspected. Source ZIP is local-only under repository policy;
unpacked LaTeX/figures are tracked. Large data,weights,ledgers stay on PFS;
backups remain deferred. Publication receipts follow required checks.

---

## 2026-10-23 Progressive RD decoder scale-up delivered

All six registered decoder runs (three arms x two seeds) completed10,000
updates and final200-source x8-endpoint evaluation with actual child exit0.
The original owner/supervisor exit1 is retained for exactly two pre-Popen
resource rejections; recovery only first-launched those two untouched jobs.
No successful scientific call or completed summary was repeated.

Reuse19,968-source training caches; total60,000updates/240,000 main exposures.
Anchored arms add80,000 full-endpoint samples across the two seeds, including
separate frozen-teacher/student forwards. Equal updates are not equal FLOPs.
Validation200 comes from historicalcal1000 excluding pilot100, not blind
or wireless evaluation. One registered source-paired bank produced400mean
CIs and720paired CIs; source CIs do not estimate training-seed uncertainty.

Anchoring preserves short-prefix gains: versus matched old-recipe control,
(7,0) PSNR +0.387529dB and LPIPS -0.005027. At full(10,0), PSNR -0.002141dB
has a crossing CI, not an equivalence claim. Versus ORIGINAL frozen Dc,
full PSNR -0.019028dB and LPIPS +0.000524 remain unfavorable. MS-SSIM also
declines at several middle/long endpoints. Report all settings and metrics.
This is a candidate for further research, not a replacement for frozen WCL.

Results: results/progressive_rd_scaleup_20261022_v1/FINAL_REPORT_zh.md;
delivery_v1 contains complete rows/statistics, figures_v3 final reviewed
vector/PNG curves, report_v1 compute and checkpoint recovery inventories.
figures_v1 clipping failure and figures_v2 title-overlap history are retained.
Final scientific/product closure: final_delivery_v1/completion_evidence.json.
Repository checks and publication have separate actual receipts on PFS.

No encoder/VAR/codebook update, PHY, new hyperparameter scan or old task
restart. Original deadline unchanged; all science exited within its window.
Checkpoint optimizer/RNG/order state is saved; resume and cross-device
restore were NOT executed. Large caches/checkpoints stay on shared PFS;
local backups remain deferred. This bounded round stops after publication.

---

## 2026-10-22 Progressive RD follow-up fully delivered

All eight original expansion shards finished2371 sources each with actual
wait exit0, no timeout, and gone processes. The original owner is gone.
The one CPU closure audited all committed wire/RX/latent hashes and complete
ledgers. No successful scientific call was replayed or budget extended.

The merged cache covers19,968 train sources and100 calibration sources at
all8 endpoints (160,544 rows). The32 stage0 sources retain their2 benchmark
endpoints separately (64 rows); do not claim a full20k x8 grid. Exact scalar
CSVs, actual throughput/I-O, storage and PFS recovery bindings are delivered
under results/progressive_rd_20261022_v1/{cache_expansion_delivery_v1,final_delivery_v1}.

Three matched5000-update decoder pilots and their single calibration100
source-paired analysis were already published. Short-prefix PSNR gains
coexist with worse full-endpoint PSNR/LPIPS; conditioning gains are small.
Keep the full-precision curves and all contrasts in decoder_delivery_v1.
This single-seed calibration is not heldout or wireless evidence and does
not replace the frozen WCL model. No encoder, codebook, VAR, PHY or extra
hyperparameter run was started. All GPU workers exited within the original
deadline. Arrays/checkpoints remain on shared PFS; local backups are deferred.

Checkpoints retain optimizer/RNG/data order, but no resumed update or
cross-device resume test has been run; recovery_manifest.json records this
limit. The one-shot training entry is not a validated resume command.
The bounded round is complete. Further experiments require a new plan.

---

## 2026-10-22 Matched decoder pilot completed

All three registered arms completed 5,000 updates x batch4 and their final
calibration100 x8 evaluation with true exit0 and gone processes. The common
initial model and every source/endpoint exposure match. Final resumable
checkpoints, optimizer/RNG/data-order state and paired calibration records are
retained on PFS. One registered bootstrap bank produced all pointwise and
direct source-paired intervals; no heldout or wireless experiment was added.

Results show a tradeoff. Multi-prefix unconditional versus the equal-update
old-recipe control improves PSNR by 0.337/0.319dB at (7,0)/(7,25), but loses
0.152dB at (10,0), with worse LPIPS at most longer endpoints. Its comparison
against frozen current Dc is also reported: +0.307dB at (7,0), -0.270dB at
(10,0). Endpoint conditioning adds only about 0.002-0.028dB over unconditional;
most LPIPS paired intervals cross zero. Do not replace the frozen WCL model
or claim broad superiority from this single-seed calibration pilot.

Full-precision results, checkpoint bindings, Chinese report and actually
opened vector figures are in results/progressive_rd_20261022_v1/decoder_delivery_v1/.
Use figures_final_v1; earlier layout/guard diagnostics remain preserved.
All eight disjoint expansion shards are now running the remaining18,968 train
sources x8; that cache is not yet closed. The absolute deadline remains
1792672233.1326199. No encoder/codebook/VAR update or new PHY is running.
Local backups remain deferred. Existing WCL materials are unchanged.

---

## 2026-10-22 Independent progressive token RD follow-up

The user authorized an independent follow-up and confirmed at least six hours
of temporary eight-GPU access. The frozen WCL paper/results are preserved.
The shared PRO host has eight available96GiB-class devices (driver label
NVIDIA RPBZZZ6), PCIe topology without NVLink. All421 restored cache files
passed actual SHA checks;20k training/1k calibration identities remain fixed.

Stage0 is actually closed:32 train sources x2 real entropy/RX endpoints and
50 bounded decoder updates, trueexit0/full checkpoint. Measured steady
training throughput is21.19samples/s; checkpoint save1.08s. This is throughput,
not a quality-improvement claim. Seedv2 is also closed:1,000newtrain+100calibration
sources x8 legal endpoints,1,100TX traversals and8,800 independent source
decodes/original predict-overwrite-advance completions. All8 shards trueexit0,
gone processes, SHA-verified FP32 latents/wires/RX tokens and quality rows.
An initial generic-module import failure occurred before model-loading bodies;
all8 failed reservations are retained, with zero science and no successful replay.

Three matched decoder pilots are actually running:multi-prefix unconditional,
endpoint-conditioned and equal-update old-recipe control. Same current Dc
initialization,1k sources,5k updates x4, identical576-parameter affine capacity,
MSE+0.1LPIPS; only a final calibration100 x8 comparison is planned. Historical
class-conditioned Fb_TX is used only by the old-recipe control training.
Another18,968 disjoint training sources x8 are expanding into persistent PFS
cache. Five GPUs start directly;three wait for their own training to close.
The original absolute deadline is shared, never reset. These active jobs are
not reported as complete. No encoder/codebook/VAR update, PHY or heldout runs.

See experiments/progressive-rd-20261022/STUDY_STATUS.md and
results/progressive_rd_20261022_v1/{stage0_delivery,seed_delivery_v1}/.
The13.7MB expansion registration is stored as2.1MB deduplicated metadata plus
an exact-byte restore utility, preserving Git's10MB limit and original SHA.
Arrays, logs and resume checkpoints stay on PFS. Local backups remain deferred.

---

## 2026-10-21 Fig.3 editorial finalization

The confirmed N1024,10-dB two-row/six-column Fig.3 now labels the original
column `(a) Pastries` and `(b) Dog`. The concise caption explicitly defines
the displayed order as LPIPS-down / DreamSim-down. Preprocessing details
remain in Experimental Setup and full-precision source records. A brief
qualitative paragraph describes the visible differences in these selected
development cases without treating them as population statistics.

All twelve embedded images and ten score pairs are identical to parent
56efef3e. Word, LaTeX and English captions are synchronized; print-size
exports and all six Word pages were opened, and LaTeX compiled successfully.
No new inference, reconstruction, channel, bootstrap, training or timing.
Fig.1/2 and Tables I-III remain unchanged. Current editable files are in
`paper/wcl_revision_20261021_v1/`.

---

## 2026-10-21 Fig.3 selected example: white long-haired dog

At the user's request, the second Fig.3 row replaces Shopping-cart75 with
the originally selected white long-haired dog21 at N1024, 10 dB,
timing_case s10_i0021_r3. Pastries95 remains the first row. The figure keeps
two rows of full views and the five-method order; labels are LPIPS-Alex /
DreamSim, both lower-is-better, to three decimals.

The six dog images and all existing LPIPS values are reused with exact
file/array identity checks. Only five missing dog DreamSim distances were
computed, using the pinned main-evaluation ensemble, FP32 and 224-pixel
bicubic-antialias float preprocessing. Actual work: one model load, one
source embedding, five reconstruction embeddings and five distances;
all12 ledger entries COMPLETE, real child/outer waits0, all processes gone.
The five Pastries scores are reused exactly. Full-precision ten-row table:
`results/figure3_dog_20261021_v1/delivery/figure3_metrics.csv`.

PDF, editable SVG, PNG, Word, current LaTeX and captions are synchronized
in `paper/wcl_revision_20261021_v1/`. Actual print-size figure and six Word
pages were opened; the existing LaTeX document compiled successfully.
Fig.1/2, Tables I-III, sensitivity and main quantitative results are unchanged.
No channel, reconstruction, training, bootstrap, timing or new LPIPS calls.
The prior shopping-cart candidate and all preceding scores remain archived.

---

## 2026-10-21 Fig.3 LPIPS-Alex / DreamSim labels

The confirmed N1024, 10-dB Pastries95 and Shopping-cart75 Fig.3 keeps the same
r3 reception records, full-view pixels and two-row/six-column method order.
Labels now show LPIPS-Alex / DreamSim, both lower-is-better, to three decimals.
Ten existing native-float reconstructions were scored only for DreamSim using
the main evaluation's pinned ensemble, FP32 and 224-pixel bicubic-antialias
float preprocessing. Archive and dtype/shape-aware array hashes match; the
existing LPIPS strings are reused exactly. Two source embeddings are cached.
All ten distances are saved at full precision with input identities at
`results/figure3_dreamsim_20261021_v1/delivery/figure3_metrics.csv`.

Actual work: one model load, two source embeddings, ten reconstruction
embeddings and ten distances; all23 ledger entries COMPLETE, child/outer
actual waits0, all processes gone. No new reconstruction, channel, old metric,
bootstrap, training or timing. Updated editable SVG, PDF,600-dpi PNG and
source are in `paper/wcl_revision_20261021_v1/`; Word and existing LaTeX Fig.3
and captions are synchronized. Print-size and all six Word pages were opened;
LaTeX compiled. Fig.1/2, Tables I-III and sensitivity results are unchanged.

---

## 2026-10-21 Compact WCL figures and closed 13-dB candidates

The revision at `paper/wcl_revision_20261021_v1/` now uses the same simplified
Fig.1 in Word and LaTeX. It retains the original schematic photos/token grids,
compresses channel/codec implementation labels and separates system flow from
boundary-scale predict/overwrite/continue. The 77.70/95.21 percent utilization
evidence remains only in Table I. Editable PPTX and vector modules/text are
7.16 x 3.1683 inches, with auxiliary type >=7.26 pt. Fig.3 is the confirmed
10-dB Pastries95/Shopping-cart75 two-row, six-column full-view comparison,
7.16 x 3.00 inches, without ROI boxes or crop rows. Image bytes and native-float
PSNR/LPIPS labels match. All six Word pages were opened; LaTeX compiled.
Fig.2, Tables I-III and the completed ARPC sensitivity results are unchanged.

The authorized four-source current-method 13-dB check is complete at
`results/selected_examples_13db_20261021_v1/delivery/` and
`paper/figures/selected_examples_13db_20261021_v1/`. Source order is 21/92/95/75.
Before execution, digital methods were assigned the largest registered SNR
not exceeding 13 dB (the frozen 10-dB endpoint/MCS), retaining actual-length
fallback. This is a declared conservative mapping, not a newly calibrated
13-dB optimum. Continuous methods use their existing 13-dB conditioning.

Actual work: 20 logical events, 12 reused digital TX payloads, 12 header and
12 body decodes, and 8 continuous TX/RX calls. All 12 complete digital received
information identities match 10 dB; their reconstruction pixels and single
realization scores are reused. The other 8 native-float reconstructions and
PSNR/LPIPS pairs are new. All20 are accepted; 24 lossless full-view PNGs and a
four-row candidate gallery are delivered. Method-specific original noise IDs
and counters remain; identical cross-method noise is not claimed.

Two initialization failures (PHY interpreter missing Sionna; metric import
search path) are retained before any corresponding science/model/pair call.
Pinned additive continuations reuse completed stages; successful scientific
calls repeated=0. Historical outer failure waits remain failures. No new
caption/tokenizer/source-probability traversal, training, calibration,
bootstrap, timing or common500 expansion was run. NeST/ARPC/BPG pixels remain
identical; Latent JSCC/SwinJSCC improve modestly on these four selected cases.
The paper keeps the confirmed 10-dB figure; 13-dB candidates remain separate.
Backups remain deferred as requested.

---

## 2026-10-21 WCL revision and completed ARPC decoder sensitivity

`paper/wcl_revision_20261021_v1/` provides the Chinese paragraph-task outline,
a six-page Word outline with actual figures/tables, a compiled LaTeX skeleton,
three final figures, Tables I-III, captions, exact inputs and visual checks.
The title is NeST-Com: Fixed-Budget Next-Scale Token Communication. The original
author-supplied system_diagram_v3_entropy.pptx was located in Downloads and
reflowed as editable native PowerPoint objects at 7.16-inch width. Its 10 dB
77.70/95.21 percent entropy-bitstream utilization uses Table I's definition.
Fig.2 is 1x4; Fig.3 retains Pastries95/Shopping-cart75 full views with identical
within-row ROIs. Table I is N1024 WHOLE/PARTIAL utilization and paired
PSNR/DreamSim; Table II is N2048 paired PSNR/DreamSim; Table III is existing
10 dB software processing latency. The old files and full eight metrics remain.

The main evidence still uses a5f8ceb0233b0e1f599ee8bb902c1a094d47119d. The only
new science is the expressly authorized matched100 ARPC decoder sensitivity:
`results/arpc_decode_sensitivity_20261021_v1/delivery/`. Original sampling
tau0.5 is reused. Explicit argmax and predeclared tau0.25 change only untransmitted
generation, with all received prefixes and protocol fixed. 592 new unique
renders cover 1800 logical setting-frames; 900 original logical results are
reused. Four metrics and direct source-paired intervals are complete, using
the original 10000x100 bank without new index draws. Actual RX/score/statistics
waits are zero; all science processes are closed. A preparation syntax error
before registration/model calls is retained; no successful science was replayed.
The old CLIP reuse matches source and reconstruction pixels for all900 frames.

At19dB both alternatives improve PSNR, LPIPS, DreamSim and CLIP-I. At4dB both
worsen LPIPS/DreamSim; at10dB low temperature improves PSNR/LPIPS while all
greedy intervals cross zero. All settings are reported and none replaces the
common500 main curve. The outline retains NeST's PSNR limitations, worse4dB
LPIPS than Latent JSCC, metric disagreements, and ARPC's better N2048/19dB
DISTS (+0.0054 NeST-minus-ARPC, CI +0.0024 to +0.0086).

No new training, source encoding, caption, channel event/decoder or timing was
run. N2048 CLIP and complete manuscript prose remain outside this execution.

---

## 2026-10-20 WCL figures and tables finalized from frozen results

`paper/wcl_figures_tables_20261020_v2/` provides Fig.2 (PSNR, LPIPS,
DreamSim, CLIP-I; five-method N1024 common500), Fig.3 (the previously selected
Pastries95 and Shopping-cart75 development reconstructions at10dB), the unchanged
N2048 paired Table I, and the existing TX/RX/Total software-processing-latency
table. English captions, results paragraph points, exact input hashes and
reproducible drawing code are included. All60 means/CIs,12 image bindings and
six N2048 metric cells were checked; actual PDF, print-size and grayscale
inspection passed. Prior figure versions and full eight-metric results remain.

The scientific baseline is a08e2517389f4df00ea4162c98417df3f52560a8, with no
intervening result change at inspection. This is a presentation-only revision:
no new training, channel, reconstruction, metric, bootstrap or timing calls.
The outline explicitly retains NeST's PSNR limitations, its worse4dB LPIPS
than Latent JSCC, LPIPS/DreamSim disagreement, and ARPC's better N2048/19dB
DISTS. Historical H800 timings are separate from the current PRO6000 table.

Fig.1 artwork is unchanged. The exact confirmed author-held
`system_diagram_v3_entropy.pptx` was not found in the checkout or local
workspace/Downloads inventory; its location is the remaining assembly item.
No older system diagram was substituted and no full manuscript was rewritten.

---

## 2026-10-20 CLIP-I supplement for the current five-method common500

`results/clip_i_common500_20261020_v1/delivery/` closes all 22,500 logical
source/reconstruction pairs: five methods, 500 original sources, 4/10/19 dB,
and each method's three original noise realizations. Native file/pixel hashes
were verified with no missing array. NeST-Com 19 dB uses T1-C's corrected m10
outputs; P100k, Swin200k, adaptive BPG and all 4,500 ARPC outcomes retain their
final versions and original denominators.

CLIP-I is raw CLIP image-image cosine similarity (OpenAI ViT-L/14, 224 input,
frozen floating tensor preprocessing, FP32). No softmax or percentage conversion
is used. One model load and 13,980 image embeddings
produced 13,480 distinct pair scores, with
9,020 exact logical-pair reuses. No training,
channel, reconstruction, caption or other neural-metric inference was added.
Original CLIP weights/source were recovered from existing static archives with
their frozen hashes; the inference environment was not upgraded.

Source means average the three original noises; the existing seed 2026100701,
10,000 x 500 bootstrap bank supplies 15 pointwise mean intervals and 30 direct
source-paired contrasts, including all 12 NeST-minus-baseline contrasts.
NeST-Com means: 4 dB: 0.8206, 10 dB: 0.8731, 19 dB: 0.9005. Results of every method are reported without
policy/checkpoint reselection. Pairing is by source; historical cross-method
noise namespaces differ. Original seven-metric summary and paired values are
copied unchanged into the eight-metric tables.

`results/clip_i_common500_20261020_v1/figures_v2/` contains the separate reviewed
CLIP-I PDF, editable SVG and PNG, caption, style and visual audit. Existing main
figures and the open Table I are unchanged. Development examples are excluded;
N2048 is not part of this first-stage supplement. New features/ledgers and
native arrays remain outside Git.

---

## 2026-10-20 Four selected development examples at current N1024 / 10 dB

`paper/figures/selected_examples_10db_20261020_v1/` contains four unchanged
original crops, twenty lossless RGB8 reconstructions, and the reviewed 4 x 6
candidate sheet. Fixed order: dog21, broccoli92, pastries95, shopping-cart75.
The current entropy-coded NeST-Com, Latent JSCC100k, SwinJSCC200k, adaptive
BPG+LDPC and ARPC+LDPC all reuse the first formal timing receive realization.
Twenty missing RX pixel arrays were recovered, and every float32 output exactly
matches its original timing hash. No source encoding, new noise/channel event,
training, calibration, timing grid or bootstrap was run. Swin's four saved
waveforms pass through its original paid-header receiver as part of RX recovery.

Twenty single-image PSNR/LPIPS pairs were saved using the original definitions
and weights. The scoring worker exited1 at its final hook-count assertion:
LPIPS directly calls its backbone forward and bypasses those hooks. This
post-output telemetry failure is retained; the separate CPU-only closure audit
validates all twenty saved pairs without recomputation. It is not relabeled as
original exit0. Metrics describe native float outputs; PNGs are uniform RGB8
display copies without crop, sharpening or generated enhancement.

These are previously selected development examples, not random common500
samples. Method-specific noise namespaces are disclosed. Existing Fig.1,
quality curves, timing results and the open Table I are unchanged.
Details: `results/selected_examples_10db_20261020_v1/delivery/`.

---

## 2026-10-20 Current five-method N1024 software timing

`results/five_method_timing_20261020_v1/delivery/` contains the current m10
NeST-Com, Latent JSCC100k, SwinJSCC200k, adaptive BPG+LDPC and ARPC+LDPC
TX/RX/software-total latency on the same PRO6000/CPU allocation. The original
fixed16 development sources, batch1, SNR4/10/19, three fresh warmups and three
formal calls per source/setting yield1440 completed calls and720 measurements.
All receiver outcomes remain in the denominator. Means, medians, p95 and
unrounded per-call ms/image, resource/software versions and receipt audit are
included. There are no new quality metrics, bootstrap, training or N2048 calls.

Two engineering failures are retained: a receiver library import before any
case/model construction, and one incomplete BPG encoder attempt that stalled
when legacy x265 created pools for180hostCPUs. The explicit continuations
reuse successful stages. BPG's unchanged binary now detects its actual6CPU
affinity through a child-only resource adapter; original presets/math/search
remain fixed. Original software stacks and this resource difference are
disclosed. These timings are separate from the historical H800 six-method set.
Source/registration/recovery entries are `experiments/five-method-timing-20261020/`.
The existing paper figures and open quality Table I are unchanged.

---

## 2026-10-20 Frozen-data WCL figure/table redraw

`paper/wcl_figures_tables_20261020_v1/` contains reviewed Fig.2 (five-method
common500, 178 x 106 mm), Fig.3 (paired PSNR gain and source-capacity utilization,
178 x 62 mm), and Table I (N2048 matched100 adaptive 4/10 dB and full 19 dB).
All values and intervals are read from the five explicitly specified CSVs at
commit a5fc2de0c0039689f2999ad0ed6a1c12c1487d06. No new experiment, reconstruction,
neural score or bootstrap was run. Fig.1 and historical artifacts are unchanged.
Vector PDF/editable SVG, 600 dpi PNG, English captions, byte-pinned input copies,
reproducible rendering code and actual visual/export checks are included.
The positive 19 dB NeST-minus-ARPC DISTS difference is retained and favors ARPC.

---

## 2026-10-20 ARPC N1024 common500 extension delivered

The fixed ARPC external100 test is extended to the original common500, retaining
original source indices and frozen 4/10/19 dB policies. All4500 logical events
are represented (1500/SNR):900 verified existing plus3600 additional events.
New400 captions/tokenizations/probability traversals produced2800 endpoint streams;
no TX clean rendering, recalibration, training or other baseline inference.
New link work used1200 encoded waveforms and3600 header+3600 body decodes.
Actual RX produced1184 new images; seven-metric scoring
covered1184 unique new pairs with all failures retained.
All105 five-method mean/CI cells and21 direct source-paired NeST-minus-ARPC
intervals use the existing10000x500 index bank, with zero new random draws.
The original four-method main means are unchanged within1e-10; NeST19dB uses
the current T1-C m10-corrected PARTIAL version. Historical noise realizations
differ across baselines: pairing is by source, not common noise across all five.

The original PHY RX exited1 after saving all3600 outputs because its final
denied-open assertion failed. Exact denied paths were lost from its memory-only
list. The original enforcing hook and failure remain unchanged; a separately
recorded saved-output audit completed without replaying a decode/model call.
This attempt-telemetry limitation is disclosed, never labeled originalexit0.
Native RX and scoring have actual exit0. The original native RX post-exit audit
exited1 on100 denied opens of its registered public wrapper source; all exact
paths persisted, with no private path. A separate unchanged-pixel audit closed
that explicit metadata lineage without any model replay. Original failures and
the corrected cross-host qualification check are retained. The admitted
PRO6000 host/runtime and resource selection, including Python build difference,
are recorded; no cross-hardware timing claim is made.

See results/arpc_common500_extension_20261019_v1/final_report_v1/REPORT_zh.md,
delivery_v1/ for all seven metrics and original100-to500 comparisons, and
figures_v1/ for the actually reviewed five-method2x2 vector Fig.2, plotCSV,
English caption and outline changes. The full manuscript is not rewritten.
N2048 remains the completed100-source scope; prior ablations/timing are retained.
Backups remain deferred. The normal commit/push is evidenced by publication_v1;
this note does not pre-fill an unobserved remote SHA.

---

## 2026-10-19 本轮实际进展：一致性修复、九单元校准与冻结测试完成

科学调用到原定范围末端停止；后续仅确定性报告、论文与正常发布。所有登记、调用、失败与复用计数按实际记录，不能将工程资格或可见目录当科学完成。

### 已完成的证据与结论

- N1024 common500 同版本审计及最小修复已闭合：9000 逻辑帧，6579 精确复用、2421 新物理事件、4842 译码，807 独特新接收输出及七指标；复用原 bootstrap 索引，新增 draw 为 0。原部分尺度 4/10 dB 和 T1-C m10 的 19 dB 结果保留，未重新选策略。4/10/19 dB 同先验 PSNR 增益为 0.61603/0.70116/0.31942 dB；19 dB agreement 为 −0.2 pp，直接配对 95% CI [−2.0,+1.6] pp，不声称全部指标获胜。
- N2048 B2 校准按前瞻九单元修订实际完成 cal1000 × 3 噪声 × 9 单元，共 27000 对象、54000 子帧、108000 头部/正文译码。NeST 的候选数是 4 dB 两个、10 dB 五个，ARPC 各一个，并非对称搜索宇宙；最终两方法均冻结为 4 dB QPSK 2/3、10 dB 16-QAM 2/3。选择依据为源内三噪声平均 DINO，未把校准均值当测试性能，未新增候选或校准 bootstrap。
- N2048 B3 冻结测试实际闭合：同 external_common100、4/10 dB、两方法、三噪声，1200 对象、2400 子帧、4800 译码。四个方法/SNR 单元均实际正确接收 300/300，null/reject/wrong 为 0。400 独特图像对中 241 新恢复/评分（NeST199、ARPC42）、159 身份完整核验后复用；1200 逻辑分母完整保留。只生成一次 10000 次来源级直接配对 bootstrap，seed=2026101902，bank SHA `2214f74adcc0e159679790b5ece0960e47aec5d933c1fec3d0969686be7846fe`。
- B3 的 NeST−ARPC PSNR 为 4 dB +3.778629 [3.493258,4.081621] dB、10 dB +3.877430 [3.687650,4.078679] dB；DreamSim 差为 −0.265833 / −0.117403；agreement 为 +39 [28,50] / +12 [2,23] pp。全部 14 个点态直接配对区间在对应指标方向有利于 NeST，未进行多重比较调整，不能解释为所有工作点上的保证或仅由终点粒度造成的因果效果。
- 固定完整 N2048 参考仍单列：64-QAM 5/6 在 4/10 dB 全拒收，在 19 dB 两方法均 300/300 正确。19 dB ARPC 的 DISTS 更好，DINO/agreement 区间跨零。自适应改善同时改变 MCS 和实际源码回退，不是纯信道码隔离消融。原共享 H800 pre-m10 计时保持旧版本说明，无跨硬件 ARPC 倍速；原 raw 两组 NO_GO 不改。

### 权威轻量结果

- `results/n1024_consistency_20261019_v1/repair_v1/delivery/`：完整七指标、源内均值、直接配对区间与闭合 manifest，SHA `8d6a29a210c300082f2ad14bc99b7334d2138d0b1f17d7f92195d21b31f4e235`。
- `results/arpc_n2048_adaptive_20261019_v1/b2b3_analysis_v1/B2/selection.json`：四冻结策略，SHA `7c3641837182d0ee7d1fd46e3372b84a8e5d751186ce7b3fb8f59a76cc561d1e`。
- `results/arpc_n2048_adaptive_20261019_v1/b3_metrics_v1/summary.csv`（28行）、`paired_differences.csv`（14行）和 `source_noise_means.csv`（2800行）：全部七指标原精度结果。科学 completion SHA `1e9a004069717e005e64257d0ef7a9896db9aed328b5652b2dc5866f7475aa90`，实际统计退出 0，PID/start 同代已退出。
- `paper/manuscripts/nestcom_entropy_v6_20261019/`：完整熵编码主稿源码，摘要、方法、结果、限制、结论、中文修改说明及主张证据表已接入真实 B3。raw 是辅助接口，actual length 不单独决定终点，RX 无 TX 真值，熵码失败输出灰图。

### 工程失败与交付边界

原测试 fixture 的 zipfile/cp437 与 dir_fd 惰性访问失败、历史 cache 元数据 schema 缺口、MAIN 读取 leo 主机局部 binary/runtime 导致的封存失败、报告输入根遗漏 5 个已登记共享 RT 源码路径的元数据失败均保留原实际非零退出和日志。资格分别按已通过 19+37、15+11 唯一测试清单闭合，未伪称单次全集通过；成功模型调用未为这些工程事项重跑。原六分析源、原七传输源及统计数学保持冻结，修正仅走明确有据的元数据兼容与正确执行主机。

当前科学结果、最终报告审图、最终全文编译与逐页审查均已实际闭合。最终9页PDF两次真实pdflatex退出0，全部页面分别由主稿任务和根任务实际打开，图1–5、表I–VI、正文和参考文献完整。PDF SHA256为 `9d3908d8cbaad7f8526fa3a1d517645c67c848af09d8cef6352fdc8dfa860f8d`，完整交付入口为 `paper/manuscripts/nestcom_entropy_v6_20261019/final_delivery_manifest.json`。报告原导出及版式修正原件均保留，修正只处理标题边界，不改变数据或区间。正常main发布由后续实际发布收据证明，此处不预填未发生的commit或push。作者姓名、单位、基金和目标期刊篇幅仍需作者输入，不伪造，不阻塞科学内容审阅。备份依用户指示延后；不恢复旧实验或拓展新候选。

---

以下保留此前历史原文；如状态冲突，以上最新实际记录为准。

## 2026-10-19 N2048 adaptive cal100 delivery: candidate coverage pending

The separate low-SNR adaptive package has completed B0/B1 and all cal100 clean
source gaps. Both equal-protection fragments pay their68-symbol header and
13-bit length,16-bit CRC and96-bit reassembly record within the2048-use budget.
35,200 existing source objects were verified; no TX/tokenizer rerun.24 physical
qualification objects used48 subframes and96 decoder calls without repetition.
The original RX final exit1 remains preserved; a metadata-only closure audited
already saved decodes after an optional Python bytecode cache read was denied.

Cal100 now has3200 clean endpoints:1407 verified existing image/score states,
1793 new NeST native recoveries and1793 new DINO values, reusing100 original
source features. All384 complete policies use actual per-image length fallback.
At10dB five NeST policies remain plausible under historical packet-proxy bounds:
QPSK2/3,3/4,5/6 and16-QAM1/2,2/3. They cannot fit the two-policy condition cap.
Per the package uncertainty rule B2/B3 have NOT run; no new test score, paired
bootstrap or low-SNR wireless improvement is claimed. The other three conditions
are covered within two proposals. Using one plausible ARPC policy perSNR would
need nine calibration cells:27000logical/54000subframes/108000decoder attempts,
still beyond the original global and per-condition scope. This is a costed
option, not an authorized expansion. Old N1024/fixed-full N2048 remain unchanged.

See results/arpc_n2048_adaptive_20261019_v1/README.md, B0_B1_delivery/REPORT_zh.md,
all source-level diagnostic tables and the actually opened vector figures_v2
(directory figures_calibration_v2). Scientific continuation waits on scope;
no training, power extension, extra methods or heavy backup is launched.

## 2026-10-19 N2048 two-method full-representation validation completed

The authorized fixed100 validation is complete under the shared two-subframe
2048-use budget, paying both headers, lengths, CRCs and96-bit reassembly records.
Complete NeST-Com m10 and ARPC e7 objects both fit; fixed64-QAM rate5/6 delivers
0/300 objects per method at4/10dB and300/300 at19dB, with no wrongly accepted
objects. Rejected objects remain gray in all denominators. This fixed full-source
layout is capacity-feasible but does not solve low-SNR reliability.

At19dB direct paired intervals favor NeST-Com for PSNR, LPIPS, DreamSim and
MS-SSIM, and ARPC for DISTS; DINO-L and source-prediction agreement intervals
cross zero. All seven metrics and pointwise intervals are reported. The single
10000-replicate bootstrap samples sources after averaging three noise draws.
Qualification and formal work used3648 transmitted subframes and7300 decoder
attempts, including exactly four explicitly authorized repeated decodes after
the preserved import failure.200 formal native reconstructions and300 distinct
seven-metric pairs completed. No training, tokenizer or TX probability rerun.

Original N1024 results and all failure lineage remain intact. Reviewed final
PDF/editableSVG/600dpiPNG are figures_v2; the first title-clipped export is retained.
See results/arpc_n2048_validation_20261019_v1/README.md and delivery/REPORT_zh.md.
No additional scientific stage is launched; restoration backups remain deferred.

## 2026-10-19 ARPC N2048 feasibility plan delivered (not executed)

Preserved the completed N1024 comparison at fbc17ae. Joined the existing cal100
700 endpoint payload/quality records and exported reviewed vector curves. A
proposed 2 x (68-header + 956-body) layout pays both length/CRC fields and two
96-bit reassembly records: 9296 source bits at common 64-QAM 5/6. Both full
NeST-Com m10 and ARPC e7 fit 100/100 calibration sources. Historical low-SNR
reliability is a warning, not new two-fragment evidence. Existing complete
source objects are checked; new TX passes can be zero. Proposed fixed100
validation is 3600 subframes, <=3648 including qualification. No new channel,
model, metric, bootstrap or training calls occurred. N2048 execution requires
a later instruction; automation remains paused and backups deferred.
See results/arpc_n2048_feasibility_20261019_v1/PLAN_zh.md and proposed_scope.json.

## 2026-10-19 ARPC fixed100 matched comparison completed

Official adapted ARPC weights passed independent source decoding and the full
finite cal1000 strategy calibration. Frozen policies then completed900 logical
frames at4/10/19dB on external_common100_v1, followed by all seven metrics and
one10000-replicate source-paired analysis. The same100 source IDs select existing
final baseline scores; NeST-Com19dB uses T1-C m10. All failures remain in their
denominators. Results in every direction are reported without test-set selection.
See results/arpc_external_baseline_20261019_v1/final_report_v1/REPORT_zh.md and
final_figures_v2/. This is not the fullcommon500 result or a new blind test.
Original failed initialization and receiver-environment evidence is preserved.
No training, old baseline scoring, old timing or reconstruction was repeated.
Restoration backups remain deferred. Historical experiments/figures are unchanged.

## 2026-10-19 ARPC A1 and A2a completed; real link calibration next

Official-weight A1 independent RX and visual/resource checks passed. The fixed
cal100 now has700 clean prefix states (676 new,24 reused),100 total source traversals
(92 new,8 reused), and800 three-metric pairs including100 failure-gray references.
All84 complete strategies were screened by actual fallback DINO-L with matched
historical PHY uncertainty; each4/10/19dB point nominates two candidates within
the original cap. No ARPC wireless frame, A2b calibration, A3 evaluation or bootstrap
is claimed in this milestone. See results/arpc_external_baseline_20261019_v1/A2A_REPORT.md.
Unknown failures and original initialization attempts remain preserved. Backups
remain deferred; old results and completed experiments are unchanged.

## 2026-10-19 ARPC external comparison: A0 engineering preparation

The user authorized the attached ARPC package through its gated final evaluation.
Initial HEAD matched `3ac13a02`; no newer result was displaced. Original cal1000
and common500 source IDs yielded fixed hash-selected cal100 and external_common100_v1
lists before image/metric inspection. A1 uses the first eight calibration IDs.
Twenty-five CPU container, boundary and real process-control tests passed. Official
ARPC/T5/BLIP assets are downloading and an isolated leo-tactile environment is being
prepared. This milestone has zero model inference, reconstruction, metrics or
wireless frames. A1 real-weight qualification has not run and is not claimed.

The prepared receiver accepts paid text, actual native source bits and public
profiles; it does not receive TX CDFs, embeddings, pixels or untransmitted tokens.
Source costs include text compression/length, native modes/boundaries, termination
and alignment. A1 actual equality, visual sanity and resource estimates gate A2;
frozen calibration gates the 100-source A3 comparison. Historical T1-C, old raw
NO_GO, existing seven-metric results and completed H800 timing remain unchanged.
See `experiments/arpc-external-baseline-20261019/README.md` and
`results/arpc_external_baseline_20261019_v1/` for scope and recorded preparation.

## 2026-10-16 T1-C corrected m10 test evaluation delivered; stop

Frozen entropy whole and partial strategies completed19dB on common500 and
independent new100, each with three original noise realizations. The600 actual
m10 source traversals retain all24 old endpoints byte-exact. Fullm10 fits6/500
and1/100 sources; the rest follow the frozen family-specific fallback chains.
The3600 logical denominator uses2454 exact reused frames (including378
cross-family equivalent events) and1125 new physical events, with
2250 actual packets within2250. All new PHY/RX/metric/statistics stages actually
exit0. Full-denominator source means and direct paired intervals use10000
source resamples, seed2026100701, separately for500 and100 sources.

Historical common500 whole-code streams from the older4090 environment differ
from the current frozen H800 implementation. Old-to-new whole-code deltas
therefore mix this version effect withm10; no solem10 attribution is claimed.
Old results remain intact. common500 retains allseven metrics; new100 retains
its originalfour. No4/10dB, other method, training, selection, LE3, HiFi or
shared-timing rerun. Existing timing remains labeled as the old version.
Scientific backups remain deferred. See
results/t1c_m10_test_20261016_v1/delivery/REPORT_zh.md.

## 2026-10-16 T1-B symmetric m10 calibration delivered; stop before T1-C

Both whole and partial families now include maximum target m10 with their
separately frozen fallback chains. All old candidates/winners are retained.
Source lengths collapse20 exact calibration aliases; four changed strategies
are completely evaluated on originalcal1000 at4/10/19dB and4101/4102/4103.
The36000 logical grid required22139 new physical events and44278
packet decodes, within44278. Exact existing events/images/scores were reused;
no source TX/tokenizer/training/test-set calls were added. One original RX
initialization failed before Torch import; its failed model-load reservation
is preserved. A separate runtime-library read boundary recovery inherited
closed PHY without rerunning it and completed RX. A later UUID spelling
assertion stopped metric initialization with zero model constructions/scores.
Exact UUID normalization retained the same physical GPU and completed
metrics and selection without repeating PHY or RX.

Frozen old/new winners and all four metrics, all new logical frame scores,
source means, paid-profile evidence and actual exits are delivered in
results/t1b_m10_calibration_20261016_v1/delivery/REPORT_zh.md.
There are2 changed family/SNR winners; calibration maximum differences
are not independent test gains or novelty evidence. Old raw R1 NO_GO,
LE0/LE1/LE2, fixed16 and shared timing stay unchanged.
T1-C/common500/new100 source-boundary work and LE3 were NOT_RUN.
No automatic successor. Scientific backups remain deferred by the user.

## 2026-10-16 Submission first batch delivered; m10 correction required

T0/T1-A/T2/T3 only. Originalcal1000 completed1000 source TX and8 independent
source RX; all24000 existing endpoints match byte-exact. Fullm10 lengths
min/median/P95/max=3823/6071/6568.05/6855bit. Capacity4273 fits4/1000;
capacity4751 fits18/1000. Both whole and partial families need symmetricm10
candidate correction. Old candidates/results are retained; no new calibration
or test link run was started. Changed-prefix subset396logical frames is not
the full calibration budget; see the explicit prospective upper bound.

Native raw fixed16 was recovered from its historical assets and scored at the
original13dB condition:48 new DISTS/DreamSim/MS-SSIM plus64 new PSNR scalars,
144 old scalars reused. Raw has lower perceptual distances and lower PSNR/MS-SSIM
means; these16 development sources do not establish population superiority.
Source roles/intersections, actual LDPC layouts, model/training facts, existing
common500 metrics and shared-H800 timing are delivered with9 reviewed figure
sets. The common500 layout is explicitly provisional pendingm10 correction.

No new channel packets, wireless images, renders, diffusion, training, bootstrap
or timing. Raw R1 NO_GO, LE0/LE1/LE2 and completed timing remain unchanged.
T1-B/T1-C/LE3/LE4 and power expansion are NOT_RUN. First batch stops here.
See results/submission_closure_20261016_v1/REPORT_zh.md.
Scientific backups remain deferred at the user's request.

## 2026-10-15 LE2 finite packet proxy delivered; stop before LE3

LE0/LE1 and old raw R1 NO_GO are preserved. The new LE2 revision corrects
the29bit length+CRC fields in each proportional allocation and nominates complete
strategies from every Pilot50 source's actual fallback prefix quality. It covers
single956 MCS and grouped reliability at7/10dB:18688 new packets,92 initial cells
and18 topups within21504; no new wireless image/VAR/clean/training calls.
Best grouped proxy J=.6919812165440816 versus single robust .7021032469582041,
delta=-.010122030414122518: NO_PROMISING_MEASURED_CANDIDATE. Unmeasured layouts
remain unknown and no global NO_GO or wireless image gain is claimed.
Nine deduplicated diagnostic/exploratory LE3 candidates are delivered, no
recommended positive proxy candidate; LE3 and power expansion are NOT_RUN.
Gray hash-convention and CSV metadata failures with their real exits remain
explicit; CPU-only final delivery exited0 without repeating any packet/score.
See results/layered_entropy_le2_20261014_v1/delivery_v2/REPORT_zh.md.
Scientific backups remain deferred at the user's explicit request.

## 2026-10-14 LE0/LE1 layered entropy source feasibility delivered

Only LE0 and LE1 were authorized and completed. See
results/layered_entropy_feasibility_20261014_v1/REPORT_zh.md.
The deterministic pilot50 from original raw cal200 completed50 TX traversals,
24 independent terminated two-stream RX proofs, and1200 clean source/endpoints
(750 exact reuse,450 missing-state renders and DINO-L scores). Wireless image
frames, packet trials, tokenization and training updates are all zero. The actual
extra segmentation information cost is median30bit, range28--32bit, including
the second13bit length field and16bit CRC. All terminal lengths are exact;
first38 streams retain an explicit missing auxiliary finish-increment field.
Source/model/CDF identities match. GPU3 resource stop and metric API failure
remain disclosed with actual exits; successful science was not repeated.
Actual Sionna group constructors and the capacity domain are independent design
evidence; original ep_phy.Backend.plan is unchanged. Prospective LE2 has at most
21504 packet trials if separately authorized. LE2/LE3/LE4 are NOT_RUN and no
automatic next stage is scheduled. All evaluated old raw two-group candidates
are dominated in the DINO and PSNR true7/10 planes; original R1 NO_GO is preserved.
Clean-prefix quality and arithmetic packability do not assert wireless gains.

## 2026-10-14 common500 supplemental quality metrics

The six frozen methods now have all27000 original common500 image pairs /
81000 DISTS, DreamSim and MS-SSIM scalar positions at N1024,4/10/19dB,
three original noises per source. See results/quality_metrics_supplement_20261014_v1/common500.
Every failure/gray frame is retained. The user separately authorized9000 missing
P100k/Swin200k pixel replays; all9000 match the old pixelSHA and header state.
No training, new samples/SNRs, checkpoint/policy selection, old-score replacement
or main-figure change. Per-source three-noise means and10000 direct source-paired
resamples(seed2026100701) report all three metrics and all15 method pairs with
pointwise95% intervals; no global-winner or simultaneous-significance claim.
The delivered13dB HiFi development16 remains separate. Original R1 NO_GO and
six-method timing remain unchanged. Full restoration increment accompanies
the normal publication; source/weights/environment prior packages are dependencies.

## 2026-10-14 saved-output HiFi metric supplement

The original N1024/13 dB development16 HiFi display set and its final Swin200k
and P100k counterparts now have all 48 image pairs / 144 DISTS, DreamSim and
MS-SSIM scalar scores. See results/quality_metrics_supplement_20261014_v1/hifi_fixed16.
All16 per method are retained, with descriptive summaries and direct paired
differences; no CI or common500 population claim. The six-method common500
supplement waits for original P100k/Swin200k pixels by explicit user choice.
No training, channel/receiver replay, selection feedback or main-figure change.
Two zero-score initialization failures and the scoped metadata/private-dependency
repairs are preserved. Original science, R1 NO_GO, old scores and timing are unchanged.

## 2026-10-14: completed six-method shared-H800 timing

The six-method timing is fully closed: 576 original v5 completed calls were preserved and 1,152 missing complete calls were executed by the registered v6 recovery, for 1,728 complete calls, 864 measurements and 2,304 packet decodes. The separate actual CPU postprocess completed successfully with the original statistics source unchanged; full-precision tables, component summaries, inherited-call origins, receiver outcomes and English/Chinese reports are available. The historical v2/v3 initialization failures and v5 partial failure remain preserved. The failed entropy warmup is disclosed separately: 1,729 known fresh visual/endpoint attempts produced 1,728 complete calls.

The user-authorized resource revision lowered only GPU0 startup free-memory admission from 18 GiB to 8 GiB. The same H800 device, utilization ceiling, 16 GiB private allocator cap, 2 GiB running free-memory floor, 6/2 threads, six CPU slots, frozen numerical protocol and remaining 13,069-second scientific budget were retained. These are two execution segments under changing shared load. Reported times are shared-environment latency; resource boundary samples do not prove absence of interference or continuous peak bounds. No cross-hardware or HiFi speed ratio is claimed.

R0 remains complete. The recovered R1 calibration remains NO_GO_STOP, with no R2, additional scientific samples, training, selection changes or bootstrap replay. These calibration results are not a blind-test conclusion. Final timing and R1 scientific backup verification are separate remaining delivery steps.

[Completed six-method timing report](results/baseline_strength_20261011/timing/six_method_v6_delivery_v1/REPORT.md).

## 2026-10-14: completed R1 calibration; NO_GO_STOP

R0 screening is complete. The recovered R1 calibration is fully closed: 32,400 physical frames and 43,200 logical VAR frames on the original 200 calibration sources. The original successful PHY/projection segments were reused; the original initialization failure and its failed model-load reservation remain preserved and disclosed separately. The recovered visual, scoring and selection stages exited successfully, followed by the actual CPU closure audit and final figures with both PNGs opened for visual review.

The frozen DINO-J gate is NO_GO_STOP. For both U minus S_robust and U minus S_back3, the stored mean difference is -0.009211461585946379 (7 dB: -0.004988394297348875; 10 dB: -0.013434528874543883). These calibration results do not establish a blind-test conclusion. R2 remains unregistered and unrun; no extra scientific samples, training, bootstrap, threshold changes or policy reselection were added. Negative and zero results are retained.

The separate six-method shared-environment timing remains unfinished; its completed 576 calls are preserved. The user has now authorized relaxing its memory admission threshold, and that resource revision is being prepared while preserving the GPU0 device and numerical measurement protocol. Final scientific backup is a separate delivery step and is not claimed complete by this publication.

[R1 calibration delivery](results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1/REPORT.md).

## 2026-10-14: reviewed CPU closure entry for the recovered R1 calibration

An independent CPU-only closure entry and one-time dispatcher now cover the recovered R1 chain. They require the original PHY/projection actual exit 0, preserve the original owner/visual exit 1, and separately verify the recovery owner and three new stages have actually exited 0. The original auxiliary resource-history attachment is separately SHA bound; the failed audit is unchanged. Completed source-grid, failure-ledger and saved-selection audits retain the original numerical and CSV functions. No reconstruction, metric model, channel, bootstrap, policy selection or R2 is invoked. Both entries passed independent review and 36 synthetic tests in the original CPU environment. Their production execution remains gated on genuine scientific closure; this milestone does not report completed quality or GO. The recovery scientific owner continues independently.

[CPU delivery contract](experiments/layered-mismatch-raw-20261013/r1_cache_recovery_delivery_v1/README.md).

## 2026-10-14: diagnosed R1 initialization failure; reviewed recovery in progress

R1 PHY completed all 32,400 physical frames and the source-free reception projection completed 43,200 logical frames, both with actual exit 0. The first visual process exited 1 during Torch initialization because a temporary-file probe reached a cache outside its permitted receiver directory. No weights or reconstruction forwards completed in that failed attempt. Its failure, ledger and actual exit remain unchanged.

A separately reviewed recovery places temporary and compiler caches inside the receiver output. It inherits the completed PHY/projection without replay, and retains the original model, source isolation, math flags, noise, frame budget and absolute deadline. Twenty-six synthetic CPU tests and a real zero-model import/flags probe passed. The earlier probe rejected a module-discovery request; that engineering failure is retained and was not a channel invocation. The single registered recovery has actually started visual reconstruction, with scoring and frozen selection following automatically. R1 quality/GO and R2 are not complete or authorized by this entry milestone. The historical CPU delivery entry requires the original failed owner to exit successfully and must not be run on this recovery; a separate closure adapter is required.

[Recovery contract](experiments/layered-mismatch-raw-20261013/r1_cache_recovery_v1/README.md).

## 2026-10-14: guarded CPU delivery entry prepared for the running R1 calibration

A separate CPU-only delivery entry has been implemented and checked with synthetic fixtures. It requires the registered owner and all five stages to have actually exited successfully, then audits the existing frame grid, identities, execution ledger and frozen selection outputs. It does not invoke a model, channel, bootstrap or policy selection. Calibration means, channel-failure outcomes and the original GO decision remain unchanged. Large derived frame tables stay in the personal runtime directory; only lightweight summaries, reports and SHA manifests are publication candidates. This milestone publishes an entry, not completed R1 quality results or an R2 launch.

[CPU delivery contract](experiments/layered-mismatch-raw-20261013/r1_delivery_v1/README.md).

## 2026-10-14: clean R0 supplement closed; finite R1 calibration started

The approved clean supplement has actually completed 5,400 VAR reconstructions, 5,400 DINO-L reconstruction scores and 200 source embeddings. Both stages and their owner exited successfully. Together with the already completed 88,896 body-packet trials and reused historical evidence, the once-only read-only finalizer authenticated the complete R0 grid and froze the calibration nominations. The screening proxy remains an approximation, not an image-level bound or a GO result; historical 4090D evidence is not relabeled as a new H800 run.

At 2026-10-14 01:35:33 Asia/Shanghai (2026-10-13 17:35:33 UTC), the separately reviewed R1 resource entry was actually registered and its owner and PHY child had launched. Its frozen grid contains 32,400 physical frames and 43,200 logical VAR images within the original limits. This is a dated initial-launch snapshot, not completion of reconstruction, scoring or calibration selection. R1 uses only the original calibration200 and frozen noise/model/math/decision protocol. No training, additional qualification frame or bootstrap is introduced; R2 remains unregistered and requires the original real GO decision.

The quality-only memory policy is 15 GiB free at admission, 14 GiB free plus verified own reservation while running, a real 12 GiB allocator cap and a 2 GiB absolute free margin. The old R1 implementation is preserved, and both versions share one authoritative attempt namespace. FP32/TF32-off, six/two mathematical threads, source-free RX inputs, exact-event reuse and all scientific budgets remain. The original GPU0 six-method timing admission and priority are unchanged. CPU source/guard tests and earlier clean memory observations do not guarantee every new input will fit or prove bitwise numerical equivalence. Unknown failures stop without automatic retry.

The fixed `r1_execution_v1/registered/model_bundle.json` is 10,043,469 bytes, above the existing 10,000,000-byte repository limit. It remains unchanged on the server and is delivered with its SHA in the registered-R1 local increment; the publication filter is unchanged. The lightweight master registration retains its exact bundle binding.

Two registration-only failures consumed no scientific calls: cross-version Python AST serialization, then optional checkpoint-size metadata in otherwise identical source pins. Their original reviews, errors and partial registration bytes are preserved. The finite registration was completed after strict checks of source path/SHA and actual optional size, retaining its original absolute deadline. These metadata corrections do not change model, channel, source selection, noise or quality scoring.

[Authenticated R0 closure](results/layered_mismatch_raw_20261013/r0_nomination_v1/closure_audit.json), [new R1 entry](experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2/README.md), and [initial R1 evidence](results/layered_mismatch_raw_20261013/r1_resource_revision_v2/dispatch_snapshot.json).

## 2026-10-14: finite clean-quality supplement started under reviewed memory limits

Following the user's request to relax resource requirements, a separate quality-only entry now enforces 15 GiB free at admission, 14 GiB free plus verified current-process reservation while running, a 12 GiB PyTorch allocator cap, and a 2 GiB absolute running margin. The previous zero-science waiter was retired before this one-time owner started. The original 5,400 clean VAR/DINO pairs, 200 source embeddings, frozen models, source identities, FP32/TF32-off and 6/2 mathematical threads are unchanged. This does not modify the registered six-method timing gate or authorize R1/R2 early.

At 2026-10-14 00:08:23 Asia/Shanghai (2026-10-13 16:08:23 UTC), the registered owner and render child had actual launch records; no successful stage exit or complete clean grid was yet observed. This is a dated execution snapshot, not completion or numerical-qualification evidence. Historical allocator reservation is not a measured live-tensor allocation peak, and the smaller cap does not prove that every frame will fit or remain bitwise identical. Unknown failure stops without automatic retry. [Resource amendment](experiments/layered-mismatch-raw-20261013/RESOURCE_REVISION_V2.md) and [registration/evidence](results/layered_mismatch_raw_20261013/r0_clean_resource_revision_v2/README.md).

## Common500 entropy-coded quality and source analysis complete; timing unresolved (2026-10-13)

## 2026-10-13: all 88,896 supplemental packet trials completed

The separately approved R0 body-packet supplement actually exited successfully: 1,389 missing cells × 64 trials, 88,896 completed and zero unresolved records. A read-only audit authenticated the full ledger, recorded waveform/receive hashes, counters, runtime and existing intervals; it ran no new scientific call. The clean 5,400-pair VAR/DINO supplement remains resource-waiting and R1/R2 remain unregistered. These packet probabilities are an IID scrambled-payload pre-screen approximation, not image-level quality or a GO result. See [closed packet segment](results/layered_mismatch_raw_20261013/r0_bler_delivery_v1/README.md).


## 2026-10-13: finite layered-raw calibration entry prepared

The complete PHY → receiver projection → reconstruction → scoring → calibration-selection entry is now implemented and reviewed with CPU-only checks. R1 is not registered or launched: the actual R0 supplement must close before the shortlist is frozen. The original calibration200, frame limits, DINO objective and GO decision are unchanged; no training, added qualification frames or automatic development launch is introduced. Source truth is isolated from the reconstruction worker, and the existing timing task retains its registered GPU0 priority. See [entry preparation](results/layered_mismatch_raw_20261013/r1_execution_preparation_v1/README.md) and its dated execution snapshot.


## 2026-10-13: finite layered-raw R0 supplement launched

The user explicitly approved 88,896 additional body-packet trials and 5,400 clean VAR plus DINO-L source/endpoint scores. The original R1/R2 limits and decision rules are unchanged. Original evidence is reused. The CPU BLER owner has actually started; its immutable milestone snapshot records 6144 completed packet trials. The clean GPU registration is complete but its initial admission returned WAIT with zero model calls; this is not reconstruction completion. R1, R2 and the six-method timing remainder are not reported as complete. See [R0 supplement](experiments/layered-mismatch-raw-20261013/R0_SUPPLEMENT.md) and [execution evidence](results/layered_mismatch_raw_20261013/r0_supplement_preparation/execution_milestone_v1.json).


## 2026-10-13: EMAIN quality delivered; shared timing recovery registered

The original common500 entropy-coded partial-scale supplement is complete:
4500 logical frames, all four stages actually exit0, 12 new single-method and
60 direct paired intervals, and five PDF/SVG/600dpi PNG figure groups. All five
PNGs were actually opened and checked. Commit `d7579a02b3021846e27de85e498332b2f735f68c`
was normally pushed and the remote SHA verified. The independent quality
increment on Lu_Data contains 16,197 payload files and 1,084,785,621 archive
bytes; archive SHA256 `c5e13427e2c03e50d30b5392397408e243198dde0c474b4c00a2911f5c6f0abb`
and every payload SHA passed. It depends on the earlier model/data/environment
recovery packages and does not contain the remaining timing results.

The shared-H800 v5 timing attempt actually exited1 after completing 576 calls
(RAW_WHOLE and RAW_PARTIAL, 288 each; 288 measured calls and 1152 packet decodes).
Its first entropy-coded warmup completed fresh visual encoding but stopped
before entropy-provider construction because a closure read a later-rebound
Python namespace. The failed reservation and all 576 completed calls remain
immutable. This is an incomplete six-method timing experiment.

The separately reviewed v6 entry fixes that provider capture and loads the
existing frozen STATIC receive table required by the unchanged public 144/360
receive domains. It inherits the 576 completed rows without executing them and
permits only the remaining 1152 complete calls. The user's plan defines 1728
complete end-to-end calls and explicitly permits restoring truly missing work
after a failure. The repeated incomplete warmup is disclosed: success would
mean 1728 complete calls, 864 measurements and at most 2304 packet decodes, with
1729 known fresh-visual/endpoint attempts including the original failed one.
The cumulative four-hour envelope deducts 1331 prior seconds, leaving 13069.

v6 registration SHA256:
`54aafaca38b177266ef2c21cac23fce4dee2a0658e9bd9a0cfc5d2836baa269a`.
At registration the GPU had only 10197 MiB free versus the unchanged 18432 MiB
initial admission threshold: WAIT, no v6 launch or new scientific call. This
is a timestamped registration snapshot; later state must be read from actual
launch/ledger/child-wait receipts. v1/v4 remain unlaunched; v2/v3 construction
failures and v5's partial failure are preserved and must not be restarted.

The independent `timing_postprocess_v1` CPU entry passed 29 synthetic tests.
Its actual v5 partial audit passed, preserving 576 COMPLETE and one RESERVED
record without issuing a six-method performance table. Complete v6 delivery
requires real child exit0 and a verified joint grid, identities and resources.
All timing remains shared-environment latency, with raw wall/audit exclusions
and the two execution segments disclosed; no old-hardware speed ratio applies.
No E4 expansion, training, quality replay or bootstrap replay is authorized.


The frozen entropy-coded partial-scale extension has actually completed all 4,500 method frames on the original 500 sources at N1024 and SNR 4/10/19 dB, with actual exit0 from source, PHY, independent reconstruction and scoring. The independent read-only receiver audit verifies all frame/counter/input/output identities, 1,500 actual source decodes and reconstructions plus 3,000 exact received-input reuses, and zero unresolved reservations. All 4,500 realized frames parsed and independently decoded successfully; no failure row was removed. Existing strong references are reused without training, reselection or altered policies.

After averaging the three noises within each source, the original 10,000-replicate bootstrap (seed 2026100701) produced 12 new method intervals and 60 direct paired intervals; 72 historical intervals remain unchanged. Entropy partial-minus-whole PSNR gains are 0.616/0.705/0.337 dB at 4/10/19 dB, with improved LPIPS and DINOv2-L at all three points. The 19 dB prediction-agreement difference is -0.2 percentage points with 95% CI [-2.0, 1.6], so there is no uniform agreement gain. Relative to the raw partial implementation, PSNR improves 0.281/0.299/0.595 dB. PSNR remains lower than Swin200k and adaptive BPG at all three points, and lower than latent-space JSCC at 4/10 dB; LPIPS is worse than latent JSCC at 4 dB, and its 10 dB paired interval includes zero. These tradeoffs are retained, not used to retune the method.

Actual source lengths, fallback attempts and prefix distributions are complete. The requested target fails to fit for 99.6/99.6/95.6% of sources, while deterministic fallback finds a legal stream for every source; this is not a channel failure. Mean payload-capacity utilization is 93.08/95.21/94.63%. Whole-scale fallback occurs for 3/190/126 of 500 sources at the three SNRs. The five figure groups have actual vector PDF/editable SVG/600 dpi PNG exports. All five PNGs were opened and visually reviewed with PASS, and all 20 figure-output hashes were verified; [the separate visual-review receipt](results/entropy_main_common500_20261013/visual_review_v1.json) records this later check without altering the original export receipt. [Results and remaining work](results/entropy_main_common500_20261013/README.md), [full-precision paired table](results/entropy_main_common500_20261013/analysis_v1/paired.csv), [source-length table](results/entropy_main_common500_20261013/analysis_v1/length_summary.csv).

E3 remains unfinished. Its first shared-H800 v2 attempt actually exited1 during endpoint/model construction because of a Python module-import path collision, with zero timing cases and zero packet calls. No latency or speedup result is available; the failed scene is retained. The independent v3 recovery entry actually exited with code 1 during construction: the runtime incorrectly accessed `s.native.args.multiple_snr`, whereas the Swin field is `s.native.multiple_snr`. It still executed zero timing cases and zero packet calls. Both failure scenes and actual waits are retained. 

Registration snapshot recorded at 2026-10-13T09:59:07+00:00 (not a live process-status assertion): v4 was retired during review before launch because its copied module globals would bypass the shared-resource metadata hook. The independent v5 recovery registration SHA is `44b14186a878621ea15e18da8cbf50e185927e6390826d019cff0865afc666f6`; four model-free CPU tests passed. At this snapshot v5 was in final review and had not launched. Original v2/v3 failure scenes remain intact. No timing cases, measured frames or digital packet calls had been consumed from the original 1,728/864/2,304 limits; numerical qualification and latency results were still pending.
 The existing `var-comm` continuation monitor is ACTIVE for authorized verification and final delivery; the legacy 15-minute `var-raw64-holdout` monitor remains PAUSED. E4 and further training remain disabled. This is completed E1/E2 quality and source analysis, not completed E1–E3 delivery.

## User-selected paper reconstruction panels (2026-10-12)

The user chose candidates06,15,16,04: white shaggy dog,broccoli,pastries,shopping cart. Five-column comparison and four-column mechanism panels were rendered from verified historical development caches atN1024/13dB. All28 native display images remain pixel-identical, with fixed noise and latent40k/Swin80k labels. User selection after gallery review is explicitly disclosed; no scientific calls or change to quantitative results. PDF/SVG/600dpi PNG were visually reviewed. [Selected panels](paper/figures/user_selected_examples_20261012/README.md).

## Eight additional reconstruction candidates recovered (2026-10-12)

The original server became accessible. Verified and exported the remaining fixed16 sources33,41,52,60,64,87,92,95 atN1024/13dB using existing caches, preserving all seven image columns and fixed first noise. Four panels havePDF/SVG/600dpi PNG, actual visual review and a SHA-verified local/Lu_Data delivery. Total available candidates now16; no new science, final user selection or claim of final100k/200k holdout reconstruction. [Candidates09–16](paper/figures/additional_candidates_20261012/README.md).

## Reconstruction examples revised at user request (2026-10-12)

Retained second-row source21 and replaced the other three rows using the first three sources of the other verified display bundle: final order0,21,25,50. Both five-column comparison and four-column mechanism panels were rendered and visually checked. All seven images in row b are pixel-identical to the prior version. Existing N1024/13dB, fixed first noises and historical40k/80k development labels remain; no scientific calls. This is a user-requested illustrative selection, not a newly preregistered sample. [Revised panels](paper/figures/reselected_examples_20261012/README.md).

## figures4papers layout trial delivered (2026-10-12)

Applied the user-requested scientific-figure-making skill to the existing common500 main comparison and fixed development photo panels. Seven layouts have vector-curve PDF, editable SVG and600dpi PNG, with actual visual review; data CSV bytes and all28 source display PNGs match the prior delivery. No scientific calls or image enhancement. Original figures remain unchanged. Historical qualitative40k/80k labels remain separate from final100k/200k performance curves. [Trial figures and reproduction](paper/figures/figures4papers_trial_20261012/README.md).

## Main-paper performance and qualitative comparison figures (2026-10-12)

A new four-method common500 figure combines the proposed raw partial-scale method, selected latent100k/Swin200k baselines and adaptive-resolution BPG + LDPC. All96 mean/interval rows are read directly from published CSVs; no new scientific or bootstrap calls. Five layouts (one composite and four single panels) have PDF/SVG/600dpi PNG exports and actual visual review. [Figure and English caption](paper/figures/final_paper_comparisons_20261012/README.md).

The accompanying existing HiFi five-column and digital four-column panels are explicitly development examples at N1024/13dB. Their latent40k/Swin80k versions must not be represented as the final100k/200k models. Same sources and fixed first noises are retained; no best-noise selection or substitute SNR. Fair H800 timing remains pending and no model was launched for this delivery.

## Redundant storage cleaned at user request (2026-10-12)

Deleted437 VAR_COMM-owned redundant files totaling10,526,892,735 logical bytes:116 superseded recovery snapshots,9 SHA-matched local-backed archives and312 synthetic test temporary files. Final complete P100k/Swin200k states, every calibration model weight, original inputs, scientific results, failure records and timing dependencies remain. All timing bindings and final recovery hashes passed after deletion. Retired intermediate optimizer/RNG snapshots are not individually recoverable; final complete migration states remain both remotely and on Lu_Data. No scientific job was launched. [Actual cleanup inventory](results/baseline_strength_20261011/storage_cleanup_20261012/README.md).

## Six-method timing entry registered; actual timing awaits exclusive resources (2026-10-12)

Training, formal common500/new100 scoring, paired statistics and visually checked figures are already delivered at commit3ccfe6e. Their completed scientific calls must not be repeated. The independent six-method timing owner is now implemented and SHA-registered, with10 passing no-model CPU tests. The actual entry returned WAIT before any model or packet call: all four GPUs still have resident jobs. Full endpoint numerical qualification and timing values remain pending; no H800 timing comparison is claimed.

The finite owner retains fixed development16, SNR4/10/19, batch1, three fresh warmups and three measured repetitions, six methods,1,728 total frames/864 measured frames and2,304 maximum digital packet decodes. It uses fresh visual encoding, real terminated arithmetic streams including every fallback attempt, original paid received fields and failure rules, and the selected P100k/Swin200k weights. Raw/entropy noise is6201 with their distinct original namespaces/counters; continuous P/Swin noise is2001 from the original timing receipt. No cached source stream or transmitter truth enters RX.

The original all-GPU/process exclusive gate remains unchanged. Only new process resource adapters were added; inverse-AST evidence, exact private dependency and native library bindings are retained. The first real fixed warmups must still pass fresh-token and repeated waveform/receiver/output identity checks. Any actual failed attempt stops without automatic replay. See[entry and boundaries](experiments/baseline-delivery-20261012/README.md) and[registered timing plan](results/baseline_strength_20261011/timing/six_method_v1/registration.json).

## Formal common500/new100 baseline supplement delivered (2026-10-12)

Both selected baselines completed all 19,800 formal frames: original common500 (18,000) and existing new100 (1,800), with actual exit-zero receipts. Source/model/noise/metric identities, complete grids, duplicates and failure denominators passed. The original source-level bootstrap produced only the registered new intervals and contrasts; all old intervals were copied unchanged. Four final comparison figure groups have vector PDF, editable SVG and 600 dpi PNG, and were actually opened for visual inspection. No further training, reselection or scientific budget was added.

[Formal report and paired contrasts](results/baseline_strength_20261011/formal_delivery_v1/REPORT.md), [common500 figure](results/baseline_strength_20261011/final_comparison_figures_v1/common500_new_old_baselines.svg), [new100 figure](results/baseline_strength_20261011/final_comparison_figures_v1/new100_new_old_baselines.svg). These are subsequent comparisons on existing cohorts, not newly blind tests. The old Swin80k/19dB range limitation and new Swin's changed training recipe/budget-limited status remain explicit.

Same-machine timing is still unfinished: pure frozen LDPC dependencies now import in the visual environment, but the six-method endpoint still needs qualification and the original exclusive resource window. Three no-science tests cover timing crash replay, finite packet charging and rejection of cached source work. Import success is not timing qualification. The prior 4090D timings are not mixed into an H800 comparison.

## Full milestone and bottleneck diagnostics delivered; common500 running (2026-10-12)

Milestones closed with actual exit0 and all96,000 logical frames (91,800 new plus4,200 verified reuses). The separate P bottleneck diagnostic closed with actual exit0 and900 frames (300 new plus600 verified reuses). Exact grids, source identities, hashes, failures and duplicate counts were checked. Three diagnostic figure groups have vector PDF, editable SVG, 600 dpi PNG and actual visual review.

On the full1,000-source calibration diagnostics, selected P100k minus parent40k improves PSNR by0.2300--0.3664dB; selected Swin200k minus parent80k improves it by0.6131--1.0876dB across six SNR values. All four parent-relative means improve, but late checkpoint DINO/ConvNeXt declines remain in the curves. No bootstrap or holdout-based selection is involved. Swin training changes and old80k/19dB range limits remain explicit.

The existing owner has automatically started common500 scoring and will continue to existing-new100. A separate once-only CPU successor waits for full scoring closure before new source-paired statistics and final figure export. Formal holdout quality and fair exclusive-window timing remain pending.

[Closed diagnostic report and tables](results/baseline_strength_20261011/diagnostic_delivery_v1/REPORT.md), [P milestones](results/baseline_strength_20261011/diagnostic_delivery_v1/figures/P_milestone_metrics.svg), [Swin milestones](results/baseline_strength_20261011/diagnostic_delivery_v1/figures/Swin_milestone_metrics.svg).

## Learning curves delivered; statistics successor waiting (2026-10-12)

The completed training histories now have actual vector PDF, editable SVG and 600 dpi PNG figures, with visual review. All60,000 P and120,000 Swin additional updates are represented; calibration objectives, per-SNR cells and learning-rate changes are retained. Full final training recovery states (P100000 and Swin200000) and raw logs have been copied to the local Lu_Data final_training_20261012_v1 increment and checked by SHA256. The increment depends on existing model/data/runtime archives.

The original GPU scoring chain continues unchanged. A separate once-only CPU successor is running and waiting for normal scoring closure, then calculate only new source-paired statistics and export diagnostic/final figures. Old intervals are reused, not resampled. Preparation and plots are not completed holdout evaluation. Final statistical/figure visual review, exclusive-window fair timing and their publication/backup remain outstanding.

[Delivery entry and reproduction](experiments/baseline-delivery-20261012/README.md), [learning curves](results/baseline_strength_20261011/training_figures_v1/training_calibration_curves.svg).

## Training closed; automatic diagnostic and holdout successor launched (2026-10-12)

Both actual training owners exited0. P selected total100000 and satisfies the registered plateau condition. Swin selected total200000 after120000 additional updates and stopped at its budget while still improving; it is not labeled converged. The frozen calibration objectives are0.03941082926963766 (P) and0.006566026536579253 (Swin), with different objective definitions and no direct cross-method loss comparison.

At the user's request, a single detached successor now runs milestone diagnostics, P bottleneck diagnostics, common500 and existing-new100 in that order. Complete original500 source/latent bytes were recovered and hash-verified from the local backup. The final evaluation budget remains19800 new method frames. Successors require actual exit-zero, complete output and identity checks; failures stop without retry. This publication records launch, not finished diagnostic or holdout quality. New statistical intervals and exclusive-window timing remain pending. The old15-minute monitor remains paused.

[Automatic execution and scope](experiments/baseline-followthrough-20261012/README.md), [launch report](results/baseline_strength_20261011/automatic_followthrough_v1/REPORT.md).

## Interim baseline evaluation completed while training continues (2026-10-11)

An independent GPU0 job completed7200 method frames: original calibration first100, N1024, six SNRs and three noises, comparing fixed P40k/P80k and Swin80k/Swin150k snapshots. Actual exit0 and all4200 overlapping original-calibration outcomes were verified. P PSNR increases0.1965–0.2801dB; Swin increases0.4161–0.9031dB. P improves all four displayed means at all six SNRs; Swin agreement decreases0.67pp at7dB and0.33pp at19dB, retained in the report. These descriptive calibration results have no bootstrap intervals and do not enter selection or stopping. Swin changes include the approved optimizer reset and expanded SNR coverage. Original80k/19dB remains out of range.

Both training jobs continued; final model selection, full milestone diagnostics and final holdout evaluation remain pending. [Results and complete tables](results/baseline_strength_20261011/interim_cal100_v1/REPORT.md).

## Baseline-strength continuation started (2026-10-11)

The user authorized P64/3 continuation and a Swin1–19dB adapted extension. R0 restored and hash-verified the original20k/1k RGB and latent caches (421 files, no dataset download). P restores complete40k model/AdamW/order/RNG state. The available Swin80k file is weights-only; the user explicitly authorized fresh Adam/order/RNG initialization, with historical80k exposure retained and the difference disclosed. Both actual H800 engineering preflights passed; the failed Swin microbatch16 allocator attempt is retained, and microbatch4 preserves effective16 within the private16GiB limit. The independent GPU0/P and GPU2/Swin owners start with full original-source calibration; dispatch and calibration are not additional completed updates.

Finite training/calibration/selection is running or preparing its initial calibration. No new final quality result, convergence claim, capacity experiment, holdout selection or timing completion is made. Milestone diagnostics and common500/new100 evaluation remain pending after selection. Historical digital/old-baseline results and paused monitors remain unchanged. New storage follows the personal code/checkpoint/tmp classification.

[Protocol](experiments/baseline-strength-20261011/PROTOCOL.md), [R0 inventory and amendments](results/baseline_strength_20261011/R0_inventory/R0_inventory.md).

## Independent new100 four-arm quality confirmation completed (2026-10-11)

All authorized quality stages for the independently fixed100 sources are now actually closed: four frozen methods at N1024, SNR4/10/19dB and three noises produce3,600 retained logical outcomes. The H800 GPU0 reconstruction owner completed1,151 new renders and538 independent entropy decodes;2,449 logical outcomes reuse identical actual inputs. The separately closed metric owner constructed three metric models, prepared100 references and scored849 unique pixel pairs, with2,751 same-run reuses. Every logical receive failure remains attached to its own row. RAW partial at10dB has13/300 body-CRC rejections, all retaining the actual hard tokens under the original KEEP rule; no fixed-gray result is substituted.

The new entropy partial-minus-whole PSNR gains are0.531dB [0.421,0.656] at4dB,0.641dB [0.503,0.785] at10dB and0.343dB [0.281,0.409] at19dB. LPIPS and DINOv2-L also improve at all three points with pointwise95% paired intervals on the improving side of zero. ConvNeXt source-prediction agreement is not uniformly better: its4dB interval includes zero, and the19dB paired vector is exactly zero. The raw within-representation PSNR gains are0.345/1.231/1.392dB at4/10/19dB; the full precision tables retain all four metrics and their intervals, including null or unfavorable effects. These are within-representation comparisons; packet formats and failure rules differ between raw and entropy, so the four arms are not a pure two-factor ablation. Entropy coding includes sender-side VAR work; raw remains the low-sender-compute implementation.

Statistics first average the three noises separately per source, then form the two registered partial-minus-whole contrasts. One unchanged published bootstrap draw matrix (10,000 replicates, seed2026100701) serves48 summaries and24 paired rows. There were46 actual marginal-interval and23 actual paired-interval calls; exact-vector reuse and exact zeros remain explicitly recorded. Confidence intervals are pointwise, with no multiplicity adjustment. The initial statistics preparation stopped before registration or any draw because it compared archive paths rather than the already authenticated same-source pixel identity. Its five original failure artifacts are preserved; a separate metadata-only admission repair completed without changing scores, statistical mathematics or the bootstrap budget.

The failed GPU3 startup remains visible as one spent, unresolved model-load reservation with zero receiver/render calls. The separately authorized GPU0 recovery completed one additional model load; the resource guard and scientific frame caps were unchanged. A read-only post-decode audit found exact token/CDF agreement for all1,800 realized accepted entropy frames, covering538 distinct actual decoder inputs. This verifies this realized sample; it does not establish universal cross-device arithmetic-coding equivalence, and no truth/CDF oracle was supplied to the receiver.

Quality and confirmation statistics are complete. Unified sender/receiver timing is still pending: no new dedicated entropy-partial timing owner has executed and the current shared-server occupancy does not meet the existing exclusive timing protocol. Older RTX4090D timing remains historical evidence, not a new H800 timing comparison. Original frozen policies, source membership, main500 results, prior failures and all scientific byte streams remain unchanged.

[Full precision summary](results/entropy_partial_new100_20261011/summary.csv), [paired differences](results/entropy_partial_new100_20261011/paired.csv), and [statistical report](results/entropy_partial_new100_20261011/REPORT.md).

## Prior publication snapshot: independent new100 source and channel stages (2026-10-11)

The independent 100-source encoder/TX owner actually exited 0 after 222.77 seconds. Its closed ledger records one model load, 100 frozen encoder calls, 100 fresh entropy source TX calls and 1,000 prior-scale calls, with zero unresolved calls. All 2,928 closed artifacts, including 200 source arrays and 1,201 actual reservation/completion pairs, were independently verified and preserved in a 20.65 MB exact archive. The source stage performs no receiver decode, reconstruction, quality score or bootstrap.

The four-arm N1024 channel owner has now actually completed all 3,600 logical frames: 100 fixed independent sources, SNRs 4/10/19 dB, three noises and four methods. All 3,600 physical observations and 7,200 actual packet decodes are accounted for with zero unresolved calls. The independent read-only audit verifies every paid packet result against the immutable ledger and retains all logical receive outcomes. It introduces no new noise simulation or decode call. RAW433, original entropy WHOLE144 and allowed-partial entropy360 catalogues remain separate, with their original frozen packet and failure rules. The portable T1 binding verifies the old 241-call qualification; it does not assert a new 241-call run.

The closed failure inventory retains 13 body-CRC rejections among the 300 raw partial-scale frames at 10 dB; those frames keep their actual hard tokens under the frozen receiver rule. All other body-CRC rejection counts, all header rejections and all received-profile mismatches are zero. The 1,800 entropy frames have parsed link payloads, but independent entropy source decoding has not yet been claimed.

These are completed source and channel stages. At this publication freeze no new100 reconstruction owner or quality evaluation is claimed complete; the next GPU stage is waiting for the existing shared-resource admission. The four policies remain frozen from the completed original 1000 calibration and no new100 result is used to choose them. Link payload parsing is not image reconstruction or final confirmation evidence. RAW body-CRC rejection keeps actual hard tokens, so it must not be relabeled a fixed-gray failure. Recorded-study exclusion, fixed 100 membership, common noise variates, finite budgets and all prior failed-attempt evidence remain intact.

[Closed new100 source stage](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_new100_source_v1.json) and [closed four-arm channel stage](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_new100_phy_v1.json).

## Prior publication snapshot: full1000 calibration and mismatch statistics (2026-10-10)

The complete original1000 four-metric calibration owner actually exited 0 after 937.03 seconds. All 36,000 logical rows preserve the original 1,000 sources, three noises, SNRs 4/10/19 dB and four admitted finalists per SNR. Its closed ledger records three metric models, 900 newly prepared references and 5,111 new source/reconstruction score pairs; 3,600 logical rows reuse exact closed-pilot pairs and 27,289 reuse pairs already scored in this run. The independent archive audit verifies all 71,995 files, 33,369 reservation/completion pairs, every logical row's actual receive outcome and reconstruction identity, and the unchanged full1000 winner computation. No score, image or bootstrap was recomputed by that audit.

The allowed-partial entropy configurations are now frozen from calibration alone: at 4 dB, m7_K25_q2_r2-3; at 10 dB, m8_K126_q4_r2-3; at 19 dB, m9_K192_q6_r5-6. Each full-calibration candidate set retained the original final whole-scale winner. Selection maximizes DINOv2-L after averaging three noises within each source and then all 1,000 sources. These are calibration results, not a new100 confirmation result; raw and entropy interfaces retain their different packet formats and failure handling.

The one-bin raw configuration-mismatch metric owner on leo GPU5 actually exited 0 after 627.41 seconds. All 5,400 logical rows were scored using three metric models, 100 source references and 2,097 new pairs, with 3,303 exact same-run pair reuses and zero unresolved calls. Its independent audit verifies all 27,895 files and 12,885 reservation/completion pairs. Every logical header/body/CRC outcome is retained separately from image caching. No historical metric value is reused and no diagnostic result changes a policy or protection threshold.

The separate source-paired statistics owner also actually closed: one original 10,000-replicate draw matrix (seed 2026100701), 48 nonmatched pointwise 95% intervals and 24 exact-zero matched controls. Independent read-only verification matches all 7,200 three-noise source means, all 7,200 source differences and all 72 means to the measured rows; every published interval matches its original completed-call receipt. This publication does not rerun bootstrap. Deltas remain mismatch minus matched at the same true SNR, LPIPS signs are unchanged, and ConvNeXt agreement deltas and intervals are converted to percentage points together. These are post-hoc diagnostics on the original common500 first100, not unseen confirmation evidence or a claim that one-bin errors are typical in deployment. The receiver still uses true noise variance. Raw body-CRC rejection keeps actual hard tokens and is not counted as a fixed-gray output.

After the full1000 policy closure, the independently owned new100 metadata selection, official-byte download and CPU content gate completed. The content owner actually exited 0 in 22.924 seconds: all 100 canonical and horizontal-flip hashes passed exact exclusion against the recorded-study registry of 24,959 sources, with no conflict or replacement. At this publication freeze the separately registered source encoder/TX owner had started and was still awaiting its terminal receipt; no new100 PHY, reconstruction, quality result or final confirmation is claimed. The newly uploaded content/source implementation is included to preserve repository completeness and exact execution identities. The 20,337 historically unavailable original-file byte hashes remain explicitly unknown; canonical exclusion does not assert those byte identities. Existing failed attempts, frozen protocols, finite budgets and paused periodic monitors are preserved.

[Full1000 metrics](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_metrics_v2.json), [exact frozen calibration winners](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_calibration_winners_v2.json), [raw mismatch metrics](experiments/entropy-partial-mismatch-20261010/reports/leo_mismatch_metrics_v4.json), and [source-paired mismatch statistics](experiments/entropy-partial-mismatch-20261010/reports/leo_mismatch_statistics_v2.json).

## Prior publication snapshot: full-calibration and raw-mismatch reconstructions (2026-10-10)

The original1000 full-calibration reconstruction owner actually exited 0 after 3,592.54 seconds. Its complete 36,000-row grid retains all original sources, three noises, three SNRs and four admitted finalists per SNR. The closed ledger records one model load, 5,110 new independent source decodes, 5,110 VAR renders, 5,110 frozen Dc calls and 91,622 prior scales, with zero unresolved calls. Of the logical rows, 3,603 reuse exact closed-pilot reconstructions and 27,287 reuse an identical actual input from this full run. All new float32 images, every logical-to-physical-to-image reference, the closed pilot/48 references and the model-call ledger were independently audited and archived.

The one-bin raw configuration-mismatch reconstruction owner on leo GPU5 actually exited 0 after 1,441.81 seconds. Its 5,400 logical rows reference the already closed 4,500 physical observations and 2,097 new exact float32 reconstructions. One model load, 2,097 VAR renders, 2,097 frozen Dc calls and 20,970 prior scales completed with zero unresolved reservations. The independent archive audit checks 57,855 members and 25,165 reservation/completion pairs, reparses actual received raw12 hard tokens, and preserves CRC-rejected tokens under the original KEEP rule. The earlier GPU0 model-load failure remains unresolved in its separate attempt; the two visual attempts have two model-load reservations in total, one of which completed. No historical image reuse or new PHY call was used by the leo visual stage.

The first leo metric attempt (v3) failed during preparation after 12.35 seconds, before any model or score call. Eighteen actual native library files were rehashed; 11 old records differed only because they retained an observed lexical-path field despite identical canonical path, size and SHA. Its four original failure receipts are preserved. The separately registered v4 engineering repair preserves lexical provenance and all original PFS file identities and rejects any actual binary difference; original metric computation functions and budgets remain unchanged.

During preparation of this publication, both metric owners actually exited 0: full1000 after 937.03 seconds and corrected leo mismatch v4 after 627.41 seconds. The full1000 complete metric grid and unchanged winner recomputation were separately checked by the run owner; their detailed metric archives and public results are a separate milestone, not part of these reconstruction reports. The separately registered mismatch statistics owner has started, but no interval result is claimed here. Its unchanged statistical implementation is included only because the repository requires uploaded self-written sources to be tracked. The new100 confirmation population is still unselected and unread and must pass the independent content gate. No new timing measurement or training is completed by this publication. The recorded-study source exclusion closure and its original-byte-hash limitations remain unchanged; historical periodic monitors remain paused.

[Closed full1000 reconstructions](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_visual_v2.json), [closed leo raw mismatch reconstructions](experiments/entropy-partial-mismatch-20261010/reports/leo_mismatch_visual_v3.json), and [preserved metric preparation failure](experiments/entropy-partial-mismatch-20261010/reports/leo_mismatch_metric_prepare_failure_v3.json).

## Prior publication snapshot: source1000 and CPU channel closures (2026-10-10)

All original1000 calibration source streams are available. The separate source900 owner exited 0 after 543.45 seconds, closing one model load, 900 fresh TX passes and 9,000 prior scales. Its 21,600 endpoint arrays were independently audited and archived; prior source100 calls and streams remain separately recorded. This source-only stage did not perform independent RX, reconstruction or scoring.

The full-calibration CPU owner exited 0 after 828.95 seconds. It completed the entire original1000-source, three-noise, three-SNR and four-finalist grid: 36,000 logical frames. There were 16,447 new physical observations and 32,894 actual packet decodes, with no unresolved calls. Forty-eight logical rows reference the closed image-link gate, 1,152 reference the closed pilot, and 18,353 reuse an identical observation from this full run. All 52,464 archived members, logical references, original1000 counters and actual decoder results were independently checked. Full-calibration reconstruction, metrics and final policy selection are not complete.

The separate one-bin raw configuration-mismatch CPU owner exited 0 after 748.43 seconds. Its 5,400 logical rows comprise 4,500 actual physical frames and 9,000 new packet decodes; 900 logical rows share an identical current observation. No historical event was reused. True SNR remains the demodulation/noise input and the original common500 source ordering and 500-stride counter are preserved. All 4,525 archive members, paid decoder events and the conversion of actual 12-bit payloads to hard tokens were independently verified. Body CRC rejection retains those tokens under the frozen KEEP rule; it is not replaced by a gray reconstruction or a software-error fallback.

The GPU0 mismatch reconstruction attempt exited 1 after 107.98 seconds at its GPU resource-margin check. The ledger contains one model-load reservation, zero completed model loads and one unresolved reservation; all source, prior, rendering, decoder, PHY and metric calls remain zero. The failed attempt and its actual waits are retained. No successful mismatch reconstruction or quality result is claimed. Any corrected engineering admission must use a separate registration; frozen science and the failed record remain unchanged.

The recorded-study source exclusion scope is now formally closed after independent review of 72 metadata artifacts and 16 registered source roles. The exclusion registry contains 24,959 source IDs with zero canonical pixel-hash gaps. It still records 20,337 missing original-file byte hashes; this limitation is retained. Historical catalog mentions are conservative exclusions, not proof of actual pixel exposure. The 337 officially recovered images are not claimed to have the original historical JPEG-byte identity. Undocumented manual uses, all pretrained-model training sets, and full-resolution or perceptual duplicate exclusion are outside this closure.

New100 confirmation remains unselected and has not been inspected. It still waits for actual full1000 strategy freezing. The new leo GPU5 mismatch attempt and full1000 GPU resource-admission revisions are local preparations at this publication snapshot; neither is reported as a successful reconstruction. The restored fixed16 timing assets are ready, but no new timing experiment has run. This publication performs no training, model execution, channel experiment or bootstrap. Historical periodic monitors remain paused.

[Closed source1000](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_source1000_v1.json), [closed full-calibration PHY](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_full_phy_v1.json), [closed raw mismatch PHY](experiments/entropy-partial-mismatch-20261010/reports/h800_mismatch_phy_v1.json), [GPU0 attempt failure](experiments/entropy-partial-mismatch-20261010/reports/h800_mismatch_visual_failure_v2.json), and [recorded-source scope closure](experiments/entropy-partial-mismatch-20261010/reports/h800_recorded_source_scope_v1.json).

## Prior publication snapshot: original100 pilot and shortlist (2026-10-10)

The original-calibration pilot has actually completed reconstruction and all four metrics for all 43,200 logical rows: 100 fixed sources, 144 frozen candidates, SNRs 4/10/19 dB and noise seed 4101. The reconstruction owner exited 0 after 1,019.01 seconds. Its separate ledger closes one model load, 1,408 source decodes, 1,408 VAR reconstructions, 1,408 frozen Dc calls and 25,142 prior scales. Exactly 1,409 new float32 image archives include the constant 0.5 failure image; 453 logical rows reuse closed48 images and 41,338 reuse an image from this pilot. All 103,369 archive members, every logical-to-physical-to-image reference and all 29,367 pairs of model-call receipts were independently verified. Failed receives remain in the population.

The four-metric owner exited 0 after 305.38 seconds. It constructed three metric models and prepared 100 references, then actually scored 1,530 unique source/reconstruction pairs; 41,670 logical rows reuse an identical pair under the same metric provider. Its 9,483 call reservations all completed, including 3,060 actual LPIPS backbone forwards. Every pair binds both source and reconstruction pixel hashes. PSNR, LPIPS-Alex, DINOv2 ViT-L14 cosine and ConvNeXt source-prediction agreement are retained for all 432 candidate-by-SNR means; agreement is not classification accuracy. No new bootstrap or holdout evaluation was performed by this pilot.

The complete-pilot DINOv2-L ranking admits the following top three candidates plus the original full-calibration whole winner at each SNR. Candidate-ID lexical order resolves exact ties. These are a shortlist for the original1000 full calibration, not its final winners:

| SNR (dB) | Complete-pilot top 3 | Retained original whole winner |
| --- | --- | --- |
| 4 | `m7_K25_q2_r2-3`, `m7_K50_q2_r2-3`, `m7_K75_q2_r2-3` | `m7_q2_r2-3` |
| 10 | `m8_K42_q4_r2-3`, `m8_K126_q4_r2-3`, `m8_K84_q4_r2-3` | `m9_q4_r2-3` |
| 19 | `m9_K192_q6_r5-6`, `m9_K128_q6_r5-6`, `m9_K64_q6_r5-6` | `m9_q6_r5-6` |

The source900 owner has now actually exited 0 after 543.45 seconds, closing one model construction, 900 fresh TX passes and 9,000 prior scales with no unresolved reservations. Combined with the previously closed source100, all original1000 source streams are available. This TX-only stage made no independent RX, reconstruction or quality calls, so it does not establish full-calibration quality. Its request SHA is `39bc1c524450ecd981a5c2dafdeb5c898bde06b2723e24f08da3efdf711354ca`.

At the checked publication snapshot, the separately registered original1000 PHY owner is actually RUNNING on CPU cores 0/1, request `81ae17e2dea3569424323d919cd84a089ed967a970a7a4cbc189bd8e48b5f2a5`. It evaluates the four admitted candidates per SNR, original1000 sources and seeds 4101/4102/4103, with at most 36,000 logical frames and 72,000 new decoder calls before exact-observation reuse. The one-step raw mismatch PHY owner is also actually RUNNING on CPU cores 2/3, request `9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf`. It retains the original common500 first100 sources, true-SNR noise and 500-stride counter, and admits 5,400 logical/4,500 physical frames with at most 9,000 new packet calls and no historical-event reuse. It uses the unchanged raw433 receiver domain; the entropy457 qualification is not presented as a raw433 qualification. Both workers keep CUDA hidden and have separate finite ledgers. These are running snapshots, not completed PHY, reconstruction or metric results.

The full1000 reconstruction and metric owners remain prepared but unexecuted at this snapshot. Final strategy freezing requires the complete original1000 population and all three registered noises. New100 confirmation remains pending: historical source-exposure/hash closure is still open, and no new confirmation pixels have been selected. No new bootstrap was performed.

The closed CPU pilot remains unchanged at 17,201 new physical observations and 34,402 actual LDPC decoder calls. Logical reuse never adds historical calls to the new-call ledger. Frozen protocol, models, actual received-input boundaries and whole-winner retention remain unchanged.

[Closed reconstruction receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_visual_v1.json), [closed four-metric receipt and candidate means](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_metrics_v1.json), [exact pilot finalists](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_finalists_v1.json), and [closed CPU pilot receipt](experiments/entropy-partial-mismatch-20261010/reports/h800_ep_pilot_phy_v1.json).

## H800 actual48 image link gate completed (2026-10-10)

The fixed first4 original calibration sources at4/10/19 dB, seed4101 and four registered targets per point completed all48 actual paid-channel and reconstruction frames. The PHY ledger closed96/96 decodes; the visual ledger closed model1, source RX48, render48, Dc48 and prior-scale866 within the960 cap, with no unresolved calls. Thirty reconstructions used a received positive-K profile and18 used a whole-scale profile; no fixed-gray output occurred in this small gate. All48 exported float32 image arrays and their frame bindings were independently verified. The CPU and GPU owners exited0 after29.18s and183.39s; these are engineering owner durations, not online timing measurements.

This closes the source, physical-link and image-link engineering prerequisites. It does not establish comparative quality. Pilot ranking, full calibration, new100 confirmation and mismatch evaluation remain **NOT_RUN** at this snapshot. Continue with the original100-source,144-candidate calibration pilot after its exact missing assets and finite execution registration are complete. Reuse only individually verified source/event states, and preserve every prior gate call. Historical source-exposure auditing remains open; no new100 pixels have been selected or inspected.

[Closed image gate report](experiments/entropy-partial-mismatch-20261010/reports/h800_ep48_link_gate_v1.json).

## H800 first32 source and real PHY gates completed (2026-10-10)

Two actual component gates are closed. The original calibration first32 source gate completed 32 fresh H800 TX passes, 576 independent partial RX decodes and 64 m4/m5 whole RX decodes; all 640 decoded-token comparisons and independently recomputed CDF witnesses passed. The exact budget closed at model load1, TX32, RX640 and prior-scale4928, with encoder/render/Dc0 and no unresolved calls. The actual owner exited0 after419.63s. Fresh whole m6-m9 streams were generated but their whole RX endpoints were outside this gate. The separate real PHY gate passed all96 body and361 header decodes (457/457, unresolved0). Its first attempt stopped before any decode because the visual environment lacks Sionna; the successful fresh attempt used the already-restored frozen LDPC environment and unchanged backend identity checks. Earlier cross-host failures remain unchanged.

These are component qualifications, not quality results. At this publication snapshot the 48-frame image link gate is pending; formal entropy-partial calibration, new100 confirmation and mismatch evaluation remain **NOT_RUN**. Proceed to the fixed48 image link gate using exact first4 new-host streams and the complete paid-profile catalogue before calibration. Historical exposure auditing remains open, and no new100 source selection is admitted. Historical periodic monitors stay paused.

[Closed source gate](experiments/entropy-partial-mismatch-20261010/reports/h800_ep32_source_gate_v1.json) and [closed PHY gate](experiments/entropy-partial-mismatch-20261010/reports/h800_ep457_phy_v1.json).

## H800 migration diagnostics (2026-10-10)

Earlier single-source H800 diagnostic outcome: **PASS_H800_SOURCE0_M4_SAME_HOST_CODEC_ONLY**. The shared PFS runtime and frozen model bytes were reused; this host's native libraries were separately observed. The earlier H800 cross-host RX loaded the models, then rejected the saved leo TX at the first exact CDF comparison (model load1 and prior-scale1 completed; one RX reservation remains unresolved). The separately registered same-host check completed one actual H800 source TX and independent m4 RX: the 30 decoded tokens and all four independently computed RX CDF hashes match its fresh H800 TX witnesses. The actual budget closed at model load1, TX1, RX1 and prior-scale13, with encoder/render/Dc0. This is one existing source and m4 only. It does not establish old-host bitstream compatibility, full source/PHY/reconstruction qualification or a timing result. Hardware and native-library effects are not uniquely separated. The v5 bitstream failure, leo SIGKILL outcome and H800 cross-host CDF failure remain unchanged. **Formal entropy-partial calibration, new100 confirmation and mismatch evaluation remain NOT_RUN.** See the [H800 cross-host report](experiments/entropy-partial-mismatch-20261010/reports/h800_single_rx_v1.json) and [same-host codec report](experiments/entropy-partial-mismatch-20261010/reports/h800_same_host_codec_v1.json).

## Entropy-partial and mismatch extension: engineering preparation (2026-10-10)

[Current migration and resume status](experiments/entropy-partial-mismatch-20261010/reports/leo_shared_migration_v4.md): the worktree and frozen private environments are restored. The v5 GPU check matched680 encoder tokens but failed old/new m4 bitstream equality; that v5 attempt ran no independent RX or reconstruction. Formal entropy-partial calibration, new100 confirmation and one-step mismatch evaluation remain **NOT_RUN**. The existing264 CPU regressions,118 extension checks and9 additional native-admission tests passed. Published WCL T0–T6 results remain unchanged and historical monitors stay paused.

## Completed WCL supplementary evidence (2026-10-10 publication)

[T0–T6 final report, results and figures](results/wcl_evidence_closure_20261009/README.md) are complete. The finite expanded calibration retains the original N1024 whole-scale winners. Same-prior VAR entropy coding improves N1024/19 dB quality at greater TX cost. The separate N2048 confirmation completed 2,700 method frames: partial-scale PSNR increments over raw whole-scale are +1.220451, +0.059652 and exactly zero dB at 4/10/19 dB. The report retains uncertainty, negative results and the different 500/100-source populations.

Original scientific results and frozen protocols are preserved. This publication follows the user's explicit push request; it runs no new scientific experiment. Restoration transfer status is tracked separately from Git publication, and historical monitors remain paused.

## Completed Kodak24 generalization experiment (2026-10-09)

[Results and delivery index](results/generalization_kodak_20261009/README.md): all 24 predetermined 256x256 center crops, N1024, 4/10/19 dB, three noises, and four frozen methods completed 864 reconstruction frames. Unconditional VAR uses the null embedding; there was no training or Kodak calibration/policy selection. Four metrics, source-paired intervals, resource/configuration tables, and fixed-source visualizations are complete. The independent 1080-call PHY budget closed; the original root ledger remains 534231.

DINOv2-L similarity and source-prediction agreement favor the proposed method against every baseline at all three SNRs in pointwise paired intervals. The report retains the tradeoffs: PSNR is below adaptive BPG at all three SNRs, and 4 dB LPIPS is worse than latent continuous JSCC. Four 10 dB body-CRC failures remain in the prescribed KEEP population. Swin 19 dB is outside its training/calibration range. These 24 fixed crops provide limited cross-collection evidence, not full-resolution Kodak evaluation or proof of pretrained-data non-overlap. Earlier common500 results remain unchanged; historical monitors remain paused.

## Latest completed supplement (2026-10-09)

[Paper supplement and result index](results/paper_supplement_20261008/PUBLICATION.md): the A-class first round is complete, including the later-added adaptive BPG + LDPC baseline (500 sources x 6 SNRs x 3 noises), four metrics and source-paired comparisons, native Swin checks, five-method single-output timings, resource/failure tables and paper figures. The original frozen common500 results at commit `252176e` remain unchanged.

Adaptive BPG has higher PSNR at every tested SNR; the proposed method has better LPIPS, DINOv2-L and source-prediction agreement. All points and negative findings are retained. Swin at 19 dB is outside its training/calibration range; native256 BPG timings do not measure the adaptive source search. External native replication and B/C extensions remain conditional and were not started. No training or new scientific work is triggered by this publication; the monitor remains paused.

## M2 scale-causal study delivered 2026-10-02

## 2026-10-04: M1 headline fixed16 and complete N2048 grid

The N1024 / 16QAM / 13 dB fixed16 replay and four-method comparison are complete: [report](reports/m1_headline_16qam13_20261004.md). The selected M1 action is m8/K0 (whole); actual 16QAM energy is reported. The new N2048 full-grid protocol and qualified execution are documented in [the experiment README](experiments/m1-n2048-full-grid-20261004/README.md). Scientific completion remains determined by the run receipts. Same-scale adapter training remains deferred.


Zero neural training updates. Frozen calibration choices, original100 development sources and failure-inclusive metrics. See [complete report](reports/scale_causal_m2_20261002.md); historical models and queues are preserved. The two-method authorized pipeline stops after this verified publication; actual-link branch status is explicit in the report.

## M1 scale-causal study delivered 2026-10-02

Zero neural training updates. Frozen calibration choices, original100 development sources and failure-inclusive metrics. See [complete report](reports/scale_causal_m1_20261002.md); historical models and queues are preserved. Method2 proceeds only after the verified method1 publication.

## N1024 extreme bandwidth study completed 2026-10-01

One fresh P1024 completed 40000 updates and selected step 40000 using complete1000-source calibration at1/4/7/13/19dB. Digital raw N1024 policies use paid68-symbol headers and were independently frozen on1000 calibration sources; P1024 receiver policies use200 calibration sources. All systems then used the same100 development sources and three noise repeats. P1024 budget_truncated=True; no convergence claim is made. QPSK/continuous have actual per-frameE2048;16QAM retains original fixed constellation scaling and actual energy. H_D/H_R decisions, paid-class conditions, distortion, source specificity, failures and computation are reported. Development is reused and the intervals do not cover training-seed variance. This authorized N1024 stage is complete. N512 remains frozen at its 40000-update budget cap; further experiments await a separate decision.

[Scientific report](reports/extreme_bandwidth_probe_20261001_R1_N1024.md).

## N512 extreme bandwidth study completed 2026-09-30

One fresh P512 completed 40000 updates and selected step 40000 using complete1000-source calibration at1/4/7/13/19dB. Digital raw N512 policies use paid68-symbol headers and were independently frozen on1000 calibration sources; P512 receiver policies use200 calibration sources. All systems then used the same100 development sources and three noise repeats. QPSK/continuous have actual per-frameE1024;16QAM retains original fixed constellation scaling and actual energy. H_D/H_R decisions, paid-class conditions, distortion, source specificity, failures and computation are reported. Development is reused and the intervals do not cover training-seed variance. N1024 remains a later decision.

[Scientific report](reports/extreme_bandwidth_probe_20260930_R1_N512.md).

## Step2 B completed 2026-09-30

The authorized single P_low arm continued selected27500 with the original populated AdamW, order and RNG. It completed 20000 added updates and selected added step 20000 only by full1000 calibration utility. The frozen evaluation then recalibrated receiver policies on200 calibration sources and evaluated100 development sources at -5/-2/1/4/13dB. G2: -5.0dB BYPASS_SELECTED, -2.0dB BYPASS_SELECTED, 1.0dB BYPASS_SELECTED. The13dB result is an out-of-training-range side-effect diagnostic.

[Report and full tables](results/rx_posterior_step2_B_20260930_R1/README.md). The development set is reused; this is not a new holdout or a training-seed replication.

## Step2 B started 2026-09-30

The user explicitly authorized B. A single P_low arm resumes selected27500 with original AdamW moments, source order and RNG. Train SNRs are -5/-2/1/4dB; full1000 calibration selects checkpoints every2500 added updates. The first milestone is10000 added updates; calibration-only extensions are capped at30000. Real-GPU populated-optimizer bitwise resume, frozen Dc/LPIPS gradients, and N4084/E8168 checks passed. The detached pipeline will evaluate the selected model and publish results after checks. Training/evaluation are in progress; no B scientific conclusion is available yet. Original A/P assets and paused old queues are preserved.

## 2026-09-30：冻结 P4084 接收端 Step2 A 完成

本轮只执行 A：P4084 第一训练 seed 2026092304 selected27500、匹配 Dc/VQ/VAR 全部冻结，原 N4084/E8168 与一次 AWGN 保持。200 源校准先冻结参数，再完成 100 源×6 SNR×3噪声×14输出的 25200 行 development 指标、108000 行 token 诊断、360 次计时和40张固定样例。新增训练更新为0，历史队列保持暂停。

1/4/7 dB 与13 dB的策略均选择真正旁路，最终输出与B2相同，未建立范围内增量。训练范围外−5/−2 dB超过B2/A1/A2并达到G1，共同A1融合权重下仍保留内容增量；相对B2的LPIPS差为−0.05076/−0.01789，DINO差为+0.19734/+0.04599。接收耗时约13→125 ms。13 dB保护来自旁路，未旁路融合的LPIPS显著恶化。

必要自检、历史B2再现、冻结身份和源图配对bootstrap已核验。结果是复用development的探索性证据；B/P_low及Step3等待独立决定。完整原始表按仓库每文件10 MB限制做可精确重建的分片，所有记录保留。

[结论与完整报告](results/rx_posterior_step2_A_20260930_R1/report_reviewed.md) · [结果及分片恢复](results/rx_posterior_step2_A_20260930_R1/README.md) · [执行入口](experiments/rx-posterior-step2-A-20260930/README.md)

> 2026-09-24 P2048: real initial20k complete, all135000 calibration rows and selected SHA verified. Both final calibration intervals support a10k extension after the existing P3060 initial run. Development/timing and full merged delivery remain pending. See reports/token_efficiency_P2048_20k_20260924.md.

> 2026-09-23 scheduler recovery: original C controller is RETIRED_VERIFIED after a blocked m6 launch, with failure evidence preserved and no m6 training. Detached coordinator_v2 resumed P2048 from verified step5967 and advanced beyond6000. Never SIGCONT the old PID. See reports/token_efficiency_scheduler_recovery_20260923.md. A is complete; B1/B2/C remain pending.

> 2026-09-23 source A COMPLETE: actual source streams/roundtrips and20 representation paths for1000 calibration+100 original development sources, calibrated quality targets frozen before development. Full source report: reports/token_channel_efficiency_20260923_source_A.md. P2048/P3060 real-device qualification passed; P2048 is training. B1/B2 communication matrices and merged short-prefix C remain incomplete; no new holdout.

> 2026-09-23 supplement accepted: A actual source representation/bitstreams, B1 P2048/P3060 and QPSK resource curves, B2 real16QAM, then merged short-prefix C. New holdout and content selectors are deferred. Code/CPU qualification is not real-result completion. See experiments/token_channel_efficiency_20260923/README.md and outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json. Existing m6 cache continues unchanged; its controller is intentionally held before training to honor A/B priority.

> 2026-09-23 short-prefix study: user-authorized new plan; v1 protocol, five-arm interface,4500 real-PHY preflight and real-GPU qualification implemented. First20k matrix is being launched; new quality/development,N3060,seed repeats,policy and new500 test are NOT_RUN. See reports/short_prefix_20260923_launch.md and experiments/var-short-prefix-hybrid-20260923/README.md.

> 2026-09-23 phase2 COMPLETE: registered bounded training, full calibration, selected-model quality/timing, strict-precision digital requalification, frozen resource lookup and PCA comparison are complete. See reports/review_20260923_phase2.md and results/review_20260923_phase2/. Original A/B and historical artifacts are preserved; no new holdout or convergence claim.

> 2026-09-23 phase1: local review repairs and affected real-weight reevaluation completed; see reports/review_20260923_phase1.md. Phase2 is separately authorized and not included in this completion.

> 2026-09-23 repository migration: the actual VAR_COMM root is the sole Git worktree. Historical research claims below are not newly certified by this migration. No A/B retraining or GPU quality validation was performed. See reports/repository_migration_20260923.md.

# VAR 通信当前进度

## 最新：latent增强后续对照全部完成（2026-09-19收口）

- 数字校准动作完成：N3572/N4084、raw/arithmetic、m7/m8/m9、D0/Dc，360000 calibration rows；m10按当前header mode域不纳入。动作及development结果位于`outputs/VAR-LATENT-ENHANCEMENT-20260917/followup/digital_policy_v2/`与`digital_policy_development_v1/`。
- 修正计时完成：650调用，旧数字重建最大像素差0；纯数字不再计算Fb_TX，Dc不再重复D0，噪声移出RX计时。产物`followup/online_timing_v2/`。
- N4084折中`3504 digital + 512 continuous`完成1500帧，五SNR均值PSNR21.5841、LPIPS0.147276、DINO0.870658；1dB body CRC失败299/300，4dB以上保留明显连续增强收益。
- 1024 Decoder适配两臂均完成10000步安全上限；development适配臂22.2190/0.131421/0.873083，控制22.0985/0.135576/0.874250，配对提升约+0.1205dB、LPIPS−0.00416，DINO区间跨0。
- 当前结论：保留1024连续增强和Decoder适配；折中分配只作为低SNR候选，不自动组成SNR自适应系统。没有新holdout、不追加训练、不启动新骨干。

## 当前：latent增强development测评与在线计时完成（2026-09-18 11:47 UTC+8）

- 100张原development图×5 SNR×3噪声、41个方法共61500行质量结果完成；未访问新holdout，冻结模型SHA未变化，所有失败保留。协议`reports/latent_enhancement_evaluation_protocol_20260918.md`，结果`reports/latent_enhancement_development_result_20260918.md`。
- 关键结果：512增强N3572为21.3479/0.157741/0.853351，1024增强N4084为22.0674/0.136225/0.874412（PSNR/LPIPS/DINO）；同资源raw m8/Dc分别20.1531/0.171924/0.867911和20.3061/0.166890/0.873181。512/1024质量优势有源图配对区间支持，但512 DINO略低，1024 DINO差异跨0。
- 1dB仍受m8基础body失败主导：m8/增强均300/300 body CRC失败；增强层不能保证低SNR感知质量。4dB及以上新增观测收益明显。
- 固定10张development图×5 SNR×seed2001的650次原始TX/RX计时完成，但发布复核发现纯数字路径统一计算Fb_TX、Dc/算术路径存在重复工作、噪声生成落在RX区间；因此计时只作诊断，不能作速度排名。原始产物`outputs/VAR-LATENT-ENHANCEMENT-20260917/online_timing_v1/`和复核说明已保留。
- 本轮不追加训练、不改checkpoint、不访问holdout。新增预算的校准自适应/可行全尺度数字对照和统一在线计时仍是缺口；当前latent增强候选保留为固定模式下有正面development证据但尚未最终定稿的系统。

## 当前：阶段B按用户决定收口，保留训练与结果（2026-09-18 10:25 UTC+8）

- 用户认为近期提升已很小，要求到当前为止。B trainer在配对更新边界安全退出，最后保存更新44510；没有强制kill、没有删除checkpoint/日志/校准结果。原controller的exit130另以`training/user_stop.json`登记为用户主动停止，不是训练失败。
- 训练固定选中点：512为30000步、1024为40000步、receiver-only为35000步；最后完整校准为40000。阶段A此前已到40000安全上限并选中38000步Dc。
- 收口报告：`reports/latent_enhancement_stage_B_training_closure_20260918.md`。B校准已显示新增512/1024观测相对同Dc无新增传输精化器有收益迹象，但同总N数字对照、development质量、配对区间和在线成本尚未做，不能宣布通信效率结论。
- 不自动续训、不扩大预算、不启动新loss/骨干/功率搜索。若继续，只做既定选中checkpoint的development和同资源对照；不访问新holdout。

## 当前：A已完成，B自动接续成功、每臂约1.2万步（2026-09-17 22:46 UTC+8快照）

- A于17:35:10达到40000安全上限结束，不宣称完全收敛；按原规则选中38000步Dc。原VAE保持不变，1000源校准连续参考Dc(F)相对D0(Fq)为PSNR+3.632995dB、LPIPS−0.052360，源图配对区间支持改善；这些不是有限带宽通信成绩。
- B控制器1605981实际自动接上。17:35:40真实GPU/原PHY/图像梯度检查通过，原基础RGB差0，D0/Dc均未改变；18:18:18实际RX缓存完成，保留全部215000帧，43954次新VAR补全。18:18起进入B训练流程，现trainer1745684每臂约12150步；旧PID仅为快照，不据此重复启动。
- 最新完整校准是每臂10000步，1000源×五SNR×三噪声、每方案15000传输，全部失败保留。五SNR平均PSNR/LPIPS：原m8+D0(N3060)18.7973/0.196227；仅Dc(N3060)19.4677/0.183585；receiver-only(N3060)19.6872/0.184131；增强512(N3572)20.9967/0.156736；增强1024(N4084)21.7291/0.137339。不同N不作等资源排名，数据是校准而非development。
- 5000→10000步，512/1024的LPIPS分别再降0.002284/0.003753；控制PSNR小增但LPIPS略退。三臂均按已登记规则继续、更新机会相同，尚不能称充分收敛。无失败记录，GPU约99%/6.6GiB，温度实时83–84℃，控制器仍监控。
- 同N3572/4084的强raw/整帧算术数字及同Dc对照、development三指标/源图配对区间、完整在线TX/RX成本尚未完成。当前只能说有新增观测的校准收益迹象，不能宣布资源效率或系统胜出。实时入口仍为`stage_B_v1/auto_001/status.json`及`stage_B_v1/training/status.json`。

## 当前：B自动接续真正挂上，等待A结束（2026-09-17 17:01 UTC+8快照）

- A实际38530步、最近选中38000，源码/配置绑定未变；此前“B未实现/未接好”是历史状态，现在已补齐独立`phase_b/`代码与条件控制器。
- 新控制器PID1605981，实时状态`outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v1/auto_001/status.json`，当前`WAIT_FOR_A`。它等A真实完成、资格通过并锁定其选中Dc，依次运行GPU工程检查、实际接收latent缓存、三臂配对训练，不需再次请求开始。A未结束，因此此时B参数更新仍为0。
- 13项CPU测试通过，含真正三臂训练循环在让卡后恢复的模型/Adam/数据序列与不中断运行逐项一致。实际数字回放：训练200000帧，校准15000帧；所有CRC/header失败保留，不修复候选。
- 模型参数：512增强527716，1024增强532328，receiver-only263776。RX公共主干相同，控制无TX和观测stem；不冒称总参数完全匹配。原loss、N/E、SNR、原两预算及Dc冻结输入梯度规则均未改。
- 独立控制器每30秒记录运行状态/GPU和异常，旧A观察器仍在。只读等待控制器曾在尚无B子进程时刷新以载入最终通过测试的版本，回执已留存；没有停止A或其他任务。任何资格/实现失败停止留档，不无限重试或自动改配方。
- 待完成项仍包括实际GPU资格、B训练与校准、development/强数字同资源对照及完整成本；不把接续完成写成通信机制成功。

## 当前：Stage A超过20000步，持续监控已挂上（2026-09-17 10:03 UTC+8快照）

- 20k训练＋1k校准FP32缓存全部完成；Stage A实际20030步，20000步完整1000源校准成为当前选中点。Dc(F)为PSNR26.1106、LPIPS0.051777，混合校准效用0.0118028；是连续信息参考，不是有限带宽通信成绩。仍按原曲线/停止规则继续，Stage B未启动。
- 用户要求持续监控，新增独立CPU观察器`experiments/var-latent-enhancement-20260917/monitoring/watch_stage_a.py`，每60秒检查进程身份、训练/校准活动、loss/梯度、选模、GPU、磁盘与冻结源码SHA。观察器不在训练源码目录，不改动已经注册的训练实现/配置/优化器，不占GPU、不停止其他任务。
- 状态`outputs/VAR-LATENT-ENHANCEMENT-20260917/monitor_001/status.json`，历史`history.jsonl`，校准/异常事件`events.jsonl`。启动PID1081377仅为快照，以实际命令及状态为准；6项监控测试通过，首检无告警、源码绑定未变，GPU100%/19690MiB/78℃。
- 异常或阶段A结束会保留本地告警/完成记录；没有配置邮件/消息/聊天自动推送，不声称已向用户发送离线通知。观察器不自动重试一般失败、不更改loss/批量/停止阈值，也不未经真实实现资格就启动阶段B。

## 当前：新latent增强实验已占用空闲GPU，缓存→阶段A（2026-09-17 01:58 UTC+8快照）

- 用户明确授权独立的m8/N3060基础＋512/1024连续latent符号，不改旧m7/RGB混合结论。协议与训练/停止规则见`reports/latent_enhancement_protocol_20260917.md`及新实验`docs/stage_A_execution.md`。
- GPU0确认空闲后，真实模型资格检查通过：F形状32×16×16，全部真token累计Fq与官方映射差0；1dB body CRC失败和19dB成功两例基础RGB均与原实现像素差0。Dc副本训练后D0/原VAE未改，冻结Dc仍有输入梯度；四样本反向实际峰值分配约14.92GiB。工程检查更新已丢弃，不是正式训练成绩。
- 八项CPU测试通过，正式流水线PID466481/cache PID466513（仅为启动快照，不据旧PID盲目重启）。已完成1000校准缓存和300/20000训练缓存，FP32源latent、全量量化latent和确定性m8基础latent；缓存后自动执行Stage A。状态入口`outputs/VAR-LATENT-ENHANCEMENT-20260917/pipeline_001/status.json`。
- Stage A使用独立post_quant_conv+decoder副本，固定50/25/25输入和MSE+0.1LPIPS。每500步固定子校准、每2000步完整校准；基于曲线判停，40000安全上限不作收敛证明。尚无实际训练/科学收益结论；Stage B及开发/数字资源对照仍待后续，整个实验未完成。
- HiFi及其自动续跑器按新任务优先级暂停，保留原1500/20源记录。之前一次非持久后台启动未留下缓存或训练更新，正式运行已改为受监控常驻session。新流水线遇其他授权GPU进程会保存让卡，不终止他人任务。

## 20源阶段91/100已保存，等待外部GPU任务释放（2026-09-17 01:01快照）

耗时排查和独立分支验证已完成，报告`reports/hifi_runtime_profile_result_20260916.md`。20源阶段在91/100帧后遇到外部CAP-VPR GPU进程，原保护规则主动退出；不是采样失败，没有删除/重选图像或修改作者参数。

新增`resource_resume_001/status.json`空闲续跑监视器（启动PID382116，实时状态为准）：只在GPU持续空闲时用原版本补余9帧，若其他GPU任务出现只中断我们自己的worker，不操作其他项目/共享环境。原1500计划及240帧仍保留暂停，不冒称完成。

停止附近19dB源47/52的原时延保留并标注资源不确定性，二者不在原共同10源时间比较中；后续争用事件继续留痕。质量仍全100计划，不因资源问题丢失败输出。说明`reports/hifi_stage20_gpu_contention_20260917.md`。分析代码仅增加该披露和标记，旧准备SHA保留，修订记录`analysis_contention_disclosure_revision.json`；物理、采样和质量统计不变。

## HiFi耗时诊断完成，20源探索队列执行中（2026-09-17）

按用户最新授权，原1500计划与240已提交帧保留并暂停；新阶段固定20×五SNR×noise2001，复用48帧，仅补52帧。入口`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/hifi_stage20_20260916/pipeline_001/status.json`，没有改采样来加速，也没有部署新分支。

性能报告`reports/hifi_runtime_profile_result_20260916.md`：原240帧=720记录视图但只有240次扩散，元数据额外恢复0。三校准工作点关闭attention checkpoint仅约0.04%–0.93%完整RX降幅，allocated显存增加约300MiB。1dB仪器轨迹的重算正向约0.98秒，已含于约51.9秒根输入反传；不能重复相加或误算整次扩散。

9个匹配状态梯度均有限/非零，6/9通过全部严格阈值；三个完整AB输出未过原RGB阈值，AA原版重复也有同量级差异。所有数组/记录保留，不改阈值，不把数值问题当科学质量负结果。原版保持。

性能图在新阶段`runtime_figures_002/`（001布局尝试保留），23项CPU检查通过。阶段分析程序已准备，完整100帧与指标后才出`hifi_stage20_positioning_20260916.md`；所有系统同20源质量，时间另统一用共同已有10源，HiFi100帧全部时延单列。无新训练、骨干或holdout。

## 本次定位材料已提交并推送（2026-09-16 22:12）

按用户“这次的先提上去”授权，一页定位、原文差异表、HiFi收尾说明/代码/测试及三类图与失败后果表已提交并立即推送到`daiqizai/var-comm`的`main`。提交`4be9faa9ec5bcb397e382c81dc34981ebaba195f`，远端SHA核对一致，发布工作树干净；23文件变更，未强推。

公开入口`results/external_baselines/positioning_interim/README.md`，首页和研究状态已加链接。9项公开测试、15项实验检查通过；用暂存区的独立副本只读公开CSV复算，三张新增数表逐字节一致，包含56项配对指标区间。发布文件/双SHA/敏感信息检查通过。

HiFi和CPU收尾等待器未修改，未发布未完成的性能结论。每次授权commit后立即push的长期约定也已同步到公开`AGENTS.md`。

## 不等HiFi：先行研究定位与失败后果复算（2026-09-16）

按用户要求，已完成`reports/research_positioning_one_page_20260916.md`阶段一页定位、`reports/external_related_work_difference_20260916.md`原文差异表，以及`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/research_positioning_interim_001/`的等资源/质量—信道次数/质量—完整RX三类图与数表。只读已有结果、CPU运行，没有新模型推理/训练或holdout。

新复算：同N4204、1dB、m8的300次传输，算术可靠正确率81.67%、raw16.67%，但全传输LPIPS算术0.19949、raw0.18613；raw−算术的源图CI为[−0.021745,−0.005649]。保留“成功率不等于全部输出图像质量”的限定证据；不把不同失败集合的均值当因果，也不改写旧算术quality=BLER的负结果。

建议数字VAR暂为限定需求的工程候选，R3保留高SNR工作点，Swin/Deep保留像素/计算取舍；当前不定稿为新增算法。普通误差韧性熵码只是登记的最小证据缺口，没有新实验队列。不把m9饱和当数字整体上限。

HiFi原worker未动，另挂CPU-only收尾等待器，状态`closure_queue_001/status.json`；完整1500帧/指标/原队列回执齐后才做同观测归因、同N4204比较并输出最终报告和最终一页定位。15项CPU测试通过。ADJSCC四固定样例的同观测SHA严格一致，RGB有约2.4–3.6e−7差异，不能假称bit-exact；按原功能容差作全量核查，详见`reports/hifi_completion_readiness_20260916.md`。

## 外部非扩散结果已同步GitHub（2026-09-16 19:53）

用户授权“都传到仓库里”后，新增代码、报告、25500开发/协议/参考行、18000校准行、完整计时与120张展示图已推送到`daiqizai/var-comm`的`main`，提交`1eb526f3c01c0891244cb46855b0346539a69289`。本地/远端一致、发布工作树干净，未强推。9项公开测试及干净克隆的119汇总/1624配对区间CPU复算通过，最大差0。入口公开`results/external_baselines/README.md`；详情`reports/github_external_update_20260916.md`。

未上传权重、完整数据集、源像素数组、环境或密钥。HiFi只有运行快照，仍按原协议继续，不冒称外部最终定位完成。

## 非扩散外部定位先行完成，HiFi已断点恢复（2026-09-16 19:05）

用户要求先完成其他部分后，已交付`reports/external_baseline_non_diffusion_result_20260916.md`。作者Swin/ADJSCC各4500次传输、原四系统6000参考、新N4204/4498数字18000次校准＋6000次冻结开发、1920对完整TX/RX计时完成。25500协议/参考/开发行汇总为119表行、1624配对区间；等N4498原图拼图、逐SNR/质量—资源/计算曲线在`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/non_diffusion_analysis_001/`。6000新PHY及183固定图像检查PASS。

相同N4498下，主区间raw数字相对Swin RA32的LPIPS−0.126653，但PSNR−5.2319dB、RX约59.64vs5.58ms；是明确取舍，不是全面胜出。R3高SNR低LPIPS参考保留。旧R3/Deep单模型峰值仍有空项，不能从旧共驻值宣称显存优势。

HiFi第52帧提交后为短任务让卡，19:05:20已用原SHA绑定队列恢复；全部提交帧与中断记录保留，没有改权重/采样/数据或启动训练。实时PID以`author_queue_001/status.json`为准。最终外部定位报告仍待HiFi全量与同ADJSCC归因，不把本阶段当所有任务结束。

## 外部作者方法实测已启动（2026-09-16）

五份用户转存作者权重全部SHA核验通过，扩散权重另与官方MD5一致。独立torch1.12.1/cu116环境建立完成；Swin三档与ADJSCC已严格加载，并与作者收发路径达到约1e-7像素一致性。HiFi完整默认235步校准推理通过，固定参数未更新；最初冻结autograd标志引发的自定义checkpoint兼容错误已保留并修复，不当方法负结果。

17:34启动有限作者测量队列，入口`experiments/external-baseline-positioning-20260916/CONTINUE.md`，实时状态`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/author_queue_001/status.json`。Swin/ADJSCC各4500帧，HiFi1500帧，原100 development、五SNR、三原噪声。HiFi10dB单图实测约78.7秒，完整默认设置预计30–36 GPU小时；没有擅自缩步数弱化基线。

四已有系统6000条只补共同SSIM，未重跑原模型。作者免费功率/mask假设与实际计费协议分表；裸ADJSCC没有被强加header。N4204/4498数字适配CPU测试通过，但必要校准、实测、时延补充和最终定位报告仍未完成，不能宣称外部定位已完成。运行输入SHA已绑定，不边跑边改源码；不启动训练或新holdout。

## GitHub发布完成，远端提交已核对（2026-09-16）

用户明确授权整理并推送到`daiqizai/var-comm`。独立发布工作树`publish/var-comm/`已提交`16dd8d7`（main，500文件、约21.8 MB），保留通信源码/配置、研究报告、失败结论、关键数据/图与CPU复算入口，排除大模型、数据像素、vendor、venv和凭据。原始文件原位只读，发布副本仅做机器路径脱敏，并记录双SHA。

5项公开结果测试、8项模式策略测试、314个Python语法检查、全部暂存blob核对及干净Git克隆后的1856配对区间复算均通过。完整Git bundle约6.1 MiB已备份。

用户提醒后找到已有非默认密钥`id_ed25519_github`，已认证为`daiqizai`。只设置本发布仓库的SSH认证命令，未新建密钥或改全局SSH配置。普通push成功，远端main/default HEAD均与本地`16dd8d78989537d80218660c21b032407798e7ad`一致，工作树干净。之前将默认SSH失败概括为需重新配置认证不完整，现已纠正；过程保留在`reports/github_publication_status_20260916.md`。没有新训练或holdout访问。

## 最新完成：三权重收尾，当前固定m7混合主线结束（2026-09-16）

报告`reports/hybrid_weight_closure_convergence_20260916.md`。四新臂各20000新增更新，旧0.01两臂同历史只读复用；全20000数据/SNR/噪声、初态/Adam及最终计数独立核对一致。六点都在校准选20000；最后5k仍改善，封顶不作为彻底收敛证明。新增训练/校准共同会话3.06138 GPU小时。

结构方面，C/0.03在主1/4/7相近LPIPS下PSNR+0.544004（CI正）；机制区间C/0.1对U/0.03在PSNR−0.05333dB时LPIPS−0.008817（CI负），保留条件读取机制。系统方面，最佳混合LPIPS0.217177，仍较raw/算术差+0.033414/+0.034597，未过原联合要求。两个预定区间六点三指标均落后校准固定比例期望参考；其非部署属性、区间越过0.005以及4dB局部PSNR例外均披露，不夸称全面支配。

9000实际工作点行＋9000旧强参考＋9000期望行完成；1856配对区间、1500 PHY及9000图像独立复核PASS，图表/CSV已导出。全部模型、Adam、失败记录与原报告保留。正式队列12:37结束，独立复核12:38完成，图表12:39完成，无后续训练/搜索或新holdout。CSI不确定性只交三篇已发表原文核查与未执行的最小验证草案，具体差异尚不够清楚，不自动定稿或启动训练。

## 本轮历史执行：固定m7混合路线三权重收尾（2026-09-16）

09:44（UTC+8）实查：正式队列3440348/trainer3440380运行正常，四个新臂已各完成700/20000新增更新。两种原接收结构与PHY不变，仅新增lambda0.03/0.1；lambda0.01原两臂完整20000步通过共同初态、Adam、训练机会及配对数据噪声审计后复用，不用旧终点作为新臂起点。固定选模、实测点匹配容差与校准拟合的帧间选择比例规则已在运行前登记。

计划`reports/hybrid_weight_closure_protocol_20260916.md`；输出`outputs/HYBRID-WEIGHT-CLOSURE-20260916/`。已完成复用资格、冻结Deep所缺15000条校准、四臂工程测速及180行质量/分析smoke；这些不计科学效果。自动流程仅训练→校准冻结→原development→数值汇总，不自动扩lambda/结构或启动CSI新题。旧报告与所有失败记录保持不变，最终收敛判断尚待完成。

## 最新完成：基图条件读取证实像素效应，未获得联合收益

2026-09-16 01:24:47注册流程结束，01:27后的独立产物核验亦PASS。两臂各新增20000，共同E/D累计30000，最后完整校准仍改善，因此按预登记预算上限停止而非宣称收敛。报告`reports/hybrid_base_conditioning_result_20260916.md`；图表在`outputs/HYBRID-BASE-CONDITIONING-20260915/analysis_001/`和`report_assets_001/`。

机制7/13/19：条件22.08019dB/0.194206/0.812539，对照21.71641/0.188881/0.816989。PSNR+0.36378、CI[+0.32971,+0.39767]；LPIPS+0.005325、CI明确为正；DINO CI跨0。空间打乱使PSNR−0.97417、LPIPS+0.040414，说明空间条件真实参与。保留有限像素/取舍证据，不把自动未通过LPIPS稳定/三指标状态等同“条件完全没用”。

系统1/4/7：条件20.43010/0.239892/0.769932；raw19.62665/0.183763/0.886743。PSNR+0.80344有CI支持、超过0.5dB点目标，但LPIPS+0.056129远超0.005，联合目标仍失败。条件比本轮控制主PSNR+0.62050而LPIPS+0.007316，不能仅和旧诊断起点比较夸大感知增量。

只加5488参数（每臂总1648125），新增空间卷积28.18M MAC/图；完整条件TX/RX约68.386/61.944ms，与同结构控制同量级，小差异不当加速证据。训练+校准两臂合计约2.271 GPU驻留小时。原资源、loss和视觉/PHY未变；3000正常+1500空间诊断行、960计时、1500 PHY/4500图片指标复核完成。

旧结果/checkpoint/失败记录均保留；无新holdout。原队列和训练/审计GPU任务均结束，没有自动扩展；当前固定条件实现停止追加，保留机制观察而非新系统赢家。

## 最新执行：实际基图中间条件读取，两臂有界训练

2026-09-15新授权与计划`reports/hybrid_base_conditioning_protocol_20260915.md`。保持m7/68+1882+1110/E6120/FEC/视觉/源残差域/loss不变，仅比较连续Decoder是否读取实际基图的空间特征。两臂同1648125参数、同起点功能与共享参数Adam，新小模块5488参数；不重开三通道gain训练。

8项CPU、真实GPU梯度、60正常+30空间打乱工程输出、20条完整计时均通过；基图与旧链像素差0。23:05:12队列2711529/trainer2711561启动（session97872），23:09正在完整初始1k校准，尚无新增更新。回执`reports/hybrid_base_conditioning_launch_20260915.md`。实时看`outputs/HYBRID-BASE-CONDITIONING-20260915/pipeline_001/status.json`及`training_001/status.json`。

新增最少10000、最多20000，任一臂改善都让两臂同机会推进；最后按完整校准选模，DINO不参与停止/选择。机制7/13/19和系统1/4/7分开，不筛失败，不访问新holdout。工程完成不等于机制成立；旧全部负结果/模型/失败只读保留。

## 最新完成：混合源残差固定分配实验及报告复核

2026-09-15 20:30:13注册流水线全部完成：每臂10000更新，126000校准记录、4500新development行与9000强参考、1500基础消融、1440新系统计时。后续CPU完整训练/Adam审计、1500 PHY重放和6000张图像PSNR/LPIPS/DINO独立复算通过。完整报告`reports/hybrid_source_correction_result_20260915.md`；图表`outputs/HYBRID-SOURCE-CORRECTION-20260915/report_assets_001/`。

主1/4/7：可靠度gain 19.69400dB/LPIPS0.236300/DINO0.782826；raw自适应19.62665/0.183763/0.886743。ΔPSNR+0.06735 CI跨0；ΔLPIPS+0.05254、CI[+0.04920,+0.05591]，未达到+0.5dB/LPIPS退化≤0.005的预设目标。三个分支均无合格校准点，10000仅是已登记的违约诊断点，不列作新强基线。

残差链相对缩短基础PSNR+2.198dB/LPIPS−0.02100，但没有追回完整系统差距。同参数可靠度输入未提供实用增量。1dB缩短数字块300/300 CRC失败，原完整raw m7正确246/300；≥7dB新基础全正确后LPIPS仍明显落后，不能仅归因于误码。匹配校准曲线仍小幅改善，不宣称已彻底收敛/混合架构无效。

新TX/RX约69.336/62.681ms，和132.017ms，主要多付TX VAR生成成本；不含空口/排队。当前所有GPU/队列已退出，无自动追加训练/搜索/新holdout。原数据/checkpoint/失败保留，通信研究总体目标仍未完成。

## 最新实际状态：源残差混合通信已进入正式配对训练

19:18更新：每臂1509/10000；首1000 checkpoint三套Adam更新计数一致，固定100校准源/4500行完整，CPU状态与覆盖审计PASS。所有运行源码绑定未改动，继续固定预算；尚未得到完整训练后的科学结论。

2026-09-15 19:07:36正式trainer2436487启动（受限队列2433287，session17285）；19:08:51每臂145/10000更新，GPU占用5326 MiB、利用率94%。21k确定性缓存完成；8项CPU回归、真实图像E/D梯度、90条工程质量及30条无缓存在线回放均通过，后者像素误差≤1.67e−6。启动回执`reports/hybrid_source_correction_launch_20260915.md`。

实时入口`outputs/HYBRID-SOURCE-CORRECTION-20260915/pipeline_001/status.json`，代码SHA锁定的有界队列含后续校准/原100 development/仅新系统计时/配对统计；出错停止，不自动换架构/额度或加训。不访问新holdout。当前仍没有新方案的科学质量结论，下方“训练尚未启动”为此前缓存阶段快照。

## 最新：固定数字基础＋源残差连续支路，工程准备已进入执行

2026-09-15用户新附件及“理解后推进”授权形成一个明确问题：同N3060/E6120下，实际传输基础图未表达的实例残差，能否提升PSNR并满足LPIPS非劣。协议`reports/hybrid_source_correction_protocol_20260915.md`，固定m7/68+1882+1110；一个小连续E/D，硬相加、同参数SNR gain、可靠度gain三个融合臂，每臂10000配对更新，不进行架构/功率/分配搜索。

18:41六项CPU协议/能量/失败候选/信息边界测试通过；4个train/calibration工程样本的无噪声数字往返正确、TX与RX旧VAR图像逐像素相同。18:42启动20k训练＋1k校准的确定性TX基础缓存，PID2407172（session77047），`outputs/HYBRID-SOURCE-CORRECTION-20260915/cache_001/status.json`。此条记录时训练尚未启动，不把缓存/梯度代码准备当成学习增益。

TX不见RX噪声/CRC；RX真实译码失败候选不删除，header失败灰图。已有完整raw/arithmetic数字、固定m7/m8/m9与感知Deep是强对照；缩短基础不能冒充强基线。旧1000 holdout已经用于提出本假设，只作历史外部参考；本轮不访问新holdout。所有旧结果/失败/权重只读保留。

## 最新完成：通信强对照、独立验证及最终收敛结论

2026-09-15 17:42（本机UTC+8）所有数值阶段结束：126000条新holdout VAR候选、84000条四冻结系统参考完成；两项CPU审计和冻结策略统计均PASS。1000张/1000类独立源严格在方法与有序名单冻结后访问，1个历史像素重复按预定规则排除，没有据holdout调整策略。最后报告`reports/communication_convergence_final_20260915.md`，正式图`outputs/COMMUNICATION-CONVERGENCE-20260915/HOLDOUT_FIGURES_001/`。

独立主1/4/7：raw质量/goodput/旧adaptive LPIPS0.179154，arithmetic三规则0.179288；后者−前者+0.000134，95% CI[−0.000262,+0.000542]。新质量策略没有超出最强arithmetic BLER控制：两者在全部七SNR同动作。6dB raw质量比raw goodput降0.022053，但arithmetic普通BLER进一步降0.010240；5dB相对arith goodput的感知改善伴随PSNR−0.3246dB，不能称无代价全指标优势。

高13/19：R3 LPIPS0.105112、PSNR23.5885，优于原VAR的0.133193/20.9821；同WeTok数字LPIPS0.086295更强，Deep PSNR26.330更高。不能说所有学习方法输。新算术链完整处理约135.33ms，raw约72.43ms，成本主要增加在TX；计时只用旧32源、不同session，不含空口/排队。

本轮任务完整结束，不自动加训或再搜索；原PID2221015/2273403/2230734及CPU审计均已终止。保留所有原报告、checkpoint、失败/中断和完整图像。新增的是有边界的通信实验发现，不是已经验证新颖性的独立调度方法；大研究目标仍未完成。

## 当前：独立holdout已启动，受限后续流水线已接续

主矩阵PID2221015（session80794）按冻结方法/名单运行1000源；16:17已204/1000、25704行，只看进度不据部分holdout改方法。受限finisher2230734（session78377）等待该实际进程，再仅运行四冻结参考、两项CPU审计和策略统计，入口`VALIDATION_FINISHER_001/status.json`。不重复队列，不重新训练或搜索。

新整帧熵编码时间2016行完成，实际TX/RX波形图像差均0、参数不变；主质量策略TX/RX约60.28/75.05ms，处理和135.33ms。raw质量策略复用旧各m计时约10.07/62.36ms；不同session/co-residency限制保留。完整开发收敛判断见`reports/communication_development_result_20260915.md`，尚未把holdout当成完成结果。

## 当前：校准/开发完成并冻结；即将按锁定方案独立验证

2026-09-15：1000源×六固定候选×七SNR×三噪声的126000条校准完成；CPU逐条重放全部PHY并核对源codec完整性/PSNR/DINO，100128个完整恢复payload全部正确还原source，审计PASS。100源12600条development也完成并同口径审计；旧raw数字复放三指标最大差≤3.82e−6，原参考未改写。

`POLICIES_001`按校准固定：raw质量规则[7,8,8,9,9,9,9]，BLER规则[7,8,8,8,8,9,9]，goodput[7,8,8,8,9,9,9]；arithmetic质量与BLER均[7,8,8,9,9,9,9]，goodput仅5dB改为m9。SNR顺序1/4/5/6/7/13/19。全局校准选择arithmetic_quality，但它与同族普通BLER策略完全等价，不得称独立质量调度增量。

Development主LPIPS raw_quality/goodput/旧adaptive同为0.183763，arithmetic三规则同为0.182580；6dB raw质量比raw goodput改善0.022872，但arith质量/BLER又优于raw质量0.012850。没有调阈值找优势，仍按原主区间与强对照报告。

15:59:05先封存`FINAL_FREEZE_001/method_core.json`与全部按类排序的source路径计划，再开始像素完整性检查；15:59:09形成1000张/1000类未用holdout清单。历史排除覆盖20k通信训练原图及flip、1k校准和3342个已用val源；预定顺序排除1个重复，未计算holdout模型指标。数据与方法SHA均封存，不可据holdout改方法。

当前新熵编码链2016条CPU端点计时运行中，PID2210452（工具session67630），只测新链、不重测七系统。随后只按`FINAL_FREEZE_001/frozen_method.json`启动1000源VAR候选矩阵和四个冻结系统参考，不训练/新架构/搜索。来源根`outputs/COMMUNICATION-CONVERGENCE-20260915/`；本轮总任务尚未完成。

## 当前：R3收尾完成，整帧熵编码/模式策略校准运行中

2026-09-15新授权见`reports/communication_convergence_execution_20260915.md`。R3原10000选定点已完成2100新development记录、原22参考复用、CPU归档/配对统计；仅自身480条同端点计时也结束，没有重跑七系统benchmark。主R3−R2 LPIPS−0.001811、PSNR−0.125949dB，DINO CI跨0；高13/19改善更多但非所有指标最优。完整阶段报告`reports/r3_completion_result_20260915.md`，不混用校准0.172232。

新整帧算术编码不是旧分组链：94 header＋2966 data，单一连续r1:m码流和一个data FEC块，溢出raw回退计费，CRC失败实际候选仍渲染。完整VAR概率/CDF、GPU无损往返、复用状态补全和失败候选与旧实现图像差均0；8项算术/FEC CPU、5项模式完整性CPU检查通过。

校准协议已固定：raw/arithmetic两族×m7/8/9，1000原独立校准图、固定4101/4102/4103噪声、七SNR，共126000记录。BLER目标10%下最大源率、LPIPS+0.25dB PSNR约束、最大源索引goodput三个规则均在结果前登记，不搜索阈值。主SNR1/4/7、高13/19，5/6过渡单列，DINO不进入新选择。

14:51核实实际校准进程2117550（工具session73155），已107/1000源、13482行，约2.7秒/源。输出`outputs/COMMUNICATION-CONVERGENCE-20260915/CALIBRATION_001/`；当前只运行这项GPU校准，尚未拟合策略或重新评测development。新holdout仅盘点文件路径，未读其像素/指标，需完成开发后冻结方法与清单再执行。不启动训练/新骨干/自动候选扩张。

## 当前完成：冻结系统完整在线代价已补齐

2026-09-15 13:09:46（本机UTC+8），七方案/32图/五SNR/seed2001/三次计时重复共3360行完整结束；每条实际TX波形、RX图像与旧结果逐字节相同，权重起止SHA不变。13:10:18独立CPU重新核对产物、故障保留、均值/P95/配对bootstrap，最大统计差1.42e−14ms。实际正式进程约283.78秒，不等于GPU利用积分。

主1/4/7 dB：数字adaptive完整TX/RX为10.090/62.284ms，R2 residual为29.388/39.535ms，Deep为2.258/3.023ms；adaptive处理和比R2多3.451ms，比Deep多67.092ms。旧数字1—2ms只是PHY，当前成本应与质量一起判断，不再称轻量或实时。计时仅32源一个噪声，不能改写100源三噪声的旧质量结果。

报告`reports/frozen_system_online_timing_result_20260915.md`；全部表格/哈希/三次保留工程失败在`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/`，正式PNG/PDF在`figures_full/`。21项CPU检查通过；`smoke_004`旧示意图标题范围误用正式模板，数据回执正确，旧图保留且另以真实2图/一次重复重绘至`figures_smoke/`，不得用于系统排名。

原正式进程1994210已结束，无追加队列；本次有限计时授权完成，训练/架构/自动搜索仍暂停，R3只保留待验证资格，研究总目标未完成。

## 当前有限执行：冻结系统完整CPU端点计时

用户于CPU准备后明确授权“开始”。2026-09-15 13:04 `smoke_004`完成70组GPU回放：发送波形和RGB最大差均0、起止模型SHA一致。动态模块归档、错误引用旧未配对WeTok数字归档、单帧batch维度适配的前三次工程失败均保留；未改旧源码、原FEC、噪声或checkpoint。11项新CPU检查和10项旧回归通过。

13:05核实正式进程1994210，已提交2/32源、210/3360条记录；同一GPU仅该任务。数字m7/m8/m9/adaptive、R2 residual、感知Deep及WeTok数字全部冻结，CPU RGB→CPU波形、CPU波形→CPU RGB连续计时。输出`outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/full_001/`；尚无正式汇总结果。新增架构/训练/参数搜索及R3质量流水线继续暂停。

## 当前补充：数字计时输入已CPU核验，未启动GPU测量

2026-09-15 11:38（本机UTC+8），固定32图/五SNR/seed2001的640条旧数字记录与R2的2400条原计时记录完成输入核验，155个输入SHA留存、10项CPU测试通过；全部失败保留，没有新传输或模型推理。报告`reports/digital_online_timing_readiness_20260915.md`，产物`outputs/DIGITAL-ONLINE-TIMING-PREPARATION-20260915/`。

新增明确限制：R2旧RX是GPU波形到GPU RGB，数字`complete_image`返回CPU；R2旧TX为两段计算之和，中间表示转换未计入。不能把旧数字PHY时间或新CPU端点时间直接与它横排。CPU准备已结束，新的同端点完整测量尚未实现/排队；暂停不解除，不扩写新架构或预检。

## 当前：收敛报告完成，数字VAR重列候选主系统

本机UTC+8 2026-09-15：`reports/convergence_report_20260915.md`完成过去24h审查（窗口截止请求时01:28:36），4组系统比较/15模型版本、64事件与22个缺少明确完成时间的旧产物分开记录。原数据只读，补做数字adaptive对固定模式/Deep/已完成学习链的图像级配对统计，没有新训练或推理。adaptive对m8主LPIPS差−0.031164，CI[−0.034927,−0.027676]；但这是开发集策略/条件系统证据，不是已完成新颖性/独立验证。

关键修正：不把有限训练判为架构无效；旧VAR parallel失格warmup回退不入强比较。数字历史receiver_seconds仅PHY译码，不含VAR补全/图像Decoder，完整在线代价是缺口。ARPC/Ada已有大量概念重合，具体单次N/E/失败输出/计费差异不能直接等同新贡献。

R3按既定上限于02:00完成10000训练和校准，属于窗口后补充、不是完成系统评测。当前VAR_COMM没有GPU任务，不干预其他项目，不启动新架构/参数搜索/大网络/复杂机制；报告完成不自动解除暂停。学习方面只留R3一个待验证候选，R2与其余合法checkpoint冻结保留；全部失败/中断记录不删除。

## 当前：按最新用户要求转入收敛报告

本机UTC+8 2026-09-15 01:28：暂停新增架构/自动搜索，保留全部checkpoint与失败记录；先审查过去24小时完成实验和数字VAR相关工作。自适应数字VAR重新作为候选主系统，不再把学习模型胜过它当唯一目标。现有R3到6700/10000，仅按预登记上限继续和保留审计，不追加新训练/搜索/新GPU队列。窗口与边界见`reports/convergence_scope_20260915.md`；当前不把尚未完成的R3系统评测列为已完成。

## 当前：R3首1000完整审计PASS，真实续10000

本机UTC+8 2026-09-15：新q向量控制保持相同2,933,775参数/三次full-grid/两次E/残差能量门控，真实原7000父点及全部10000配对批次已核验。CPU初始函数与GPU波形/RGB差0，图像损失确实回传E/R；4项CPU和实际driver断点测试PASS。00:35首1000全校准/模型Adam/数据/功率/选模审计PASS，同1000校准prediction LPIPS0.198349、residual0.202011，尚非最终结论。

00:38从实际1000模型/Adam续到10000，1099666/1099667/1099668运行中；00:46读取实际2000、global9000，原1000完整历史和已绑定源码/校准未改。报告`reports/wetok_r3_activation_20260915.md`，入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。23方法48300质量行与16模型2560独立计时的协议已登记，**评测驱动和自动收尾仍需实现，尚未排队**。不访问新holdout，不称全网格为next-scale。

## 当前：R2完整结果封存，R3向量归因登记

2026-09-14 23:30，全部46200质量行、48400图像/8800特征CPU审计、3744主/270支持点区间与2400独立计时/70区间完成并复核。MS−普通full-grid主LPIPS+0.002350、95% CI[+0.000808,+0.003969]，DINO更低，无明确速度优势；停止当前固定4/8/16的机械续训，不否定所有层级通信。

残差版主LPIPS0.220543，较普通full-grid0.230345改善，但仍差于数字自适应0.183763/感知Deep0.205291；相对同WeTok数字的高SNRLPIPS仍差。它另增6417参数、两次内部E与1.68758ms RX成本，不能直接宣称残差独立或next-scale贡献。完整报告`reports/wetok_r2_result_20260914.md`。所有旧进程已退出，19项CPU测试PASS。

新入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`：仅登记同参数/回算次数/完整训练历史的q vs y−q向量对照，门控保留同一标量残差能量。尚未实现或启动R3，不从R2最优残差点切换短训，不修改旧权重或哈希。研究仍缺强系统优势、充分机制归因、训练重复与新holdout，不标记完成。

## 当前：R2全部10000训练完成，进入22方法质量评测

2026-09-14 23:09:03完整审计PASS，实际四模型/Adam10000、global17000，原数据/损失/资源不变。single/MS完整校准最优仍是7500；ordinary full-grid和innovation选10000，LPIPS分别0.190455/0.174380。不能将相同10000机会误写为全部使用10000权重。当前quality983552、finisher880742、observer880743，23:12真实提交23/100源；原880740/880741已正常退出，不重启。`reports/wetok_r2_milestone10000_20260914.md`。19项CPU测试PASS，当前无完整R2 development/新holdout结论。

2026-09-14 22:09：四臂R2至8400总更新，880740/880741/880742/880743继续正常运行。7500封存端点已生成完整校准、逐SNR、训练分项和成本四类PNG/PDF及CSV；15项CPU测试PASS，绑定训练/评测源码不变。成本对账1.414581h分项＋0.010964h未细分开销＝1.425545h进程时间，不是GPU利用率积分。逐SNR也保留1 dB部分PSNR回退与MS19 dB微小LPIPS回退，不称全面改善。`reports/wetok_r2_training_curves_20260914.md`。

## 当前：R2的7500审计PASS，四臂实际继续10000

2026-09-14 21:35完成全部1000图校准与独立CPU审计：单次/多尺度state/普通full-grid/innovation的LPIPS为0.192001/0.192695/0.193974/0.176768，分别较5000改善0.005124/0.001984/0.000994/0.007951；是五训练SNR的校准均值，不与development主区间混排。实际模型/Adam7500、global14500、新2500数据/噪声配对、功率与完整校准选模全部通过。

21:43由实际7500模型/Adam继续到10000，trainer **880740** / reviewer **880741**，finisher **880742**与被动observer **880743**已实际运行，21:46保存7600总更新。质量与独立计时实现完成，11项CPU/真实driver断点测试通过；后续22方法46200主行、旧18参考不变，32图15模型2400独立计时行。共享只用于显式质量准入，计时遇竞争等候。报告`reports/wetok_r2_milestone7500_20260914.md`，入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。10000/质量尚未完成，不把排队当成结果；下方769675等均为历史。

## 当前：四臂R2真实续训已启动

2026-09-14 20:09从四个实际5000模型/Adam启动至7500，trainer **769675** / reviewer **785157** / observer **785158**。CPU逐值恢复与真实driver断点测试通过；20:10保存5001，20:26读取实际5500保存点确认四臂Adam5500、global12500、前5000历史及来源不变。20:32已到5800总更新。没有重跑起点全校准、重置Adam或更改loss/lr/N/E。`reports/wetok_r2_activation_20260914.md`；入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。当前没有R2完整校准或development结论，下方“尚未启动”是历史。

## 当前：Grid完整结果已结束，准备匹配充分性续训

2026-09-14 Grid5000的37800质量主行、39600图像行CPU复核和1760条独立计时全部完成。4/8/16 state对普通full-grid state的LPIPS差−0.000230、95% CI[−0.001220,+0.000759]，DINO更低，时延也无可靠增量；不能将原对single的改善直接称为next-scale优势。最佳学习链Joint full-grid innovation LPIPS .231342，仍弱于Deep .205291/数字自适应 .183763。报告`reports/wetok_joint_grid_result_20260914.md`。

下一步不加模块，准备single、multiscale state、普通full-grid state、full-grid innovation四臂从实际5000模型/Adam共同继续至10000，保留相同data/loss/lr/N/E以检查充分训练和适配。四端点CPU核验PASS（同global12000、同末批次/噪声/冻结视觉及Adam5000），**新训练尚未实现/启动，当前无活动训练队列**。入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。以下训练/评测PID均为历史，不能按旧等待状态重启。

## 当前：Joint网格控制已实际训练

2026-09-14 18:29 Grid2500完整校准/审计PASS，普通full-grid .198209、Joint innovation .194304；均选2500，仍只是校准。两臂已从实际末端续5000，trainer650391/reviewer650392，19:05核实4600。新18方法质量/统计与11模型独立计时链路通过18项CPU测试，原100源图像素和33600参考行核验通过；finisher689944/observer689945已排队自动收尾，不能重复启动。共享只用于新质量阶段，速度另行独占测量。`experiments/wetok-joint-grid-controls-r1/docs/evaluation_preparation_20260914.md`。

2026-09-14 17:55首1000全校准/审计PASS：普通full-grid state LPIPS .200014，Joint innovation .202011；同1000机会R-only full-grid .199371，尚无强结论。两臂已从真实1000模型/Adam共同续2500，trainer **612129** / reviewer **612130** / observer **612131**；旧首1000进程均结束。报告`reports/wetok_joint_grid_milestone1000_20260914.md`，设置和配对规则不变。

2026-09-14原Joint5000于16:48完整结束：主1/4/7 dB条件版LPIPS 0.236337，对Joint single差−0.002615、95% CI[−0.003608,−0.001680]，对R-only full-grid差−0.000295且CI跨0；仍落后数字自适应/感知Deep的LPIPS。全部33600主行、配对统计与CPU波形/图像复核完成，原49张回执保留。报告`reports/wetok_joint_result_20260914.md`。旧恢复器/评测/观察器均结束，不再重启。

17:23新两臂真实GPU零更新profile PASS，6项CPU/断点测试通过。17:27启动普通16/16/16 state及既有full-grid innovation的Joint控制，先1000更新；trainer **570837**、reviewer **579794**、观察器 **585355**。实际读取300步保存点核实E/R、Adam和原配对数据，非空进程。当前入口`experiments/wetok-joint-grid-controls-r1/docs/CONTINUE.md`。旧三Joint不重训，不把普通迭代称为next-scale；新开发结果尚未产生。

## 当前：Joint E/R同起点控制已激活

16:25最新核实：第一次恢复已从49张推进至**55/100**。16:19新evaluator480961因另一GPU任务482008再次触发原保护退出，恢复器473779/观察器473780仍在等空闲，未重启训练、未改校验规则。原49张回执及冻结源码复查不变；当前不根据部分样本给最终结论。

2026-09-14 16:01，评测在第50张源图末端发现其它GPU进程455090，按原保护退出；前49张/16464主行已提交且保留，训练5000和校准结果不受影响。16:12恢复监督器**473779**与观察器**473780**启动，资格和31项CPU测试通过，等待GPU空闲后自动用原`--resume`继续，再做统计。未停止其它任务、未改原代码/hash、未重启训练。入口`experiments/wetok-joint-sender-r1/docs/evaluation_recovery_20260914.md`。**旧evaluator443167/finisher348503已退出，下方运行中条目为历史。**

2026-09-14 15:50，三臂5000训练/完整校准/审计PASS，均选中5000。校准LPIPS single .197125、no-history .197853、state .194679；条件版开始优于两个基础版，但仍不及R-only full-grid .187311，不能据校准宣称方法成功。原trainer/reviewer已结束，evaluator **443167**、finisher **348503**、观察器 **357923**正在运行，首张336行已保存。报告`reports/wetok_joint_milestone5000_20260914.md`，配对学习曲线`reports/wetok_joint_calibration_figures_20260914.md`。**下方训练中状态均为历史，不重启原trainer。**

15:10已实际核实3600保存点，3500监控完成；state监控LPIPS有波动，不据子集改配方或选模，仍等5000完整校准与原自动评测。三主进程与只读观察器存活；当前GPU仅有本训练，已观测到的早前并发记录不抹去。

2026-09-14 15:00核实保存3100/5000，trainer/reviewer/finisher仍为348501/348502/348503。运行期间其它GPU任务加入，新增只读观察器357923记录并发与硬件背景；不停止其它任务、不改原训练。最终时延若受并发影响，不作独占速度结论。`experiments/wetok-joint-sender-r1/docs/passive_timing_context_20260914.md`。新增5项观察器测试后，共27项CPU测试通过，原训练与收尾绑定源码均未变。

等待训练期间完成HJSCC作者8个小型源文件的只读协议核查，未下载权重或运行新GPU模型。已定位变长CBR、布局/归一化尺度传递、示例固定10 dB与RGB导出要求；不把未适配的示例冒充3060-use强对照，也不据此删除HJSCC。报告`reports/hjscc_source_protocol_review_20260914.md`。这些是强对照接入准备，不是新性能结果或已完成复现。

2026-09-14 14:32完成Joint2500全校准及审计PASS。选中LPIPS single .197908、no-history .200233（仍选0）、state .198812；相对同机会R-only分别−.001012/0/−.002979，但Joint state仍不及single，不能称尺度结构成立。14:37从实际2500模型/Adam共同续5000，trainer **348501** / reviewer **348502** / finisher **348503**，自动评测/统计已排队，勿再手动启动重复任务。曾检测到其它GPU任务，待空闲才启动，未抢占。报告`reports/wetok_joint_milestone2500_20260914.md`；原13:32及更早条目为历史。

新增自动收尾工具与6项CPU测试，连同原16项共22项通过，训练/评测数学和旧绑定SHA不变。5000完成后保留三Joint、全部六R-only及七系统参考，共33600主行＋600支持点＋900无噪诊断；当前没有新development或holdout结果。实时入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。

2026-09-14 13:32 Joint1000全校准/审计PASS：single/no-history仍选中0，state选中1000，尚无明确整体收益。三臂从实际末端模型/Adam共同续至2500，trainer **274550** / reviewer **274551**，13:43保存1100。`reports/wetok_joint_milestone1000_20260914.md`。旧188608/188609已退出；一次GPU占用拦截后重新核实空闲才启动，未抢占别人。
原RX资料已新增逐SNR/配对图和固定样本原图，Deep离支持点异常单列，72图块验证内容不变。入口`reports/wetok_receiver_figures_20260914.md`。这些不是新推理结果。

2026-09-14 12:51已完成三臂起点完整校准并实际保存Joint step1；复读断点确认三臂E都已更新、Adam/训练历史位置一致，源码绑定通过。仍在首1000里程碑训练，不据首个更新宣布质量改善。

2026-09-14原RX-only六臂5000已完成全部训练、27300主行＋600支持点回放与CPU复核。主区间multiscale innovation .242830未胜single .242495，full-grid .236632有局部改善但仍不及数字自适应/感知Deep的LPIPS；高SNR可胜旧骨干参照，仍弱于同WeTok数字链。停止继续扩展该固定E多尺度候选，保留全部强对照。`reports/wetok_innovation_result_20260914.md`。
12:39从与R-only相同原父点激活三结构Joint E/R，trainer **188608** / reviewer **188609**，先step0完整校准再1000更新。只让通信E也优化，loss/结构/数据噪声/新更新机会/N/E不变；原控制与reference资格已冻结。当前入口`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。未有新Joint性能结论，未访问新holdout。

## 当前：固定共同发送器的接收创新量实验

2026-09-14 11:10恢复训练已保存4100/5000，三个新进程均正常。等待5000及其自动评测/分析，Joint未启动。后续Joint严格评测代码已补齐并通过16项CPU检查，保留全部六个冻结接收器与强系统，不声称所有方法同y；`experiments/wetok-joint-sender-r1/docs/evaluation_preparation_20260914.md`。现有冻结绑定仍通过。

2026-09-14 10:15实际核查：旧三个进程已消失，未完成5000或评测；本机10:10重新启动。已完整核验3000保存点（六模型/Adam/数据噪声/冻结E与源码），10:22恢复原5000流程，trainer **25991** / reviewer **25992** / finisher **25993**，10:26保存3100。不要将旧WAIT状态当成仍在跑。`experiments/wetok-innovation-r1/docs/recovery_20260914.md`，新日志和原现场在`outputs/WETOK-INNOVATION-R1-RECOVERY-20260914/`。未改实验设置或访问新test。

07:28完成2500与全校准审计：多尺度innovation .200449，优于state .201791/prediction .201300，但仍不及single .198920和full-grid .196310。不能称next-scale成立。六臂已从真实2500末端共同继续5000，trainer **3205198** / reviewer **3205199**，07:44保存2700。收尾进程 **3211225** 已排队自动做5000评测/统计/绘图，不再手动启动重复评测。报告`reports/wetok_innovation_milestone2500_20260913.md`。
Joint准备于07:32真实GPU零更新检查PASS，图像梯度到E/R、权重不变、分项梯度/成本已记录；**没有启动Joint训练**，先等当前完整控制及自动评测结果。训练、profile和排队评测源码已绑定SHA，不边跑边改。

新增只读训练审计：7000父点共28000次曝光，但图像阶段20000次曝光只覆盖15197张不同训练图，4803张仅接受过表示监督；R-only续训不更新E。single/no-history末段校准仍改善，conditioned近乎持平，不能统一宣称充分收敛。强Deep为30751643个通信参数、40000次ImageNet微调曝光，另有COCO历史，仍保留为强系统参照而非等参数控制。`reports/wetok_training_sufficiency_audit_20260913.md`。
据此准备`experiments/wetok-joint-sender-r1/`的同起点、同数据/更新机会Joint与R-only控制；7项CPU测试和真实初始化权重核验PASS，新的GPU训练未启动，不抢当前六臂任务。RX图像评测已补去no-history不影响图像的辅助读取，并逐帧核对原输出；17项CPU检查通过，当前训练/校准图没有改变。

06:00首1000更新、完整校准和共享y/固定E/Adam/选模审计PASS。选中校准LPIPS single .198920、no-history .200233（选中0）、state .201941、prediction .201300、innovation .202062、full-grid .199371；创新量尚未体现图像增益，不把初始化偏移修复当成功。已从实际1000末端模型/Adam共同续至2500，trainer **3136686** / watcher **3136687**，原配置与物理预算不变。报告`reports/wetok_innovation_milestone1000_20260913.md`；完整校准/监控/训练曲线已生成。旧3072510/3077070已正常结束。

geometry完整开发评测已完成：新204×30显著改善旧映射，但新geometry条件对无历史LPIPS +.003614、CI[+.002575,+.004700]，DINO也更差；仍未胜强系统LPIPS。不能把共同geometry收益当next-scale成功。完整解读`reports/wetok_geometry_result_review.md`。
基于训练源波形解释误差诊断及ISEC先例核查，新实验只改RX：冻结同一已选中的single发送器和视觉模块，六臂读取逐样本相同y，设置预测特征与全网格创新量控制。CPU/真实GPU无更新检查及首1000训练均已完成，尚无新机制系统成功结论。
入口`experiments/wetok-innovation-r1/docs/CONTINUE.md`，状态`outputs/WETOK-INNOVATION-R1-TRAINING/status.json`。重编码/迭代本身不是原创，冻结E保留输入梯度但不更新其参数。最终27300主行+600支持点评测、配对统计与实际波形/图像CPU核验执行器已补齐；15项CPU测试及441条旧图/21父点信号只读检查通过，未访问新模型development或新holdout。GPU负载约94%并出现间歇温控降频，不提高负载、不改硬件设置；新时延评测增加次序平衡和硬件状态记录。

## Geometry总4500完成，当前推进匹配完整预算总7000

image2500完整校准/审计PASS：204×30三结构LPIPS .203079/.204214/.203916；同预算153×40为.257791/.254603/.256377。改善首先是geometry共同效果，当前条件版未稳定优于同geometry的其它结构；不与不同数据/SNR的development强基线混排。
已从实际4500末端模型/Adam共同续训至总7000=image5000，trainer **2973427** / watcher **2974929**；设置不变，结束后做已准备的21000主行+600支持点匹配开发评测与交互/成本分析。`experiments/wetok-comm-v2-20260912/docs/geometry_continue_7000.md`。

## Geometry同预算校准改善，继续总4500

总3000（表示2000+image1000）已完整校准及审计PASS；204×30的新LPIPS single/no-history/conditioned为.206725/.206953/.207850，对同预算153×40为.259012/.261040/.256377，同时PSNR提高约1.14–1.34dB。这是calibration，不能与另一批development强基线混排，也尚未证明新geometry的条件增量。
已从实际总3000模型/Adam共同继续至总4500，trainer **2941742** / watcher **2943268**，配置和N/E不变。决定`experiments/wetok-comm-v2-20260912/docs/geometry_continue_4500.md`。
新geometry最终评测/统计入口已实现：两组共六模型同场重测、四强参考、21000主行+600支持点、1008配对+48交互区间。34项CPU/合成整链测试及441条旧图PSNR/配对复核通过，正式新模型development GPU评测仍未执行。

## 本机UTC+8 2026-09-13：接口5000完成，转入受控geometry训练

原接口5000开发评测完成：continuous条件主1/4/7 LPIPS .279289，对无历史差-.007812、95% CI[-.009552,-.006112]；PSNR/DINO配对也支持改善，但仍输数字自适应.183763和感知Deep.205291，noiseless约.190仍远离native约.0868。不是系统成功；报告`reports/wetok_interface_milestone5000_2026-09-13.md`。
新204×30三结构已在00:47启动，trainer **2902777** / watcher **2907659**，目标先总3000(表示2000+image1000)，再按证据到同控制image5000。当前已完成2000表示并fresh Adam重置，正在image0全校准。保持6120实坐标、连续接口、视觉/loss/SNR不变。
153×40原控制的真实初始化、全部7000步数据/噪声、阶段优化器与选择机会审计通过，不重复训练它们。初始化2线程曾因QR数值差被严格拒绝，已恢复历史8线程而完全匹配，没有改SHA或阈值。31项CPU检查及真实GPU梯度/功率profile通过，新geometry没有开发集结论。
新入口`experiments/wetok-comm-v2-20260912/docs/geometry_execution.md`；当前状态`outputs/WETOK-GEOMETRY-20260913-TRAINING/status.json`。独立holdout和训练重复仍未完成，研究目标active。

## 2026-09-12：v2持续研发已激活，WeTok A0/A1三臂正式训练

21:36九臂仍向5000运行，已保存3100；当前训练代码与资源不变。只准备了一个后续发送geometry候选153×40→204×30（同6120实坐标），旧控制参数/前向完全一致、新行正交构造及CPU图像梯度检查通过，共30项测试。候选参数2927358对控制2895944、计算也不同，尚未GPU profile/训练或测图像收益，不因准备好就自动切换。详见`experiments/wetok-comm-v2-20260912/docs/geometry_candidate_preparation.md`。

21:12向5000续训的实际2700断点审计PASS（global data4700、Adam2700，存储独立/采样配对），进程继续。2500结果另有13/19dB条件相对无历史的局部LPIPS改善-.00313/-.00564，但PSNR/DINO区间跨0且仍未赢强系统；不把主区间负结果说成所有SNR无效，也不据此改主评价区间。

**20:57已进入总追加5000的共同续训，2500开发结果为接口改善但非结构/系统成功。** 新trainer **2730582**、watcher **2733118**，从实际2500模型/Adam继续原配方和账本；未改网络/packing/功率或增加数据。决定`experiments/wetok-comm-v2-20260912/docs/interface_continuation_5000.md`。
20:47完成33600主无线行、600 Deep支持点及3024+162配对区间。主1/4/7 continuous条件LPIPS .296130、无历史 .290503，差+.005627、95% CI[+.003947,+.007379]，条件版更差；无历史也仍明显输数字自适应.183763和感知Deep.205291。未把接口收益当next-scale胜利。报告`reports/wetok_interface_milestone2500_2026-09-12.md`。
独立统计复算3024区间最大误差<4e-15，12600个选中hard与原父点图像hash比较一致；这是统计/文件审计，不是第二次神经推理。600支持点GPU重放与旧图像素差0，未用5/6异常制造优势。新holdout仍未访问。

19:21的2000固定监控重新改善：continuous三结构LPIPS .256390/.254504/.253020，较1000同源分别-.001166/-.004966/-.002473。此前1500回退不能视为持续恶化的定论；保持九臂到2500完整校准，不改配方、不从监控选模。`experiments/wetok-comm-v2-20260912/docs/interface_monitor2000_review.md`。

19:00补充CPU训练梯度监测：1600点、4训练源×3个continuous，MSE/LPIPS局部2/12反向，但注册总梯度对LPIPS均正向；范数比不构成回退因果证明，Adam方向亦未据此假定。未更新参数/占GPU/访问新test，继续原2500检查点。结果`experiments/wetok-comm-v2-20260912/docs/objective_probe_result_1600.md`。

**18:49：1500固定监控的continuous LPIPS回退，不能宣称训练继续单调改善。** same-source的1000→1500变化为single +.003228、无历史+.008029、条件+.005612；当前值.260784/.267499/.261105。既有最佳不变，仍保持九臂配置等机会至2500全校准；若非有限按既有安全规则停并保留现场。`experiments/wetok-comm-v2-20260912/docs/interface_monitor1500_review.md`。
CPU整链评测/源图恢复测试已加入，修复续跑后的预热计时，并累计session耗时、检查外来GPU竞争；27项测试通过，训练源码/参数和物理账本未改。新模型development及新holdout仍未执行。

18:24续训恢复审计PASS：九臂共同保存追加1100（global data3100、Adam1100），模型/优化器存储独立、数据/增强/SNR/噪声完全配对；当前仍向2500目标推进。见`experiments/wetok-comm-v2-20260912/docs/interface_resumed_checkpoint_audit.json`。

**18:16：1000完整检查点已结束并通过审计，九臂继续到总追加2500。** 三个continuous全1000图/五SNR校准LPIPS为0.259012/0.261040/0.256377，选择追加1000；六个hard仍选择合法追加0，不能当作新条件结构成功。条件continuous对无历史的校准差-0.004663，不是独立development结论。
从实际1000末端模型与完整Adam继续，不从最佳点回退。新trainer **2614845**、watcher **2616385**；保持LR/loss/接口/网络/N/E/数据与噪声，追加1500到2500，1500/2000监控、2500全校准。决定`experiments/wetok-comm-v2-20260912/docs/interface_continuation_2500.md`。
最终评测补上600条固定支持Deep5/6单列：原始曲线不改，NN条件4/7而物理SNR5/6。600旧图PSNR/预算检查、4次实际CPU模型重放通过，最大像素差3.40e-6；25项CPU测试通过，GPU新模型评测仍未执行。另完成16训练源的只读TX占用诊断，记录线性skip/实际发射信号方向问题，但未据此改packing或宣称容量界。

**17:09首500追加步监控出现连续接口改善信号，训练继续。** 固定100 calibration图×五SNR，continuous三结构LPIPS分别0.257093/0.257975/0.257031；native有界-ST约0.301–0.306，identity约0.330–0.335。原native同源起点约0.287–0.288；不把这批监控与development强基线混排。
条件continuous对无历史差仅约-0.000944，对single-pass约-0.000062，尚未确认next-scale独立增量。保持九臂等机会，继续当前1000检查点的完整校准，不按monitor选模或改loss。细节`experiments/wetok-comm-v2-20260912/docs/interface_monitor500_review.md`；23项CPU测试及441条旧参考复核通过，新模型最终development评测仍未执行。

**16:44接口九臂已完成并审计共同100追加更新。** global data step2100、fresh Adam逐参数step100，九臂模型/Adam状态存储独立、数据/增强/SNR/噪声完全配对，最大能量误差5.96e-7；原始训练/profile源码未变化。训练及watcher真实存活，不根据100步loss宣称有效。
新development执行器`evaluate_interfaces.py`和统计`analyze_interfaces.py`已补齐：continuous保持连续解码，16方法/33600行、63组/3024个配对区间、PSNR/LPIPS/DINO图表；旧7方法原图只读引用。CPU完整合成网格与441条真实旧对照复核通过，尚未执行新模型development评测，不能把准备完成当方法成功。详见`experiments/wetok-comm-v2-20260912/docs/interface_execution.md`。

**16:06起接口九臂流程实际运行。** 三结构×hard_identity/hard_bounded/continuous_mean；19项CPU测试和真实九臂零更新profile通过。两个hard前向像素相同、train/eval一致、image梯度到E/D、视觉权重冻结；此时只是工程通过，不是模型收益。
trainer **2518473**、校准监督器 **2518976**，实际日志`interface_train_0001000_launch2.log`。首1000流程从完整起点校准开始，profile估计九臂更新主体0.944小时，完整校准另计；首1000只是检查点，初步目标5000再按走势调整。
实时`outputs/WETOK-COMM-INTERFACE-20260912-TRAINING/status.json`；详细恢复和下一步`experiments/wetok-comm-v2-20260912/docs/interface_execution.md`。当前不访问development，首里程碑自动完成CPU配对/选模审计和calibration曲线。最终新接口development评测入口仍需实现，不冒用旧hard-only输出。

**15:52更新：bit权重六臂已探索性提前停止，不是完成5000。** 固定100张calibration monitor在追加1000时，原joint/bit权重1的LPIPS为single_pass 0.68509/0.83198、无历史0.72811/0.90610、条件版0.53653/0.76036，均继续恶化。保留六臂共同global6000的完整模型/Adam；SIGINT时已执行至共同1113追加步，未提交部分不纳入比较。没有新development结论，不声称所有bit权重无效。
自有trainer与监督器均已退出，不触碰其它任务。回执`outputs/WETOK-COMM-BIT-SUPPORT-20260912-TRAINING/exploratory_early_stop/termination.json`；决定`experiments/wetok-comm-v2-20260912/docs/bit_support_early_stop.md`。原监督器中断状态保留，不伪造完成。
接续准备`configs/interface_study.yaml`：从真实2000模型出发、各臂统一fresh Adam，三结构×native identity-ST/native有界均值-ST/真正连续接口，冻结视觉/网络/物理预算/原joint权重。下一步先做真实接口与梯度profile再配对训练；尚无新接口收益结果。

以下bit权重条目是此前启动记录，已由上述停止决定取代：

**bit监督修订对照已启动。** 2026-09-12 15:06起，三种结构×原joint/bit-BCE权重1，共六臂从同一5000模型/Adam续训；父点SHA已锁定，optimizer深拷贝，唯一变化为bit权重0.01→1。首个追加5000更新仍是研究里程碑。
已验证六臂各250追加更新、global/Adam step5250；各模式内模型/Adam初值相同，所有batch/SNR/噪声指纹一致，源代码未漂移。训练PID2473858；评测/分析监督器PID2483685。
新结果将同时比较同预算原joint、原实验2000最佳、同配方条件/无历史、以及强数字与Deep参考。16项CPU测试及378条历史最佳/旧对照CPU回放通过；尚无本轮质量增益结论。
实时状态`outputs/WETOK-COMM-BIT-SUPPORT-20260912-TRAINING/status.json`，日志`outputs/WETOK-COMM-V2-20260912-LOGS/bit_support_*`；代码入口`experiments/wetok-comm-v2-20260912/scripts/train_bit_support.py`。

**首5000里程碑已完成，研究继续。** 三臂完整校准均选择2000步，joint阶段到5000反而恶化LPIPS/bit恢复。主1/4/7 dB条件版LPIPS0.328912，对无历史差+0.000860、95% CI跨0，未证实条件增量，也未赢旧强系统。
noiseless仍约10.1%bit错误/LPIPS0.210，而同图native参考0.08681；19dB噪声只再增加很少损失。按v2优先修订通信恢复目标而不是先改功率。
下一项已登记`configs/bit_support.yaml`：从共同5000模型与Adam起点，仅改joint bit BCE权重0.01→1，原配方/修订配方在三结构内等新增训练配对，旧2000最佳另保留。详见`reports/wetok_a0a1_milestone5000_2026-09-12.md`；修订尚未产出结果，不将首里程碑当研究完成。

### 初始启动记录

用户指定根目录`next_scale_communication_research_brief_v2.md`为持续执行任务书（SHA 27648fbc73485deb74a0695f3f7840bd01c561c1b0b4f6f67284fefcfb6d2e3f）。旧研发时间/金额/轮数上限取消，但只使用已授权本机GPU，不租新服务、不动其他用户任务，通信公平与数据隔离不变。
新实验`experiments/wetok-comm-v2-20260912/`。选定WeTok是确定性CNN Encoder/Decoder，原生32×16×16±1量化特征、4组8-bit；不套VAR的4096类CE、不另加生成噪声，也不把分组当语义尺度。
三臂single_pass／multiscale_no_history／multiscale_conditioned，每臂2895944参数、相同初值和配对训练。全部读取完整y，不称逐包渐进；中间4/8状态是连续通信估计，最终16网格为native硬符号。新增模型不需要类别/动态mode，0 header+3060 data，总能量6120；与旧68+2992分账同N/E比较。
20000训练图×两种真实翻转、1000校准及100 development的原生组索引缓存完成；100图native重建回放像素差0；校准/development源像素精确去重检查无重叠。实际梯度通过冻结Decoder到E/D，全部冻结权重不变，表示阶段不调用Decoder。
2026-09-12 13:00启动三臂训练，PID2380048；已完成2000表示恢复并进入图像联合训练。首个5000步只是评估里程碑，后续按固定校准走势继续/调度，不自动因达到上限停研究。
8PSK+soft-bit FEC对应数字参考已在CPU生成：8192payload+CRC16+tail6、9180coded bits=3060 8PSK符号，恒定每复符号能量2，CRC失败候选仍渲染；这不是最优数字系统宣称。
首里程碑自动评测/统计监督器PID2400192。结果将比较同WeTok基线/无历史控制、WeTok数字、原数字固定m8/自适应与感知Deep，统一PSNR/SSIM/LPIPS/DINO与严重失真事件；不访问新holdout，不采用不合格warmup fallback。
实验协议`experiments/wetok-comm-v2-20260912/docs/protocol.md`；实时训练`outputs/WETOK-COMM-A0A1-20260912-TRAINING/status.json`，日志`outputs/WETOK-COMM-V2-20260912-LOGS/`。当前尚无最终通信增益结论。

## 2026-09-12：有界骨干选型已完成，暂不正式迁移到XQ；WeTok完整codec参照为正面结果

旧2×2流水线于01:43:30自然完成；自动队列确认空闲后，选型评估01:46:16启动、01:49:15完成（均UTC+8），GPU已释放，新增训练0步。
同100张development，四模型1800正式逐图记录全部完成：旧官方full PSNR22.8359/LPIPS0.10575；fidelity23.8304/0.12089；XQ配套full22.1412/0.13194@6864rawbits；WeTok23.7393/0.08681@8192rawbits。
WeTok相对旧官方8160bits，源表示仅多32bits，PSNR+0.9034dB（95%CI[0.7915,1.0171]，95/100图提升）、LPIPS-17.91%（100/100图提升）。相对fidelity，PSNR均值低0.0911dB、CI跨零，LPIPS改善28.19%；不是重复更平滑工作点。但WeTok仅测完整codec，未证明next-scale支持。
XQ直接prefix有真实收益，但补全没有形成支持迁移的证据：2424bits补全19.7391/0.20490，对旧3060bits补全19.6916/0.17714，PSNR近似但LPIPS恶化15.67%；3960bits补全20.9408/0.16594，对旧5088bits21.2437/0.13590也有感知退化。XQ3960bits优于旧3060bits，但多29.41%源bits，非等预算证据。
XQ2424bits补全相对自身直接解码PSNR反降0.2783dB；3960bits仅增0.0792dB。作者三seed采样未扭转结果。不同自然预算不能当完全匹配，也不能据此断言XQ所有码率都差；本轮结论是**不足以支持付出正式迁移成本**。
独立CPU产物/统计复核PASS：1836个含smoke输出tensor哈希、67份源码/config、7模型资产一致，20组条件统计/80组配对指标及10000次按图bootstrap复算差0；没有第二遍GPU推理/指标网络重算。XQ配套tokenizer与独立checkpoint EMA552/552张量完全相同，非配错权重。
旧基线重算与历史非bit-exact（包括旧p9直接个别图1.6552dB差异），未受控归因；本轮选型用本次统一重算基线，不混历史均值。DINO暴露、类别side信息、无信道/非3060complex uses边界保持。
**决策：本轮不正式迁移到这对XQ；WeTok仅保留高保真完整codec参照；尚未选出满足全部要求的next-scale替代。不是停止VAR通信研究，不自动续训或增加其他模型。**
复核报告：`reports/backbone_selection_review_2026-09-12.md`；自动结果、原始float、图表及审计：`outputs/ei-liulu-xqvar-eval-20260912-v1/`。

## 历史排队记录：2026-09-12凌晨选型准备（下述为当时状态）

用户先指定公司开发机，SSH两次认证前TCP超时，未操作远端文件/进程；随后明确有认证，改为本机既有GPU任务结束后运行。
未向任何GPU任务发送停止信号；原2×2仍按原计划自然运行，不在train/evaluate间隙抢卡。之后不自动新增原版VAR训练/模块设计。
新隔离入口 `experiments/backbone-eval-20260912/`，协议 `docs/protocol.md`，旧项目/checkpoint/基线保持只读。
已核对官方XQ源码配置与配对权重清单；双PQ的自然prefix8/9/full静态raw为2424/3960/6864bits（必须进一步实际索引核验）；WeTok四256码本、16×16对应8192bits，指定公开checkpoint已定位。
新候选权重单流限速下载、私有venv不改变当前训练环境。原100图清单及官方/fidelity模型定位完成；全重建四臂，XQ两个自然prefix直接/argmax及预定三seed作者采样敏感性，无训练/信道试验。
CPU-only队列安全测试41项通过。01:02:14启动监视器PID1856483，命名 `ei-liulu-xqvar-eval-20260912-v1`；实际队列当前 `WAITING_ASSETS` 且旧supervisor仍存活，**新模型GPU评测尚未开始**。
新队列状态 `outputs/ei-liulu-xqvar-eval-20260912-v1/queue-status.json`；资产完成需源码/配置/模型校验receipt，旧整条pipeline终止+3次空闲才启动。遇外来GPU任务只退出自己的评估进程组。
没有新PSNR/LPIPS结果、没有迁移决定；XQ DINO训练暴露明确披露，不用论文指标代替同协议测量。接下来只执行这一次有界选型，不自动重训通信。
01:22代码/配置/SHA锁定完成；78项CPU-only测试通过（含tiny随机模型API/补全审计，不是预训练质量）；独立XQ权重已通过LFS SHA和CPU元数据读取。处理了作者checkpoint的YAML元数据兼容性，只用精确SHA限定的受限loader；未退回不受限pickle。CPU准备任务PID1869580串行等待后续资产，monitor PID1856483仍在等待。详见`reports/backbone_selection_queue_2026-09-12.md`。

## 2026-09-12：通信骨干2×2已完成，未证实增强骨干的稳定增益

本轮01:43:30（UTC+8）训练/16800行development评测/统计全部完成，四臂各20000更新、80000曝光；三阶段产物和源代码SHA复核通过，GPU已释放，不自动追加训练。
两个next-scale均选总16000步；增强对小骨干LPIPS差-0.001586，95% CI [-0.004106,+0.000725]，PSNR/DINO差的区间也跨0。增强LPIPS 0.236827，仍差于数字m8 0.214927。
两条parallel没有满足共同PSNR约束的候选，按协议回退到10000步warmup端点；不能把主表相对这些回退项的巨大差值当作B配方下的next-scale结构增益。旧B继续只读参考，增强版不自动升为主模型。
解读与全部边界：`reports/token_backbone_result_2026-09-12.md`；原始结果在新代码仓库`outputs/VAR-TOKEN-BACKBONE-20260911-ANALYSIS/`。整个正式训练均为microbatch2，测速4未正式启用。本节不更改上方另行授权的视觉骨干选型任务。

### 本轮历史启动记录（已结束）

用户给定的token域Transformer方案已实现：增强192维/6头/4发送+4共享接收，实际5139662参数；小骨干128维/2层并匹配SNR调制，1559054参数。旧B固定只读；不换视觉模型、不改m8/3060预算、不加功率/全尺度/FEC。
四个严格比较臂统一从头10000步warmup+10000步B、每臂80000曝光；同一家族parallel/next-scale初始化相同，fresh Adam、teacher=0、数据/噪声/增强配对。旧B仅工程参考，训练历史不混排。
CPU检查及16800行合成统计测试通过（合成数据不是实验结果）；真实GPU四臂hard输出与原接收算法差0，image/state梯度到E/D非零，prefix阶段不构建后缀/RGB，frozen权重未变，profile新增更新0。
实测四臂训练主体约8.17 GPU小时，计划校准1.85小时，合计10.02小时，另计optimizer/I/O/最终评测；不是精确工期。
代码和配置在`../var-next-scale-comm/`，协议`reports/token_backbone_protocol_2026-09-11.md`，入口`scripts/run_token_backbone.py`。当前没有新架构收益结果，不按参数规模推断必胜。
14:44首次启动仅遇到空历史分支目录未创建，新增更新0次；目录创建逻辑已修复并补回归检查，失败现场保留。14:46以`configs/token_backbone_20260911_r2.local.yaml`重启，只改训练输出到`TRAINING-R2`，科学设置不变。
14:53四臂各250更新已共同保存，Adam均step250，数据/噪声/增强指纹相同，loss/梯度有限、teacher=0，最大功率误差5.96e-7，旧B及新训练源码snapshot均核验未变。监督器PID1408947；流水线完成后自动评测/统计，异常停止。
启动说明`reports/token_backbone_launch_2026-09-11.md`；最新步数见新仓库`outputs/VAR-TOKEN-BACKBONE-20260911-TRAINING-R2/status.json`。主体约10小时，粗估2026-09-12凌晨完成，不是时刻保证。

## 2026-09-11 13:42：本机m8 A/B训练、开发评测和统计全部完成

`VAR-M8-LOCAL-20260911`于13:42:19（UTC+8）完成全部流水线；训练13:34:52结束，GPU已释放，不自动追加训练或新变量。
双方各新增10000更新/40000图像曝光；按预定完整校准及PSNR约束选A step2000、B step6000，并非双方只训练到选中步数。
主1/4/7 dB、100张development×3噪声：B相对A的PSNR +0.31210 dB、LPIPS -0.090893（相对改善27.60%）、DINO +0.11685，逐项配对95%区间支持改善。
B的LPIPS 0.238479，仍差于固定数字m8 0.214927、数字自适应0.183763和感知DeepJSCC 0.205291；不能宣布强系统胜出。主区间原起点对数字m8的LPIPS差距追回79.59%，仅为描述性比例。
三阶段产物及源码SHA复核通过，12600行评测无重复键，全部3060 uses；216组配对统计复算差0。没有重新做全量独立模型推理/图像指标计算，不访问新test。
结果摘要`reports/m8_local_result_2026-09-11.md`；完整报告`../var-next-scale-comm/reports/prefix_refinement_remote_result.md`，图表和CSV位于新运行ANALYSIS/EVALUATION目录。

## 历史启动记录：2026-09-11首个250步共同断点通过

用户明确表示本机显卡空出并要求开始；此前只准备/暂停限制对本次固定m8实验解除，不扩展至全尺度、功率分配或parallel重训。
RTX 4090 D已确认空闲；原13类资产及官方VAR源码SHA校验通过，不复制大型资产、不改历史结果。
新配置`../var-next-scale-comm/configs/m8_local_20260911.local.yaml`，各新增10000更新，B前2000 prefix/后8000 joint，科学设置与已准备协议一致。
新运行根名`VAR-M8-LOCAL-20260911`；无更新GPU profile/工程检查通过，1000图×5 SNR起点校准相对旧结果MSE/LPIPS差均为0。
2026-09-11 10:30:38（UTC+8）进入正式更新；10:33:23双方各新增250步、各1000图曝光并保存共同断点，Adam step均10250。首250个batch指纹逐一配对，loss/梯度有限、teacher=0，功率误差最大4.77e-7，训练源码snapshot未变。
后台依次执行训练→development评测→分析，异常停止；监督器PID 1113754。日志`../var-next-scale-comm/outputs/VAR-M8-LOCAL-20260911-PIPELINE/train.log`，实时更新数以TRAINING/status.json及共同resume为准。
实测两分支训练加计划校准约2.98 GPU小时，不含optimizer.step/I/O/最终评测，不是完成时刻保证；尚无新方案最终效果结论。详情`reports/m8_local_launch_2026-09-11.md`。以下准备阶段记录为历史状态。

## 当前准备：固定m8配对A/B，后续在其他服务器训练

按用户最新要求，论文主线保持有限带宽/总能量下的next-scale学习通信；当前先准备现有m8映射的训练修订，不同步改全尺度/功率/FEC/网络，也不重训parallel。
准备入口`../var-next-scale-comm/docs/REMOTE_M8_HANDOFF.md`，配置`configs/prefix_refinement_remote.yaml`；同epoch2/Adam状态，每分支新增10000更新初始计划，B为2000+8000。
每500步固定100图监控、预定6个步骤完整1000图选模，监控不选模；真实header、自身历史、硬选择/ST和冻结骨干不变。
资产清单587文件、8.025 GiB已校验，不复制大型资产；核心/配对采样/状态目标/optimizer副本/CLI防误启动CPU检查通过。
正式训练、预训练模型推理、GPU测速均未在本机执行；新服务器须先验证环境和profile/step0，然后由用户显式开训。准备不等于新训练结果。

## 2026-09-10：研究方向更新，不恢复旧m8续训

用户将主线调整为：受带宽、总能量和合法CSI约束的多尺度通信编码、保护分配与next-scale条件接收联合训练。
全尺度信息可参与发送，旧m8和数字自适应继续作为参考/强基线；不把换tokenizer的共同提升当通信贡献。
首版固定组长度、学习SNR条件保护，无反馈、不强制逐尺度CRC；补充噪声固定、等功率/等能量区别和TX/RX信息边界。
方案及小范围原始资料核查见`reports/multiscale_joint_direction_2026-09-10.md`；代码整理仓库的研究说明和协作规则已同步。
**本轮只更新方向文档，不改训练源码或配置；未实现全尺度链、未租卡、未测速、未下载checkpoint、未训练/推理。**旧两阶段实验仍暂停，不自动作为新方向起点。

## 专用资料：VAR原理、通信动机和四个研究问题

用户给出的第二至四部分已整理到`../var-next-scale-comm/outputs/var_report_core_materials_2026-09-10/`。
含修订正文、18张逐段配图、完整五方法主表、CSV及24条口径核对；无PPT、无新增训练/推理。
重点区分源payload与信道次数、Deep的0+3060预算、不同旧流程的m8数值，以及联合训练增量与固定y接收器消融。
已保留全部七SNR原记录和Deep5/6补充，两阶段训练仍暂停。原实验结果未改写。

## 图片素材入口：用户要求只整理对比图，不继续做PPT

可用图片集中在`../var-next-scale-comm/outputs/figure_library_2026-09-10/`，双击其中`index.html`浏览。
检索11个相关项目根，整理124张主图、14张精选：完整猫图m1–m10、当前学习链与强对照、数字尺度/SNR和旧跨数据集图片。
中止产物、无效管线和NOT RUN占位图不纳入可用图；实际历史负结果另设分类。来源与使用边界随图附带。
这只是离线素材整理，不运行模型、不新增训练；两阶段方案仍暂停。代码整理版的`reports/figure_library_2026-09-10.md`记录详情。

## 2026-09-08：两阶段prefix恢复（用户要求暂停；新增训练0步）

用户随后要求“先暂停一下”，先对齐VAR通信研究目标。已检查本轮队列和孤立子进程，当前没有存活的本轮实验进程，
没有训练resume或完成receipt；运行状态已记为PAUSED_BY_USER。未经用户新的明确授权，不重新启动、恢复队列或推进训练。

用户确认两阶段和公平续训对照并要求开始。协议`reports/prefix_refinement_protocol_2026-09-08.md`，配置`configs/prefix_refinement.yaml`。
两分支同next-scale epoch2模型及10000步Adam状态；各新增10000更新，B前2000步CE+0.1state，后8000步image+0.01CE+0.01state。
固定3e-5、全程自身历史、m8/3060预算；每1000步完整1k校准，LPIPS选模且每SNR PSNR相对起点退化不超过0.2 dB。
新实现仅记录和监督前八尺度累计state，不改旧模型/源码；prefix-only不生成后缀/RGB。训练、评测和统计报告入口已实现。
CPU自检已验证原输出/原loss/梯度/官方state目标差0、state梯度到E/D非零、prefix-only仅8次VAR尺度前向/0次图像Decoder。
额外CPU协议检查验证阶段边界、PSNR guard、header屏蔽、optimizer值相同且不共享存储、168组配对统计聚合。它们不是新增训练结果。
当前GPU被另一账户常驻服务占用，禁止擅自停止；正式GPU训练开始前等待至少11000 MiB可用显存。
自动入口`bash scripts/run_prefix_refinement.sh`，具有重复启动锁和断点续接，训练完成后接开发评测及报告。
准确运行状态见`outputs/VAR-PREFIX-REFINEMENT-TRAIN-001/status.json`；日志`outputs/logs/prefix_refinement_pipeline.log`。
不能把已实现、CPU自检通过或GPU排队记成完成10000次更新；正式结果尚不存在。

## 现有训练结果补充：校准端点与固定m8数字对照（复核完成）

仅分析已冻结产物，没有训练/模型推理、改checkpoint或新数据。校准只有epoch1/2两个端点，不是密集学习曲线。
next-scale第二轮校准MSE下降0.8986%，但LPIPS从0.29041462变为0.29998300（恶化3.2947%，五SNR都变差）；
MSE+0.01LPIPS目标仅均值改善0.3143%，新增事后配对区间跨零。原epoch2选择符合协议，未擅自替换为epoch1。
100图开发对照双方固定m8/3060：next在全部七SNR的LPIPS、DINO落后数字m8；4—19 dB三指标均落后，1 dB只有PSNR占优。
19 dB next/digital LPIPS=0.262843/0.177141；不能把差距只归因于数字自适应发m9。next对parallel的逐SNR LPIPS增量仍成立。
核对20000行校准、6300行相关开发数据及72组配对结果（最大差0）；原源码、CSV、配置与选择记录SHA不变。
报告：`reports/prefix_calibration_fixed_m8_review.md`；两张新图及配对表：`outputs/VAR-PREFIX-CALIBRATION-M8-REVIEW-001/`。
本项只读分析任务完成，不据两个校准点宣称收敛，也不自动续训或调整loss/架构。

## 2026-09-07：固定m8全局prefix-JSCC（训练、评测与复核完成，最新）

用户改为真正训练通信Encoder/Decoder，VQ/VAR/官方image decoder全部冻结；不要求无训练方法先成功。
已完成20k训练/1k校准prefix缓存，包含实际重新tokenize的水平翻转；与100图development无精确内容交叉，类别覆盖1000。
图像MSE、LPIPS对E/D梯度和通过冻结VAR的suffix→prefix梯度均通过；训练/评测hard forward像素差0。
两版本通信参数完全相同，均1,508,750；仅global E/D及融合模块训练，DINO不参与loss/选择。
按`reports/learned_prefix_training_protocol_2026-09-07.md`完成各2epoch/40k曝光、10000更新，多SNR、68+2992=3060。
最后6000更新全部self-history；VAE/VAR/LPIPS权重前后完全相同。训练与选择审计通过，两个版本均由校准选择epoch2。
独立校准image MSE+0.01LPIPS目标：parallel 0.02289528，next_scale 0.02077703；不是开发集结果。
100图×7SNR×3噪声、5方法共10500行完成。预定1/4/7 dB主区间，next-scale相对同参数parallel：
PSNR +0.71616 dB，LPIPS -0.03235360（相对8.921%，95% CI=[-0.04079135,-0.02418770]），DINO +0.06787497，三项CI均支持改善。
但当前学习链仍落后于原数字自适应三指标；相对感知DeepJSCC，DINO更高而PSNR/LPIPS更差，不能宣称整体胜出。
这证明了本实现中next-scale参与联合训练的增量，不是宣称学习通信整体已经优于强系统。

两项审计通过：预算/选择/冻结权重，10500行、7348图重算指标、4200个hard-prefix官方重建像素差0，80个学习RX/40个Deep/200个错配body重放一致。
发现旧Deep高频Fourier SNR条件在未训练5/6 dB异常；补充固定最近训练条件（5→4、6→7）且真实噪声不变，避免把基线异常当优势。
主1/4/7结论不变，补充对照不是事前主实验的一部分。错配body显著恶化图像，说明模型利用了源相关数据。
完整报告：`reports/learned_prefix_result_2026-09-07.md`。
本轮固定训练任务已完成，不重启旧state纠正，不擅自改loss/网络或续训；保留训练型next-scale增量证据，原数字自适应仍为主基线。

## 2026-09-07：整帧末尺度辅助搜索（完成，停止本候选，最新）

按用户“开始吧”完成4/5/6/7 dB、100图×3噪声、8臂共9600行；发送/FEC/3060预算、模型及5.5 dB阈值未变。
28组穷举、16个边界、真实码长及神经自检通过；4/7 dB的旧m8/m9共1200条接收记录完全复现。
主区间5/6 dB，VAR−列表65的ΔLPIPS=+0.00005630，95% CI=[-0.00040866,+0.00059500]，无独立优势。
6 dB完整接收API补测（20图/一噪声）约152.84 ms vs列表65的9.90 ms；VAR未把预定达标SNR从7降到6。

前缀瓶颈：4/5 dB真r1:8覆盖0；6 dB为65/296，7 dB为86/100。被覆盖的151帧VAR全部恢复正确整帧，
但不能补回不在4候选中的前缀。固定m9诊断5 dB有1次CRC误接受；冻结adaptive_var该点发m8，未受此样本影响。
大列表效果更好但有44次误接受，且710次受候选/堆上限限制，不能包装成严格等算力或无风险升级。

独立审计通过：3,222,659条候选、3984个重算概率表、1240张图全部重算指标，638张新图旧实现像素差0。
480次完整接收API补测均复现主输出；全模型冻结，未训练/访问新正式测试。
另120次波形到RGB连续调用像素差0；6 dB VAR/list65约211.97/69.38 ms，完整图像链路约3.1倍，不是API的15.4倍。
报告：`reports/whole_frame_prior_result_2026-09-07.md`。
最终`STOP_THIS_FINITE_PREFIX_VAR_RECEIVER`：停止当前4前缀/末尺度候选，不扩大L、训练或搜索，原整帧自适应保留。
本轮结束，不代表停止VAR通信主线；强整帧熵编码未实施，不作总体或新颖性优势声明。

### 此前只读核查

用户提供新方向：整帧FEC/发送预算不动，ML数据CRC失败才启用有限r1:8假设下的r9先验辅助搜索。
这不是继续选择上轮的两张输出。已核查现有600条正确m9接收记录：同口径clean LPIPS=0.13589562、
PSNR=21.24372801；整帧7 dB仅差0.00157537 LPIPS / 0.07404158 dB，支持把关注点移到4—7 dB交界。
历史冻结候选阈值5.5 dB意味着4/5/6/7 dB原模式为m8/m8/m9/m9；不需要根据新目标结果重挑阈值。

技术边界及实施顺序：`reports/whole_frame_prior_review_2026-09-07.md`。
目前仅只读产物/代码及历史文献记录核查；本轮外网访问被拒绝，未完成原文核实。
此前没有新信道搜索、神经推理、训练或资源调整；随后获得实施授权，现已按预注册完成，结果见本节开头。
上轮二候选输出选择路线的停止决定保持不变。

## 2026-09-07：7 dB CRC失败输出回放（已完成）

用户限定只复用100图、7 dB、原三噪声和固定m9，回放首个CRC失败硬候选是否应仅用于图像。
按`reports/crc_failure_replay_preregistration_2026-09-07.md`完成900条分组记录回放及300条整帧对照，
共3000条输出/诊断行，仅新生成219张独特图。没有训练、换FEC、搜索资源或新增信道候选，可信前缀不变。

VAR的33次失败输出全部改善，LPIPS从0.15322436降至0.14302625，追回原整帧差距的64.74%。
但整帧m9仍为0.13747099，VAR retain−整帧为+0.00555526，95% CI=[+0.00281896,+0.00866255]。
VAR二候选oracle与固定retain相同，也不能追平。4次基础失败占VAR追回量52.47%，说明丢弃有用候选的代价很大但非唯一问题。
双方均retain后，VAR比ML低0.02282401 LPIPS，比熵编码低0.00798687，两个配对CI均完全低于0；仅限此7 dB开发回放。

独立审计通过：1200条可信接收轨迹与波形核查、219张新图全部用旧实现复渲染像素差0、580张图全部重算指标；四模型冻结。
完整总体/失败分层表：`reports/crc_failure_replay_result_2026-09-07.md`。
决定`STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9`：停止当前两张输出间的选择路线，不训练选择器或扩大搜索；
固定m9分组原型保留为机制证据，原整帧自适应继续保留。本轮结束，不自动启动新实验，不代表停止VAR通信主线。

## 2026-09-07：三关及独立审计均完成（较早记录）

用户明确要求继续并执行到全部完成。已完成第9尺度7500行同观测译码对照与独立审计：
4 dB正确接受恢复率从0提高到9.6%；7 dB从87.8%提高到97.8%；10 dB各主臂均为100%。
VAR臂未观测到CRC误接受。状态`PASS_SINGLE_SCALE_GATE_ONLY`，不是整体系统收益结论。

独立审计重算波形似然、GF(2) CRC、所有指标/CI，并核对9个实际码长NumPy MAP最优分数。
第三关完成分离20图资源选择和100图完整3060-use对照，八臂共12000行，header/class/mode/CRC/tail均计费。
4/7 dB主比较中，VAR相对同分组ML的LPIPS改善6.79%，PSNR提高0.6778 dB，DINO提高0.01809；
但明显落后于原整帧自适应，与同概率实际熵编码＋FEC也没有显著优势。

最终状态：`MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN`。
保持原整帧自适应主基线；保留整token译码器和机制证据，不将当前固定m9分组原型替换主系统。
没有看到目标曲线后继续调参，也没有训练或访问新测试图像。

第三关独立复核12000条接收轨迹和1555张独特重建的全部图像指标；四个模型冻结，
4个旧argmax（包括有误码前缀）回放像素差0。两关审计均为`AUDIT_PASS`。
完整报告：`reports/next_scale_decoding_result_2026-09-07.md`；对比图在`outputs/VAR-NEXT-SCALE-DECODING-FIGURES-001/`。

## 2026-09-07：第一关先验诊断（较早记录）

按用户“可靠恢复一个尺度，再帮助译下一个尺度”的方向，完成第一关；时间按本机Asia/Shanghai记录。
新代码、配置、原始输出及报告都在VAR_COMM，旧JSCC/VAR资产只读复用，没有训练或新正式测试。

- 100张既有development目标＋100张同类不同图donor；独立于它们的1000张旧calibration拟合静态频率。
- 第8/9尺度真实token NLL：真实前缀9.44293/8.97015，静态频率11.96977/11.91021 bit/token。
  相对静态改善2.52684/2.94006，配对95% CI为[2.36202,2.70979]/[2.79774,3.09341]。
- 两尺度都胜过均匀、静态、同类错误前缀三个对照，超过事前门槛；状态`PASS_PRIOR_GATE_ONLY`。
- 独立bit边缘只保留相对均匀NLL改善的3.32%/2.79%，下一步需保留整token评分。
- 800行指标、频率表、输入内容SHA和131项输出哈希经独立NumPy/PIL缓存审计，状态`AUDIT_PASS`。

结果与边界：`reports/next_scale_prior_result_2026-09-07.md`。
当时据此进入第9尺度同FEC/同观测/同搜索预算对照，先穷举校验，再做后续两关。
第一关本身的正确前缀是oracle条件；第三关已改用真实PHY恢复的前缀，结果见最新记录。
第一关单独不代表通信、图像、实际码长或新颖性结论。

## 2026-09-06：独立工作入口确立

用户明确：当前主要研究 **VAR用于通信**，不要继续把记录写入旧 diffusion-JSCC 目录。
本目录从该次整理起承接VAR通信的新进度、方案、代码和实验记录。
七份已有报告原样归集到 `reports/`；历史代码和原始输出不搬动、不覆盖。

## 已有基础

| 实验线 | 实际状态 | 在当前主线中的位置 |
|---|---|---|
| VAR同前缀补全价值 | 已完成prefix-only / completion / Full-VQ消融 | 已有基础证据，不重复当新机制 |
| 固定3060-use m7/m8/m9 + FEC | 已跑通，策略为1dB→m7、4dB→m8、7/13/19dB→m9 | 保留基线 |
| Receiver-prior跨数据集验证 | 2026-09-04已完成ImageNetV2/CLIC/Kodak结果 | 保留历史正式结果和使用记录 |
| Decoder-only微调 | PSNR提高但LPIPS/DINO退化 | 负结果/权衡，不作为无损成功 |
| 同前缀量化cell约束修正 | 20张自检、100张主消融和3种子诊断已完成，未过继续投入门槛 | 停止该具体候选 |

cell约束修正的主消融LPIPS相对改善仅0.64%/0.92%，不如简单prefix混合；
三种子诊断改善1.30%/1.69%，仍低于预注册5%门槛。完整结果见
`reports/var_prefix_consistency_result_2026-09-06.md`。

## 当前边界

- 已完成先验辅助FEC/CRC、真实逐尺度接收和同概率熵编码对照；未做新的神经训练或全新正式测试。
- 新接收器、PHY和审计代码独立写入本项目；历史完整系统未搬迁，旧模型/数据保持原路径。
- 自身公平预算已核查；优化LDPC/Polar、完整近作复现和系统新颖性审查尚未完成。
- 按用户追加授权完成7 dB首失败硬候选的图像回放；二候选oracle仍未胜过整帧，该选择路线停止，不训练选择器。
- 不根据本次100图结果继续追调配置；其他模式/失败输出机制尚未执行，不是本轮已取得的收益。

后续判断和实验都围绕VAR用于通信展开；某个模块停止不代表自动退回旧diffusion/JSCC主线。

## 数据边界

原ImageNet-100永久为development；2026-09-04的ImageNetV2 2000 query和10000 gallery已经使用。
后续若需要全新正式测试，先核对各项目历史曝光记录。
本轮另有100张同类donor作为development曝光；旧manifest虽标为test，也不能再声称未见。
第一关没有访问ImageNetV2/CLIC/Kodak的新测试图像。

2026-09-24: P3060 initial20k and135,000 calibration rows verified; selected20k. Both budget milestones are complete; the existing delivery_chain has begun P2048 calibration-driven30k extension. P3060 also satisfies the initial extension criterion but awaits its serialized decision. Full evaluation/delivery remains pending. See [P3060 milestone](reports/token_efficiency_P3060_20k_20260924.md).

2026-09-24: P2048 completed30k and stopped by the registered calibration rule (last interval0.186047%<0.2%); selected30k. All195,000 calibration rows checked,60,000 new rows published. The existing scheduler has started P3060 until30k. Development/quality/timing and whole-study delivery remain pending. See [30k milestone](reports/token_efficiency_P2048_30k_20260924.md).

2026-09-24: P3060 completed 30k; selected 30k and 195,000 calibration records verified, with 60,000 new rows published. Last intervals improved 0.650284% / 1.597301%; the existing scheduler continues to 40k under the registered rule. P2048 remains finalized at 30k. Full development/evaluation/delivery remains pending. See [P3060 30k milestone](reports/token_efficiency_P3060_30k_20260924.md).

2026-09-24: P3060 finalized after 40k; selected 37.5k. Last interval changes +0.080443% / -0.470516% do not support extension. All 255,000 calibration records verified. P2048 remains finalized at 30k. Shared real-weight interface acceptance passed (selected trained-model replay remains pending); the existing scheduler is executing QPSK full calibration. See [training stop and acceptance](reports/token_efficiency_P3060_40k_and_shared_acceptance_20260924.md).


2026-10-04: Registered prior-aware UEP N1024 with real Sionna5G LDPC, continuous accepted-prefix RX, independent ConvNeXt validation and conditional N2048 replication. CPU BLER shards running; GPU source-Q waits for existing M1 N2048 delivery. No new quality result or training. See [Stage A](reports/prior_aware_uep_stage_a_20261004.md).


2026-10-04: Qualified M1 N2048 source-parallel GPU execution on the same six completed calibration sources with exact full-row parity (image metrics, link decisions and waveform/observation hashes; no reconstructed-pixel hashes were recorded). Selected 4 workers at 2.021x measured source throughput; preserved 286 prior source checkpoints. Original science and delivery paths continue. See [execution benchmark](reports/m1_gpu_workers_20261004.md).


2026-10-04: Prior-aware UEP N1024 actual-link evaluation, independent ConvNeXt validation, matched P1024 10 dB, fixed examples and cost measurements completed. DELIVER_AND_STOP_NO_EXTENSION. See [UEP report](reports/uep_prior_aware_20261004.md). No new model training or holdout access.


## 2026-10-12 paper redraw and fixed final-checkpoint examples

Registry-based NeST-Com redraw: 24 figure groups / 30 PDF-SVG-600dpi PNG figure versions in `paper/figures/redraw_v1/`. Published full-precision intervals are reused;722 numeric plot rows verified, no bootstrap or scoring. Eight registered final-checkpoint inference frames (P100k/Swin200k, sources21/92/95/75,N1024,13dB,noise2001) actually exit0; unchanged digital images reuse their verified original observations. Final and historical HiFi galleries stay separate. Six-method timing remains WAIT under the original exclusive-resource gate; no timing values fabricated. See `experiments/figure-redraw-20261012/README.md` and delivery validation.


## 2026-10-12 final-link HiFi fixed16 delivery

Complete16 HiFi recoveries now use the final Swin200k encoder, confirming decoder and identical paired received waveform at N1024/13dB. All16 full253-step calls and24 missing final-baseline frames actually exit0; eight earlier baseline frames were verified and reused. No training, scoring or bootstrap. Full16 and selected five-column figures are delivered separately from the explicitly historical Swin80k fixed16 table; common500 curves remain unchanged. Shared-GPU runtime is not a fair speed comparison. [Report](results/hifi_final_fixed16_20261012/REPORT.md).


## Kodak24 final-baseline supplement — 2026-10-13

Completed actual432 fresh P100k/Swin200k method frames using all24 previously frozen RGB256 center crops, N1024,4/10/19dB and the original2001/2002/2003 method-specific noise rules. Inference, final scoring and source-statistics/plot children actually exited0. NeST-Com raw and adaptive BPG reuse432 verified historical method frames; four raw10dB BODY_CRC_REJECT_KEEP outcomes remain. No training, Kodak selection/calibration, extra bandwidth, entropy coding or HiFi was added.

Delivery: [four-method results](results/generalization_kodak_final_20261013/README.md), [full48-row table](results/generalization_kodak_final_20261013/analysis_v1/summary.csv), [source-level paired intervals](results/generalization_kodak_final_20261013/analysis_v1/paired.csv), [four-metric vector figure](results/generalization_kodak_final_20261013/analysis_v1/kodak24_N1024_final_four_methods.pdf), and [concise main-text paragraph](results/generalization_kodak_final_20261013/manuscript_results.tex). Complete1152 source means and3456 metric cells are included. The original Kodak10000 source-bootstrap indices andseed2026100901 were reused; old intervals were not recomputed. PNG was actually opened and vector PDF/SVG checks passed.

The results are a tradeoff: raw has lower PSNR than Swin200k at allthree SNRs but better LPIPS/DINO/prediction agreement; against P100k,4dB PSNR/LPIPS are worse and19dB are better. Agreement10/19dB against P has pairedCIs includingzero. This24-center-crop comparison does not establish native-resolution performance, full-corpus disjointness or uniform dominance. Swin200k retains its weight-initialized additional-training/reset-state and expanded1–19dB provenance; convergence is not claimed. The first scorer metadata failure occurred before any model call; the failure, originalregistration and zero-call recovery mapping are preserved. No completed scientific call was repeated. Other pending audits/timing are not declared completed by this supplement.


## Prior launch snapshot: common500 entropy-partial supplement and shared-H800 timing (2026-10-13)

The user authorized only the frozen EC_VAR_PARTIAL supplement on the original 500 sources, N1024, SNR 4/10/19 dB, and three original T1 noise realizations: 4,500 new method frames. E0 verified all 500 pixel/token/source-order bindings and complete matching references. Published strong-reference quality and intervals are reused; no training, strategy change, test-based selection, new bandwidth or second-stage expansion is authorized.

The source/PHY/independent receiver/scoring owner has actually started under registration_v4; completed scientific calls will not be replayed. Follow-up statistics require actual exit-zero receipts from all four stages and the complete 4,500-frame grid. The finite statistical supplement creates only 12 new marginal and 60 new paired intervals with the original source-level bootstrap; existing intervals remain unchanged. Actual stream lengths, every attempted fallback, transmitted prefix and packet failures are retained. See [scope and execution](experiments/entropy-main-common500-20261013/README.md) and [reuse audit](results/entropy_main_common500_20261013/e0_reuse/audit.json).

The user's latest instruction explicitly removes the exclusive-server requirement for the registered six-method timing. A separately bound v2 waits for this supplement's actual scoring closure, then runs all six methods serially on the same H800 in a shared environment, recording resident workloads and load. The fixed16/SNR4,10,19/batch1/three-warmup-plus-three-measured contract and 1,728-call/2,304-packet caps remain. Shared measurements must not be described as uncontended hardware latency. Original v1 and its historical exclusive gate remain untouched. [Shared timing revision](experiments/baseline-delivery-20261012/timing_revision_v2/README.md).

This entry records audited reuse, real execution and automatic handoff, not completed quality, statistics or timing delivery. The existing `var-comm` continuation monitor is ACTIVE for authorized completion; the legacy 15-minute `var-raw64-holdout` monitor remains PAUSED. No E4 expansion will start automatically.

## Layered raw SNR-mismatch R0 - 2026-10-13

The separately authorized finite raw two-group mismatch study completed source
and historical-result auditing and CPU preparation. Original calibration1000
and development100 pixels/tokens were authenticated; fixed cal200 uses source
ID hashes only. The real CPU backend admits all433 single profiles plus201 U
and55 EEP profiles (689 IDs). A targeted exact-received-token audit reuses3400
historical clean-Q values. The remaining screening gap is5400 clean source/state
pairs and1852 exact BLER layout/SNR points (1389 at selection SNR4/7/10).

No new scientific packet, reconstruction, real-image score or bootstrap was
run. No R1 shortlist, scientific registration or GO result exists. A bounded
supplement proposal is awaiting the user's decision; it is not charged to the
original R1/R2 budget. The shared timing v6 remains unchanged and resource-
waiting;576 prior complete calls remain preserved. Historical automations are
paused. [Audit and scope](results/layered_mismatch_raw_20261013/README.md),
[gap proposal](results/layered_mismatch_raw_20261013/R0_GAP_DECISION.md),
[protocol](experiments/layered-mismatch-raw-20261013/PROTOCOL.md).
