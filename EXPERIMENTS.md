## 2026-10-28 — WCL representation and fixed-frame narrative revision

The anonymous [manuscript v2](paper/manuscripts/nestcom_wcl_n512_20261028_v2/README.md)
revises the abstract, introduction, method transitions, reduced-budget discussion
and conclusion from `ccf4084165cc57e356157ae074c9bfdedb03d2fa`. The argument now
connects coarse-to-fine residual representation, endpoint/capacity mismatch,
actual-length fitting and the two reconstruction roles of received boundary tokens.
WHOLE−Ada differences are descriptive frozen means; interval claims reuse the
existing NeST−WHOLE source-paired statistics. Unfavorable Ada comparisons remain.

The reviewed double-column PDF has 5 pages and a 91-word abstract; the
single-column PDF and editable Word have also been synchronized and inspected.
All three figures, three tables, prior manuscripts and scientific results remain
unchanged. No experiment, inference, timing or bootstrap was run in this revision.
Publication checks are recorded in the manuscript's `validation/repository_checks/`.
Completed scientific tasks and their automations remain stopped.

## 2026-10-28 — N512 evaluation and complete WCL manuscript

The [N512 study](results/nestcom_n512_20261028_v1/delivery/REPORT.md) is complete:
500 original common500 sources, three methods, three SNRs and three noise
realizations give 13,500 formal logical frames. Policies were independently
calibrated and frozen before testing. All 36 mean CIs, 24 source-paired CIs and
12 budget interactions use the existing source-bootstrap bank. The 17 logical
CRC failures remain in the denominator; no no-fit or NULL event occurred.

NeST−Ada PSNR differences at 4/10/19 dB are +4.180/+4.055/+1.740 dB.
DreamSim favors NeST at 4/10 dB and Ada at 19 dB; CLIP-I favors Ada at all
three points. NeST−WHOLE PSNR gains are +0.392/+0.399/+0.792 dB. Cross-budget
comparisons describe independently calibrated systems, not an isolated effect
of N or token ordering. Historical N1024 results and figures are unchanged.

The anonymous [revised manuscript](paper/manuscripts/nestcom_wcl_n512_20261028_v1/README.md)
includes a reviewed 5-page, 10 pt double-column PDF, a double-spaced review PDF,
editable Word, LaTeX, unchanged Figures 1–3 and the new N512 Table III.
Original failed attempts, exact reuse and actual computation are documented.
All scientific stages are closed; their launchers must not be rerun. Backups
remain deferred as authorized. Repository-check receipts are under
`results/nestcom_n512_20261028_v1/publication_validation/`.

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

## 2026-10-19 本轮实验登记与实际执行结果

本轮没有训练新模型、重跑已完成六方法计时或扩大为五方法全带宽扫描。预算如下；N1024 修复与 N2048 校准/测试是独立范围，不合并为一个统计总体。

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

> 2026-09-23 source A COMPLETE: actual source streams/roundtrips and20 representation paths for1000 calibration+100 original development sources, calibrated quality targets frozen before development. Full source report: reports/token_channel_efficiency_20260923_source_A.md. P2048/P3060 real-device qualification passed; P2048 is training. B1/B2 communication matrices and merged short-prefix C remain incomplete; no new holdout.

> 2026-09-23 supplement accepted: A actual source representation/bitstreams, B1 P2048/P3060 and QPSK resource curves, B2 real16QAM, then merged short-prefix C. New holdout and content selectors are deferred. Code/CPU qualification is not real-result completion. See experiments/token_channel_efficiency_20260923/README.md and outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/status.json. Existing m6 cache continues unchanged; its controller is intentionally held before training to honor A/B priority.

> 2026-09-23 short-prefix study: user-authorized new plan; v1 protocol, five-arm interface,4500 real-PHY preflight and real-GPU qualification implemented. First20k matrix is being launched; new quality/development,N3060,seed repeats,policy and new500 test are NOT_RUN. See reports/short_prefix_20260923_launch.md and experiments/var-short-prefix-hybrid-20260923/README.md.

> 2026-09-23 phase2 COMPLETE: registered bounded training, full calibration, selected-model quality/timing, strict-precision digital requalification, frozen resource lookup and PCA comparison are complete. See reports/review_20260923_phase2.md and results/review_20260923_phase2/. Original A/B and historical artifacts are preserved; no new holdout or convergence claim.

> 2026-09-23 phase1: local review repairs and affected real-weight reevaluation completed; see reports/review_20260923_phase1.md. Phase2 is separately authorized and not included in this completion.

> 2026-09-23 repository migration: the actual VAR_COMM root is the sole Git worktree. Historical research claims below are not newly certified by this migration. No A/B retraining or GPU quality validation was performed. See reports/repository_migration_20260923.md.

# VAR 通信实验索引

## VAR-LATENT-ENHANCEMENT-20260917（新授权，运行中，非旧混合复活）

2026-09-18更新：训练已按用户决定收口后，选定checkpoint的development质量和在线端点计时均完成。结果报告`reports/latent_enhancement_development_result_20260918.md`；质量61500行、41方法、无新holdout，计时650调用。当前证据支持保留latent增强候选，但低SNR和DINO取舍仍需如实披露，外部/R3/Deep只复用既有结果做后续定位，不重新训练本轮模型。

2026-09-18更新：已启动选定checkpoint的development评测，协议`reports/latent_enhancement_evaluation_protocol_20260918.md`，配置`experiments/var-latent-enhancement-20260917/evaluation/config.json`。100图×五SNR×三噪声，raw/算术各N3060/3572/4084、m7/8/9及D0/Dc全部保留；不访问holdout、不重新选checkpoint。当前评测结果尚未完成。

2026-09-17 17:01更新：独立`phase_b/`三臂训练及条件接续已实现，13项CPU测试/21.5万真实PHY回放通过；控制器等待A完成及其所选Dc资格，再自动GPU自检→实际RX缓存→配对训练。报告`reports/latent_enhancement_stage_B_handoff_20260917.md`。当前尚未发生B参数更新；不重复启动旧A/HiFi或把准备冒称全链完成。

入口`experiments/var-latent-enhancement-20260917/`，配置`configs/experiment.json`，协议`reports/latent_enhancement_protocol_20260917.md`，启动`reports/latent_enhancement_launch_20260917.md`。固定原m8/N3060，再追加512/1024连续latent观测；先适配Dc，再根据真实连续参考决定是否进入轻量通信E/D。原D0与数字基础不变，校准选模、不访问新holdout。8项CPU及真实GPU接口自检通过；当前精确缓存→Stage A流水线运行，Stage B/同总资源数字/Dc/refiner/development评测尚未完成。新缓存和模型只写`outputs/VAR-LATENT-ENHANCEMENT-20260917/`，不更改旧结果。

## HIFI-RUNTIME / EXPLORATORY-STAGE20（9月16日登记，2026-09-17执行）

协议`reports/hifi_runtime_stage20_protocol_20260916.md`；性能结果`reports/hifi_runtime_profile_result_20260916.md`；有界AA补查`reports/hifi_repeatability_check_scope_20260916.md`。三固定校准点原版/直接attention分别完整计时和拆分；未改权重/精度/完整采样，未训练，未切换原实现。

原240已提交帧及1500计划只读保留，旧全量收尾等待器暂停。新`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/hifi_stage20_20260916/`固定20源、五SNR、noise2001；复用48旧合法结果、用原版补52，之后同子集所有系统统计。小样本探索不冒称1500完成。原脚本/vendor的六个绑定不变，性能诊断和scope wrapper独立留存。

## 只读推进：不等待HiFi的研究定位

阶段一页报告`reports/research_positioning_one_page_20260916.md`，原文差异`reports/external_related_work_difference_20260916.md`。新增CPU脚本`experiments/external-baseline-positioning-20260916/scripts/position_completed_results.py`只复用已完成输出，三类图及失败后果分层在外部输出根的`research_positioning_interim_001/`。没有重跑历史测量或开新支线。

`finish_hifi_positioning.py --wait`是CPU收尾等待器，不占GPU、不改原推理。完整回执后才出`external_baseline_positioning_result.md`和`research_positioning_one_page_final.md`；出错留存/停止，不自动新训练或Git推送。真实同数据观测与同预算系统对照分开，独立ADJSCC不重复计数。

## 已完成阶段：外部非扩散质量、共同预算数字与计算补充

`reports/external_baseline_non_diffusion_result_20260916.md`；主要产物`outputs/EXTERNAL-BASELINE-POSITIONING-20260916/non_diffusion_analysis_001/`。Swin/ADJSCC公开模型完成；N4204/4498仅原数字编码族和m7/m8/m9进行18000校准＋6000冻结开发；1920新完整CPU端点计时逐字节对齐质量。119表行、1624源图级配对指标区间和固定图片原尺寸/等预算拼图就绪。

执行登记`reports/external_baseline_priority_work_20260916.md`，新入口`evaluate_digital_budgets.py`、`time_digital_budgets.py`、`summarize_available.py`和`audit_non_diffusion.py`均在外部定位实验`scripts/`内。9项CPU测试、6000新PHY和183图像/指标子集复核通过。HiFi原队列19:05恢复，尚未完成全量；没有新训练或holdout，也没有重新打开已结束支线。

## 运行中：EXTERNAL-BASELINE-POSITIONING-20260916

作者SwinJSCC、ADJSCC及同ADJSCC的HiFi-DiffCom，只使用公开预训练权重。五权重核验、作者commit锁定、独立兼容环境、小样本完整HiFi推理与Swin/ADJSCC收发一致性通过；内部四系统6000参考只补SSIM。入口`experiments/external-baseline-positioning-20260916/CONTINUE.md`。

作者测量队列17:34启动；程序实际保留原100 development×五SNR×三噪声。默认HiFi单张235步约78.7秒，长任务不等于训练。原假设/实际计费分表，预算不强塞3060，生成seed固定23。还需新公共预算数字校准/评测、成本与最终报告；此时没有外部胜负结论。已完成/停止的旧研究不重跑。

## 已完成：HYBRID-WEIGHT-CLOSURE-20260916

六个实际工作点均获得同源20000新增训练机会；四新臂实际训练、0.01两臂经严格审计复用。完整校准五节点/固定子集曲线、校准冻结选点及比例、9000开发工作点、9000参考与9000期望行、1856配对区间和完整PHY/图像独立复核均完成。新训练校准会话3.06138 GPU小时；不访问新holdout、不重跑无关计时。

结论见`reports/hybrid_weight_closure_convergence_20260916.md`：有相近感知/像素条件下的局部结构增量，但最低混合LPIPS仍较算术自适应差0.034597，原系统目标未过。结束当前固定m7混合主线，不扩大loss/网络/分配搜索；保留全部数据与模型。帧间选择仅期望参照，不是新的部署系统；4dB局部取舍与训练预算边界如实保留。CSI新题只是文献报告，未启动实验。

## 启动快照：HYBRID-WEIGHT-CLOSURE-20260916（已结束）

原两接收器×lambda{0.01,0.03,0.1}，仅改变MSE＋lambda LPIPS权重；每点从原共同暖启动/Adam获得20000新增更新机会。0.01原历史通过审计后只读复用，0.03/0.1四臂实际新训。m7/68+1882+1110/E6120、五SNR、实际RX基图和失败规则不变。完整校准在0/5k/10k/15k/20k、固定子集其余每1k；DINO只报告。校准冻结实测checkpoint、跨lambda匹配及固定帧间选择比例后才评测原100 development及三噪声。

协议`reports/hybrid_weight_closure_protocol_20260916.md`，配置`configs/hybrid_weight_closure.json`；09:44队列3440348/trainer3440380，四臂各700/20000。缺失的冻结Deep校准15000行已补齐，旧强系统结果/计时不重复。最终只判断结构增量、系统取舍和当前固定路线去留；期望帧间选择不是实际部署，也不访问新holdout。任何新题仅允许文献报告，尚未启动。

## 已完成：HYBRID-BASE-CONDITIONING-20260915（2026-09-16收口）

两臂同1648125参数、同功能/Adam起点，原loss及m7/68+1882+1110/E6120不变。每臂新增20000更新（共享参数累计30000），0/5k/10k/15k/20k完整1k校准和固定子集曲线完成；按上限结束，不宣称收敛。3000正常开发输出、1500空间打乱诊断、9000冻结参考、960新系统计时及完整PHY/图像复核均结束。

报告`reports/hybrid_base_conditioning_result_20260916.md`。7/13/19条件对控制PSNR+0.36378有支持，但LPIPS+0.005325，DINO未确认改善；空间打乱显著损伤输出，保留有限像素侧机制证据。原1/4/7对raw PSNR+0.80344，但LPIPS+0.056129，联合目标未过。原负结果不改写，不自动续训/改资源/加模块，不访问新holdout。

## 当前：HYBRID-BASE-CONDITIONING-20260915

仅训练unconditioned/conditioned两臂，实际基图或零图通过同一个小空间模块，在连续Decoder的32/64恢复层注入；公共y派生特征保证控制的共享模块也可学习。同参数1648125，零投影同初态，旧add10000/Adam仅诊断暖启动。E/D骨干、MSE+0.01LPIPS、m7、N3060/E6120、原FEC不变。

计划`reports/hybrid_base_conditioning_protocol_20260915.md`。CPU/GPU/质量/计时工程检查通过，23:05正式受限队列启动；详见`reports/hybrid_base_conditioning_launch_20260915.md`。新更新10000–20000由固定完整校准规则有界判停，两臂同预算。机制与系统判断独立，另有不训练的固定空间条件打乱诊断；不新测holdout、不自动换资源或加模块。

## 已完成：HYBRID-SOURCE-CORRECTION-20260915（预设目标未通过）

三臂各10000更新及完整模型/Adam保存，原20k/1k/100角色不变；126000校准记录、4500新开发行、9000冻结参考、1500短基础消融、1440完整新系统计时全部结束。1500数字PHY与6000图像指标独立复算PASS。报告`reports/hybrid_source_correction_result_20260915.md`，正式图`outputs/HYBRID-SOURCE-CORRECTION-20260915/report_assets_001/`。

没有合格校准点；可靠度gain主LPIPS0.236300，对raw差+0.052536，PSNR仅+0.06735且CI跨0；同参数SNR gain控制未显示有实用价值的可靠度增量。局部4dB有PSNR收益但LPIPS退化，不能包装为联合优势。完整限制/学习曲线/失败分层见报告。停止当前固定点自动追加，不抹除失败、不否定全部混合JSCC、不启用旧holdout作新测试。

## 新授权：HYBRID-SOURCE-CORRECTION-20260915（固定分配，不搜索）

实际激活：2026-09-15 19:07正式三臂配对训练开始，trainer2436487/队列2433287。21k缓存、8项CPU测试、GPU梯度/90条质量和30条计时工程回归已完成；详见`reports/hybrid_source_correction_launch_20260915.md`。实时状态以队列`status.json`为准；仅工程回归完成，不是新方法性能通过。

协议`reports/hybrid_source_correction_protocol_20260915.md`；配置`configs/hybrid_source_correction.json`；缓存/训练入口`scripts/prepare_hybrid_correction.py`、`scripts/train_hybrid_correction.py`。原官方冻结m7基础与1110-use真实源图残差同占N3060/E6120，接收端实际PHY错误进入训练；add、snr_gain与reliability_gain仅比较融合信息，不另建大骨干。

本次先完成工程回归和确定性缓存，再做每臂10000的有限配对训练及校准。原20k/1k/100 development角色不变；上一轮1000 holdout仅历史参考，无新holdout授权。主目标、LPIPS0.005工程非劣容差及0.5dB期望增量已先于新性能登记；没有可行checkpoint时如实标不合格，不回退冒充强基线。不自动扩展资源或模型候选。

## 已完成：COMMUNICATION-CONVERGENCE-20260915 全部收敛与独立验证

R3：2100新development质量行与480自身计时；实际整帧算术码：126000校准、12600 development、2016新链计时；独立holdout：1000源126000 VAR候选与84000冻结强参考，所有PHY/资源/源codec/图像统计审计完成。最终报告`reports/communication_convergence_final_20260915.md`。

主结论不通过“独立新质量调度方法”：arithmetic质量与BLER完全同表，主LPIPS对raw无确认增益；局部6dB源编码/保护收益明确。没有把失败记录删除、短期训练泛化为架构无效或使用失格fallback。全部源码/配置/数据/图像/统计链保留，新的训练/架构/搜索/holdout不自动启动。

## 当前：COMMUNICATION-CONVERGENCE-20260915

- R3冻结收尾已完成：2100新质量行＋46200旧参考，100个development源，原三噪声与七SNR；仅其自身480条同CPU端点计时完成。见`reports/r3_completion_result_20260915.md`，没有训练或重选checkpoint。
- 实际整帧VAR算术码已实现并CPU/GPU资格通过。两族×三个m、1000校准源×七SNR×三固定噪声的126000行实际运行中，PID2117550；不是新增架构或参数搜索。协议`reports/whole_entropy_and_mode_policy_protocol_20260915.md`与配置`configs/communication_decision_study.json`先于结果冻结。
- 三种预定校准规则为目标BLER最大源率、最终LPIPS且PSNR最多退0.25dB、最大源索引goodput；最后一项只复用同一批数据，不新增传输。完成全部校准再拟合并固定，不用development逐图质量选模式。
- 开发/独立holdout尚未开始；只登记可用文件路径，不读holdout像素。最终方法及数据清单冻结后才进入独立验证；不据本阶段更新称研究已完成。

## 当前有限执行：FROZEN-SYSTEM-ONLINE-TIMING-20260915

用户新授权仅完整计时；七个现有冻结系统、32个既有development源、五SNR、原seed2001，每条件三次计时重复，3360行已于2026-09-15 13:09完整结束，13:10独立CPU复核通过。实际波形/图像差0，所有模型起止SHA相同。70行`smoke_004`通过，前三次归档/接口工程失败完整保留。报告`reports/frozen_system_online_timing_result_20260915.md`，协议`reports/frozen_system_online_timing_protocol_20260915.md`。

主1/4/7 dB数字adaptive CPU端点TX/RX10.090/62.284ms，R2为29.388/39.535ms，Deep为2.258/3.023ms；补齐成本不等于质量创新或新holdout验证。不启动任何训练/新架构/搜索或R3评测，有限任务完成后GPU队列为空。正式图用`figures_full/`，工程smoke图用正确样本标注的`figures_smoke/`，不将其作为正式性能图。

## 当前补充：DIGITAL-ONLINE-TIMING-PREPARATION-20260915（CPU只读）

固定32个既有development源/五SNR/seed2001，m7/m8/m9/adaptive共640条旧传输输入与R2原2400条计时记录核验通过；10项CPU回归检查、155个输入SHA。不计作新传输实验，没有重新渲染/计算质量或启动GPU。报告`reports/digital_online_timing_readiness_20260915.md`。

查清旧R2设备驻留RX与数字CPU输出端点不一致，旧R2 TX也非连续端到端计时。未来需要现有冻结模型同端点测量；当前仅输入准备完成，没有GPU计时driver或自动队列。收敛报告revision 2不变，新增架构/搜索/训练仍暂停。

## 当前：CONVERGENCE-AUDIT-20260915（只读收敛审查）

窗口2026-09-14 01:28:36—2026-09-15 01:28:36（本机UTC+8），4组已完成系统比较/15个模型训练版本；输出`outputs/CONVERGENCE-AUDIT-20260915/`，报告`reports/convergence_report_20260915.md`。没有新传输/模型推理/训练或搜索，完整源图、checkpoint和失败记录保留。合法完整校准早期最优与失格warmup fallback分开。

数字VAR自适应重新作为候选主系统，学习支线仅保留R3待验证资格，其他冻结为控制。R3窗口后02:00已按原10000结束训练/校准，但没做最终系统评测；当前VAR_COMM无活动GPU队列，不干预其他项目。新增架构/自动参数搜索/新大网络/复杂机制继续暂停，不按下面历史计划自动启动。

## 当前：WETOK-REENCODING-VECTOR-CONTROL-R3（已激活）

本机UTC+8 2026-09-15：实现/CPU原始初始化/GPU图像梯度/4项CPU与实际driver断点测试通过。首1000完整校准和模型Adam/数据/功率/选模审计PASS，00:38实际续到10000，1099666/1099667/1099668；00:46核验2000。唯一融合向量q对y−q，均保留标量残差能量门控，整体向量大小/方向都可能变化，不单独宣称方向效应。`reports/wetok_r3_activation_20260915.md`。

最终23方法/48300主质量行和16模型/2560独立计时只完成协议/配置登记，驱动与收尾尚待补齐；当前没有新R3 development或正式test结果。旧R2完整参考只读，后续入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。

## 当前：WETOK-REENCODING-VECTOR-CONTROL-R3（登记，未训练）

R2于2026-09-14完整结束：主MS−普通full-grid LPIPS+0.002350且CI全正，残差版−0.009803优于普通full-grid但仍输主区间数字自适应/Deep。46200质量主行、48400图像CPU审计和2400独立计时完整；报告`reports/wetok_r2_result_20260914.md`。不重启旧R2队列。

后续只补向量归因对照：三次16×16读取、两次E、同参数/门控/10000训练机会，仅融合q而不是y−q。两者门控都保留残差能量，不能称完全无残差；Joint训练后不同s/y只匹配源/噪声/N/E。原7000父点和完整初始化/Adam/数据资格需核验，当前未实现或启动。入口`experiments/wetok-reencoding-vector-control-r3/docs/CONTINUE.md`。

## 当前：WETOK-JOINT-SUFFICIENCY-R2的10000系统评测

2026-09-14 23:09四臂10000训练/全校准/模型Adam/数据与选模审计PASS。single/MS部署选择7500，两个full-grid选择10000；各臂均拥有10000机会。当前质量983552/收尾880742/观察880743，23:12提交23/100原development源图；完整质量和独立计时尚在流程中。19项CPU测试PASS；报告`reports/wetok_r2_milestone10000_20260914.md`，入口`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。不重启已结束训练，不访问新test，不省略同WeTok数字强参考。

R2辅助报告（2026-09-14 22:09）：7500完整训练/校准曲线与可对账新增成本已封存，`reports/wetok_r2_training_curves_20260914.md`；原15模型质量/计时准备不变，新增只读绘图工具通过4项测试，全套15项CPU测试PASS。实时训练至8400，并未执行新的R2 development或正式test。

## 当前：WETOK-JOINT-SUFFICIENCY-R2，7500→10000

2026-09-14 21:35，四臂7500完整校准与实际模型/Adam、历史数据、功率、选模审计PASS，四臂均有LPIPS改善；21:43从真实7500保存点共同续到10000，21:46已保存7600。trainer880740/reviewer880741，自动质量/统计/独立计时finisher880742与被动GPU observer880743已运行；不重开旧769675队列。

评测/计时与断点测试准备完成，11项CPU测试PASS。最终四R2＋18旧参考＝46200主行，78个比较/3744主配对区间；独立15模型32源×五SNR×seed2001＝2400计时行/70区间。模型选择使用完整校准，100图monitor不选模；多训练、同机会结构、历史强系统三类比较明确分开。质量可显式共享、计时单独准入，未有新R2 development结果。`reports/wetok_r2_milestone7500_20260914.md`；`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。

## 当前准备：WETOK-JOINT-SUFFICIENCY-R2

2026-09-14 20:09已进入真实续训，四臂继承model+Adam5000到7500；20:26核实5500总更新、500新增、global12500，旧5000历史不变。trainer769675/reviewer785157/observer785158，CPU恢复与真实driver中断等价通过，损失/学习率/预算未改；无R2新开发结论。`reports/wetok_r2_activation_20260914.md`。

在Grid5000完整质量/计时结果后，准备四臂相同新增5000更新的训练充分性对照。四个实际5000模型/Adam、global12000、原数据/噪声/冻结身份已CPU核验；没有启动GPU训练，driver尚待实现。`experiments/wetok-joint-sufficiency-r2/docs/CONTINUE.md`。

前轮Grid的18方法质量与11模型独立计时完整完成，4/8/16未展示独立LPIPS/PSNR/时延优势；Joint全网格innovation最佳但仍落后强系统LPIPS。`reports/wetok_joint_grid_result_20260914.md`。所有旧结果与模型保留。

## 当前：WETOK-JOINT-GRID-CONTROLS-R1

2500更新/完整校准审计于2026-09-14 18:29完成，两臂实际续5000（650391/650392）。新评测保留18方法、37800质量主行，新增模型推理与冻结参考复用分明；与固定32图/11模型的1760行独立计时分开。18项CPU测试通过，自动finisher689944已挂载；当前尚无新Grid开发结果，不把准备/校准当系统胜出。`docs/evaluation_preparation_20260914.md`位于该实验目录。

2026-09-14从原7000父点开始两臂匹配Joint训练：full_grid_state_history和full_grid_innovation，仍N3060/E6120、原loss/数据/噪声/预算。6项CPU测试及真实GPU零更新图像梯度检查PASS，17:27实际激活，已核实300步模型/Adam；trainer570837/reviewer579794。新Grid还没有development结果，不将参数相同冒充计算量相同。入口`experiments/wetok-joint-grid-controls-r1/docs/CONTINUE.md`。

原Joint5000完整评测/统计已于16:48结束，33600主行全部保留。条件版对single有小幅LPIPS增量，但对强全网格差值CI跨0，仍不及数字/Deep的LPIPS。结果`reports/wetok_joint_result_20260914.md`。

## 当前：WETOK-JOINT-SENDER-R1

2026-09-14 16:01原评测因GPU竞争退出，49张已提交、5000训练与校准完整保留。恢复监督器473779/观察器473780于16:12启动，等待空闲再原协议`--resume`，之后自动统计；资格核验和31项CPU检查通过，不重新训练、不变更原源码/参数/样本。`experiments/wetok-joint-sender-r1/docs/evaluation_recovery_20260914.md`。

5000更新与全校准审计于2026-09-14 15:50完成，三Joint均选5000。条件版校准LPIPS .194679优于Joint single .197125，但R-only full-grid .187311更强。当前自动回放三Joint＋六R-only＋七系统，evaluator443167/finisher348503，尚未得到完整Joint development结论。报告`reports/wetok_joint_milestone5000_20260914.md`。下方训练中状态为历史，不重复启动。

2026-09-14续训正常，15:00已核实3100保存点。被动GPU并发记录从14:44开始，不改变质量实验；新增5项观察器测试，共27项CPU测试通过。HJSCC小型源码/协议核查单独完成但尚无模型或性能复现，详见`reports/hjscc_source_protocol_review_20260914.md`，不取代当前注册的强控制。

2500完整校准及审计已PASS；2026-09-14 14:37三臂从真实末端模型/Adam继续5000（348501/348502/348503），全量评测与CPU统计已排队。single/state选中2500，no-history仍选0；对匹配R-only有小幅校准改善，条件结构尚未胜single。报告`reports/wetok_joint_milestone2500_20260914.md`。新增收尾工具后22项CPU测试通过，当前未有新Joint development或holdout结果。

2026-09-14 12:39已激活Joint三基础结构，首1000里程碑；相同原7000父点和fresh Adam规则、相同数据/噪声与5000新更新机会，唯一训练因素为E是否也优化。原R-only5000控制、全量评测/分析、reference资格、GPU梯度与16项CPU检查均PASS。训练尚无新性能结果。`experiments/wetok-joint-sender-r1/docs/CONTINUE.md`。
原WETOK-INNOVATION-R1已完成冻结，结果见`reports/wetok_innovation_result_20260914.md`；以下运行条目为历史，不据旧PID重启。

## 当前：WETOK-INNOVATION-R1

2026-09-14 11:10已从恢复点继续到4100；原5000评测仍由恢复后的finisher自动接续。后续Joint评测准备完成：16方法、九模型新测、全部六R-only对照，16项CPU测试PASS；实际Joint训练与GPU评测尚未执行。

2026-09-14恢复：旧队列中断于3000保存点附近；旧进程消失、系统已重启。核验六臂真实模型/Adam/数据噪声后，从3000续至5000并重新挂上审计/评测队列，3100已保存。`experiments/wetok-innovation-r1/docs/recovery_20260914.md`。不把中断时间计为训练，不误称5000已经结束。

2500完整校准/审计已结束；full-grid .196310优于多尺度innovation .200449。六臂共同续到5000，自动收尾评测进程已排队（3211225）。`reports/wetok_innovation_milestone2500_20260913.md`。Joint仅准备与真实GPU零更新profile通过，未训练，须等待完整控制与结果归因。

后续准备（未启动GPU训练）：`experiments/wetok-joint-sender-r1/`只改变通信E是否接受梯度，与现有R-only三基础结构同起点、同新训练预算；7项CPU检查及真实权重初始SHA核验通过，等待实际完整控制与GPU profile。动机来自父点训练曝光与校准曲线审计，不把多训练本身或可训练参数变多当论文创新。

首1000更新/完整校准/审计已于本机06:00完成，创新量选中LPIPS .202062，尚不胜single .198920或prediction .201300。六臂已从实际末端模型/Adam共同继续2500；no-history虽选中0，也从真实1000末端续，不重置训练历史。`reports/wetok_innovation_milestone1000_20260913.md`。

geometry同预算开发评测已完成并保留负交互结果。当前固定同一发送器，进行single/no-history/state-history/prediction-feature/multiscale-innovation/full-grid-innovation六臂RX-only比较；相同y、N/E、数据/噪声和训练机会。CPU/真实GPU检查通过、首1000训练/审计已完成，尚无新机制系统成功结论。`experiments/wetok-innovation-r1/docs/protocol.md`。

评测准备已补齐：27300主行、600固定支持点、1968主配对区间和108支持点区间，逐帧共享s/y与冻结E/选模核验、全部失败保留。15项CPU/合成整链测试通过，旧参考441条PSNR和21波形只读预检通过；训练/profile绑定源码和配置未改。`experiments/wetok-innovation-r1/docs/evaluation_protocol.md`；尚未执行新接收器development评测。新时延评测轮换六臂次序，并保存源图前后温度/时钟/节流记录。

## 当前：WETOK-GEOMETRY-20260913

总4500/image2500全校准审计已完成，geometry收益保持但条件结构未确认。当前三臂从真实末端继续总7000/image5000；旧控制历史不变，之后运行匹配完整预算开发评测。

首总3000已完成，匹配image1000预算下全校准LPIPS约.207，对旧geometry约.256–.261改善；只作校准证据。现继续总4500(image2500)，两个geometry的比较训练资格保持一致。新最终评测入口只接受总7000匹配预算，CPU检查通过，未提前访问新模型development。

本机UTC+8 2026-09-13启动204×30三结构训练，固定6120实坐标/N3060/E6120、连续接口、视觉模型与原loss；153×40完整2000+5000历史控制经过资格核查后只读复用。31项CPU与真实GPU无更新profile通过，初始目标总3000，未形成新geometry性能结论。入口`experiments/wetok-comm-v2-20260912/docs/geometry_execution.md`。
前接口5000比较已完成：主区间出现同接口条件增量，但仍不胜强系统；`reports/wetok_interface_milestone5000_2026-09-13.md`。以下v2接口运行条目均为历史，不再默认重开。

## v2当前执行：WeTok原生通信A0/A1（2026-09-12）

20:47接口2500开发评测/统计完成，33600+600行、独立3024区间复算通过；条件continuous LPIPS不胜无历史或强系统。20:57九臂从真实末端共同恢复到总追加5000，配置和N/E不变；不把继续训练当收益保证。报告`reports/wetok_interface_milestone2500_2026-09-12.md`。

18:16接口九臂完成1000更新+全校准/审计并继续至总追加2500；完整calibration的continuous LPIPS single/no-history/conditioned为.259012/.261040/.256377，不是development系统成绩。25项CPU测试及Deep支持点补充检查通过；当前运行入口train_interfaces --until-additional 2500 --resume。详见docs/interface_continuation_2500.md。

16:06已启动接收接口九臂流程（首1000检查点）：三结构×native identity-ST/native有界ST/continuous。相同2000模型父点、统一fresh Adam、原joint loss、固定N/E，19项CPU及九臂真实GPU无更新profile通过。正进行完整起点校准；后续按完整校准共同续训，最终development评测尚未做。协议`experiments/wetok-comm-v2-20260912/docs/interface_protocol.md`，接续`docs/interface_execution.md`。

此前bit权重六臂15:52因固定监控继续严重退化而探索性提前停止；共同追加1000现场保存、计划5000未完成，不把该停止当整个研究终点。报告`reports/wetok_bit_support_stop_2026-09-12.md`。

以下为之前的启动记录：

首5000原配方里程碑已完成，联合图像训练出现感知/bit退化，未支持条件历史的明确增量。已于15:06启动单变量bit-BCE 0.01→1的六臂配对续训：同5000模型/Adam、同新增预算/数据/噪声，不改网络/物理/接口。协议`experiments/wetok-comm-v2-20260912/docs/bit_support_protocol.md`；首250追加更新审计通过。

原生±1 Fq全局发送，single-pass／三阶段无历史／三阶段条件三臂，同N3060/E6120/训练机会；不是codec换代本身的创新，不是逐包渐进。
原生缓存、100图identity回放与真实image-gradient检查通过；13:00已正式训练，5000为首里程碑而非总上限。旧暂停/200元/半天限制由根目录v2任务书覆盖。
入口`experiments/wetok-comm-v2-20260912/`，配置`configs/study.yaml`，协议`docs/protocol.md`；native cache/8PSK数字对照及日志位于本项目outputs，旧结果只读。

## 已完成并复核：ei-liulu-xqvar-eval-20260912-v1（无训练骨干选型）

- 授权：本机已有GPU整条流水线自然结束后；不再使用需要认证的开发机、不停止任何已有任务。
- 数据：同100张development，旧预处理/PSNR/SSIM/LPIPS/DINO；非新test。
- 全重建：旧官方、旧fidelity、XQ实际配套tokenizer、WeTok指定8192bit完整codec。
- 条件恢复：XQ真实自然prefix8/9（静态2424/3960rawbits）直接/配套VAR补全/full；原3060/5088raw点只作离散源表示参照，非复信道等资源。
- 状态：01:46自动启动、01:49完成，1800正式行；独立CPU产物/统计复核PASS。**暂不正式迁移到XQ；WeTok完整codec得到保真/感知共同改善，但不直接接通信。** 本轮新增训练0步。
- 协议：`experiments/backbone-eval-20260912/docs/protocol.md`；结果解释：`reports/backbone_selection_review_2026-09-12.md`；输出 `outputs/ei-liulu-xqvar-eval-20260912-v1/`，不覆盖旧记录。

## 已完成通信骨干2×2：VAR-TOKEN-BACKBONE-20260911

2026-09-12 01:43（UTC+8）完成四臂各20000更新、16800行development评测及统计。增强对小next-scale的LPIPS差-0.001586，95% CI跨0；仍未追平固定数字m8的LPIPS。
两条parallel无合格PSNR候选，回退warmup；不将该回退对照的巨大差值当作B配方结构增量。摘要`reports/token_backbone_result_2026-09-12.md`，原主表和receipt保留不改写。以下为启动时记录。

小/增强×parallel/next-scale均按完整10000 warmup+10000 B配对训练；保留旧B为冻结工程参考，不用其继承历史替代严格比较臂。只改通信E/D。
实现、CPU合成检查与真实GPU无更新检查已通过；增强5139662参数、匹配SNR小骨干1559054参数。新结果尚未形成；入口`../var-next-scale-comm/scripts/run_token_backbone.py`，新协议在该仓库`reports/token_backbone_protocol_2026-09-11.md`。

## 2026-09-11本机运行完成：VAR-M8-LOCAL-20260911

固定m8的A同预算原配方续训/B两阶段通信训练于13:42:19（UTC+8）完成训练、12600行development评测及统计；每分支新增10000更新，B为2000+8000，不引入其他方法变量。
执行位置`../var-next-scale-comm/`，配置`configs/m8_local_20260911.local.yaml`；选中A step2000、B step6000。主1/4/7 dB的B相对A LPIPS改善27.60%，但仍未超过固定数字m8/自适应/感知Deep的LPIPS。摘要`reports/m8_local_result_2026-09-11.md`；所有旧准备/暂停条目为历史记录，不自动追加实验。

## 当前独立工作区实验

| 实验 | 状态 / 主要作用 | 本地报告 |
|---|---|---|
| 固定m8原配方续训 vs 两阶段通信训练（远端准备） | 当前指定步骤，准备与CPU自检完成；未训练/未GPU测速，代码在新仓库 | `../var-next-scale-comm/docs/REMOTE_M8_HANDOFF.md` |
| 多尺度通信×保护分配×next-scale条件接收（2026-09-10方向更新） | 用户更新方向；设计口径已记录，尚未实现/训练/测速/租卡；旧m8续训仍暂停 | `reports/multiscale_joint_direction_2026-09-10.md` |
| 两阶段prefix恢复对照（2026-09-08） | 用户要求暂停；实现/CPU检查完成，新增训练0步，无存活队列；重新启动须用户明确授权 | `reports/prefix_refinement_protocol_2026-09-08.md` |
| 已有校准与固定m8对照复核（派生分析） | 完成，无训练/推理；next第二轮校准LPIPS全SNR退化，固定m8数字链在开发七SNR的LPIPS/DINO仍更好；72组原配对复算差0 | `reports/prefix_calibration_fixed_m8_review.md` |
| m8全局prefix-JSCC训练（2026-09-07） | 两版本完成各10000更新/40k曝光，冻结VQ/VAR/image decoder，最后6000更新全self-history；均选epoch2 | `reports/learned_prefix_result_2026-09-07.md` |
| 图像梯度工程自检（同日） | 通过；MSE/LPIPS到E/D梯度非零、梯度穿过冻结VAR、train/eval硬前向像素差0；不作科学性能gate | `outputs/VAR-PREFIX-JSCC-GRADIENT-001/selfcheck.json` |
| 训练预算/校准选择审计（同日） | 通过；相同初值/更新数，独立image-only校准，DINO未优化，冻结权重不变 | `outputs/VAR-PREFIX-JSCC-TRAIN-AUDIT-001/audit.json` |
| 学习prefix开发评测与审计（同日） | 10500行完成；next相对parallel有8.921% LPIPS增益，绝对性能未胜数字自适应；7348图/4200硬前缀重建复核 | `outputs/VAR-PREFIX-JSCC-EVAL-AUDIT-001/audit.json` |
| DeepJSCC条件支持检查（同日） | 主结果冻结后补充；实际5/6 dB、NN固定条件4/7，600行，无训练/物理噪声变化；不把原off-grid异常算优势 | `outputs/VAR-PREFIX-DEEP-COND-SANITY-001/summary.csv` |
| 整帧4前缀/末尺度辅助译码（2026-09-07） | 完成9600行；未胜列表65，条件真前缀恢复饱和但覆盖不足；`STOP_THIS_FINITE_PREFIX_VAR_RECEIVER` | `reports/whole_frame_prior_result_2026-09-07.md` |
| 整帧列表独立审计（同日） | `AUDIT_PASS`；322万候选/3984概率表/1240图复核，638新图旧实现像素差0 | `outputs/VAR-WHOLE-FRAME-PRIOR-AUDIT-001/audit.json` |
| 完整接收API时延补测（同日） | 20固定图×4SNR×6方法=480调用，输出全复现，不修改主统计 | `outputs/VAR-WHOLE-RECEIVER-LATENCY-001/summary.csv` |
| 波形到RGB时延补测（同日） | 20固定图×2SNR×3核心方法=120调用，像素差0；6 dB VAR/list65约212/69 ms | `outputs/VAR-WHOLE-IMAGE-LATENCY-001/summary.csv` |
| 7 dB首个CRC失败候选图像回放（2026-09-07） | 完成；固定m9/3060/原波形与FEC，VAR追回64.74%差距但二候选oracle仍输整帧；`STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9` | `reports/crc_failure_replay_result_2026-09-07.md` |
| CRC失败回放独立审计（同日） | `AUDIT_PASS`；1200条原轨迹、219张新图用旧实现复渲染像素差0、580张图完整重算指标 | `outputs/VAR-CRC-FAILURE-REPLAY-AUDIT-001/audit.json` |
| 第9尺度同观测译码（2026-09-07） | `PASS_SINGLE_SCALE_GATE_ONLY`；100图×3SNR×5噪声×5先验=7500行，7 dB正确恢复率87.8%→97.8% | `outputs/VAR-SINGLE-SCALE-CHANNEL-001/completion.json` |
| 单尺度独立审计（同日） | `AUDIT_PASS`；波形、CRC、7500行统计与9个独立NumPy MAP最优值复核 | `outputs/VAR-SINGLE-SCALE-CHANNEL-AUDIT-001/audit.json` |
| 真实逐尺度与强对照（同日） | 12000行完成；同分组机制通过，整帧自适应/熵编码强对照未胜出 | `reports/next_scale_decoding_result_2026-09-07.md` |
| 完整链路独立审计（同日） | `AUDIT_PASS`；12000条可信状态轨迹、1555张独特重建全部重算指标 | `outputs/VAR-PROGRESSIVE-CHANNEL-AUDIT-001/audit.json` |
| Next-scale先验诊断（2026-09-07本机日期） | `PASS_PRIOR_GATE_ONLY`；100图×2尺度×4先验，完整token log-loss过门槛；没有FEC/CRC或图像质量结果 | `reports/next_scale_prior_result_2026-09-07.md` |
| 独立缓存审计（同日） | `AUDIT_PASS`；重算800行指标/CI、静态频率及1200图内容SHA，不是第二遍模型推理 | `outputs/VAR-NEXT-SCALE-PRIOR-AUDIT-001/audit.json` |

最新整帧原型预注册：`reports/whole_frame_prior_preregistration_2026-09-07.md`；未训练/改变发送/FEC或5.5 dB阈值。
4/5 dB无真前缀进入4候选，6 dB仅65/296；条件先验有效但不胜普通CRC列表的图像收益/成本。
大列表受候选/堆上限限制且有误接受，未声称严格等算力或直接替换主系统。
强整帧熵编码未实施，不把保留旧分组熵编码记录当成完成了该强对照。

前轮回放预注册：`reports/crc_failure_replay_preregistration_2026-09-07.md`；产物`outputs/VAR-CRC-FAILURE-REPLAY-001/`。
3000行包含900个分组接收记录的discard/retain/oracle输出与300个原整帧对照，不是3000次独立传输。
100张原development、7 dB、原三噪声，未训练/换FEC/重分资源/新搜候选。两候选选择路线按预定oracle界限停止。

第一关预注册：`reports/next_scale_prior_preregistration_2026-09-07.md`。
主输出：`outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/`；同类donor明确作为development，旧calibration仅用于频率拟合。
第二/三关已接续执行，详见上表；先验门槛通过不等于系统门槛通过。

第二关协议：`reports/single_scale_channel_preregistration_2026-09-07.md`。
第三关协议：`reports/progressive_channel_preregistration_2026-09-07.md`。
该前轮三关状态`MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN`，不替换原整帧自适应主基线。

工程记录：首个C++自检在穷举例中发现初始化顺序错误，修复后SELFCHECK-002通过；
第三关smoke001后补全mode误接受计数，smoke002通过，实验参数不变。失败/smoke目录保留，不计科学样本。

## 历史归集

2026-09-06归集。下列结果此前已在旧工作区执行，**不是本次重新跑出的结果**。
历史原始输出根目录及各报告SHA见 `historical_sources.json`。

| 实验 | 状态 / 主要作用 | 本地报告 |
|---|---|---|
| Decoder-only full/m8/m9微调 | 已完成；失真提高但感知/语义退化 | `reports/var_decoder_only_m89_result_2026-09-03.md` |
| VAR-m8真实无线诊断 | 已完成；分离接收误码与补全损失 | `reports/var_m8_wireless_diagnostics_result_2026-09-04.md` |
| 固定3060-use自适应scale/FEC | 已完成开发机制验证；保留m7/m8/m9基线 | `reports/var_fixed3060_adaptive_scale_result_2026-09-04.md` |
| 同前缀语义/码率贡献消融 | 已完成；同payload下比较prefix-only与VAR completion | `reports/var_semantic_rate_contribution_result_2026-09-04.md` |
| Receiver-prior跨数据集验证 | 已完成历史正式验证；不得再用该数据声称未见测试 | `reports/var_receiver_prior_cross_dataset_result_2026-09-04.md` |
| 同前缀cell约束修正 | 完成并停止；6400行argmax主消融、3600行三种子诊断 | `reports/var_prefix_consistency_result_2026-09-06.md` |

cell约束修正预注册：`reports/var_prefix_consistency_preregistration_2026-09-06.md`。
该候选最终状态为 `STOP_THIS_LATENT_CORRECTION_CANDIDATE`，不是VAR通信主线整体停止。

## 后续记录约定

新实验在本目录登记数据角色、共享信息、真实预算、模型/源码版本、随机种子、对照、判据和输出位置。
新输出写入本项目的 `outputs/`，不覆盖历史输出，不默认写回旧项目。
未运行的建议不得记为已完成；只做目录整理不记作新算法实验。


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
