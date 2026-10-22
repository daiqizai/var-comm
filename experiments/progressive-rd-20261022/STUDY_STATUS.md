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

# Progressive RD follow-up (2026-10-22)

Independent research from frozen WCL publication. Read user_request.txt, then
results/progressive_rd_20261022_v1/stage0_delivery and seed_delivery_v1.
Stage0 and seedv2 are actually closed. The original failed seed import remains
preserved with zero completed model/source/RX/metric calls.

Current decoder registration: decoder_pilot_registration.json. Three arms,
5000 updates x4 each, common Dc initialization and matched576-parameter FiLM.
One final calibration100 x8 per arm. No checkpoint selection or held-out claims.
Only decoder weights update; old recipe historical Fb_TX is training-only.

Expansion: expansion_registration.json is a13.7MB runtime metadata file, omitted
from Git's10MB/file limit. The deduplicated expansion_registration_compact.json
and restore_expansion_registration.py recover its exact bytes and original SHA.
Do not modify the active runtime registration. CPU synthetic accounting checks
and original seed runtime are reused. Successful source calls are not repeated.
Workers3-7 start on free GPUs;0-2 depend on the real decoder actual_wait0 and
gone process. All share the original absolute six-hour-window deadline.

Never hot-edit bound code. Inspect PID/start_ticks, actual_wait, failure and
ledger before action. No automatic science retry. All data/checkpoints remain
outside Git on PFS; normal exact-scope repository checks/publication required.
No new environment, training data download, tokenizer, PHY, selector or encoder
fine-tuning is authorized by an old record. See the current user request.
