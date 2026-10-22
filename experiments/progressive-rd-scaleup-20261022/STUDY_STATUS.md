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

