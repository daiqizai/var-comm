# T5 and T6 figure coverage at the entropy-quality export

Scope: the actual requested figure deliverables in sections7 and8 of `NeST_Com_WCL_supplemental_experiments_v1_20261009.md`. This is a read-only local asset audit, not a new scientific result or a claim that T6 has run. Paths below are relative to the repository.

| Requirement | Actual coverage | Location / remaining condition |
|---|---|---|
| Original N1024 complete-scale and partial-scale curves, all six SNRs, four metrics | Complete; original CSV means and CIs | `results/wcl_evidence_closure_20261009/T5_figures/fig_t5_same_prior_N1024.{pdf,svg,png}` plus four independent panels |
| Original partial minus complete source-paired differences | Complete;7dB zero and13dB small increments retained | `results/wcl_evidence_closure_20261009/T5_figures/fig_t5_partial_minus_complete.{pdf,svg,png}` plus four panels |
| External system-positioning curves | Existing complete500-source supplement; no redraw needed | `paper/figures/adaptive_bpg_holdout500/fig_adaptive_bpg_main_N1024.{pdf,svg,png}`; adapted Swin19dB extrapolation marker/caption retained |
| New complete-scale entropy quality contrast | Complete in this directory; both frozen entropy families and original partial-scale raw,4/10/19dB | `fig_t5_entropy_quality_N1024.{pdf,svg,png}` plus four panels; post-hoc same-source explicitly labelled |
| Direct paired entropy minus partial-scale contrast | Complete in this directory; existing paired intervals | `fig_t5_entropy_minus_partial_N1024.{pdf,svg,png}` plus four panels; LPIPS unchanged; VAR19dB agreement mean0 and nonzero CI retained |
| Mechanism examples before entropy addition | Complete; all16 fixed development sources at1/10/19dB | Original T5 root `fig_t5_mechanism_N1024_*`; four four-row pages and all16 PDF perSNR |
| Mechanism examples including entropy fifth column | Complete; same fixed16 at4/10/19dB, actual frozen VAR-conditional entropy family | `results/wcl_evidence_closure_20261009/T5_figures/mechanism_entropy_completed_v2/`;12 pages, three all16 PDFs.4dB is the actual low-SNR entropy point; no4-to1dB substitution |
| External five-column fixed examples | Complete; all16 fixed development sources at1/10/19dB, original/BPG/Swin/latent/NeST-Com | `results/wcl_evidence_closure_20261009/T5_figures/external_completed_v1/`;12 pages, three all16 PDFs. Original13dB views remain in T5 root |
| Small main-text sample subset plus all16 pages | Complete | Page01 uses the first four entries of the historical fixed order. No outcome-based reselection; remaining pages expose all16 |
| HiFi subset when source/workpoint/noise match | Existing separate matched13dB development subset; no additional generation needed | `paper/figures/hifi_development_N1024_13dB_group02/`; source indices4,21,24,29 in original order. It is not relabelled as holdout,1dB,10dB or19dB |
| Editable vector data plots,600dpi PNG, raw plot data, captions, sample mapping | Complete for the finished T5 groups | Each completed version has completion/seals and captions. Photo panels embed measured raster reconstructions with vector text; data curves themselves remain vector |
| Optional ImageNet true-label top1 | Not a mandatory figure deliverable | Prediction agreement remains correctly named. No new classifier evaluation is authorized or started by this plotting task |
| N2048 summary, paired results and cross-budget mechanism figure | Pending actual T6 scientific closure | Consumer implementation `t6_score.py` and `t6_statistics.py` is ready, engineering-checked only. Final cross-budget rendering must wait for frozen calibration, the actual new100×3noise×3SNR×3method outputs, scoring and statistics |

## Remaining figure work

The only mandatory T5/T6 figure still dependent on uncompleted science is the T6 cross-budget mechanism figure. It must use actual N2048 summary/paired CSVs alongside the existing N1024 values, identify different source populations (old500 post-hoc supplement versus registered new100 confirmation), retain all failures/zeros, and restrict comparisons to the measured4/10/19dB points. There must be no interpolation-based bandwidth-saving estimate or invented equal-quality crossing.

T4 TX/RX cost summaries and the T1 quality-versus-TX-cost deliverable still require the actual unified timing closure before cost graphics can be finalized. Those are outside the specific T5/T6 figure completion line and are not imputed here.

`mechanism_entropy_completed_v1/` is a rejected layout with a clipped heading; it is explicitly superseded byv2 and must be excluded from publication. The earlier T5 root README records its original asset availability at the time of export; the new completed subdirectories supply the subsequently missing10/19dB examples without rewriting that historical artifact.
