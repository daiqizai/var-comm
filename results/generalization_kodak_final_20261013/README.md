# Kodak24 final frozen-baseline results (2026-10-13)

All24 original RGB256 center crops; N1024; SNR4/10/19; original noise labels2001/2002/2003 and method-specific generators. P100k/Swin200k add432 fresh actual method frames,432 reconstructions and1728 scores. Verified NeST-Com raw/BPG scores reuse432 historical method frames. No training, Kodak tuning, entropy/HiFi or bandwidth expansion occurred.

`analysis_v1/summary.csv` is the four-method full-precision mean/95%CI table; `source_means.csv` has1152 three-noise source means; `rows_all_four_methods.csv` retains all3456 metric cells; `paired.csv` has36 source-paired intervals, including12 historical raw−BPG and24 new raw−final-baseline intervals. `paired_new.csv` isolates the latter. LPIPS is not sign-flipped; agreement entries are fractions, the figure uses%, paired agreement differences convert to percentage points by multiplying100. ConvNeXt agreement is not classification accuracy.

Original Kodak analyze.py,10000 source bootstrap indices andseed2026100901 are reused. Old raw/BPG interval rows are preserved byte-for-byte. Original files remain unchanged. Four raw BODY_CRC_REJECT_KEEP frames at10dB remain in the evaluation. All216P100k and216Swin200k frames were RECEIVED_OK. The full four-metric figure is `analysis_v1/kodak24_N1024_final_four_methods.{pdf,svg,png}`; PDF is vector, SVG retains text, PNG is600dpi and was actually opened. `manuscript_results.tex` is the concise main-text paragraph; the analysis directory also preserves the automatically generated mean-based paragraph.

`registration.json`, `frames.json`, `inference_grid_audit.json`, `anchor_reuse_audit.json`, `delivery_audit.json` and actual_wait receipts establish scope and actual exits. Final scores are in `metrics_final_v1`. The first `metrics_v1` attempt exited1 on a missing metadata alias before launch or model/scoring calls; its failure and wait record are retained. `scoring_recovery_v1/registration.json` adds only the alias and new owner binding, with no extra scientific budget. No inference or completed scoring call was repeated. The original registered executable bytes remain intact.

The final P uses the frozen source VAE andDc. Swin200k is the previously selected weight-initialized additional-training version with reset optimizer/order/RNG and expanded1–19dB training range, not a claim of convergence or an official optimum. This replaces neither the historical Swin80k archive nor its19dB outside-range label. The H800 scorer retains original metric math/preprocessing/weights and a separate runtime identity; old-host bitwise equivalence is not claimed. No absence of overlap in foundation-model pretraining corpora is inferred from this benchmark.

The existing statistics entry refuses duplicate scientific execution. For a fresh output directory using already completed rows:

```sh
python experiments/generalization-kodak-final-20261013/postprocess.py --rows results/generalization_kodak_final_20261013/metrics_final_v1/rows.csv --completion results/generalization_kodak_final_20261013/metrics_final_v1/completion.json --old-root results/generalization_kodak_20261009 --reuse-audit results/generalization_kodak_final_20261013/anchor_reuse_audit.json --out /absolute/fresh/output
```

This command includes the original bootstrap and is documented for independent reproduction, not automatic rerun of the completed experiment. A failed layout can recover through --render-only without bootstrap. See the source README_STATS for schema and safety details. Reconstructed pixels/IQ/noise/call ledgers stay outside publication as runtime evidence and in the verified local incremental backup; frozen models/environments depend on earlier restore packages.
