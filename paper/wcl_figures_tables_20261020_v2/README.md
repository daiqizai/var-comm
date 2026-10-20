# WCL final figure/table set v2

Frozen scientific baseline and observed checkout: `a08e2517389f4df00ea4162c98417df3f52560a8`, `main`. There were no newer commits, tracked edits or local input differences at inspection. This revision changes presentation only. No training, channel/decoder calls, reconstruction, neural metric inference, bootstrap, timing or sensitivity experiment was run.

## Deliverables

- `figure2.pdf`, `.svg`, `.png`: 178×106 mm, four metrics, five methods, exact 4/10/19 dB positions, all 60 saved means and pointwise 95% CIs.
- `figure3.pdf`, `.svg`, `.png`: 178×80 mm, Pastries95 then Shopping cart75, two originals plus ten frozen reconstructions. Native 256×256 image pixels are preserved; labels are vector/editable. The 600 dpi composite export does not increase image detail.
- `figure2_data.csv`, `figure3_mapping.csv`: unrounded plotted data and individual image/version/noise/score mapping.
- `table_i.tex`: byte-identical copy of the v1 N2048 table. Its historical source comment remains accurate; the same six metric cells were checked against the unchanged paired CSVs at a08e251.
- `timing_table.tex`: existing 15 mean rows, all three SNRs; `timing_notes.md` explains boundaries, versions and load.
- `captions.md`, `results_outline.md`: English captions, definitions and paragraph points, without a full-paper rewrite.
- `draw.py`, `style.json`, `input_manifest.json`, `inputs/`: reproducible frozen-data drawing and SHA-pinned inputs. No font files are distributed.
- `data_validation.json`, `visual_review.json`: targeted data/artist checks and actual visual inspection.

## Reproduction

Use an existing Python environment with NumPy, Matplotlib and Pillow:

```text
python -B paper/wcl_figures_tables_20261020_v2/draw.py --output /your/new/output/directory
```

The renderer verifies all shipped input hashes, reads frozen summary/paired fields and embeds existing PNG pixels. It never imports a model, decoder or bootstrap implementation. Supply a new output directory to keep reviewed exports intact. Styling is adapted from figures4papers; actual paths and SHA identities of the skills read are in `skills_used.json`.

## Provenance and field mapping

All repository input paths, byte counts and SHA-256 hashes are in `input_manifest.json`. Exact copies under `inputs/` allow deterministic rendering without network or heavy scientific caches.

| Output | Frozen input | Fields / rule |
|---|---|---|
| Fig. 2 | `results/clip_i_common500_20261020_v1/delivery/eight_metric_summary.csv` | `mean`, `ci_low`, `ci_high`; `psnr_db`, `lpips_alex`, `dreamsim`, `clip_image_cosine`; `source_count=500` |
| CLIP comparison wording | Same delivery, `nest_minus_baselines.csv` | Direct source-paired differences and pointwise intervals |
| Fig. 3 | `results/selected_examples_10db_20261020_v1/delivery/display_manifest.csv` | Source95/75, frozen image SHA/version/noise/status; stored `psnr_db`, `lpips_alex` |
| Original image bindings | Existing selected-example delivery manifest and 20261012 selection | Exact original PNG hashes, unchanged source IDs and crop lineage |
| Table I | Adaptive/full N2048 `paired_differences.csv` | Original direct paired means/CIs; adaptive4/10, full19 |
| Timing table | `results/five_method_timing_20261020_v1/delivery/summary_ms_per_image.csv` | `tx_ms_mean`, `rx_ms_mean`, `total_ms_mean`; no re-estimation |

Method map: `EC_VAR_PARTIAL` → NeST-Com; `P100000` → Latent JSCC; `Swin200000` → SwinJSCC; `BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024` → BPG + LDPC; `ARPC_ADAPTED` → ARPC + LDPC. The selected-example/timing aliases `NEST`, `BPG`, `ARPC` are mapped explicitly in the script. No historical raw NeST, old P or old Swin result is substituted.

NeST19 dB retains corrected T1-C PARTIAL output. CLIP-I is raw image–image cosine: no percentage or distance transformation. Lines only join observed working points. All CI endpoints fit the chosen axes, including CLIP-I's displayed 0.5–1.0 window. Very narrow CIs can be partly covered by markers; their actual widths are not inflated.

## Preserved material and assembly note

The v1 tables/figures, complete eight-metric results and full four-example gallery remain unchanged. The old paired-gain/utilization Fig. 3 is archived supporting material; current Fig. 3 is the reconstruction grid.

The user-confirmed `system_diagram_v3_entropy.pptx` was not found in the current Git checkout or local workspace/Downloads inventory. Its artwork is outside this delivery; the agreed Fig. 1 is retained by reference, with an aligned caption. No older diagram is substituted. Final manuscript assembly needs the location of that exact author-held asset.
