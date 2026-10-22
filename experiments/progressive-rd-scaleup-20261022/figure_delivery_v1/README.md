# Closed-statistics figure preparation

Prepared during the six registered training jobs. This code has not rendered actual results and is not a scientific completion receipt. It is separate from all 15 registered runtime files and from the already qualified statistics entry.

Run only after the existing `run_summary_lineage_once.py` has genuinely returned exit0, its child is gone, and the closed delivery manifest verifies. Deploy this independent directory to `experiments/progressive-rd-scaleup-20261022/figure_delivery_v1/` only at that point. Use the existing CPU plotting interpreter with CUDA hidden; record real argv, source SHA, PID/start, stdout, stderr and child wait. Do not run the previous pilot plotters or old summary entry.

The entry reads only registration, the closed summary wait, the sealed delivery manifest and saved scalar CSVs. It neither loads a checkpoint nor scores images, samples a bootstrap bank, or repeats the scientific grid audit. It exports `results/progressive_rd_scaleup_20261022_v1/figures_v1/` exclusively; if that directory exists, diagnose before doing anything else.

Outputs: three 178 ×70 mm rate–distortion plots (mean over two seeds and each individual seed, with common axis limits), plus two 178 ×108 mm direct-paired comparisons covering all six registered contrasts. Every plot has vector PDF, editable-text SVG, 600 dpi PNG, 160 dpi print preview and grayscale preview. The plotted metrics are PSNR, LPIPS and MS-SSIM; all five metrics remain in the original complete result CSVs. No endpoint is dropped and no horizontal coordinate is shifted. Pointwise CI endpoints are drawn independently of means so an unusual percentile interval need not contain its mean.

Read `scientific-figure-making/SKILL.md`, design-theory, common-patterns and demos on 2026-10-22. Applied blue/green/red/gray semantic colors, redundant line/marker encodings, shared legends, white background, compact paper typography and true vector export. Did not copy the examples' ultra-wide canvases or oversized fonts.

Export does not mean visual approval. Open all actual print previews and grayscale previews at the final dimensions after a successful render, inspect labels/legend/CI/cropping/negative values, and save a separate visual review with the exact exported file hashes. `PENDING_ACTUAL_OPEN` remains until that real review; never claim this preparation as final figures.
