# Fixed development examples with native256 BPG

Sources `[4, 21, 24, 29]`, N1024, 13 dB. Old five columns are pixel-identical to the historical group02. New BPG scores cover all fixed16 sources at this one SNR and first measured repetition; the combined CSV contains the displayed four sources.

`SOURCE_UNFIT` is a gray fallback with no received frame. It remains in the per-image quality table. Native256 BPG and adaptive-downsampling BPG are distinct baselines. No same-observation claim is made across methods.

Core fields: `psnr_db`, `lpips_alex`, `dinov2_vitl14_cosine`, `convnext_top1_source_prediction`. Agreement is a per-image 0/1 value, not classification accuracy. Existing PSNR values retain the precision and computation of their original cache. Source/reference cells are `REFERENCE_NOT_SCORED`; missing method metrics remain empty with explicit `MISSING` reasons, never zero.

Originally missing metric cells: 8; remaining: 0. See `missing_original_metrics.json` for the original inventory and `missing_convnext/completion.json` when filled. No existing metric, reconstruction, channel simulation, bootstrap, or policy was rerun. The BPG job scored 16 cached outputs, with 16 source preparations counted separately. The independent missing-only ConvNeXt supplement adds 8 reconstruction predictions and 0 source predictions; it does not recompute existing PSNR, LPIPS, or DINO-L.

Reproduce with the frozen request: `python experiments/paper_supplement_20261008/a5_bpg_examples/a5_cached_examples.py --request <request.json> --mode collect`, then scheduled `--mode score`, `--mode render`, and `--mode export`. Only score needs the GPU. The score stage does not rerun completed source checkpoints.
