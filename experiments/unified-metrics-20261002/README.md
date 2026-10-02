# Unified metrics for frozen reconstructions

This extension implements the user's October 2 request to add the same image metrics to historical N512/N1024 results and both scale-causal experiments. It performs no training, calibration, policy selection, or new holdout evaluation.

## Default evaluation for subsequent results

Keep the original PSNR, LPIPS-Alex, DINOv2 ViT-S/14, latent recovery error and mismatch checks. Add a separate DINOv2 ViT-L/14 image cosine column, OpenAI CLIP ViT-L/14 image cosine, official DISTS, torchvision ResNet50 IMAGENET1K_V2 true-label top1 and original-prediction agreement, official DreamSim ensemble, and MS-SSIM. The two DINOv2 columns use the same original floating RGB preprocessing (224-square bicubic, align_corners=False, antialias=False, ImageNet normalization) and normalized class-token cosine. Record model weights, implementation revisions and transforms. New evaluation metrics must not silently enter calibration or model selection.

All class-conditioned methods are labelled as such, including failed headers; their classification results are supplementary. Main classification conclusions use unconditional methods. D0 and oracle outputs remain reference results. KID and FID remain deferred to a larger independent holdout.

## This registered backfill

The inventory covers N512 and N1024 full 30-method tables, M1 selected policies and equal-token controls, M1 dense source-only rate curves, M2 oracle projections and controls, and M2 actual links if the existing gate admits them. It includes every frozen development row. Calibration candidates and timing measurements are not additional development reconstructions.

Noisy groups retain the registered 100 sources and three noise seeds. M1 source-only curves and clean M2 oracle outputs each have one observation per source. Pairing and bootstrap resample the 100 sources, after averaging a source's noise repeats. Historical image outputs are regenerated using their exact frozen choices and original batching, then checked against original metrics, latent errors and available hashes/events before any new score is accepted. Original numbers remain in the final table.

## Background execution

`prepare_assets.py` downloads pinned public evaluator sources and weights into an ignored output cache, with a separate environment. `supervisor.py` qualifies these models on CPU using synthetic fixtures, waits for the original experiments to finish and publish, then publishes this extension's source, restores/scans frozen reconstructions, computes paired summaries, and publishes results. It exits after verified publication. It never starts another old experiment queue.

`runner.py` uses an exact-pixel cache within each source and commits a checkpoint only when that source's complete expected row inventory has passed replay checks. Duplicate method rows are retained. Changed inputs or mismatched reconstructions stop the run. Completion receipts distinguish engineering fixtures, real model qualification, actual scoring and verified repository delivery.

After loading both the frozen reconstruction models and all new evaluators, `batch_speed.py` qualifies batches 1/2/4/8/16 on 16 fixed synthetic reconstructions. Each candidate gets one warmup and three timed passes. It must match scalar floating scores within absolute `2e-5`, preserve every classification prediction and flag, and stay within 88% of GPU memory by peak process reservation. The fastest valid median time per image determines the registered batch; batch one must pass. No development image is used to choose this execution setting. The selected size and exact qualification receipt hash appear in the scoring registration and completion receipt.

Only new metric scoring is batched. Each source's reference features and classifier baseline are prepared once. At most the selected number of distinct pending RGB images are retained; pending and completed duplicate images share scores while every original row remains in place. Incomplete batches use only smaller qualified powers of two. A scoring error stops execution rather than changing the registered setting. The original replay shapes, random stream and scalar metric definitions stay fixed.

Results are written to `results/unified_metrics_20261002`. Large CSVs are published as lossless shards; `publish.py --restore-tables` reconstructs their original bytes. Read `METRIC_PROTOCOL.md` and `MODEL_SOURCES.md` for model relationships, interpretation limits and source links.
