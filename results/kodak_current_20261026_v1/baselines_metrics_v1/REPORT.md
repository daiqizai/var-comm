# Kodak24 baseline reuse and four-metric scoring

All 1296 logical frames across six configurations, three SNRs and three original noise labels have four finite full-precision scores in `frame_metrics.csv`. Each retains its source ID, receive status, image path, file SHA, array SHA and original input record. No receive-result filtering occurred.

Final P100k and Swin200k reuse 432 stored native outputs. BPG reuses 216 logical frames represented by 72 unique native archives. The BPG archives were recovered by checking the original migration-volume hashes and every selected blob hash. No BPG encoding, channel decoding or reconstruction was repeated. Native BPG scoring-array hashes and embedded source RGB match; the historical native-output serializer remains separately documented.

All 648 baseline PSNR values and 648 LPIPS-Alex values are reused after source, reconstruction, formula, preprocessing and weight identity checks. Historical hardware provenance is retained; this is not a claim of cross-device bitwise equivalence. Old raw NeST images or scores are not substituted for new PARTIAL/WHOLE outputs. Only old full source tokens were separately restored for the NeST lane.

Actual new scoring:

- 703 distinct source/reconstruction pairs and 727 distinct feature inputs, including 24 source images.
- 199 PSNR evaluations and 199 LPIPS-Alex pair evaluations for new native outputs.
- 703 DreamSim distances and 727 FP32 ensemble embedding calls.
- 703 CLIP-I cosine scores and 727 FP32 ViT-L/14 image embedding calls.
- 593 duplicate logical pairs reuse exact score identities in each role.
- Model constructions: LPIPS 1, DreamSim 1, CLIP 1.

DreamSim and CLIP use the exact existing main loader, weights, RGB-float preprocessing and tensor 224 bicubic antialias rules. PSNR/LPIPS use the original AST-projected formulas and frozen weights. CLIP-I is an unscaled image-image cosine; no caption or softmax is involved.

The H800 three-metric role completed in its original Torch 2.11.0+cu128 environment; the PRO CLIP role completed in its original Torch 2.10.0+cu130 environment. Both use batch 1, FP32, TF32 off and deterministic flags. Host identities, PCI buses and distinct CUDA/NVML UUID contracts are recorded. All role owners, supervisors and scientific children actually exited 0 and were verified gone on their own execution host. CPU merging also actually exited 0; `merge_closure.json` binds the evidence.

No new training, source coding, channel experiment, reconstruction or bootstrap ran in this lane. The parent performs the registered Kodak24 source-bootstrap analysis. The statistical frame denominator remains 1296 regardless of compute deduplication.

Metadata-only local inline-command syntax errors were retained before affected remote bodies executed. The original 16 synthetic tests passed once; four admission guards and four host-closure guards were subsequently tested once each. No scientific score was retried. NeST and ARPC scientific-lane historical initialization, saved-packet finalization and denied-source-code-read observations are preserved and pinned in `closed_inputs_v1.json`; original failures were not rewritten as successful runs.

This metric package does not compute bootstrap results or scientific rankings. Use the parent report for source means, intervals and manuscript conclusions.
