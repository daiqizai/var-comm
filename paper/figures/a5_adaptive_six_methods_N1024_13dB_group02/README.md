# Fixed development examples with adaptive BPG

This is a separate six-column figure for sources `[4,21,24,29]`, N1024, 13 dB. The first five columns and all 64 existing method-metric cells are reused unchanged, including the previously filled eight ConvNeXt values; the 16 source-reference cells are not scored. The original native256 BPG six-column figure is retained.

The new baseline follows the existing adaptive source rule and the modulation/coding choice frozen on 100 calibration sources. A separate fixed16, single-SNR, single-noise receive job generated its actual float RGB under a 32-PHY-call cap. This figure script does not reconstruct or simulate a channel. All failures remain present.

New quality calls: 15; new reference feature preparations: 15; original-reference ConvNeXt predictions reused: 15; exact native image metric reuses: 1. No original scores or bootstrap were recomputed. The source feature count is separate and does not claim to remove legacy reference computations internal to the frozen scorer.

The four core fields are PSNR (dB), LPIPS-Alex, DINOv2 ViT-L/14 cosine, and ConvNeXt source-prediction agreement (per-image 0/1, not label accuracy). Every metric uses original float RGB. Same source and SNR do not imply the same received observation; see the caption and per-image metadata. These examples do not estimate holdout success rates.

Reproduction: use the prepared request with `adaptive_cached_examples.py --request <request.json> --mode collect`, then scheduled `--mode score`, CPU `--mode render`, and `--mode export`. The read-only dependencies and exact receive completion SHA are in `collection.json`.
