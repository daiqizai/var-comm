# Native BPG source-paired comparisons

This supplementary table adds72 new comparisons: native-resolution BPG+LDPC minus each of the published proposed partial-scale+VAR, latent continuous JSCC and adapted SwinJSCC-80k methods, at N1024 and SNR1,4,7,10,13,19, for PSNR, LPIPS-Alex, DINOv2-ViT-L/14 cosine and ConvNeXt source-prediction agreement.

The500 source IDs and their exact order are identical in the two actual completion receipts. Input source means already average each arm's three frozen noise realizations. No missing source intersection, imputation, new metric evaluation or failure filtering is used. All96 selected source-mean groups/48000 values are present. Input tables are verified against their original completion hashes, and all read-only input hashes remain unchanged after computation.

For each new contrast, subtract the two existing500-element source-mean vectors, then use the unchanged pure-NumPy statistics module's interval function. Its draws function is called exactly once to generate one10000x500 source-resampling index array with seed2026100701, shared by all72 new comparisons. These are pointwise95% percentile intervals, with no simultaneous/multiplicity claim. No original mean/CI or old bootstrap result is recomputed or overwritten; single-method CIs are never subtracted.

Direction is always native BPG minus reference. Positive favors BPG for PSNR,DINOv2-L and prediction agreement; negative favors BPG for LPIPS. Agreement values and intervals remain fractions in the CSV; multiply by100 for percentage-point differences, never relative improvement percentages. Agreement is not classification accuracy.

This is post-hoc supplementary analysis, with no policy selection on holdout. Pairing is by source, not a claim of identical transmitted waveforms or shared noisy observations. Native BPG uses2001/2002/2003; the proposed digital system uses6201/6202/6203; latent continuous and Swin use2001/2002/2003 in their respective method-specific channels. Swin19dB remains outside its training and calibration range. No training, model inference, channel simulation, PHY decoding, old-result mutation or new weight download occurs.

Published original result commit:252176e041758ecb2d3e81b6fde5b587e7e17bb7. See pins.json for exact input/code hashes and ordered source IDs; completion.json binds the new outputs.

Reproduce into a fresh directory from the repository root:

```text
python experiments/paper_supplement_20261008/native_bpg_paired.py --root . --out results/paper_supplement_20261008/native_bpg_paired_reproduction
```
