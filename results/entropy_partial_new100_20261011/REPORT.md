# Independent new100 N1024 four-arm confirmation

All four frozen methods use the same 100 previously selected sources at 4, 10 and 19 dB, with seeds 9301, 9302 and 9303. Every actual failure is retained. This table does not subtract old 500-source means.

First average three noises separately per source using the original numpy.mean float64 expression. The unchanged published draw and interval functions use 10000 source bootstrap replicates, seed 2026100701 and pointwise 2.5/97.5 percentiles. One draw matrix serves 48 marginal summaries and 24 predeclared paired comparisons. Exact zero vectors and identical vectors are retained with explicit interval provenance. There is no multiplicity adjustment.

Paired differences are partial minus whole separately for raw and entropy methods; the two contrasts are not added. LPIPS signs are unchanged and negative paired differences favor partial. ConvNeXt measures source-prediction agreement, not accuracy; fraction estimates and interval endpoints are all multiplied by 100 for display in percent or paired percentage points.

Policies were frozen before this population was selected. No source, policy, model, channel, reconstruction or score was rerun by this statistics stage. Original metric files and earlier confidence intervals remain unchanged.
