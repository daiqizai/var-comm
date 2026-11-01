# V9 captions (same LaTeX text)

## figure1

NeST-Com. (a) The protected profile specifies the endpoint and MCS; both terminals reconstruct coding probabilities from the shared frozen VAR and known coarse tokens. (b) Independently terminated token-wise prefixes fit the body capacity; bars are schematic. (c) Received boundary tokens overwrite predicted positions. The merged map updates the residual latent and finer-scale context. Unreceived boundary positions retain their coarse-conditioned predictions.

## figure2_float

Source-referenced reconstruction on common500. (a,b) Seven systems at $N=1024$, averaging three channel outcomes per source including failures. (c,d) Error-free equal-information budgets, including 57 shared overhead bits and Ada's additional 10-bit class; WHOLE/NeST share ceiling $(10,0)$. Points and bars are equal-source means and pointwise 95\% source-bootstrap intervals. Ada is the predicted-class fixed-frame adaptation. Lower values are favorable.

## figure3_float

Selected advantage example at $N=1024$ and 7 dB, illustrating preservation of source structure. Columns show source, WHOLE, adapted Ada and NeST. Values are full-image DISTS$\downarrow$ / DreamSim$\downarrow$ for the first prescribed channel realization. Supplement~S6 gives source IDs, selection rules and additional examples.

## table_i

Within-scale transmission evidence. (A) NeST minus WHOLE under the frozen family policies. (B) Matched granularity: quality on control100 and TX processing on timing32. (C) Ordered receiver interventions on the actual NeST prefixes. Brackets are source-paired 95\% intervals; each block uses its stated population.

## table_ii

Auxiliary quality and processing costs at $N=1024$, 10 dB. Quality uses common500; timing uses 32 sources, one RTX PRO 6000, two CPU cores, batch one, and complete TX/RX processing boundaries. BPG uses CPU only.

## table_iii

Reduced-budget reconstruction on common500 at $N=512$. Means include all failures; PSNR differences use direct source-paired 95\% intervals. All five metrics and paired comparisons are in Supplement~S2.
