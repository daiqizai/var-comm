# NeST-Com figure and table captions

## Figure 1

NeST-Com. (a) The selected endpoint and fixed MCS determine the protected profile, combined with the coded body before the channel. Both terminals reconstruct coding probabilities from shared VAR and known coarse tokens. (b) Independently terminated prefixes undergo actual-length fitting up to the calibrated target; bars are schematic. (c) Received tokens overwrite boundary predictions. Each remaining scale recursively updates the accumulated state and constructs the next context before final decoding. Unreceived boundary positions retain their coarse-conditioned predictions.

## Figure 2

Source-referenced reconstruction on 500 ImageNet images. (a,b) Six methods at $N=1024$, including the shared-model WHOLE reference; three channel outcomes per source include gray failures. (c,d) NeST-Com, WHOLE and Ada-TokenCom at equal, error-free information budgets, including 57 shared protocol bits and Ada-TokenCom's additional 10-bit class; WHOLE/NeST share ceiling $(10,0)$. Points and error bars are equal-source means and pointwise 95\% source-bootstrap confidence intervals after within-source averaging. Ada-TokenCom is the predicted-class, fixed-frame adaptation. Lower values are favorable.

## Figure 3

Selected reconstruction examples at $N=1024$ and 10 dB. Columns show source, WHOLE, SwinJSCC, the predicted-class fixed-frame Ada-TokenCom adaptation, and NeST-Com. Values are full-image DISTS$\downarrow$ / DreamSim$\downarrow$ for each method's first prescribed channel realization. Supplement~S6 gives source identities, selection details and additional examples.

## Table I

Within-scale transmission. (A) Shared-model NeST--WHOLE differences. (B) Matched endpoint granularity. (C) Ordered receiver interventions. Brackets are source-paired 95\% intervals; sample sizes are specified for each block.

## Table II

Auxiliary quality and processing costs at $N=1024$, 10 dB. Quality uses 500 sources; timing uses 32 sources, one RTX PRO 6000, two CPU cores, batch one, and complete TX/RX processing boundaries. BPG uses CPU only.

## Table III

Reduced-budget reconstruction on 500 sources at $N=512$. Means include all failures; PSNR differences use direct source-paired 95\% intervals. All five metrics and paired comparisons are in Supplement~S2.
