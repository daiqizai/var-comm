# Full calibration scoring and freeze

Consumes only closed actual RX output. Each of the original1000 sources retains
two candidates perSNR, three original noise realizations at4/10/19dB, including
gray failures. No evaluation image or score enters calibration.

Original PSNR, LPIPS-Alex and DINO-L setup, checkpoints, numerical flags and pair
computation are retained. The native float32 image and source uint8/255 inputs
are unchanged. Old800 pairs are reused only by the original full pair key and
source proof. The only permitted metric metadata translation changes its
selection-population description fromcal100 tocal1000, checked by exact equality
of all remaining metadata. Original100 reference embeddings can likewise be
carried forward with source/feature hashes and unchanged numerical identity.

At most900 new reference embeddings and7200 new image-pair evaluations. Same
actual pair is scored once and mapped to all its logical frames. Gray remains
float32 0.5. No old score is changed or repeated. Only the original A2b absolute
deadline is used; this phase is bounded2700s inside that window.

Each source first averages its three noise outcomes. Candidate comparison then
averages1000 sources. Select only maximum DINO-L; exact ties use lexicographic
candidate ID. PSNR and LPIPS are diagnostics. No bootstrap, confidence-based
selection, retraining, channel or native reconstruction call occurs here.

Six new synthetic tests include the exact original compute AST, metadata-only
identity translation and complete paired source/noise accounting. All model
imports are blocked in qualification. Real child waits and source-host closure
are required separately. The frozen policy is eligible for A3 only after actual
score/freeze/owner/supervisor exits and root review, never merely file presence.
