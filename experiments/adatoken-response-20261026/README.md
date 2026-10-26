# Ada comparison and controlled endpoint evidence

The user authorized bounded A/B/C studies on 2026-10-26 and subsequently put
the predicted-class Ada fixed-frame comparison and controlled endpoint study
at equal first priority. `C_PREDICTED_CLASS_PLAN.md` supersedes the unlaunched
null-class main-comparison route in the historical `PLAN_zh.md`.

## Completed evidence

- A: four nested endpoint sets, fixed models/MCS/maximum endpoint, 100 existing
  sources, N1024, 4/10/19 dB and three noises. All 3,600 logical results,
  48 means and 72 direct source-paired intervals are closed. Read
  `../../results/adatoken_response_20261026_v1/a_delivery/REPORT.md`.
- B: direct correction versus subsequent propagation, 100 existing sources,
  10/19 dB and three noises. All 1,800 logical arm results and four-metric
  statistics are closed. No new B transmission occurred. Read
  `../../results/adatoken_response_20261026_v1/b_delivery/REPORT.md`.
- C: official LlamaGen-L/VQ16, predicted and paid class, independent source
  decoding and actual-frame qualification have passed. The actual24 frames
  include22 accepted receptions and2 gray rejections. Qualification is an
  operational gate, not a comparative performance result.

The remaining992 calibration sources were uniquely launched after reusing
the eight qualified conditional source encodes. Calibration selection and
common500/Kodak formal testing are not complete at this milestone. Never
infer completion from a registration. Read actual control waits and ledgers
on PFS before continuing; do not duplicate active or completed stages.

## Interpretation

At10dB, quarter-minus-whole PSNR is +0.660149 dB. Eighth-minus-quarter is
+0.122946 dB, while DreamSim and CLIP-I intervals for that increment cross
zero. Better packing does not establish monotonic improvement in every
quality metric or justify changing the main configuration after testing.

At19dB the B propagation increment is exactly zero: the positive within-scale
prefixes are already in the final scale. The old raw R1 NO_GO remains intact.

The null-class qualification is named LlamaGen + LDPC (Ada-inspired), not
an unqualified Ada reproduction. The main C comparison is named
**Ada-TokenCom (predicted-class, fixed-frame adaptation)**. Model, tokenizer,
prior and image decoder differ across systems; their total difference is
not an isolated endpoint-granularity effect.

No training, new channel model, ARQ, controller reproduction or main-curve
replacement is included. Large arrays, weights and ledgers remain on PFS;
backups are deferred. Failures and additive numerical/import revisions are
retained. Independent rendering helpers only consume closed statistics.
