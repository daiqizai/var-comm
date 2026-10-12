# Final-link HiFi update: fixed16 development examples

This finite update retains the generative receiver comparison without expanding
the external baseline roster. Main common500 performance curves remain unchanged.
The human instruction chooses a bounded final-link update as the first step;
this registration does not authorize a600-output quantitative study.

## Roles and claims

| Comparator | Question answered | Interpretation |
| --- | --- | --- |
| Adaptive-resolution BPG + LDPC | Traditional digital compression and error protection | External adapted system |
| SwinJSCC (adapted) | Learned image transmission | Final200k checkpoint; budget-limited, not claimed converged |
| Latent-space JSCC | Continuous versus digital transmission with shared visual representation/decoder | In-house controlled baseline, not a native model from another paper |
| HiFi-DiffCom (adapted) | Continuous reception augmented with a generative prior | System-level comparison; not proof that VAR is intrinsically better than diffusion |
| Whole/partial raw and entropy-coded variants | Transmission granularity under matched within-interface permissions | Internal mechanism comparisons |

HiFi conditions posterior sampling on the channel observation with a confirming
constraint. It is not generic postprocessing of an independently selected blurred
image. This study retains the already adapted paid-header implementation, frozen
ImageNet ADM, and original resource-dependent schedule; it does not claim native
HiFi optimality. [DiffCom paper](https://arxiv.org/html/2406.07390v2).

ARPC is relevant prior work: next-scale progressive coding, prediction of unsent
scales, and autoregressive probabilities for entropy coding. Its released recipe
uses a group-masked bitwise quantizer, Infinity-2B, and text conditioning. We do
not relabel an internal variant as ARPC or claim to outperform it. A native
wireless comparison would require its own model, payload and side-information
accounting. No ARPC reproduction or additional external baseline is launched.
[ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a255ea760cfedf42a988d9de6ec902cf-Abstract-Conference.html),
[author code](https://github.com/Joanna-0421/ARPC).

## Fixed finite scope

- Original16 development indices:0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95.
- N1024,13dB; original channel seed2001 and posterior seed23, one realization.
- Exactly16 complete HiFi recoveries. Reuse4 final Swin outputs and4 final latent
  outputs from the previous redraw; generate only12 missing outputs per baseline.
- Frozen final Swin200k supplies BOTH encoder and confirming decoder. Each paired
  Swin/HiFi arm receives the identical actual full waveform. HiFi decodes its paid
  header; no source image, TX truth mask, or free gain enters the receiver API.
- One existing calibration source qualifies native numerical parity and full
  image gradients, using inherited tolerances. No extra diffusion probe, training,
  hyperparameter search, scoring, bootstrap, or model selection is performed.
- Digital reconstructions retain their existing seed6201 observations; cross-family
  noise draws differ. No method chooses its best noise realization.
- Fixed-gray header failures remain included. A runtime failure leaves its case
  reservation and actual process exit intact; no automatic retry is permitted.
- Batch1, FP32, TF32 off, deterministic operations,6/2 CPU threads,6-core affinity,
 16GiB process allocation cap. Shared-server elapsed times are diagnostic only.

The four dog/broccoli/pastries/cart rows were selected by the user after viewing
historical examples. They are display examples, not an unbiased evaluation set.
All16 outcomes are preserved. No mean advantage over these four is computed.

## Historical and final versions

Historical HiFi remains explicitly adapted to Swin80k with a same-observation
Swin80k reference. Its16-source published statistics remain separate. Final-link
photos are not merged into that old table, and neither is inserted into the
common500 main performance curve. No HiFi speedup ratio is reported: comparable
hardware, TX/RX boundaries, and the original exclusive timing gate are still
required before such a claim.

## Reproduction

`run_fixed16.py prepare` registers inputs once; `run_fixed16.py run` is the
one-attempt scientific entry, not a figure rerender command. Do not rerun it after
completion. `plot_final.py` uses saved outputs only and refuses an incomplete or
failed inference run. Delivery retains PDF, editable-text SVG,600dpi PNG, native
pixels, full16 figure pages, English captions and exact provenance.
