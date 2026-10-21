# Current five-method 13 dB development candidates

Status: COMPLETE. This is a fixed four-source qualitative check, not common500.
Sources and order are dog21, broccoli92, pastries95, shopping-cart75 from the
original selected-development identity. All source pixels/crops are unchanged.
The current entropy-coded NeST, final P100k, adapted Swin200k, adaptive BPG and
adapted ARPC are used. Old 13-dB raw/HiFi outputs are not substituted.

## Policy fixed before execution

The frozen digital policy grid has 4/10/19 dB, without a separate calibrated
13-dB entry. For this check, select the largest registered SNR not exceeding
13 dB, i.e. the 10-dB endpoint/MCS, and evaluate the actual channel at 13 dB.
Per-image actual terminated-length fallback and every paid field are unchanged.
This is a declared conservative mapping, not an assertion of optimal 13-dB
adaptation. N=1024, 68 header + 956 body symbols, Es=2. Continuous methods use
their existing 13-dB SNR input. Original fixed noise IDs, counters and method
namespaces are retained; cross-method noise identity is not claimed.

## Actual work and reuse

- 20 logical events: 12 digital and 8 continuous; all accepted, 0 rejects/unfit.
- Reuse 12 exact digital source payloads; 0 captions, tokenizers or probability
  traversals. Actual digital decode: 12 header + 12 body calls.
- All 12 digital received-information identities exactly match 10 dB, including
  received payload, profile and status. Their native reconstructions and single
  realization PSNR/LPIPS are reused without model calls.
- Continuous methods: 8 encoder calls and 8 saved receiver outputs; 8 missing
  PSNR/LPIPS pairs computed using the frozen native-float scoring path.
- 4 original + 20 reconstruction lossless PNGs; all manifest scores refer to
  the actual displayed single realization. No PNG-based rescoring, bootstrap,
  calibration, training, timing, or main-grid expansion.

Two initialization failures are retained: the first PHY interpreter lacked
Sionna, and the first scoring import lacked the original search path. Both
failed before their first scientific/model/pair call. Additive pinned recovery
entries preserve the original attempts and completed stages. The historical
outer waits remain failures; required scientific stages and the scoring
continuation have actual wait 0. Successful scientific calls repeated: 0.

## Interpretation and deliverables

NeST, ARPC and BPG retain identical pixels for all four sources. Raising SNR
alone did not change their already-correct digital payloads under this mapping.
Latent JSCC and SwinJSCC show small PSNR increases and LPIPS decreases on each
of these four sources. See comparison_10_vs_13.csv for full-precision values.
These selected cases are not population statistics or evidence of a newly
optimized 13-dB policy. Keep the confirmed 10-dB pastries/cart figure in the
paper; the complete 13-dB four-row gallery is available for comparison.

- display_manifest.csv: source, method/version, noise, status, native and PNG
  identities and PSNR/LPIPS.
- image_manifest.json: all 24 individual lossless PNG checksums.
- completion.json: actual counts and disclosed initialization failures.
- ../execution_receipts/: launch, actual waits and preserved failures.
- ../../../paper/figures/selected_examples_13db_20261021_v1/: complete gallery,
  individual PNGs and deterministic drawing script.
