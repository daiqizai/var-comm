# HiFi-matched development examples

**Population: development, not the unified 500-image holdout.** All 16 historical fixed sources have matching N1024 / 13 dB cache entries for the five requested columns. Group 2 of four consecutive entries in the original order was exported; no quality-based reselection.

Development-only qualitative examples at N = 1024 and SNR = 13 dB. Rows follow entries 5 through 8 of the pre-existing fixed16 source order (indices 4, 21, 24, 29). All columns match the same original uint8 source pixels. Neither source selection nor noise selection uses reconstruction quality. The fixed first noise realization is used: seed 2001 for the continuous, SwinJSCC and HiFi-DiffCom systems, and 6201 for digital transmission. SwinJSCC and HiFi-DiffCom share the exact received waveform; cross-family noise draws are different. These images are development examples and do not provide an independent evaluation on the unified 500-image holdout. All images come from verified existing reconstruction caches; no inference, channel simulation or metric evaluation was repeated. 

Columns show the original, adapted SwinJSCC image transmission, continuous latent-space joint source-channel coding, adapted HiFi-DiffCom diffusion-prior recovery, and partial-scale digital transmission with VAR completion. SwinJSCC uses the frozen 80k-step model and a 256-use protected header plus 768 body uses. HiFi-DiffCom uses this received frame and the frozen unconditional diffusion prior with its full posterior schedule. It has no label or text side information. The digital configuration at this point is m = 8, K = 9; its selected policy and frozen visual-model identity match the final release. No official-optimal or full-convergence claim is made for the adaptations.

Columns show the original, complete-scale transmission with VAR completion, partial-scale transmission without generative completion, and partial-scale transmission with VAR completion. Complete-scale means the selected prefix of complete scales (m = 8, K = 0, 255 tokens), not all source scales. Partial-scale transmission sends m = 8, K = 9 (264 tokens). Both digital configurations have the same modulation and coding permissions. The last two columns use the same actual received tokens and frozen decoder Dc; absent residual contributions are zero in direct decoding, not codebook index zero, and no extra source-image ground truth is supplied.

## Files

`availability_16.csv`: complete matched-source inventory; pixel verification is explicitly marked for the four exported sources. `provenance.csv`: exact archive/key/slot and original float-image hash per column. `selection_and_integrity.json`: source bindings, fixed noise and same-reception proofs. `individual/`: 28 native 256×256 PNG images. PDF/SVG preserve vector text over raster photographs; PNG previews are 600 dpi. No scientific result file or prior holdout figure was changed.

## Reproduction

With NumPy, Pillow and matplotlib installed, from the repository root:

```sh
python scripts/plot_paper_hifi_development_examples.py --group 2 --render
```

The small display bundle is portable. On the original cache host add `--collect` to revalidate the source caches before rendering. Internal method names remain in the scripts and provenance, not on the figure. The companion `plot_paper_visual_comparisons.py` supplies display helpers and labels.
