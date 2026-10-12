# Final Swin200k + HiFi fixed16 delivery

The original16 development sources at N1024,13dB have complete final-link
Swin/HiFi paired observations and final latent-space outputs. All16 full HiFi
posterior calls actually finished, with253 reverse steps each and no header
failure. The owner actually exited0. Eight previously completed final baseline
frames were verified and reused; only24 missing baseline frames were generated.

Both HiFi's forward encoder and confirming decoder use the frozen Swin200k
checkpoint. The measured waveform is identical to its paired Swin reference;
received paid-header fields supply the mask and gain. One calibration source
passed native forward parity (zero maximum difference) and inherited numerical
gradient checks before the full recovery. Original23 author-code files match
historical qualification hashes. Frozen ADM weights match the original digest.
No training, quality scoring, bootstrap, hyperparameter selection, new external
model, or common500 quantitative curve was run or changed.

Complete16 examples and the user's selected dog/broccoli/pastries/cart panels are
in `paper/figures/hifi_final_fixed16_20261012/`. Display selection does not define
a quantitative sample. The old Swin80k/HiFi table retains the original full-
precision means and paired intervals in separate CSVs, with explicit version and
sample-size labels. We report system-level generative reception, not superiority
of VAR over diffusion in a controlled backbone experiment. ARPC is discussed as
related work; no native reproduction or superiority claim is made.

All layouts have PDF/SVG/600dpi PNG and native-pixel provenance. HiFi diagnostic
runtime receipts came from shared-GPU execution; they are not qualified timings
or a basis for speedup ratios. The original six-method timing still requires its
unchanged exclusive-resource window. The optional100-source/600-recovery HiFi
study is not started.
