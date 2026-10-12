# Final-link HiFi qualitative comparison

Final-checkpoint qualitative comparison on the original fixed16 development sources at N=1024 and13dB. Latent-space JSCC uses the final100k checkpoint; adapted SwinJSCC and the HiFi forward encoder and confirming decoder all use the final200k checkpoint. Swin and HiFi share exactly the same actual received waveform and decoded paid auxiliary information. HiFi uses the frozen unconditional ImageNet ADM and full registered posterior schedule with sampling seed23. Channel seed2001 is fixed for continuous methods; unchanged digital outputs retain seed6201. There is no best-noise selection or image enhancement. All16 sources are delivered, including failures. Dog, broccoli, pastries and cart were user-selected after viewing earlier examples and do not define a quantitative test population. This compares complete generative reception systems, not isolated VAR versus diffusion backbones. Historical Swin80k/HiFi statistics are separate. Shared-server elapsed times are not a fair speed comparison.

All16 pages: `all16_final_link.pdf`. The selected4 and selected3 panels are display options; all outcomes remain available. Each panel has PDF, editable-text SVG and600dpi PNG. Native pixels remain256x256.

`historical_Swin80k_fixed16.csv` reuses full-precision published statistics and intervals, not new final-link statistics. No scoring or bootstrap was run.

Reproduce layouts only: `python experiments/hifi-final-fixed16-20261012/plot_final.py`. Do not rerun the inference entry.
