# Figure 2 caption and accessible description

Use the caption in `caption.tex` with either style. Full source values and bounds
are in `plot_inputs/fig2_plot_data.csv`; A/B use the same file SHA-256.

Alt text: Four panels compare five methods at 4, 10 and 19 dB on common500.
PSNR and prediction agreement generally rise with SNR, while LPIPS and DreamSim
decrease. Adaptive BPG and SwinJSCC have higher PSNR than the NeST-Com methods;
the NeST-Com methods have lower perceptual distances and higher prediction
agreement over these points. Entropy-coded and raw NeST-Com agreement nearly
coincide at19 dB. Error bars are existing source-level pointwise95% intervals.

Long description: The x-axis distances reflect the actual6 dB and9 dB gaps.
Each panel has its own metric units and unbroken linear y-axis. No intermediate
SNR observation, fit, smoothing or significance mark is added. Dark-blue solid
circles identify entropy coding; medium-blue dashed open circles identify raw;
orange squares identify latent-space JSCC; green dash-dot triangles identify
adapted SwinJSCC; purple dotted diamonds identify adaptive BPG + LDPC. A places
the shared legend above a compact grid; B uses taller panels and places it below.
Markers show500-source means after each source's three noises are averaged.
The intervals describe each mean, not a paired method difference. Small or
overlapping intervals remain at their exact values rather than being inflated
or offset. The four panels share the same observation/cohort definitions.
