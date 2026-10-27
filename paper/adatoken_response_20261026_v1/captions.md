# Figure captions

**Fig. 2 (replacement).** Image communication on common500 at N=1024 complex
symbols per image over AWGN channels. Each source averages three fixed noise
realizations; error bars are pointwise 95% source-bootstrap confidence intervals
over 500 sources. Ada-TokenCom denotes the predicted-class fixed-frame adaptation,
including paid class signaling. LPIPS uses AlexNet; CLIP-I is the OpenAI CLIP
ViT-L/14 image–image cosine similarity. Lower LPIPS and DreamSim and higher PSNR
and CLIP-I are preferable.

**Supplementary Kodak figure.** The same frozen systems evaluated separately on
24 fixed Kodak RGB256 crops, with three original noise realizations per source.
Error bars use the existing 24-source bootstrap bank. The WHOLE family remains
in the complete tables but is not a seventh system in this figure. Ada-TokenCom
has the same predicted-class, fixed-frame meaning as in Fig. 2.

Neither curve uses smoothed or shifted observations. Direct source-paired
NeST-minus-Ada contrasts are supplied separately; cross-method noise namespaces
may differ. The adaptation does not reproduce long-term-budget control or ARQ.
