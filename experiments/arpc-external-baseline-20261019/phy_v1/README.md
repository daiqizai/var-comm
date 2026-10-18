# ARPC paid single956 PHY qualification

Independent adapter; the old ep_phy Backend, QAM mapping, convolutional paid
header, LDPC implementation and standard AWGN function retain their source bytes.
The old360-profile factory is not used with the new catalogue. Its completed457
decoder qualification and source identity are verified, never rerun. The new
24-frame independent-process test qualifies this additional ARPC interface.

Public96 profiles are endpoint0(null) or1..7 x q2/4/6 x four rates. IDs fit the
paid12-bit header. Each frame uses68 header+956 body symbols, nominal Es2, actual
LDPC k/n. Body pays13-bit payload length and16-bit CRC, including all known
padding. Complete ARPC byte containers pay text, mode, boundary and termination
bits. No shortening the physical frame for null or no-fit, no per-frame energy
renormalization. Public counter/session scrambling is independent of image data.

The fixed plan is three first hash-selected cal100 sources for each of six
candidate/SNR combinations (18), one fourth-source case at19dB for each of five
distinct selected MCSs (5), and one explicit null-profile full frame (1). These
24 engineering frames never enter image-quality calibration or evaluation. Each
uses original registered standard noise seed4101. At most24 header and24 body
decodes, zero neural-model or scoring calls. Actual received profile chooses the
body decoder; erroneous but accepted fields remain actual. All failures persist.

TX serializes observation files and an image-independent catalogue. Independent
RX has only observation/SNR/public counter, actual paid profile, shared backend,
and source grammar. Its original PFS audit hook blocks source archives, TX files,
registrations, source CDFs, quality scores and model outputs. Post-exit audit may
compare hashes; it never changes received bytes. Parser rejection yields the
registered fixed-failure behavior; no KEEP or truth repair.

On the original frozen_ldpc_py311 CPU environment, first run the15 new synthetic
tests, then register and execute exactly once via execute.py. Controls are
exclusive; stage waits record PID/start/argv/actual exit and retain failure.
600-second engineering wall limit, CPU2/1, CUDA hidden. Source and original native
library hashes are checked before any packet call, so no native rebuild occurs.
Unknown failure stops; no automatic retry or completed-call replay. Registration
and real PASS closure are required before separate cal1000 implementation.
