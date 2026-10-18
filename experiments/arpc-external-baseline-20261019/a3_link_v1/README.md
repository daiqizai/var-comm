# Fixed900-frame evaluation link

This is a population/finite-limit adapter of the qualified and closed calibration
PHY. The original backend, full96paid profiles,13bit length,16bit CRC,68+956symbols,
N1024,Es2, source parser, QAM, LDPC and standard AWGN functions remain unchanged.
The frozen calibration policy contributes exactly one candidate at4/10/19dB.
Actual length fallback and null-profile handling remain unchanged. Failures keep
all900logical frames. Three registered evaluation seeds6201/6202/6203 replace
calibration seeds, with a distinct public counter namespace3000000.

No old physical observation is reused for a different ARPC waveform. Within this
evaluation, an identical waveform can be encoded once, while each SNR/noise has
its own actual observation and recorded receive operation. Receiver inputs stay
observations, SNR, public counter and complete public catalogue only.

Prepare requires source100closure, fullcal1000frozenpolicy and the original24frame
engineering proof. A3 shared deadline is inherited; no new registration window.
Five synthetic tests are new population/limit/adaptation checks. No repeated PHY
qualification, old experiment, training or numerical-policy change is allowed.
