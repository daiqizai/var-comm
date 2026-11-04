# P1 controller reference, frozen before its results

The first batch uses N=512/1024 and SNR=4/10/19 dB. The P4 grid is not
started. All twelve QPSK/16-QAM/64-QAM and 1/2,2/3,3/4,5/6 MCS are retained.
For each MCS the source alternatives are longest-feasible native prefixes
under a target of slice14 or slice28. These two targets are fixed in advance;
no test-quality-dependent candidate expansion is allowed.

The new 64-complex-symbol control header uses a real (k=38,n=128) LDPC code
and QPSK. Its information fields are version2, MCS4, source packet byte
length16 and CRC16. Model/resolution/CDF are preconfigured for the study.
The body contains the complete independently terminated source packet,
including its72-bit wrapper and CRC32, plus known zero padding to the real
LDPC information dimension. No additional body CRC is silently omitted:
the source CRC32 and known-padding checks are the body acceptance rule.
An endpoint is paid inside the source packet. MCS is recovered from the
current received header, never passed as transmitter truth. The synchronized
frame counter and public session are declared shared context for scrambling.

Real no-noise checks admitted every MCS for both frame lengths (48 frames).
The original wider qualification completed68/72 frames before N2048 q6 r3/4
exceeded the single-codeblock information limit. Those successes and the
original failure are retained; this is not a whole-attempt PASS. N2048 high
rate support is deferred, not repaired by changing the registered code.

Reliability is measured on the first100 calibration images, one registered
noise realization per image and operating point, seed2026110421. Shared
standard noise is used across MCS and targets for each source/frame length.
The resulting grid is2 frame lengths x3 SNRs x12 MCS x2 targets x100 sources
=14,400 actual frames. CPU process time is separately capped at96 hours;
each worker has a12-hour maximum. These are CPU PHY calls, not GPU savings.

Both header and body are actually decoded. Accepted source bytes must map to
an already independently decoded native packet; a new accepted packet without
that identity requires an independent source-decoder stage and is not silently
marked successful. Receiver rejection is gray. Report Wilson95% intervals;
zero observed failures is not a claim of zero underlying failure probability.

A content-independent MCS/target policy minimizes calibration mean DISTS
under this reliability table. The source-informed reference evaluates the
same candidate successful reconstructions at TX and minimizes expected DISTS
for each image. It cannot see future noise. Its complete candidate computation
is recorded; it is an expensive reference, not a learned efficient selector.
Results on the fixed500 test images are an expected-distortion diagnostic
conditional on the measured reliability table, not a new14,400-frame test or
an unconditional deployment bound. Source-paired uncertainty and finite
reliability-sample uncertainty are reported separately. No complex controller
training or P3/P4 extension is triggered automatically.
