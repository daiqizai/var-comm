# Frozen-policy formal PHY, common500 and Kodak24

Engineering entry only until source closure, qualification, registration and
actual one-time execution. Population loops do not imply completed events.

The complete cal1000 policy is already frozen with SHA
68d12e9b19006e8509b3ae5a62132e73044b88ad29ca041f96b436144923ae8c:
4dB q2 r2/3; 10dB q4 r2/3; 19dB q6 r5/6. Target256 with actual longest
independently terminated prefix fitting. There is no selection on formal scores.

Use original source_index0..499 for common500,0..23 for Kodak24, never combine
their indices. Original stable standard_noise(source_id,noise_seed), common500
6201/6202/6203 and Kodak2001/2002/2003, Es2 and observed=wave+noise*10^(-SNR/20).
New paid-frame event namespaces explicitly identify this Ada formal experiment;
counter=6000000+4096*index+profile for common500 and9000000+4096*index+profile
for Kodak. Event identity cannot collide with the completed calibration run.
Noise realization identity remains source/noise-seed based, not claimed identical
to all other methods. Statistics are source paired.

Three fixed MCS/SNR choices and three noises give4500+216=4716 logical/physical
events, at most9432 header/body decodes. No source model, RX generation, metric
inference or bootstrap runs in this worker. Calibration's18000 events and original
24 PHY qualifications remain preserved: cumulative22740 events/45480 packets.

Original paid profile catalogue, length13+class10+CRC16, 68header+956body,
LDPC/QAM math and source parsing are unchanged. Reuse original choose and receive.
The receiver sees only the actual observation, publicSNR/counter/catalogue and
ledger. All gray/null/failure logical frames remain. Post-output exactness audit
never corrects received tokens or classes. Unknown failures retain their ledger
and stop; no automatic retries. Independent RX and four-metric stages follow
only actual closed PHY outputs.
