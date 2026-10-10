# EC partial source and paid-PHY engineering review, v1

Status: local engineering implementation and independent source review passed. No real VAR source qualification, real PHY qualification, calibration search, model inference, channel simulation, or new-source selection was performed for this review. This is not an experiment-completion or launch receipt. The complete calibration/render orchestration is not implemented.

## Source syntax and independent recovery

The new `ep_source_codec.py` preserves the original SHA-bound 24-bit positive integer CDF and 32-bit arithmetic interval. VAR uses null class 1000 and the original frozen model states. Each scale's probabilities depend on received complete coarser scales. At the last partial scale, the first K raster CDF rows encode/decode the selected tokens; the incomplete scale never advances the probability provider. K is one of floor(next-scale size times 1/4, 1/2, 3/4). For m=4..9 these are (6,12,18), (9,18,27), (16,32,48), (25,50,75), (42,84,126), and (64,128,192).

A fresh independent RX provider reconstructs the CDF from recovered tokens. RX accepts only actual family/m/K and received arithmetic bits: no TX tokens, TX probability table, target policy, source ID or ground truth. Canonical re-encoding and bounded 30-bit terminal lookahead are checked. Model/software errors propagate; only source-syntax errors may enter the registered channel-failure rule. Whole K=0 and erroneous-but-valid STATIC header profiles delegate to the unchanged original decoder. Formal construction must load the frozen original train20k static CDF as well as VAR so a received STATIC ID remains decodable.

One fresh TX probability pass can flush multiple canonical endpoints without mutating the arithmetic encoder. The whole endpoint bitstreams match the original encoder in the synthetic regression. This is not slicing a terminated whole stream. Partial fallback tests endpoints in descending actual token count until the first actual bitstream fits; all smaller whole actions remain available. A whole action retains the original whole-only fallback. Missing stream evidence blocks preparation and is not a fabricated transmission failure. Selection uses actual encoded lengths only, with no quality input and no raw-versus-arithmetic substitution.

## Paid header, body and public catalogue

The first 144 public profile dictionaries and IDs remain identical to T1. The extension appends 216 positive-K profiles, IDs 144..359. A paid 12-bit profile ID determines actual received family, m, K, q and code rate. Raster ordering is a fixed protocol constant. No free m/K or position mask enters RX. The original 68-symbol header and 956-symbol body total N1024. The body pays 13 length bits, all arithmetic bits, known-zero padding and CRC16, with source capacity k-29.

The new receipt namespace is separate. The original wire namespace and default session are retained for scrambling and standard AWGN variates; engineering checks confirm identical old-profile metadata, information bits and seeded randomness. Receive uses the full public catalogue and chooses the body/source decoder from the actually received header, including a wrong but legal family. The formal factory rejects the old qualification schema before constructing any backend; it requires the new sealed 360-profile/12-layout qualification and 457 actual decoder calls (96 body plus 361 header), unresolved count zero and all source/input/output bindings. These 457 real calls have not been run.

## Exact reuse boundaries

Unchanged whole transmitted bits or random variates do not alone prove that the old receive result is reusable. The expanded catalogue changes acceptance for CRC-valid IDs 144..359. The old Header.receive receipt replaced an unknown ID with null, losing that ID. Therefore old CRC-valid/unknown-header outcomes require a new metered header decode under the full new catalogue; the old receipt cannot determine their new outcome. Old accepted IDs 0..143 and CRC-rejected outcomes can be reused only when the relevant original received waveform, decoder result, source/model/numerical identity and decision semantics are proven exact. The old 241-call qualification is never presented as the new 457-call qualification.

Old source streams or images may be reused only after actual received state, payload bits, model and numerical identity match. Source ID alone is insufficient. A partial stream is a new independently verified canonical stream, not a truncated whole payload.

## Finite intended budgets and remaining work

The source engineering grid has 24 endpoints: six whole and 18 partial. The proposed first32 real source gate costs at most 32 TX provider passes plus 576 independent partial RX decodes; missing whole m4/m5 endpoints add at most 64 independent RX decodes. These are planned bounds, not completed calls.

The candidate grid is 144: target m7/m8/m9, four K choices including zero, and 12 modulation/rate choices. The fixed first100 calibration pilot is 43,200 logical frames with an 86,400 packet-call upper bound before exact reuse. Per SNR, pilot top3 union the original final whole winner gives at most four full-calibration finalists, at most 36,000 logical frames / 72,000 packet calls over the original1000 and three noises. Every original whole action remains in the pilot and each original final whole winner remains in the full shortlist. The separate new100 confirmation has 3 SNRs x 3 noises x 4 methods = 3,600 frames / 7,200 packet-call upper bound. Qualification is separate. Actual source/cache reuse must be recorded rather than assumed.

The end-to-end source owner, image recovery, actual calibration/confirmation orchestration, gates, durable call budgets and dedicated TX timing consumer remain to be implemented and admitted. Online TX timing must include fresh tokenization, VAR probabilities, arithmetic/fallback processing and paid channel encoding; prepared source cache reads are not online transmission latency. Shared probability computation may improve the algorithm but requires explicit disclosure and comparable whole-method timing. No launch is authorized by these engineering passes.

## Actual local validation

- `test_ep_source_codec.py`: 17 tests passed, 0.655 seconds, actual exit 0. All 18 partial independent roundtrips used a positive fake integer CDF conditioned only on explicit recovered coarse tokens. Tests cover all24 endpoint/plan agreements, all144 fallback agreements, original whole bits, STATIC dispatch, other valid payload acceptance without truth rejection, noncanonical/invalid streams and software-error propagation.
- `test_ep_phy.py`: 4 tests passed, 0.305 seconds, actual exit 0. No backend, real decoder or channel was executed.
- Additional local read-only engineering probes passed: all144 original profiles equal; 9 exact scramble cases; 6 exact noise cases; 6 actual-profile family dispatch cases; old qualification rejected before model/backend construction.
- Test source roots prefer the normal repository `src/var_comm` files when present, otherwise the local audit snapshot. The original source SHA checks remain mandatory. The snapshot is a local test fallback, not a required private runtime dependency.

No tests were rerun to write this report. The companion JSON pins the reviewed bytes. Historical snapshot paths describe locally inspected evidence, not a fresh remote-HEAD inspection. Real source and PHY qualification remains pending.
