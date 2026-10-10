# T1 entropy WHOLE physical framing and qualification

Protocol identity: `WCL_T1_ENTROPY_WHOLE_PHY_20261009_V1`.
This is new WCL evidence; it does not modify the frozen raw64 protocols, results, or ledgers.

## Frozen paid frame

Every transmitted frame contains exactly 1,024 complex channel uses: the original paid 68-use convolutional header followed by 956 body uses. Header payload is a 12-bit public profile ID, protected by the original CRC16, six termination bits, and original convolutional coding/rate matching. There is no free family, scale, modulation, rate, length, or per-image information.

The complete public catalogue has 144 profiles in this exact nested order: family `EC_STATIC_WHOLE`, `EC_VAR_WHOLE`; actual m = 4, 5, 6, 7, 8, 9; q = 2, 4, 6; nominal rate = 1/2, 2/3, 3/4, 5/6. IDs are consecutive 0 through 143. Target m remains 7, 8, or 9. The source layer can fall back only because the actual encoded arithmetic stream does not fit, decreasing m one at a time to m4. It must not choose a fallback using reconstruction quality, change rate/modulation after observing a channel outcome, or replace arithmetic bits by raw indices. m4 contains 30 tokens; with positive 24-bit CDF counts its approximately 722-bit worst-case stream fits the minimum 927-bit source capacity. This fallback endpoint was fixed before qualification was run.

For each of the 12 q/r combinations, n = 956q and k = floor(nr). The information word is 13-bit unsigned actual arithmetic length L, the L arithmetic bits, known zero information padding, and CRC16 over all preceding information bits. Source capacity is k - 29; 2 <= L <= k - 29. There are no body termination or frame padding symbols. Source canonical decoding is separate from the PHY parser.

The body uses the original actual Sionna 2.2.0 LDPC5GEncoder/LDPC5GDecoder classes with 20 flooding iterations, boxplus-phi check nodes, sum variable nodes, float32 precision, LLR maximum 20, hard information outputs, and the pinned original decoder configuration. All 12 layouts must pass their actual constructors; unsupported layouts are recorded and stop qualification. No fake encoder, substitute LDPC, theoretical-only admission, or silent candidate-grid shrinkage is permitted. The new wrapper removes the old wrapper's q2/q4-only admission restriction without changing Sionna itself. The existing `main_raw64_packet_adapter.modulate/demap` mappings support q2/q4/q6. The header is the existing `uep_phy.Header`.

## Receiver contract

The actual accepted paid header selects family, actual m, q, rate, k, and n from the whole public catalogue. An incorrectly accepted but legal header is processed using that received profile. The receiver does not receive the transmitted profile, original source tokens, arithmetic stream, target m, or original image. Rejected headers produce no body decode. Body CRC rejection, illegal length, or nonzero known padding produce no source payload. A physically parsed payload still needs the source codec's arithmetic/canonical validation; CRC acceptance alone does not imply correct source recovery.

```python
rt = t1_phy.create_runtime(root, qualification=completion_path)
wave, tx_metadata = rt.transmit(profile_id, payload_bits, counter)
z = t1_phy.standard_noise(source_id, noise_seed)
observed = wave + (10.0 ** (-snr_db / 20.0)) * z
rx = rt.receive(observed, snr_db, counter, rt.profiles, ledger,
                {"event_id": stable_event_id, "phase": "calibration"})
# rx['rx_profile'] describes the actual received profile.
# Pass rx['body']['payload'] to the corresponding source decoder only when
# rx['body'] exists and rx['body']['parser_accepted'] is true.
```

`counter` is a public nonnegative integer. The shared public session defaults to the protocol identity. Scrambling is deterministically derived from protocol, session, and counter. Standard full-frame Gaussian variates have shape (1024, 2), dtype float64, and are seeded only by protocol, stable source identity, and registered noise seed. The same source/noise seed uses the same variates across family, m, q, and rate; SNR scaling is explicit. Fixed nominal constellation energy is 2 per complex symbol, without per-frame normalization. `tx_metadata` records the actual header energy, actual body energy, E_frame, and rho = E_frame / 2048. All scientific frame energy summaries must use actual transmitted image frames, not qualification random bits.

## Independent real qualification

The actual qualifier has a separate durable ledger and a hard cap of 241 decoder calls: each of 12 actual LDPC layouts receives four noiseless and four 60 dB AWGN body roundtrips (96 calls), followed by each of the 144 known paid header IDs and one CRC-valid unknown ID 4095 (145 calls). The four payload lengths are 2, capacity, half capacity, and capacity minus one; patterns include zeros, ones, and fixed-seed random bits. Payload seed is 20261009041. No image is read, no neural model is called, no source quality is measured, and no bootstrap is run. Qualification uses CPU only with two PyTorch threads, one interop thread, deterministic algorithms, and TF32 disabled. Random payload identity, noise identity, actual hard decoded bits, CRC/length/padding parsing, layout/configuration, and measured frame energy are retained per case.

The request binds all relevant legacy PHY helpers, all repository `src` Python/C++/header sources, both new T1 PHY scripts, the independent ledger implementation, actual backend source/version identity, original qualification SHA, actual layout catalogue, execution commit, decoder configuration, and exact budget. Before and after execution these bindings are checked. The formal runtime factory requires a completed 241-call real qualification and verifies the request, complete source closure, actual current constructors/backend/catalogue, and hashed qualification outputs. The local selfcheck is explicitly only format/dispatch engineering; it does not count as real PHY evidence.

The qualifier uses a nonblocking lock and durable reservation before each decoder callback. Repeating an exact completed ledger event reuses its saved result. An unresolved reservation or existing failure blocks automatic rerun; diagnose and preserve that evidence before any separately versioned attempt. A completed qualification is reused without new decoder calls. Global project STOP and qualification STOP files are respected between calls. No old ledger is opened.

## Remote execution by the owning coordinator

Use the existing pinned Sionna CPU environment, not a substitute environment. The following commands are sequential; the owning coordinator supplies its already verified CPU Python path as `CPU_PY`.

```sh
R=/home/liulu/projects/VAR_COMM
O="$R/outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/phy_qualification_v1"
"$CPU_PY" "$R/experiments/wcl-evidence-closure-20261009/scripts/t1_qualify_phy.py" prepare --root "$R" --out "$O"
"$CPU_PY" "$R/experiments/wcl-evidence-closure-20261009/scripts/t1_qualify_phy.py" run --request "$O/request.json"
```

Do not describe the qualification as complete until its real `completion.json` reports PASS, all 12 actual layouts and 144 profiles, 96 body and 145 header decoder calls, and an independent ledger with total 241 and unresolved 0.
