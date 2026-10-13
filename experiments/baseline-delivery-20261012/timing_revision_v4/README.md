# Six-method timing v4: constructor compatibility recovery

## Actual attempts and remaining budget

Both v2 and v3 ended with actual worker exit 1 during construction. Both immutable journals have zero reserved/completed timing cases and zero digital packet decodes. v2 constructed VAE/VAR/Dc before the strict module-path failure. v3 passed that point and also loaded the P100k/Swin200k states before encountering the Swin field error. These are real model construction attempts, not zero-model checks. No fresh timing warmup has completed.

v4 remains a separately registered recovery, with no retry loop. All v1-v3 sources, receipts, failures, logs and registrations are preserved. The 1,728 total/864 measured case budget and 2,304 digital packet cap remain entirely unused. Models, policies, data, noise, math, CPU 6/2 threads and FP32/TF32 settings are unchanged.

## Minimal correction

The frozen upstream `vendor/SwinJSCC/net/network.py` constructor parses `args.multiple_snr` into `self.multiple_snr`, a list of integers. It never stores `self.args`. The historical timing build incorrectly addressed `s.native.args.multiple_snr`.

The independent v4 wrapper changes only that one assignment in an in-memory copy of the original build function:

```python
s.native.multiple_snr = list(range(1, 20))
```

An exact replacement and inverse AST equality prove the remaining build body is unchanged. The original source file is never edited. The inherited v3 exact-byte module alias adapter remains constructor-only and restores both loaders before any timed frame.

## Complete interface review

- **Visual/native:** the actual frozen Backend constructs `native.torch`, `native.loaded` containing VAE/VAR/Dc/device, and `backend.models`, `render_function`, `g`, `native_start`. This matches fresh-token encoding, rendering, parameter reporting and native-linkage checks. The new attempt still must actually pass final linkage and warmup checks.
- **Raw whole/partial:** construction reached both endpoints in v3. `cat.entry`, 433-entry header map, backend/legacy/header, serializer and RX `present_actual` return fields match the endpoint. RX derives its prefix and partial values only from the received state. A rejected header returns gray; a rejected body CRC retains the original hard-token KEEP behavior. No TX truth argument enters RX.
- **Entropy whole/partial:** v3 passed the unmodified qualified 144/360-domain constructors with the exact-byte aliases. `transmit`, `receive`, received `rx_profile`, body `crc_accepted`/`parser_accepted`/`payload`, `PartialSourceCodec.encode_endpoints`, inherited whole `encode`, and independent `decode` signatures match. Whole fallback remains whole-only; partial fallback uses the original ordered endpoints and fresh terminated streams. Every failed larger attempt stays inside TX time.
- **P100k:** v3 successfully constructed the selected `BudgetContinuous` from its actual state. Its `transmit(F)` returns B×1024×2 and `receive(y,snr)` requires a per-image SNR vector; the endpoint supplies that shape. `ContinuousDecoder` returns clamped [0,1] images. The existing energy and wave/output identity checks remain authoritative during real warmups.
- **Swin200k:** v3 successfully built the official adapter and loaded its selected state. `swin_replay.transmit` calls `SwinCodec.encode_data`; that calls `native.encoder` with the explicit SNR. `receive` recovers paid context and calls `native.decoder` with the explicit SNR. This route never calls upstream `native.forward` and does not sample from `multiple_snr`. The corrected list is constructor metadata, not a change to the transmitted signal or decoder calculation.
- **First-case bookkeeping:** the fixed order, per-case reservation, two-digital-decoder cap, source token check, wave/RX/output hashes, nesting of TX/RX meters and audit exclusions remain unchanged. New v4 tests use synthetic objects and exercise actual endpoint RX and fallback interfaces; they do not qualify real numerical results.

The review found the Swin attribute mismatch above; no other field/signature mismatch was found. This is bounded static/synthetic evidence, not a claim that all real paths have passed. Actual construction completion, source-token equality, three fresh warmups and three identical measured repetitions remain pending.

## No-model tests

The CPU tests execute the authoritative upstream constructor with inert stand-ins, reproduce the missing `args` failure, validate the corrected integer-list field, verify inverse AST equality, trace explicit 19 dB through actual Swin encode/receive methods, exercise raw/entropy accepted and failed RX branches using synthetic objects, preserve raw KEEP, and confirm fresh fallback accounting and RX argument isolation. No Torch import, real network, actual channel, LDPC packet or image reconstruction is invoked.

`register-recovery` and `verify` only bind/check evidence. `run` is separate and awaits coordinator review and dispatch. It holds predecessor owner locks, uses the same shared-H800 resource gate, and stops on unknown failure without automatic retry.
