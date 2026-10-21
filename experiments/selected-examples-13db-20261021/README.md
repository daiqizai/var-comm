# Four selected development examples at 13 dB

This extension was requested on 2026-10-21. It does not change the paper's
10-dB examples or common500 results. Source indices remain 21, 92, 95, 75.
The exact original preprocessing, checkpoints and native receive mathematics
are bound through the completed five-method timing and 10-dB pixel deliveries.

## Policy fixed before execution

There is no calibrated 13-dB digital-policy entry in the frozen 4/10/19-dB
table. For this qualitative check, use the conservative lower-grid rule:
select the existing policy at the largest registered SNR no greater than the
requested SNR. Therefore 13 dB uses the 10-dB target endpoint and MCS,
including each source's original actual terminated-length fallback. This is
an explicitly declared extension, not a newly calibrated optimum at 13 dB.
Reuse the exact 10-dB terminated NeST, ARPC and BPG source payloads. Generate
and actually decode the 13-dB observations with the existing seed/counter
derivation. Do not select seeds or retune policy after seeing images.

Latent JSCC and SwinJSCC use their existing 13-dB SNR-conditioned receivers.
Swin's original native supported SNR list already includes 13. Fixed seeds
are 6201 for NeST/ARPC and 2001 for P/Swin/BPG, with the original method-
specific namespaces (not identical cross-method noise realizations).

## Finite work and reuse

20 logical N1024 events, each 1024 complex symbols, original Es=2 convention.
12 digital source payloads reused: zero caption/tokenizer/probability calls.
At most 8 continuous encoder calls and 20 new reconstructions; initially
execute only the eight continuous reconstructions. For a digital result,
reuse old pixels and scores only after equality of the entire actually
received information, profile, acceptance and status against the 10-dB
receiver input. A different digital receiver input stops assembly for a
separately recorded receive-only continuation; no TX truth repair is allowed.
Only PSNR and LPIPS-Alex are evaluated for missing image/source pairs, at
most 20 pairs. No training, bootstrap, timing measurement, common500 run or
ARPC sensitivity rerun. Model setup is not a timing result.

RX runs separately with only actual observations and shared model/config.
Its audit boundary denies source archives and TX evidence. CPU assembly
and quality evaluation read source truth only after outputs are fixed.
Keep each reservation, completion, failure, PID/start identity and actual
child.wait. Exclusive directories prevent silent reruns. Never overwrite
an engineering failure or repeat a completed scientific call.
