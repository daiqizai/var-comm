# Frozen-policy common500 evaluation at 7 dB

This is the user-authorized extension of Fig. 2 to a fourth SNR, following the separately completed 100-calibration-source/two-development-image study. Six systems, the original common500 source identities, three channel repetitions per source, N=1024: **9,000 logical results**. No training, new calibration, or parameter search is part of this extension. The new GPU ceiling is 12 device-hours.

The seven-system development comparison is not the population used for Fig. 2. WHOLE is excluded from this six-system extension. Existing 4/10/19 dB results, internal controls, and paused research branches are preserved.

## Frozen inputs and interpretation

- Exact policy and source identities are in `results/nestcom_7db_common500_20261030_v1/protocol.json`, `source_manifest.json`, and `cache_identity.json`.
- NeST, Ada and ARPC reuse all 1,500 existing digital source caches. NeST uses the 25-quarter endpoint catalogue, Ada all independently terminated prefixes 1..256 and its paid predicted class, and ARPC its paid complete source containers. Frozen 7-dB MCSs are 16-QAM rate 1/2, 16-QAM rate 1/2, and QPSK rate 3/4, respectively. BPG retains its pre-existing 7-dB, 16-QAM rate 1/2 policy.
- The first three policies were selected using 100 calibration images and one channel realization; BPG's historical policy used 100 images and three realizations with PSNR selection. These are not the 1,000-image calibration procedure used for the original main points. None is reselected on common500.
- All physical/channel outcomes are retained, including failures as fixed gray outputs. Identical actual received inputs may share an exact receiver reconstruction, but all three original events remain in the statistical denominator. Method-specific noise identities are preserved; BPG's new stage key is explicitly `COMMON500_7DB`. Equal seed labels do not imply shared AWGN vectors across methods or SNRs.
- Metrics are the frozen native-float PSNR, LPIPS-Alex and DreamSim definitions. No CLIP-I result is added. Fig. 2 uses LPIPS-Alex and DreamSim; Table II retains its original 4/10/19 dB scope.
- Average the three repetitions within each source, then use the original fixed 10,000×500 source-bootstrap bank. Paired differences are calculated directly from per-source differences, not from marginal intervals.

## Finite execution and preservation

`prepare_v2.py` verifies source identity and the registered cache keys. The original `prepare.py` preflight failure is retained: additional unused NPZ keys had no quarter-grid hash entry, and the failure happened before any science. `g_import_v1.py` preserves the original import-path bytes; the independent prelaunch import correction occurred before any registration or subprocess launch. No successful scientific call was repeated for either engineering correction.

`chain.py` is the unique finite owner of transmitter, channel and receiver stages. `dispatch.py` registers immutable source hashes and GPU reservations; `control.py` records real PID/start ticks, waits, timeout and exit status. Source generation and reception use separate processes. Actual receiver inputs cannot be repaired using transmitted source truth. Unknown failures are preserved and stop automatic continuation.

After the chain actually exits successfully, `prepare_metrics.py` verifies all 9,000 records and native arrays before a single registered `metrics.py` stage. `summarize.py` performs the one planned bootstrap pass. The local bounded continuation has an exclusive claim and never retries scientific stages. It is not a recurring scheduler.

Run evidence, numeric results and actual compute/reuse counts are delivered under `results/nestcom_7db_common500_20261030_v1/`. Arrays, full channel records, feature caches and logs remain on PFS under `nestcom_7db_common500_20261030_v1`. A completed registration must not be relaunched; these files document the executed lineage rather than a command to duplicate it.
