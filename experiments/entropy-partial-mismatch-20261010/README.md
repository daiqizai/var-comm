# Entropy partial scales and one-step configuration mismatch

## Original100 source streams complete; pilot PHY running (2026-10-10)

The original calibration first100 now have actual H800 arithmetic streams: the closed first32 streams are reused, and the remaining68 sources completed fresh TX with exactly one model load and680 prior-scale calls. The actual owner exited0 after159.87s; all reservations closed. The68 new sources add no encoder, independent RX, reconstruction, PHY or quality calls. Their1,632 stored endpoint streams were independently checked against the exported receipts. Earlier source32, PHY457 and image48 engineering gates remain closed and are not replayed.

The separate CPU-only pilot PHY owner has actually started for100 original calibration sources,144 frozen candidates,3 SNRs and seed4101:43,200 logical frames, with at most86,400 new packet decodes before individually verified observation reuse. It uses2 CPU cores, hides CUDA and has a finite6-hour execution window. Its request SHA is `34a3538a57650dcc54c00abcf9ca45b3cde2dfc11e02d125b96e42555accd478`. This publication records a running phase, not pilot completion. Independent reconstruction and the four quality metrics remain pending; finalist ranking, full1000 calibration, new100 confirmation and mismatch evaluation remain NOT_RUN. The original whole winner is mandatory in the later finalist union.

The pinned DINOv2-L, ConvNeXt and LPIPS assets are restored for a minimal four-metric adapter; restoration itself performed no model inference. Historical source-exposure auditing is still open, so no new confirmation source has been selected. Published science, frozen protocols, all previous failures and paused periodic monitors are preserved.

[Closed source100 receipt](reports/h800_ep_pilot_source100_v1.json).

## H800 actual48 image link gate completed (2026-10-10)

The fixed first4 original calibration sources at4/10/19 dB, seed4101 and four registered targets per point completed all48 actual paid-channel and reconstruction frames. The PHY ledger closed96/96 decodes; the visual ledger closed model1, source RX48, render48, Dc48 and prior-scale866 within the960 cap, with no unresolved calls. Thirty reconstructions used a received positive-K profile and18 used a whole-scale profile; no fixed-gray output occurred in this small gate. All48 exported float32 image arrays and their frame bindings were independently verified. The CPU and GPU owners exited0 after29.18s and183.39s; these are engineering owner durations, not online timing measurements.

This closes the source, physical-link and image-link engineering prerequisites. It does not establish comparative quality. Pilot ranking, full calibration, new100 confirmation and mismatch evaluation remain **NOT_RUN** at this snapshot. Continue with the original100-source,144-candidate calibration pilot after its exact missing assets and finite execution registration are complete. Reuse only individually verified source/event states, and preserve every prior gate call. Historical source-exposure auditing remains open; no new100 pixels have been selected or inspected.

[Closed image gate report](reports/h800_ep48_link_gate_v1.json).

## H800 first32 source and real PHY gates completed (2026-10-10)

Two actual component gates are closed. The original calibration first32 source gate completed 32 fresh H800 TX passes, 576 independent partial RX decodes and 64 m4/m5 whole RX decodes; all 640 decoded-token comparisons and independently recomputed CDF witnesses passed. The exact budget closed at model load1, TX32, RX640 and prior-scale4928, with encoder/render/Dc0 and no unresolved calls. The actual owner exited0 after419.63s. Fresh whole m6-m9 streams were generated but their whole RX endpoints were outside this gate. The separate real PHY gate passed all96 body and361 header decodes (457/457, unresolved0). Its first attempt stopped before any decode because the visual environment lacks Sionna; the successful fresh attempt used the already-restored frozen LDPC environment and unchanged backend identity checks. Earlier cross-host failures remain unchanged.

These are component qualifications, not quality results. At this publication snapshot the 48-frame image link gate is pending; formal entropy-partial calibration, new100 confirmation and mismatch evaluation remain **NOT_RUN**. Proceed to the fixed48 image link gate using exact first4 new-host streams and the complete paid-profile catalogue before calibration. Historical exposure auditing remains open, and no new100 source selection is admitted. Historical periodic monitors stay paused.

[Closed source gate](reports/h800_ep32_source_gate_v1.json) and [closed PHY gate](reports/h800_ep457_phy_v1.json).

Earlier single-source H800 diagnostic outcome: **PASS_H800_SOURCE0_M4_SAME_HOST_CODEC_ONLY**. The shared PFS runtime and frozen model bytes were reused; this host's native libraries were separately observed. The earlier H800 cross-host RX loaded the models, then rejected the saved leo TX at the first exact CDF comparison (model load1 and prior-scale1 completed; one RX reservation remains unresolved). The separately registered same-host check completed one actual H800 source TX and independent m4 RX: the 30 decoded tokens and all four independently computed RX CDF hashes match its fresh H800 TX witnesses. The actual budget closed at model load1, TX1, RX1 and prior-scale13, with encoder/render/Dc0. This is one existing source and m4 only. It does not establish old-host bitstream compatibility, full source/PHY/reconstruction qualification or a timing result. Hardware and native-library effects are not uniquely separated. The v5 bitstream failure, leo SIGKILL outcome and H800 cross-host CDF failure remain unchanged. **Formal entropy-partial calibration, new100 confirmation and mismatch evaluation remain NOT_RUN.** See the [H800 cross-host report](reports/h800_single_rx_v1.json) and [same-host codec report](reports/h800_same_host_codec_v1.json).

Historical leo single-RX v2 outcome: **STOPPED_SIGKILL_DURING_MODEL_LOAD_CAUSE_UNDETERMINED**. The GPU child was terminated by SIGKILL during model loading, before that call completed. The terminating actor and cause are not established; this is not attributed to OOM. Independent RX did not run. The unresolved model-load reservation is preserved. Actual owner exit 1; registered RX GPU 7. Actual reserved/completed calls: model_load 1/0, encoder 0/0, source_tx 0/0, source_rx 0/0, var_render 0/0, prior_scale 0/0, decoder_forward 0/0; unresolved 1. This is one existing source and m4 on the new platform only. The original v5 cross-host bitstream failure and v1 resource stop remain unchanged; old CDF and original dist.py byte identity remain unavailable. **Formal entropy-partial calibration, new100 confirmation and mismatch evaluation remain NOT_RUN.** See the [closed v2 diagnostic report](reports/leo_single_rx_diagnostic_v2.json).

Prepared on published `ca26ceb383e93fc4ee56298dcf896720188f86f6`. The shared-server worktree and frozen private environments are restored and import-checked. CUDA device enumeration has been diagnosed and repaired; the v5 numerical check matched680 encoder tokens but failed old/new m4 bitstream equality (137 of311 bits differ). That v5 attempt ran no independent RX or reconstruction, and its exact failure remains preserved. See the [current migration report](reports/leo_shared_migration_v4.md), [CPU checks](reports/shared_host_cpu_checks_v1.json) and [resume instructions](RESUME.md). New entropy-partial calibration, new100 confirmation and one-step mismatch evaluation remain **NOT_RUN**. Historical monitors remain paused.

A separate one-source m4 RX diagnostic is implemented and CPU-tested. Its first actual owner stopped at shared-GPU resource admission before launching a child: model and RX calls were both0. That first attempt produced no numerical RX result, and v5 cross-host failure is unchanged. See the [actual diagnostic status](reports/leo_single_rx_diagnostic_v1.json).

## Scope

Priority is the missing entropy partial-scale cell at N1024 and 4/10/19 dB.
Keep the tokenizer, unconditional VAR (null class 1000), Dc, raster order, numeric
settings, modulation/rate permissions, metrics and original calibration sources.
There is no training or quality-dependent per-source encoder choice.

The four confirmation arms are frozen raw whole, frozen raw partial, frozen
VAR entropy whole, and the calibrated entropy extension allowing partial scales.
All four use the same newly registered 100 sources and three noise repetitions:
3,600 method frames. Averages from this population must never be subtracted from
old 500-source averages as if paired. The primary contrasts are within each
representation. Cross-representation differences include framing and failure
handling, and are not pure component interventions. The published 19 dB entropy
whole quality/TX-cost tradeoff remains part of the evidence.

The independent diagnostic uses the original common500's first100 registered
sources, the original [6201,6202,6203] noise seeds and original 500-based counter.
True SNR 4 uses lookup 1/4/7; true 7 uses lookup 4/7/10; true 10 uses lookup 7/10/13.
Lookup only chooses the frozen whole/partial configuration. True SNR controls
channel scaling and both header/body demodulation. Report each deviation from
the matched condition, including header, CRC and fixed failure outcomes. No
threshold is tuned after observing these results. This is a one-step configuration
mismatch diagnostic, not a claim about typical deployment estimation error.

## Preparation and launch boundaries

The new scripts are independent of the published WCL files. `ep_plan.py` defines
the finite grid and whole-winner retention. `ep_phy.py` and `ep_qualify_phy.py`
adapt the existing real Sionna implementation and paid header, without editing
the original files. Their local selfcheck measures only packet-format engineering.
It performs no real encoder construction, channel simulation or packet decoding.

Before a new stage, read the current repository instructions and actual status,
inspect existing owners and their actual waits, and verify deployed source hashes.
Register a new bounded output namespace and separate ledgers; do not touch old
ledgers or infer server availability from the last successful publication.
Then execute independent real source and PHY qualification before calibration.
The calibration/execution orchestration and runtime admission still need to be
bound to the current remote receipts; this directory is not an unattended launcher.

New confirmation content is unavailable for policy selection. A complete exposure
registry, source-ID and content deduplication gate must close before claiming
that the new100 is disjoint from all study-used sets. The historical holdout1000,
completed T6 sources, Kodak sources and previously inspected/rejected candidates
must be audited as well as the commonly referenced train/cal/dev/holdout lists.
Missing old content hashes must be completed with the original preprocessing;
they cannot be silently ignored or described as a full deduplication check.

The finite protocol and budgets are in [PROTOCOL.md](PROTOCOL.md). Independent
source-codec and mismatch reviews are delivered alongside the new scripts.
After actual results close, run the existing repository checks, update the release
manifest through its existing tool, and normally push the reviewed files. Weights,
large caches and arrays stay outside Git under the existing release rules.

## Reproduce local engineering checks

From the repository root:

```text
python -m unittest discover -s experiments/entropy-partial-mismatch-20261010/scripts -p "test_*.py" -v
python -m unittest discover -s experiments/entropy-partial-mismatch-20261010/tests -p "test_*.py" -v
python experiments/entropy-partial-mismatch-20261010/scripts/ep_qualify_phy.py selfcheck
```

These commands produce no scientific result, winner, quality interval, timing or
claim of real-link acceptance.
