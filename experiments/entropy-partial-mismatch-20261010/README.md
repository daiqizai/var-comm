# Entropy partial scales and one-step configuration mismatch

Prepared on published `ca26ceb383e93fc4ee56298dcf896720188f86f6`. The shared-server worktree and frozen private environments are restored and import-checked. CUDA device enumeration has been diagnosed and repaired; the v5 numerical check matched680 encoder tokens but failed old/new m4 bitstream equality (137 of311 bits differ). No independent RX or reconstruction has yet run, and the exact failure remains preserved. See the [current migration report](reports/leo_shared_migration_v4.md), [CPU checks](reports/shared_host_cpu_checks_v1.json) and [resume instructions](RESUME.md). New entropy-partial calibration, new100 confirmation and one-step mismatch evaluation remain **NOT_RUN**. Historical monitors remain paused.

A separate one-source m4 RX diagnostic is implemented and CPU-tested. Its first actual owner stopped at shared-GPU resource admission before launching a child: model and RX calls were both0. The numerical RX result remains **NOT_RUN**, and v5 cross-host failure is unchanged. See the [actual diagnostic status](reports/leo_single_rx_diagnostic_v1.json).

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
