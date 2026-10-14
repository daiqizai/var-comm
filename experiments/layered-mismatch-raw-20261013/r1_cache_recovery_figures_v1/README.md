# Recovered R1 calibration figures — prepared, not run

This entry is prepared for the one fixed recovered R1 delivery. Preparation and
synthetic CPU checks do not certify actual rendering or scientific completion;
use the later execution receipts, artifact manifest and visual-review record to
establish what has actually completed.

## Fixed inputs and closure

- Recovery master: `ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c`.
- CPU delivery source: `f37df9e92520a1f45bdc7954dbbd6eaf5c1a7cab4721d3b5df7ec5df65c6de01`.
- CPU dispatcher source: `38a7a17dc2d6473972472b05a1fccd9c6954bba0269d013625ee25c28ce7207d`.
- Input result directory: `results/layered_mismatch_raw_20261013/r1_cache_recovery_delivery_v1`.
- Actual dispatch control: personal temporary `r1_cache_recovery_delivery_dispatch_v1`.

Before parsing quality, the entry requires the exact source/master/command
bindings, actual CPU delivery exit zero, matching positive PID/start identities
that are gone, an unheld dispatcher lock, no dispatcher or publication failure,
and the original pending/final completion hardlink. It then checks the completion
manifest SHA and every light/heavy delivery artifact SHA and byte size. Manifest
paths must stay directly inside the two fixed delivery directories. An active
scientific runtime path cannot be supplied as a table.

The already verified CPU closure must retain the original owner/visual exit one,
one failed initialization and zero original forwards/images, inherited successful
PHY/projection, and the three successful recovered stages. These checks read the
completed delivery's small closure report; they do not reopen or recompute the
original scientific audit or arrays. The old source/master and failed segment
are never relabelled as successful.

The only quality inputs are the saved `frozen_winner_summary_csv` and `selection`
from the verified manifest. Exactly 90 cells are required: five frozen methods,
two transmitter SNR estimates, three actual SNRs, and three metrics. The saved
200-source/three-noise denominators and each candidate/receiver-rule identity
must match. All failures remain in the saved means. There is no source filtering,
policy ranking, bootstrap, confidence interval, ConvNeXt score or R2 launch.

## Figure contents

1. `r1_calibration_quality`: three metric rows (PSNR, LPIPS-Alex, DINOv2-L cosine)
   and two estimate columns (7 and 10 dB). The x axis is true SNR; the dotted
   vertical guide is the matched condition. The left/right points show transmitter
   overestimate/underestimate by 3 dB. All five frozen methods use descriptive
   labels, distinct markers and line styles. The LPIPS sign is unchanged.
2. `r1_saved_gate_comparisons`: the stored two-group-minus-robust and
   two-group-minus-backoff DINO-J differences for each estimate and their saved
   mean. The saved gate is shown without being recalculated from rounded display
   numbers or from the plotted quality table.

The scientific-figure-making skill supplies the publication conventions: blue,
red, teal, violet and green palette, minimalist axes, shared metric limits,
separate frameless legends, and explicit line/marker distinctions. Each figure
is exported as vector PDF, editable-text SVG and 600-dpi PNG. CSV input precision
is preserved; a byte-identical quality CSV and stored gate-value CSV accompany
the figures. No uncertainty shading or error bars are added.

## Future execution and publication

Only after the actual recovered CPU delivery is fully closed, deploy these
reviewed files and run once with CUDA hidden in the original CPU environment:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/layered-mismatch-raw-20261013/r1_cache_recovery_figures_v1/figures.py --expected-registration ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c
```

The new result directory is
`results/layered_mismatch_raw_20261013/r1_cache_recovery_figures_v1`.
The personal temporary root holds its exclusive claim, staging and Matplotlib
cache. The original frozen `r1_delivery_v1/deliver.py` is imported only for its
file-publication helpers; its `main`, audit, metric or selection functions are
never invoked. No active recovery source is imported. Existing outputs are not
overwritten, and failed attempts are retained without automatic retry.

Successful file generation still requires a witnessed process exit of zero and
actual opening of both PNGs for visual review. The manifest and completion remain
explicitly `VISUAL_REVIEW_PENDING` until that separate review is recorded. Figure
generation does not by itself establish a scientific GO or a completed R2.

## Synthetic checks

The local suite fixes the previous Windows implicit-text-encoding failure and
tests full-precision values, negative/zero results, saved gate preservation,
all 90 cells, source/noise/policy identity, altered manifests, foreign runtime
paths, exit/identity metadata with a mocked lock gate, publication failures, and mixed-stage
failure disclosure. It imports no model and performs no SSH or real-data read.

```text
python -B -m unittest discover -s .research/layered_mismatch_raw_20261013/r1_cache_recovery_figures_v1 -p test_figures.py -v
```
