# N1024 consistency: closed-delivery reporting entry

This entry is separate from the 22 registered, frozen scientific files in
`../repair_v1`. It imports none of them and must not be run while scientific
stages or their parent controllers are active.

## Scope and gates

`report.py` accepts only registration
`4789b2595190262a5970f985d25f420663d8225321830a770c4400fc82487670`.
It requires all four scientific stages, their inner controllers, and their root
controllers to have real successful waits and to be gone by PID/start identity.
It verifies the registration-bound source files and the completed light-delivery
manifest. It requires no unresolved PHY, receiver, or metric calls, and records
the original successful receiver's zero truth and zero TX-CDF reads.

Inputs are the closed files at
`results/n1024_consistency_20261019_v1/repair_v1/delivery/`, together with the
SHA-bound 3,000-record source-utilization CSV and the SHA-bound historical scores.
The program verifies the full common500 × 2 families × 3 SNR × 3 noise grid,
checks all 9,000 frame endpoints and terminated lengths against those 3,000
source records, and verifies the saved source means against the saved scores.
This arithmetic does not rescore images or draw bootstrap samples.

The output is created exclusively at
`results/n1024_consistency_20261019_v1/completion_report_v1/`.
An existing output directory is an error, not permission to overwrite or retry.
Failures after output creation leave `failure.json` in that directory.

## Invocation, only after root confirms actual delivery closure

Run from the unique remote repository, in the frozen LDPC Python environment
with CUDA hidden and bytecode disabled:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python -B experiments/n1024-consistency-20261019/completion_report_v1/report.py --expected-registration 4789b2595190262a5970f985d25f420663d8225321830a770c4400fc82487670
```

Root must record persistent stdout/stderr, the exact argv, parent and child
PID/start identities, and an actual wait. This entry has no launch loop,
successor, scientific worker, or retry. Preparing and deploying it is not
scientific completion or figure delivery.

## Outputs

- Chinese report and a 21-row quality/direct-paired comparison table, plus a
  LaTeX three-line table. Display rounding is separate from the CSV values.
- `legacy_whole_version_delta.csv`: old versus repaired WHOLE means for 4/10 dB
  and all seven metrics. No new old-versus-new confidence intervals are made.
- `actual_compute_and_reuse.json`: real scientific ledgers, exact reuse counts,
  score origins, and the saved bootstrap-bank identity.
- `quality_main`: PSNR, LPIPS, DreamSim, and prediction agreement.
- `quality_supplement`: DINO similarity, DISTS, and MS-SSIM.
- `same_prior_paired`: directly paired PSNR gain, LPIPS reduction, DreamSim
  reduction, and agreement difference. All values and interval endpoints,
  including negative values and crossing-zero intervals, are retained.
- Actual 180 × 55 mm vector PDF, editable-text SVG, and 600 dpi PNG for each
  multi-panel figure; separate 60 × 55 mm panels and grayscale checks.
- Unified style, English captions, axis limits, input manifest, output manifest,
  and an export completion receipt.

The completed 3,000-record utilization ECDF is referenced unchanged. This entry
does not redraw it or manufacture a CDF from means.

## Statistics and interpretation

Both entropy families use common500 only, N1024 with 68 header and 956 body
symbols, nominal Es=2, original noise seeds 6201/6202/6203, and the frozen
4/10/19 dB policies. The repair changes only the 2,421 source-version-different
WHOLE 4/10 dB events. All 6,579 other logical frames are reused. Both 19 dB
families retain the T1-C m10 results.

Quality and paired intervals come directly from the successful scientific
delivery. That delivery reuses the existing 10,000 × 500 bootstrap index bank;
this reporting entry does not load or recalculate that bank. For a
lower-is-better paired plot, the saved partial-minus-whole estimate is negated
and interval endpoints exchanged, so the display is a reduction. This is an
exact sign conversion, not a new bootstrap or subtraction of marginal CIs.
Agreement is displayed as a percentage and its paired difference in percentage
points. Failures remain in the registered denominators.

The old-versus-new means describe an implementation-version correction. They
must not be described as new training, m10, endpoint selection, or policy gains.
No new100, cal1000, Kodak, fixed16, raw, JSCC, ARPC, or N2048 result is pooled.

## Visual review

The following skill files were actually read for this reporting design:

- `C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md`
- `C:/Users/11946/.codex/skills/scientific-figure-making/references/design-theory.md`
- `C:/Users/11946/.codex/skills/scientific-figure-making/references/common-patterns.md`
- The trend-demo reference previously read for this audit at
  `.research/figures_redesign_v1/references/figure_ophthal_review__plot_trend.py`.

The canvas, type sizes and markers use the paper scale rather than the demo's
large canvas. Family names, color/line/marker redundancy, real numeric SNR
spacing, white-filled baseline markers over interval lines, and vector exports
are explicit. No interpolation beyond straight observed-point connections is
performed.

Export success sets `REPORT_EXPORTED_VISUAL_REVIEW_PENDING`, not visual PASS.
The reviewer must actually open the color and grayscale PNGs, inspect the
180 × 55 mm layout and independent panels, verify page sizes, editable SVG text
and absence of raster image embedding in the scientific PDF, then record an
honest `actual_visual_review.json`. Preserve source/output bytes and root's
real execution receipts; deterministic layout repair may use a new explicit
report version without changing data or statistics.
