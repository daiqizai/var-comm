# Reproduction and evidence map

This directory contains reporting helpers, separate from the immutable P0–P2
scientific launch bindings. They read closed outputs and do not rerun inference,
metrics, bootstrap statistics, radio frames, or training.

The scientific entry points are in the parent directory. The original launch
requests contain absolute PFS paths and must not be run again against their
existing output directories. A new reproduction requires a new scratch root,
verified public assets, and new explicit registrations. The existing run is
owned by `first_batch_continuation.py`; do not start its individual stages by
hand while that owner is active.

## Model and numerical route

- Upstream: CVL-UESTC/RDVQ, commit
  `a2744eedd803d9e20b55b4b0c935ec211a07d184`.
- Public initialization: `lam_4.8_tau_0.1.pt`, Hugging Face revision
  `d6223fb30f23f3805ec1d0a925c592fc43d223bd`; SHA-256
  `827aaab038626bb306fc30fac37634eece418dd02bd4617f88e65d8d19e92294`.
- PyTorch visual environment: Python 3.10.12, torch 2.11.0+cu128; isolated
  task dependency overlay, no edits to the shared environment or pinned model.
  Exact installed versions and wheel hashes are in `environment_identity.json`.
  DreamSim is loaded from the hash-checked source/assets in `metric_assets.json`;
  a missing installed-distribution version does not mean that metric is absent.
- FP32, TF32 disabled, math SDPA, two CPU threads, deterministic causal argmax
  suffix. This is the declared evaluation route, not the upstream sampling
  default. The entropy model and native slice dependencies are preserved.
- Training uses native CUDA bicubic backward. It is not guaranteed bitwise
  deterministic; the qualified checkpoint continuation tolerance is `1e-6`
  (observed maximum absolute difference `1.1920928955078125e-7`).
- The source packet has a 72-bit wrapper and native rANS streams. Termination
  states are already included in those stream lengths. The dynamic PHY adds
  a separate protected 64-symbol header. Old NeST 68-symbol/29-bit fields and
  its arithmetic-code monotonicity identity do not apply.

The GPU driver reports `NVIDIA RPBZZZ6` with 97,887 MiB per device. The plan's
PRO6000 naming is not substituted for that observed driver identity. Device
admission verifies the NVML index, UUID and PCI address. This host has different
NVML and CUDA-runtime UUID strings, so the qualified route uses numeric CUDA
visibility with PCI bus ordering. Other users' devices and processes are not
altered.

## Data and fair comparison

Official COCO ZIP filenames define fixed source identities: 20,000 training,
1,000 development, 1,000 calibration, and 500 validation-split test images.
Training/development/calibration order uses seed 20261104, test order 20261105.
Every selected JPEG is checked against official ZIP CRC/size and a SHA-256.
Canonical native RGB hashes are also retained. One exact duplicate pair occurs
within training; no exact native-pixel pair crosses the four splits. There is
no near-duplicate certificate or image-level exclusion against upstream
pretraining. The public checkpoint has no such pretraining manifest.

Evaluation uses RGB, shorter-side 256 bicubic resize and 256 center crop.
Training uses the same resize plus seed/source/epoch-keyed random crop and
horizontal flip, shared by matched arms. These are mechanism experiments,
not a reproduction of the author's full-resolution benchmark tables.

All ten P2 arms share initialization, 4,000 optimizer updates, global batch 16,
microbatch 4 with four accumulation steps, learning rate `1e-5`, Adam betas
`(0.9, 0.95)`, and a frozen codebook. R1 has one training seed; R2, R3 and
estimated-only R3 each have three. R2/R3 comparisons are paired by training
seed and source. The common fresh GAN-inactive adaptation is explicitly not
the author's original full training trajectory. No checkpoint is selected
from final test quality.

The full and two prefix branches use the same MSE, official VGG-LPIPS and
soft rate losses, plus the same codebook commitment/stability terms. R3
calibrates estimated lengths from actual packets on 128 fixed development
sources every 500 updates. Its estimated-only control performs the same audits
but does not apply the correction. The discarded qualification optimizer
updates do not enter any P2 checkpoint.

## Reading the outputs

- `source/summary.csv` and `source/paired_deltas.csv`: the frozen public model's
  actual-source-packet budget curves and direct paired source effects.
- `control/reliability.csv`: 14,400 actual calibration frames, with Wilson
  frame-error intervals. Accepted packet bytes are matched to independently
  decoded native source inputs.
- `control/expected_*.csv`: test-source **conditional expected** metrics using
  that empirical reliability table. They are not additional measured test
  radio frames. Source confidence intervals condition on the reliability
  table; its uncertainty is reported separately. The source-informed reference
  evaluates candidate reconstructions at TX and is expensive, not a trained
  runtime selector.
- `matched/summary.csv`, `paired_deltas.csv`, and
  `training_seed_variation.csv`: fixed-checkpoint P2 results when closed.
  Source-bootstrap uncertainty and independent-training-seed variation are
  separate. Do not treat channel repeats as training seeds.
- `rates/`: packet lengths, paid overhead and initial rate-proxy diagnostics
  from existing native records. Correction diagnostics on the same 128 audit
  sources are labelled in-sample; final fitting always uses actual bytes.
- `resource_receipts.csv`: actual process waits and charges. Parent CPU
  receipts include child usage; do not add them to leaf CPU receipts. GPU
  charges are recorded at scientific leaves. The separate P0 probe correction
  is documented in `resource_snapshot_p0_v2.json`.
- `retained_failures.json`: original failed attempts and evidence locations.
  Completed caches were reused only after identity checks. Failed HTTP asset
  transfers are distinct from model or radio failures.
- `source_identities.csv`, `split_identity.json`, `copy_provenance.json`, and
  `raw_evidence_index.json`: data order, canonical identities, delivered-file
  hashes and PFS-only raw evidence. Arrays, weights, model outputs and full logs
  are deliberately not copied to Git.

The raw matched-analysis column called `training_GPU_hours` is misnamed: it
contains the total arm charge. The reviewed delivery renames it
`total_GPU_device_hours` and separates training from evaluation using actual
leaf receipts. Raw PFS bytes and all scientific numbers remain unchanged.

## Report-only commands

Use an unused output directory. These commands require actual successful
closure, preserve raw data, and do not launch the scientific queue:

```bash
python reporting/collect_closed_evidence.py --root "$RUN_ROOT" --repo "$REPO_ROOT" \
  --output "$REPO_ROOT/results/rdvq_prefix_20261104_v1/p1_delivery_v1"
python reporting/summarize_native_rates.py --root "$RUN_ROOT" \
  --output "$REPO_ROOT/results/rdvq_prefix_20261104_v1/p1_delivery_v1/rates"
python draw_actual_bit_curves.py \
  --p1 "$RUN_ROOT/p1_source_analysis_v1/summary.csv" --output "$FIGURE_OUTPUT"
```

After all ten arms close, use `--include-p2` in a fresh delivery directory and
add `--p2 "$RUN_ROOT/p2_matched_analysis_v1/summary.csv"` to the figure command.
Figures still require actual visual inspection; successful exports are not
automatically a visual-review PASS. P3/P4 are not launched by these tools.
