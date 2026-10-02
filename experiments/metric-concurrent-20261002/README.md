# Concurrent scoring of sealed experiments

R4 resumes the six sealed R3 source caches and changes only the scheduling
performance limit. Three consecutive warm M2 intervals above 1.75 times the
registered exclusive baseline pause concurrent scoring. This provisional limit
allows the measured 1.46 ratio while gathering evidence about combined completion
time. Memory reserves, thermal checks, foreign GPU identity checks, parity,
model formulas, scalar batch size and exclusive scientific timing stay fixed.

This is a scheduling trial, not a demonstrated total speedup. The final runner
still recomputes original reconstructions and metrics; its cache saves only the
new metric forwards. Saved new metric time must exceed additional M2 time and
handoff costs to shorten overall completion. The controller's completion records
do not claim that this condition has already been met.

Revision R3 preserves the R1 failure and the complete R2 pilot. The historical digital `latent_sq_error` alias comes from the independently recomputed `latent_sq_err_final` value before the unchanged parity check. The original N512 and N1024 analyses define the same alias; neither expected measurements nor reconstructed pixels are changed. Immutable JSON records are compared by their canonical JSON identity so Python tuples and their saved JSON arrays resume correctly; changed numerical values still fail. A registered peer that has finished CUDA cleanup may remain briefly in the driver listing: only its verified cleanup identity receives a bounded five-second allowance, while unknown or reused process identities remain blocked.

This extension scores the already completed N512, N1024, M1 and M1 source-only rate rows while M2 runs. It makes no training or policy-selection updates. Original experiment sources, registrations, outputs and communication formulas stay unchanged.

The reviewed runtime bundle lives under `outputs/METRIC-CONCURRENT-R4-20261002/runtime` during M2. Its entire Python and Markdown inventory is registered. After M2 is published and exits, the same bytes are published under `experiments/metric-concurrent-20261002`. Scheduling peer admission is explicit and separately recorded; unknown GPU users and original thermal checks remain enforced.

## Run and resume

The controller records the launch PID/start ticks and supplies a verified immutable admission file. Use the metric asset environment and original experiment import paths.

```
python runtime/concurrent_runner.py --root ROOT --cache-dir ROOT/outputs/METRIC-CONCURRENT-R4-20261002 --admission ROOT/outputs/METRIC-CONCURRENT-R4-20261002/admission.json --previous-cache-dir ROOT/outputs/METRIC-CONCURRENT-R3-20261002 --max-sources 1 --gpu-memory-fraction 0.45
```

After checking the one-source pilot, resume with the same arguments without `--max-sources 1`. Only the four registered completed studies are loaded. The worker preserves original numerical runtime flags, uses the original read-only replay and checks every original row's metrics, latent errors, flags and available waveform hashes. New metrics use float32 scalar batches. The worker caps its own PyTorch allocation at45 percent of GPU memory before loading reconstruction and evaluation models. It does not change the cap or formulas after an allocation failure.

A `pause_requested.json` marker with `status=PAUSE_REQUESTED` causes exit75 at the next yielded replay-frame boundary. Completed source checkpoints remain; the unfinished source is discarded and recomputed on resume. Each immutable checkpoint binds all original row hashes, policy/data/model/source files, exact reference and reconstruction RGB fingerprints, classifier labels and baseline prediction, evaluator identity, numerical runtime and scalar qualification. Partially written temporary files are never reused.

`partial_completion.json` inventories completed checkpoints. A partial snapshot is useful; the final report still requires all100 sources and all authorized studies.

## Final cache handoff

After the original experiments and extension source publication are complete, the controller invokes:

```
python runtime/final_runner.py --root ROOT --cache-dir ROOT/outputs/METRIC-CONCURRENT-R4-20261002
```

The bridge runs the unchanged final runner and replays every original scientific row again. It only seeds the new scalar metric cache. Reuse requires exact float32 reconstruction/reference fingerprints, the identical evaluator and manifest, unchanged frozen input files and runtime flags, and the same true label and original-image classifier prediction. No cached method label, class-conditioning annotation or old quality metric is injected into the final table.

The final synthetic batch qualification compares the selected batch with scalar outputs at an absolute tolerance of2e-5 for each floating metric and exact equality for classifier predictions. Final registration explicitly reports concurrent cached batch1 values and the selected batch used for uncached values. It binds cache provenance before hashing final checkpoints, and the scoring completion and published results include that provenance. Classification conclusions continue to use the unconditional rows; label-conditioned rows remain descriptive.

Engineering tests use synthetic fixtures and no real model weights. Their results are not scientific measurements.
