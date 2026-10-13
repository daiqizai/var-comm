# RX cache/import/flag qualification only

This one-attempt engineering probe does not recover or execute the experiment.
The original receiver source is pinned at `55cd5886...`; its file firewall,
numeric-flag reader and eight pre-CUDA numeric-setting statements are extracted
with exact AST checks. No receiver module, model loader, PHY or score entry is
imported. The source-free model-bundle SHA must be explicitly supplied.

The probe output is exclusively
`L/r1_cache_recovery_probe_v1`, where `L` is the experiment's personal temporary
root. It creates no future visual-stage directory. Private `tmp`, `inductor` and
`triton` cache subdirectories are installed before Torch import. Both environment
variables and Python's already-cached `tempfile.tempdir` point inside that output.
The original receiver firewall is unchanged; an additional probe-only guard
rejects every file write outside the private output, new symlinks and process or
network requests. Source images, tokens, TX/PHY files and the old parent cache
remain forbidden.

Use the original visual Python 3.10.12 with CUDA hidden and bytecode disabled:

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <frozen_visual_py310/bin/python> -B probe.py --model-bundle-sha256 <actual-original-bundle-SHA>
```

The probe actually imports Torch and NumPy and applies the original six/two
thread, FP32/highest, TF32-off, deterministic settings. A request profile rejects
observed public tensor factories, weights, model constructors and CUDA setup;
TorchDispatchMode rejects all ATen operations during numeric configuration.
CUDA must remain uninitialized before and after. Imported interpreter/NumPy/Torch
files are checked against the existing runtime projection. This is Python audit
evidence, not an OS sandbox, inference result or numerical model qualification.

No model weights are whitelisted or loaded. No tensor, model, metric, channel,
registration, or new qualification frame is requested. A failure retains its
blocked-path/request counts and does not automatically retry. Even a swallowed
denial makes the qualification fail. Explicit original source/entry hashes and
the actual zero exit must accompany the eventual completion receipt.

`test_probe.py` uses only the original source text and synthetic stubs. Its
negative firewall cases are separate from the real probe and never hidden from
the real probe's `blocked=[]` requirement. Run the tests with bytecode disabled.
Deployment to another source directory requires the same original repository
source to be accessible; no scientific input or active quality result is read.
