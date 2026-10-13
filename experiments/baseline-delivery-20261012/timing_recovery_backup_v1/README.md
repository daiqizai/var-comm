# Frozen timing recovery increment

This independent backup entry only preserves reviewed source, immutable v2/v3/v4
lineage, v5's 576 completed calls and failure evidence, v6 registration and
authorization, and the original STATIC receiver CDF. The existing result-only
postprocessor and its CPU evidence are included. No active v6 runtime or result
file is eligible; no model weight, original dataset, or full environment is copied.

`python experiments/baseline-delivery-20261012/timing_recovery_backup_v1/backup.py plan`
reports file count, uncompressed size, and required missing evidence without
creating an archive. After a normal commit/push is verified and its receipt is
saved, the coordinator may issue the one-time command:

```sh
python experiments/baseline-delivery-20261012/timing_recovery_backup_v1/backup.py build --release-receipt /absolute/path/to/normal_push_verified.json
```

The receipt must specify matching `head` and `verified_remote`, `normal_push: true`,
and `branch: main`. The local Git HEAD and every archived source must match that
published revision. Existing backup attempts are preserved and never overwritten.
Each file and the final archive carry a SHA256. This package is neither a complete
six-method timing result nor an independent full recovery package. Earlier model,
data, timing-entry, and environment packages remain explicit dependencies.
