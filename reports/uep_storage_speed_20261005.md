# UEP N1024 quality-table storage acceleration — 2026-10-05

The existing quality-table run now saves new reconstructions as uncompressed NPZ archives containing the original float32 arrays. Measured source throughput increased by **1.47×**. This is an engineering change; no additional scientific method or quality result is claimed.

## Measured cost

| Measurement | Original | New |
|---|---:|---:|
| Sequential source time | 23.295 s (20 sources) | 15.867 s (6 sources) |
| New-source wall interval | — | 15.986 s |
| Archive write + fsync + SHA, calibration source 0 | 8.176 s | 0.422 s |
| Archive write + fsync + SHA, calibration source 1 | 9.017 s | 0.406 s |
| Archive bytes, calibration source 0 | 197,518,465 | 231,289,906 |
| Archive bytes, calibration source 1 | 193,679,796 | 231,289,906 |

Twenty one-second GPU samples after continuation averaged 75.3% utilization, with at most 7776 MiB allocated and 64°C. This short sample is not a whole-study utilization measurement.

At the observation, 216/1000 source checkpoints existed. The estimated remaining quality-table time was 3.48 hours, saving approximately 1.59 hours at the measured rates. These estimates exclude BLER refinement, actual-link evaluation, independent metrics, receiver timing, report generation and publication.

## Exactness and scope

- The original source-quality code, model weights, numeric settings, VAR generation, state set, calibration population, metric batch 8 and scientific registrations remain unchanged.
- Two existing complete calibration sources were written with both containers. Array names, dtype, shape and raw bytes were exact, including the original float32 reconstructions, reference, row IDs and image mapping. All 295 scientific rows per source were retained, and the original checkpoint reader accepted both containers.
- Only the `source_quality` module's local NumPy reference uses the storage proxy. The global NumPy module and other phases are unchanged.
- The run paused after source 208 (209 completed sources). Their checkpoint SHA256 values remain unchanged. Source indices 209 onward use ZIP_STORED; each new checkpoint records the archive's actual SHA256.
- The remaining archives require roughly 28 GB more disk than the measured compressed equivalent. Free disk at qualification was about 932 GB. Single-writer locks, temporary-write/rename/checkpoint ordering, and the original GPU safety checks are retained.

## Automatic continuation

The storage supervisor holds the original owner lock while the quality wrapper runs. It checks the frozen inputs, verifies the complete 1000-source output hashes, then launches the unchanged R3 owner and configuration. That owner continues calibration refinement, actual-link evaluation, metrics, isolated receiver timing, reporting and push. Unexpected failures preserve evidence and require review; no automatic retry or scientific scope expansion is enabled.

This note records a running engineering change, not completed UEP delivery. The full experiment's completion and remote publication receipts remain the delivery criteria.

[Measured throughput](../results/uep_storage_speed_20261005/throughput_observation.json), [CPU exactness proof](../results/uep_storage_speed_20261005/qualification.json), [execution registration](../results/uep_storage_speed_20261005/execution_registration.json), [preserved checkpoint hashes](../results/uep_storage_speed_20261005/pause_receipt.json), [source snapshots](../results/uep_storage_speed_20261005/execution_sources/).

Source snapshots have a `.py.txt` extension to preserve the frozen tracked-Python inventory during the running study. The execution registration binds the actual independently stored `.py` files.
