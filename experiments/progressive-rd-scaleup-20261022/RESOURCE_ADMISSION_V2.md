# Shared resource admission, before any new science

The first metadata registration exited1 at the idle-device utilization<=10 gate after creating the immutable train_index. It started zero model/science jobs. Original script, logs and actual wait are retained.

Actual shared workload now uses about25GiB/device; all eight still have about70GiB free. The new bounded decoder-quality study admits free>=60000MiB and utilization<=80%, records actual resource snapshots, and retains48GiB training allocator/2GiB free floor and all original scientific/window caps. This is not latency evidence or a claim of exclusive resources. No other workload is signaled or modified.

register_resource_v2 reuses the exact already-created train_index instead of re-enumerating or overwriting it. owner_resource_v2 changes only the qualification path and admission comparison. supervisor_resource_v2 changes only the owner entry path. All model/data/loss/training and native receiver sources remain exactly as qualified. Five focused CPU tests verify boundary conditions and exact owner/supervisor changes; the completed20-test math qualification is reused.
