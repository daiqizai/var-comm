# Six-method timing v2 admission and call-path audit

Status: registered, not executed. No v1/v2 launch, model construction, real channel, packet decode or timing case exists at this audit. Nine synthetic CPU tests pass.

The old raw transmit and receive paths each call `MixedPublicCatalogue.verify`, hashing all 433 profiles and 523 aliases in the measured window. The v2 timing wrapper executes the same check before and after each full case. Per-frame protocol checks are retained. Individually reviewed pure provenance array/metadata digest helpers still compute their real outputs in explicitly excluded audit sections. Raw wall and excluded audit durations are both retained; no global clock or hash implementation is changed. CRC, scrambling/noise seed generation, received-field parsing, arithmetic canonicality and all fresh neural/source work remain timed.

The latest user instruction explicitly permits shared resources. V2 uses the inherited H800 GPU0 device binding, a 16 GiB private PyTorch allocator cap, 2 GiB free safety margin and initial GPU utilization at most 25 percent. Other workloads are recorded and left running. Results will be labeled shared-environment measurements, not uncontended latency. The original CPU affinity, 6/2 threads, FP32, TF32-off and numerical implementations remain.

The same 1728-call budget includes 864 warmups and 864 measured calls; the four digital arms retain the 2304 packet limit. A root-coordinator release binding E1 stage/closure evidence is required before the finite owner may start. The old v1 lock is shared and any actual prior attempt blocks replay. No scientific call has been authorized by the CPU tests or this registration alone.

Entry: `experiments/baseline-delivery-20261012/timing_revision_v2/timing_owner_v2.py`. Preserve v1 registration and use this wrapper for the authorized next window. Final numerical qualification, actual measurements, raw tables, summaries, figures and actual child exit remain pending.

A preserved zero-call prelaunch revision explicitly permits concurrent E2 CPU-only statistics under the shared-resource authorization. The separately registered once-only successor waits for all four E1 exit-zero receipts and the complete 4500-frame scoring receipt, then checks admission and exec-hands off to the timing owner. It never retries a real failure.
