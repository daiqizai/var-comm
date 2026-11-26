# Four-user image uplink replication

The experiment uses the SwinJSCC-based four-user receiver in the 24 November [replica study](../overloaded-semsic-20261124/). The three-step nonlinear estimator and all three checkpoints are frozen. Each user sends a different RGB image. The shared transmitter adapter produces its latent. The slot selects the Walsh signature during waveform embedding. All four transmitters are active in this experiment.

## Inputs

Use the physical study directory under liulu/analysis for code and light results, and a separate empty PFS scratch directory for the snapshot, caches and job controls. The November 24 study provides a physical snapshot of five packages, its evaluation module, the three checkpoint hashes, and the per-group receiver records in its events and gate_events directories. Paths to the frozen backbone and checkpoints come from that evaluation module. Source IDs and hashes, not demonstration PNGs, specify the images. Use the released FP32 PyTorch environment with TF32 disabled. The study never trains weights.

## Run order

Here STUDY points to a new physical working directory containing this experiment's code and protocol. OLD points to the previous study's complete working directory. SNAP24 points to its PFS source snapshot. SCRATCH is separate. PY is the frozen visual runtime's Python. The launch owners create unique directories and record actual process exits.

~~~sh
python3 -B "$STUDY/prepare_study.py" "$SNAP24" "$SCRATCH/snapshot" "$STUDY/protocol.json"
python3 -B "$STUDY/launch_worker.py" mechanism "$SCRATCH" "$STUDY" "$OLD" "$PY" --gpu 3
python3 -B "$STUDY/launch_worker.py" development "$SCRATCH" "$STUDY" "$OLD" "$PY" --gpu 2
python3 -B "$STUDY/continue_gpu.py" "$SCRATCH" "$STUDY" "$OLD" "$PY"
~~~

The continuation waits for both prerequisites before formal scoring and for formal scoring before timing. The secondary protocol is separate. Its development-only selection excludes schedules for which ranking cannot affect the selected subset in any round. Running the following freeze on the release verifies the recorded logical decision. In a fresh replication it creates the new decision. The additional owner waits for both primary GPU exits.

~~~sh
python3 -B "$STUDY/freeze_secondary.py" "$STUDY"
python3 -B "$STUDY/start_secondary.py" "$SCRATCH" "$STUDY" "$OLD" "$PY"
"$PY" -B "$STUDY/analyze.py" "$STUDY" mechanism
"$PY" -B "$STUDY/analyze.py" "$STUDY" routing
"$PY" -B "$STUDY/secondary_stats.py" "$STUDY"
"$PY" -B "$STUDY/verify_results.py" "$STUDY"
"$PY" -B "$STUDY/plot_geometry.py" "$STUDY"
~~~

The primary and secondary freezes, per-image identities, stage-level observations, original latency measurements and paired estimates are in the [results directory](../../reports/replica_routing_20261127/). The historical fixed-gate study provides independent numerical anchors for all three mechanism arms. Model parameters and image arrays stay in their original storage.

## Measurement boundaries

For the mechanism intervention each replica replaces only the interference reconstruction. All twelve native gain tensors are replayed, and the actual four LMMSE matrices for the weak slot are checked for exact agreement. Ground-truth signals and channel gains enter an offline scorer only after the receiver has returned. The first two normalized error measurements use transformed interference energy in their respective domains. The noise-inclusive measurement uses combined desired signal power, while latent and frontend measurements use the transmitted source latent and its frontend transform under the receive metadata. The image metric uses the true source pixels.

The practical receiver selects users using its estimated channel powers, weak-user combiner and, in one development candidate, denoiser innovation. Production scheduling recomputes its own gates. Each RGB call counts one local reconstruction plus one encode, for one user and one round. Every timed receiver also decodes all four final source reconstructions. Timing begins with the received GPU waveform and ends at four clamped GPU RGB outputs, with synchronization at both ends. Each timing session uses four distinct physical CPU cores. Shared GPU occupancy is recorded in the hardware JSON. The two spreads are per-user received-power contrasts, not run SNR labels.
