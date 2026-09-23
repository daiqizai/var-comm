# User-requested GPU memory utilization adjustment

P2048 was safely paused at step8862 before disposable real-weight benchmarks.
The checkpoint SHA is d817aab3c9df0458604683a22c474ef6506bbff0a34a52e94b6c7992b4d57d35.
Model/data/loss/FP32, effective batch16, optimizer and update count are fixed.
The benchmark uses16 calibration images and the actual populated AdamW state;
no benchmark updates or weights are saved into training. Two opposite-order
rounds with3 warmup+10 timed updates per configuration reduce simple order bias.
They do not establish sustained hot-state or end-to-end throughput.

| microbatch | update seconds | allocated peak GiB | throughput change vs4 |
| --- | ---: | ---: | ---: |
| 4 | 0.394519 | 5.466 | reference |
| 8 | 0.381368 | 10.598 | +3.45% |
| 16 | 0.378431 | 20.848 | +4.25% |

The benchmark's initial automatic >=5% time-reduction rule recommended keeping4;
that result is preserved. In response to the user's explicit memory-utilization
request, the operational decision is8: it uses more memory with a small measured
speed benefit and substantially more headroom than16. This is a disclosed override
of the performance threshold, not a changed benchmark result or quality selection.
No claim of large acceleration or long-term quality equivalence is made.

At micro8, loss max absolute difference was7.45e-9, gradient relative L2
1.68e-4 and parameter-update relative L2 1.12e-4. All pass the predeclared
1e-5/1e-3/1e-2 tolerances. Floating-point accumulation order changes, so the
continuation is explicitly versioned and not claimed bitwise identical to micro4.
Real-device qualification at effective batch16/micro8 also passed frozen-Decoder,
input-gradient, actual2N energy, populated optimizer isolation and bitwise resume
within micro8. These are engineering checks, not new formal image-quality results.

Original registrations/checkpoints remain unchanged. New checkpoints carry an
execution_identity binding the new runner/helper, runtime specification and actual
benchmark; selected loading validates this identity against checkpoint and historical
registration. Only the exact recorded8862 checkpoint may cross the initial boundary.
New checkpoints have a microbatch8-v1 suffix to preserve old events. P3060 remains
on its original micro4 entry pending its own qualification.

Launch uses run_budgets_microbatch_guard.sh -> thermal_guard_v2 -> coordinator_v3.
Both thermal flags are checked, with the same safe pause/cooldown policy. Earlier
thermal/controller evidence remains available. No power/clock/fan changes were made.
Evidence: results/token_channel_efficiency_20260923/microbatch/.
