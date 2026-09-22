# Persistent stage-A observer

`watch_stage_a.py` runs on CPU and polls every60seconds. It lives outside the registered training source directory; no frozen implementation, configuration, RNG, optimizer or checkpoint is edited.

Checks include exact worker command/PID identity, training heartbeat, full calibration curves, the selected checkpoint, stopping-rule progress, GPU utilization/memory/temperature, free disk space and registered source hashes. A ten-minute absence of training/calibration/checkpoint activity raises a local alert; a deliberate resource yield is not treated as a crash. It never kills jobs, changes clocks, increases batches, restarts failed training or launches stageB.

Artifacts live in `outputs/VAR-LATENT-ENHANCEMENT-20260917/monitor_001/`: current `status.json`, minute-by-minute `history.jsonl`, deduplicated calibration/alert `events.jsonl` and the stage-A completion notice. All alerts are local artifacts/stdout; no email, messaging channel or automatic chat notification is configured. The daemon can continue without further progress queries while this machine/session remains running.

Run with the existing Python environment:

```bash
experiments/backbone-eval-20260912/.venv/bin/python -u experiments/var-latent-enhancement-20260917/monitoring/watch_stage_a.py
```

The observer remains alive after stageA ends to preserve its terminal status. An eligible continuous-reference result is not a finite-bandwidth communication result and does not mean stageB has started or the full experiment is finished. New stage-B monitoring will be registered with its actual implementation rather than guessed now.
