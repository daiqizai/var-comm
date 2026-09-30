# Step2 B: selected P_low receiver study

Completed added updates: 20000. Selected added step: 20000 (global 47500).

- [Scientific report](report.md)
- [Frozen configuration](config.json)
- [Calibration-only training evidence](provenance/training/completion.json)
- [Receiver calibration policy](selected_policy.json)
- [G2 decisions](decisions.json)

Training checkpoints, models and tensor caches remain outside Git. Full paired tables are stored as exact byte-preserving shards below10MB. Restore them with:

```bash
python3 experiments/rx-posterior-step2-B-20260930/tables.py restore
```

P_low receiver methods share one P_low observation. Comparisons with the old P4084 pair source/noise seed/N/E and retain each model's distinct waveform and observation hashes.
