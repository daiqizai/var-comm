# N512 extreme bandwidth study

- [Scientific report](report.md)
- [Frozen protocol and identities](config.json)
- [H_D/H_R decisions](decisions.json)
- [Training evidence](provenance/training/completion.json)
- [Full digital calibration evidence](provenance/digital_calibration_archive.json)

Large measured tables are stored as exact byte-preserving parts below the repository file limit. Restore them with:

```bash
python3 experiments/extreme-bandwidth-20260930/tables.py restore
```

Models, checkpoints and tensor caches remain on the execution host. Source-level intervals use the reused development set and one training seed.
