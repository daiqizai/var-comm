# Unified additional metrics

[Complete report](METRICS_PUBLISHED_REPORT.md). The original analysis report and its hashes are retained.

Large CSVs are stored as standalone parts under `table_shards`. Their manifest preserves the exact original bytes and SHA256. The experiment's `publish.py --restore-tables` restores these original CSVs. Full provenance-verifying analysis additionally requires its registered local input files. No model weights, caches, tensors or image datasets are included.

The real-weight qualification uses synthetic pixels and is not a scientific result. The final tables use the registered 100 sources and paired source bootstrap.
