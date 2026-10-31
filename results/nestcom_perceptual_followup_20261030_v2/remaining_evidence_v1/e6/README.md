# Closed E6 evidence

All named execution roles actually exited successfully; owner and child identities were checked as gone. See execution_closure.json and delivery_manifest.json.

The original frame_metrics.csv exceeds the unchanged repository 10 MB limit and remains on PFS, untracked. Published frame_metrics_parts/part_*.csv retain every full-precision scalar string in original order. csv_shard_identity.json binds the original and parts; one header plus their rows reconstructs the original CSV bytes. No scoring, generation, PHY, or bootstrap was repeated for publication.

These are 30,000 actual receptions on the registered 100 sources and frozen 10 dB policies. Overall metrics retain all gray rejections. Successful-only quality is descriptive and has an SNR-dependent event denominator; it is undefined at 8 dB, where all receptions fail.
