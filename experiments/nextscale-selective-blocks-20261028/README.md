# Same-scale selective blocks: P0–P3 complete

The [complete report](../../results/nextscale_selective_blocks_20261028_v1/REPORT_zh.md) records NO_GO_STOP. No P4/P5 or new training is authorized. Original PROTOCOL.md, scope and historical records stay unchanged.

Execution order: nominate_p2.py -> prepare_p3.py -> four registered p3_phy.py shards -> p3_rx_replay_v2.py -> summarize_p3.py -> audit_p3.py. These stages have already executed and must not be relaunched as a generic reproduction command. Registers, input hashes, actual waits and closed ledgers establish execution. Native arrays/observations remain in the pinned PFS study directory.

The original p3_rx_replay.py preserves a CPU-only pre-replay cache-identity failure. Independent v2 and test_p3_replay_v2.py correct the reference-pixel-hash interpretation without using TX truth or repeating PHY/model calls. See CACHE_IDENTITY_CORRECTION.md in the result directory.

plot_p3_v2.py exports the final PDF/SVG/PNG from saved CSVs, without inference or new statistics. Original plot_p3.py and write_report_p3.py retain their executed bytes; the final report has documented editorial corrections and figure-v2 links.

See history_and_reuse.md for prior evidence. Successful scientific stages are never automatically rerun.
