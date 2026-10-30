# Closed E1/E3/E4 scalar delivery

These three stages are actually closed. E2/E5/E6 and the final V7 manuscript are not certified by this package. `delivery_manifest.json` preserves the original inputs, true wait receipts and original export hashes.

The repository limits each tracked file to 10 MB. Therefore the two large E1 tables are published as `e1/dists_per_frame_parts/part_*.csv` and `e1/native_output_inventory_parts/part_*.csv`. `csv_shard_identity.json` lists exact order, fields, rows, hashes and original file identities. Concatenating parsed rows in that order and writing the single original header reconstructs the original CSV bytes exactly. The original unsharded files remain at their pinned runtime paths and are not staged. All scalar strings, including full-precision metrics, are unchanged; no scientific computation was repeated.

Join frame records to native inputs by `dists_pair_key`. There are 66,000 frame rows,28,253 unique native pairs and22,000 source means. The full five-metric summary and direct paired tables have220 rows each. `dists_summary.csv` and `dists_paired_deltas.csv` are DISTS-only subsets. E3/E4 each use their explicitly recorded100-source population, not the500-source population. All methods, failure-gray outcomes and unfavorable directions are retained.

The original BPG7dB stored-gray annotation is preserved separately in `e1/failure_count_annotation.json`; no old metadata or quality score was overwritten. The reports distinguish a confidence interval spanning zero from evidence of a direction. Arrays, weights and logs remain on PFS.
