# Lossless text parts for three large scientific tables

The repository's published-file rules prohibit `.gz` and files above 10 MB.
These seven UTF-8 text parts preserve the exact original bytes of three tables;
each part is at most 8,000,000 bytes and splits occur only at newline boundaries.
This is a storage representation, not revised scientific data. Original files,
their completion receipts, and their original hashes remain unchanged.

`manifest.json` identifies the original repository-relative paths, byte lengths
and SHA-256 values, followed by each part's exact order, length and SHA-256.

From a clone, restore the original paths with Python's standard library:

```text
python results/paper_supplement_20261008/large_table_parts_v1/restore.py --output-root .
```

The restored tables are adaptive-BPG's 9000-frame quality `rows.json` and A4's
`frame_diagnostics.csv` and `quality_by_frame.csv`. Restoration verifies every
part and the reconstructed original before placing the file. An existing file
with the exact original hash is reused; a different existing file is rejected.
No model, bootstrap, channel, decoder or original scientific script is invoked.

Restored files exceed the repository limit and are local working artifacts;
do not stage them along with these parts. The adaptive result curves need only
the already published `comparison_summary.csv`, `paired.csv` and `completion.json`,
so plotting them does not require restoring these frame tables.
