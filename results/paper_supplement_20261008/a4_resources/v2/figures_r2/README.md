# A4 rendering revision 2

This directory only rerenders four existing, completion-hash-verified CSVs. Original data, statistics and first-render files remain unchanged. No source checkpoints, model, channel, RNG or bootstrap are accessed.

The source-information figure omits point labels that overlapped at 10/19 dB; the caption supplies SNR ordering and the original table retains exact coordinates. The state-quality caption now calls out varying subset counts and the single CRC-rejected KEEP frame at 7 dB. subset_counts.csv copies existing count fields; conditional curves have no gain or subset confidence intervals.

Reproduce: `python experiments/paper_supplement_20261008/a4_resources/export_resources.py --render-existing /home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a4_resources/v2 --output <new-empty-render-directory>`
