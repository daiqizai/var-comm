# Kodak24 compact resource export

This is a read-only export of actual completed reconstruction JSON records: 864 frames (four methods, 24 sources, three SNRs, three noises), 72 BPG source/SNR configurations and 12 frozen method/SNR configurations. Failure states are retained. No model, channel, codec, new statistical test or resampling is executed.

`frame_resources.csv` preserves actual frame status, paid channel uses, code parameters, saved total energy and receiver diagnostics. Empty fields mean not applicable or not separately recorded, never zero failure. In particular BPG header/body energy was not separately saved in its per-frame JSON; the export retains measured total energy without reopening waveforms or inventing a split. Neural component energies are the values already saved by the inference job. Code-rate fractions use the recorded k and n, not the nominal rate label.

`bpg_source_configs.csv` contains each selected complete-stream size, resolution, QP and hash once per source/SNR, with consistency checked across all three noises. It retains the previously frozen adaptive source-encoding rule; Kodak does not select a new MCS. This is distinct from the earlier fixed native-resolution BPG baseline.

`frozen_configs.csv` covers all four methods at 4, 10 and 19 dB. VAR uses the null embedding, with no real class input. Swin's 19 dB point remains outside its training/calibration range. Post-reception correctness checks are audit diagnostics and never receiver inputs.

`evidence/` contains byte-identical copies of all four actual frame indexes and completion receipts, the executed neural request r2 and other metadata requests, the common dataset manifest, and available completed supervisory child-wait receipts. A launch record alone is not completion. If score supervision was not yet complete when this export ran, it is omitted; this resource export does not claim that scoring or plots completed. The earlier prepared but unused neural request r1 is omitted.

This compact delivery does not include NPZ reconstructions, source PNGs, model checkpoints, all BPG search candidates or an executable migration environment. Original server paths and hashes remain in the receipts. The exporter authenticates the JSON consumed and checks completion-bound image descriptors; it does not reopen or rehash large binary caches. It must not be described as a complete server migration.

Reproduce into a fresh export destination: `python experiments/generalization_kodak_20261009/export_results.py --root /home/liulu/projects/VAR_COMM --out NEW_EXPORT_DIRECTORY`.
