# Swin/ADJSCC actual PHY identity and timing scope

The CPU audit verified all9000 real paid-protocol frames across six existing workpoints. Original completion CSV hashes, development source/adapter/protocol bindings, four actual checkpoint files,9000 frame receipts and every regenerated standardized data-noise SHA passed. Source pixel identities and the pre-shared numerical frame counter agree with the original100 development sources. Pixel/PSNR archive integrity was separately audited in reference_audit_v1; this audit does not pretend that stored waveform hashes are fresh waveform recomputation.

| Method | Data N | Paid metadata N | Total N |
|---|---:|---:|---:|
| Swin RA32 |4096|402|4498|
| Swin RA64 |8192|562|8754|
| Swin RA96 |12288|664|12952|
| ADJSCC C2 |4096|0|4096|
| ADJSCC C4 |8192|0|8192|
| ADJSCC C6 |12288|0|12288|

Swin's float32 normalization power and combinatorial active-mask rank consume protected channel uses. RX reads the decoded metadata, never replaces it with TX truth. The original protocol retains a legal decoded hard candidate even if CRC rejects it: one such inexact RA96 frame remains in the results. Bare ADJSCC receives fixed power1 plus a pre-shared frame counter, not free TX normalization power. These are natural resource points and must not be renamed N3060 or N4084. Maximum absolute recorded E-2N error is0.00604, within the audited0.01 floating tolerance.

The recorded native-model qualifications used adapter SHA230ffbc..., whereas actual development transmissions bind SHAec1be3a... . This whole-file revision mismatch is not proof of a model error, but the old qualification cannot certify the exact later adapter without review or replay. Fresh native parity is still required. The pinned author environment is available and imports the actual root adapter successfully: PyTorch1.12.1+cu116, CUDA11.6. This CPU import did not initialize CUDA or execute a quality model.

Historical timing used four CPU threads and one warmup per rate in the pinned author runtime. Current VAR timing uses six threads and a different warmup scope/runtime; these old times stay explicitly separate. The next actual work is native parity plus uniform CPU-RGB/waveform/RGB endpoint timing in the pinned author environment, after the already registered serial GPU stages. Do not change author runtime/checkpoints to conceal this difference. Common LPIPS/DINO scoring is already queued separately and must not be duplicated.

Artifacts: [audit](../results/token_channel_efficiency_20260923/author_phy_audit_v1/audit.json), [resource/timing scope](../results/token_channel_efficiency_20260923/author_phy_audit_v1/resource_and_timing_scope.csv), [runtime availability](../results/token_channel_efficiency_20260923/author_phy_runtime_v1/availability.json). The9021-entry binding manifest stays local, with its SHA published. This closes receipt/noise/resource identity checks, not full GPU parity or merged delivery. N3060 DeepJSCC/R3/digital full PHY/timing compatibility remains a separate pending item. No training, new holdout, diffusion run or hardware change occurred.

Vendor source inventory: diffcom is at the registered a8cc4d6 commit with no tracked changes. Swin is an archive checkout rather than a nested Git repository; all10 files match the local cached source archive associated with the registered a6d0e6d commit. Archive/file hashes are recorded in `author_phy_runtime_v1/swin_source_archive.json`; no upstream network refetch or vendor mutation was performed.
