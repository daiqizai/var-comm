# Adaptive BPG holdout resources and failures

This is the later-added adaptive-downsampling BPG + LDPC baseline, distinct from
the original native256 BPG experiment. The frozen source rule chooses among
256/128/64/32-pixel BPG inputs and QP candidates using source-side MSE and actual
complete-file size; restored images use fixed Pillow bicubic to 256. MCS was
selected on exactly 100 original calibration sources with three noises, not
1000 sources and not the holdout. All 500 holdout sources x six SNRs x three
noises (9000 rows) are retained, including every failed source or link outcome.

This export reads JSON and fixed existing BPG bytes only. It imports no source
codec, model or PHY implementation, reads no reconstruction NPZ, draws no noise,
reruns no decoder, and performs no bootstrap. Quality and paired confidence
intervals are delivered by the separate metrics stage. Original results and
selection files are never changed.

resource_summary.csv records each frozen profile, qualified actual LDPC k/n,
nominal label, filler/rate-matching facts, complete-file byte distribution and
frame energy. B_src is complete BPG file bytes x8, including its format header.
The container also pays 13 length bits, known zero information padding, and a
16-bit body CRC. k includes all of these, rather than being pure source bits.
The header pays 12 profile bits, 16 CRC bits and 6 tail bits, encoded into136 bits.
All transmitted frames allocate68 header +956 body +0 physical-padding complex
uses. Information-container zero bits are encoded inside the body; they are not
extra unused physical symbols. Frame-use fraction is1 for actually transmitted
frames. SOURCE_UNFIT has no waveform/PHY call; energy and codeword quantities
specific to its actual transmission are NA and actual_transmitted_uses is0.

The saved actual_frame_energy is the measured transmitter waveform sum. Header
energy136 is an exact derivation from the hash-pinned Header/scale_channel
mapping:136 real coordinates are +/-1, exactly preserved in float32. Derived
body energy is the saved total minus136; physical padding energy is0. No signal
is regenerated to obtain these components. Energy distributions exclude absent
SOURCE_UNFIT frames and explicitly report their counts. These are waveform
energy units, not hardware joules or a claim of strict per-frame energy equality.
The selected profile k/n and allocated budget remain listed even for SOURCE_UNFIT;
the separate actual transmission fields show zero uses and no transmitted bits.

source_configurations.csv has3000 source/SNR entries, avoiding triple-counting
the same source-side choice across noise seeds. source_setting_distribution.csv
reports QP and resolution distributions over500 sources at each SNR; its separate
dimensions are alternative descriptions and must not be summed together.

failure_counts.csv distinguishes mutually exclusive final statuses from
overlapping diagnostics. Header rejection, body CRC rejection, parser rejection,
BPG decoder rejection and unsupported decoded format retain their original
statuses. header_wrong_accept compares actual accepted profile with the frozen
transmitter profile offline. parser_accepted_payload_different compares actual
received bytes with the selected TX stream offline. A CRC-accepted parser failure
or byte mismatch is reported as detected wrong content; a parser-rejected body
has no saved received BPG file to pretend was decoded. These diagnostics never
rescue, replace or change a receiver output. A wrongly accepted stream which
decodes remains a non-gray decoded image, with its wrong-content flag retained.
Zero-count states remain explicit; they do not establish zero population risk.

streams/ contains the original complete TX BPG files for source index0 at each
of the six SNRs, fixed by index without looking at quality or noise outcomes.
stream_samples.csv gives source identity, resolution, QP, original path and hash.
If source0 is unfit, the manifest retains that fact without selecting another
source. No best noise sample is chosen: these are pre-FEC source code streams.

Reproduce after actual holdout completion:

    python experiments/paper_supplement_20261008/a1_bpg_adaptive/export_resources.py --request <frozen-a1b-request.json>

The default output is resources_v1 under the frozen experiment directory; an
explicit --output must be a new empty directory. Existing publication filters
are unchanged; binary BPG examples remain local when the filter excludes them.
