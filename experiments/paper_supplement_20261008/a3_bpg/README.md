# A3 in-memory BPG source codec and endpoint

This is a separate deployment-timing adapter for the existing native256 BPG
baseline. It leaves the official0.9.8 source, CLI binaries, scientific policies
and old results unchanged. It does not implement the separately studied
adaptive-downsampling baseline.

The encoder includes the unmodified `bpgenc.c` with its CLI main renamed. RGB
is converted to the same lossless PNG bytes in memory, passed through the
original `read_png` via `fmemopen`, then encoded with the original x265 archives
and the frozen settings: compression level8, YCbCr, 4:2:0 and8-bit. The complete
BPG container is collected by a memory callback. The receiver uses the original
libbpg decoder API to restore RGB from the actual received complete bytes.
There is no source image or TX payload input to the receiver.

`build_memory_codec.py` compiles only into the specified independent output
directory. It rebuilds decoder objects as PIC with the original Makefile's
numerical options and links the existing PIC x265 archives. Compilation and
library loading occur outside all timed windows.

`qualify_memory_codec.py prepare` freezes the first20 original calibration
images and QPs29/51. Its `run` command performs40 actual comparisons against
the original CLI. Every complete container byte and every decoded uint8 RGB
value must match. This diagnostic performs zero PHY decodes and no channel
draws. CLI file I/O is confined to qualification; its elapsed time is not a
deployment-latency result.

`MemoryBPG.fit` creates a fresh PNG and performs the frozen QP51 feasibility,
binary search and boundary±2 checks on every TX call. Within-call repeated QPs
reuse their just-computed bytes; there is no cross-call or disk cache. PNG
construction and source search are included in TX timing.

`BPGEndpoint(r, calls, helper)` integrates with the separate unified A3 runner.
`r['bpg']` contains `{path, sha256}` descriptors for `original_request`, `freeze`,
`parity_completion`, `build_receipt`, `phy_module`, and `memory_module`.
All actual header/body decodes go through the independent bounded Calls ledger.
No original scientific ledger is changed. The class exposes `t`, `storage`,
`source_unfit`, `gray`, `last_fit`, `last_source_bytes`, and `last_outcome`.

At an unencodable source point, TX measures the actual failed source-codec
attempt and returns no waveform. No channel or receiver is invoked. RX and
link end-to-end latency must be marked not applicable. A fixed gray image may
be exported outside the timing window for the declared qualitative failure
display, but must not be presented as a received reconstruction.

The runner preserves the first measured output at each fixed source/SNR outside
the timing window for later development-only qualitative comparisons. These are
new frozen-policy development outputs, not old500-holdout cache hits. They never
select the MCS, source rule or best noise. The new BPG timing/display noise uses
the explicit `A3_DEVELOPMENT_TIMING` namespace and seed2001.

Public wrapper interface:

```python
codec = MemoryBPG(library_path)
stream = codec.encode(rgb_uint8_hwc, qp)
restored_uint8_hwc = codec.decode(stream)  # None only for actual source rejection
stream_or_none, source_status = codec.fit(rgb_uint8_hwc, capacity_bytes)
```

Build/parity scripts do not start background work. The coordinator performs the
serial benchmark after qualification and records the common CPU affinity,
threads, GPU environment, per-case outcomes and independent decode budget.
No learned BPG weights exist; compiled library storage is reported separately.
