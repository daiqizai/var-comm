# T0 entropy inventory — historical evidence, not a new experiment

Audit date: 2026-10-09. Requested scientific snapshot: `14b09ecd72984fb39c683d1a62bcd3221c69142f`; original common500: `252176e041758ecb2d3e81b6fde5b587e7e17bb7`.

Only local source/metadata/CSV reads were performed. No SSH, model call, token generation, arithmetic encoding, LDPC, channel simulation or bootstrap was run. Current remote file presence is left for the root agent. No migration archive was read.

## Practical conclusion

The VAR-conditioned source coder does **not** need to be rebuilt. Historical H has a complete independent null-class arithmetic encoder/decoder and a completed 1,000-source original-calibration codec cache. The original calibration source IDs and order exactly match both the frozen M1 registration and P1024 training registration. The 32-source T1 audit can first verify and extract historical records; it should not automatically repeat 32 GPU encoding jobs.

The historical H final quality table is **not** the new T1 result. Its physical framing, failure rule, selected candidates, selection metric, SNRs and populations differ from the requested DINO-L-selected common500 comparison. Existing code and exactly matching frames can be reused, while new protocol/selection/test cells remain necessary.

A complete original-train20k static count cache through scale 9 has **not been verified**. The located training-token preparation source writes only scales 1–8 (255 tokens), retains original+horizontal-flip views, and may remove overlapping sources. The old scale-frequency diagnostic fits on calibration images and only scales 8/9: those fitted counts cannot be relabelled as train20k statistics.

## Classification

| Target | Status | Actual evidence / missing step |
|---|---|---|
| Arithmetic integer coder and null-class provider | REUSE_COMPONENT | Eight local implementation files match the historical H completion-bound SHA exactly; current remote SHA is pending. |
| Original calibration1000 source streams m6–m9 | REUSE_COMPONENT | Historical completion has 200 reused sources + 800 newly encoded, 3,200 new independent roundtrips, exact original calibration IDs. Remote checkpoint and stream NPZ bytes still require fresh hash verification. |
| 32-image T1 VAR source audit | REUSE_COMPONENT | Select a fixed rule from the original order, then validate existing source checkpoints/NPZ, lengths and roundtrip receipts. `remote_checks.json` supplies the first32 as a candidate only, not an independently imposed freeze. |
| Historical source200 clean images/PSNR | REUSE_COMPONENT | Existing CSV has 800 unique source×m rows; source200 NPZ contains clean reconstruction arrays. The later full1000 source stage explicitly made 0 new images, so it cannot by itself establish 1,000 clean-image caches. |
| Historical H result as new DINO-L common500 baseline | NEED_NEW | Old H used PSNR selection and 13/19 dB, and DROP/canonical parsing, not raw KEEP. Compatible source components are reusable; old means are not interchangeable. |
| Full original-train20k m9 static entropy statistics | BLOCKED_MISSING_ASSET | Local evidence is insufficient; inspect remote manifests/NPZ headers before estimating any tokenizer-only work. This is a missing-asset verification state, not evidence that the historical training was absent. |
| Static coding algorithm | REUSE_COMPONENT | Existing positive-frequency CDF and integer coder can consume a frozen scale-shared CDF; new training-derived tables/smoothing/identity need registration. |
| Entropy end-to-end timing | NEED_NEW | H timing adapters exist; new deployment protocol timing remains needed. Preparation timers are not online TX/RX latency. |
| N2048 entropy / m10 | NEED_NEW | H wrapper accepts only m6–m9; catalog hardcodes 68+956 symbols. A new N2048 protocol is required later. |

## Exact entropy semantics

- `h64_source.CODEC_PROTOCOL`: 32-bit arithmetic interval, total CDF mass `2^24`, alphabet 4096, minimum frequency 1, raster order, one continuous complete-prefix stream, no CFG, null class 1000.
- `entropy.probability_cdf`: float64 mass from full log probabilities; counts = `floor(p*(2^24-4096))+1`; remaining count assigned to the first argmax probability. No top-k/top-p truncation.
- Encoder's termination is the legacy pending-plus-two rule; actual terminal bits are in the stream. The stream is bit-addressed, with no mandatory byte alignment on wire in H. Storage in a NumPy byte per bit does not make its wire size 8× larger.
- Decoder creates its own `_Prior`, obtains probabilities from already decoded coarser scales only, and advances only with decoded values. It accepts exactly 30 deterministic zero-lookahead reads beyond the transmitted bits; these are arithmetic syntax, not truth padding. It independently re-encodes for canonical bit equality. No truth token comparison is used as a receiver failure detector.
- Legacy `whole_entropy.VarScaleStream` requires a true/received class in 0…999 and belongs to the old class-conditioned N3060 protocol. Reuse **only** its integer `ArithmeticEncoder`/`ArithmeticDecoder`, through `h64_source.reference_primitives`; do not reuse its class stream or N3060 wire wrapper.
- The frozen prior in `partial_receiver._Prior` uses FP32, `class_emb(1000)`, disabled dropout and full-codebook logits. The decoder provider does not receive source pixels, source labels or transmitter CDFs.

## Existing H wire protocol and gaps against current task

| Field | Actual H behavior |
|---|---|
| Total / header / body | 1024 / 68 / 956 complex symbols. |
| Paid header | 12-bit public profile + CRC16 + 6 convolutional tail bits; rate-matched to 136 bits, QPSK, 68 complex symbols. |
| Profile | Encodes m/K, raw versus arithmetic, q, actual k/n and LDPC layout through a public frozen catalog. Receiver chooses layout from the actual accepted header. |
| Body | Fixed profile-known k; 13-bit stream length + actual source bits + deterministic internal zero fill + CRC16. Source capacity = k−29; actual LDPC n = 956q. |
| MCS | 16QAM/64QAM only; rates 1/2, 2/3, 3/4, 5/6. QPSK body support is not available from this H catalog and must not be assumed for the low-SNR task. |
| Raw/entropy choice and overflow | At each target m down to m6, H-A chooses arithmetic only if shorter than raw, then tests actual capacity. A shorter fitting m is selected without image-quality search. Mode is paid in profile. |
| Failure | Header rejection, CRC rejection, invalid length/padding or canonical parsing failure yields fixed gray 0.5. An accepted wrong valid stream remains an actual erroneous reconstruction. Software/model errors are raised, not relabelled channel loss. |
| Power | Fixed constellation normalization has nominal mean Es=2, not exactly 2N energy per QAM frame. `transmit_body` already records actual body energy. |
| Selection | Historical finalizer explicitly states all3-noise per-source PSNR. Whole candidates were selected on hash200 and retained; full1000 then validated. This is not the present DINO-L top3-on100/full1000 selection design. |

The raw main KEEP path remains frozen. A same-observation raw CRC-DROP diagnostic should be added only for relevant actual CRC-rejected cached raw frames. Its difference cannot be conflated with the new entropy interface.

## Verified historical metadata and existing CSV content

`H/full1000_source/completion.json` SHA256:
`e4bd19e04b74db88e91baec4539d7a305db4e07ad0d456b1bf399ad0d99605b5`.

It records 1,000 sources, 200 prior cache reuses, 800 new encodes, 3,200 new canonical roundtrips, 0 new image renders, 0 new metric calls and 0 new packet decodes. Its elapsed 539.061 seconds was a historical preparation stage, not an ETA or online sender latency.

The 200-source length CSV was actually read with original precision. Existing m9 arithmetic lengths range 2,809–4,516 bits, mean 4,040.97 bits; raw m9 is 5,088 bits. This is descriptive historical calibration information, not a new common500 result, and does not establish reliable reception. `existing_source200_length_summary.csv` includes all four m values; no bootstrap was run.

Model **state** identities are distinct from model **file** SHA:
- VAE state `6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b`.
- VAR state `b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5`.
- Dc state `bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1`.
- Historical file VAE `7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c`, VAR `4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb`.
- Historical environment: PyTorch 2.11.0+cu128, CUDA12.8, RTX4090D, FP32/highest, TF32 off, deterministic cuDNN, six torch threads/two interop. Recheck environment before replay.

## Minimal remote access

`remote_checks.json` lists six precise metadata/data targets plus eight source hashes; do not scan or unpack migration archives. First inspect source checkpoint0000 schema, verify completion hash, and inspect train cache manifests and member headers. Only then fetch the selected 32 small checkpoints/stream NPZ needed for local extraction, or perform a read-only hash pass remotely. Full model checkpoints and all reconstruction arrays need not be downloaded for T0.
