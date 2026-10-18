# ARPC standalone container v1 (pre-result design)

The paid, image-independent profile supplies native preset and endpoint e∈1..7.
The container carries no source ID, probability table, original pixels, embeddings
or unsent tokens. The fixed caption is UTF-8, generated once per source with the
pinned BLIP base checkpoint, deterministic beam count 3, minimum length 5,
maximum length 20 and sampling disabled. Its payload is never free side data.

Bit order is MSB-first. Fields are:

1. One text-format bit, then encoded caption byte length (9 bits, 1..511), then
   those bytes. Format 0 is raw UTF-8. Format 1 is raw DEFLATE (zlib level 9,
   wbits=-15, memLevel=8, default strategy, Z_FINISH), used only when its complete
   encoded byte count is smaller; ties use raw. The decoded UTF-8 limit is 511
   bytes. RX requires a complete stream without unused data or over-limit output.
2. One metadata-encoding selector bit.
3. A mode/length tag for each native 16-bit group in the paid endpoint prefix.
   Tag 0 means the author's raw 16 bits; tags 1..15 mean that many actual native
   arithmetic bits, including its `finish()`. Original native compression/raw
   decisions and bit strings are unchanged.
4. Actual visual group bits concatenated in scale/raster order.
5. Zero fill to the next byte, fully counted as payload alignment.

For metadata mode 0 the tags use four bits each. Mode 1 sends a 13-bit exact tag
stream length followed by a separately terminated adaptive 16-symbol arithmetic
stream. Counts start at one for all 16 symbols and increment after each symbol;
at most 521 symbols are present, so no rescaling is needed. RX reconstructs these
counts solely from decoded tags. The fixed rule selects mode 1 only if its 13-bit
length plus actual terminated bits are shorter than four bits per tag; ties use
mode 0. This is lossless container coding, not a learned probability model or
quality-based selection. The same upstream integer arithmetic core is used.

The number of tags is known from the paid endpoint's public scale schedule;
individual boundaries follow from actual decoded tags. A canonical re-encoding
check uses only received data. Truncated streams, invalid UTF-8, noncanonical
metadata and nonzero/excess trailing padding are rejected. A changed but valid
raw block is accepted as changed received data, not compared against TX truth.

`payload_bits = caption_bits + 1 + 9 + 1 + mode_and_boundary_bits + visual_bits +
alignment_bits`. The combined mode/boundary field is not counted again as
separate flags. Native and metadata `finish()` costs are already inside their
actual coded lengths. The 13-bit tag length (mode 1 only) is distinct from the
outer 13-bit total-body payload length; neither can be omitted.

The future PHY separately pays outer length 13 and CRC16 inside actual k, plus
the 68-symbol protected public header. Body n=956q, total N=1024, nominal Es=2.
No per-image caption or token may enter the public profile directory. A1 has no
wireless transmissions; fit checks at 1245/2520/4751 payload bits are diagnostics.
The actual cost decomposition will be reviewed before A2. Unreasonable metadata
dominance triggers a fair-serialization review, not a claim of poor ARPC quality.
