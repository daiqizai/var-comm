# P3 RX cache identity correction

The first CPU-only cache preparation stopped before saving a cache or processing a P3
reception. It asserted that every `preprocessing` field was identical. This field
actually records SHA256 of each source's preprocessed uint8 pixels, as defined by
`experiments/nestcom-n512-20261028/prepare_assets.py` and `nest_source_worker.py`.
It is a reference-image identity, not a numerical image-processing pipeline version.

The original p3_rx_replay.py, failure.json and exit-1 receipt remain unchanged.
Independent p3_rx_replay_v2.py indexes already verified received-token states by
all renderer inputs: coarse tokens, positions, values, model hashes, numeric route,
completion settings and endpoint. The original full state and its source pixel
hash remain stored and hash-checked. Collisions must have identical native pixels.
The renderer never reads the `preprocessing` label or source pixels; it consumes
the actual tokens and native model state. This is verified against the preserved
renderer source, not assumed from equal token counts.

No source ID or expected TX tokens enter the replay function. Unknown accepted
wire bits are preserved as pending native RX work, never replaced with expected
tokens or gray. There were zero such misses in the observed P3 events. Native
images are looked up only after exact independent-RX stream memoization. Actual
source/image metric pairs are joined after saved RX; original reference hashes
and all 286 used native arrays were independently verified. No PHY, successful
source decode, image generation or metric inference was repeated by this repair.
