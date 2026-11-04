# P2 bounded execution details

This supplements P1_P2_PROTOCOL.md without changing its scientific settings.
Four currently idle identical-architecture GPUs are available; the other four
belong to unrelated work. Independent single-GPU matched arms use the qualified
microbatch4 / accumulation4 implementation. This prioritizes paired arms and
resumability over a new DDP performance study. It is not an eight-GPU speed claim.

Training uses shorter-side256 bicubic resize, a random256 crop and horizontal
flip. Image order is an epoch-specific permutation of the original20k manifest;
crop/flip are keyed by training seed, epoch and complete source ID. Both are
identical for matched arms and independent of prefix sampling RNG.

The4,000-update horizon is split at the already registered2,000-update
development checkpoint. Each segment has its own actual process exit and
resource receipt. The second segment restores model, optimizer, RNG, rate
correction and exact data cursor. A planned segment boundary is not a failure
retry and does not reset an arm's total resource allowance. An unknown failure
is retained and is never restarted automatically.

Development evaluation uses all128 fixed audit sources at steps2,000 and4,000.
Final testing uses the original500 test sources at4,000 only. All28 native
prefixes plus NULL are retained, with independent RX and four native-array
metrics. Report fixed and unseen budget curves using actual packet lengths.
No checkpoint, training budget or loss is selected using test results.

Each arm reserves at most10 GPU-device-hours for both training segments
combined. Native evaluation and scoring have separate reservations; the complete
P2 allowance stays256 hours. The owner allows600 seconds for a requested
checkpoint stop, so an in-progress128-image rate audit can finish before save.
The worker checks stop requests at an optimizer boundary. Rolling checkpoints
are atomic and alternate two files, retaining the last complete checkpoint.

The upstream manual describes ImageNet256 training followed by OpenImages/DF2K
resolution adaptation. The downloaded model contains no image-level pretraining
manifest or optimizer/discriminator state; image-level exclusion against all
upstream training data cannot be certified. Our COCO IDs and native pixel hashes
are checked across all newly constructed splits, and this provenance limitation
is retained in the report. The common fresh4000-step GAN-inactive adaptation is
not presented as an exact replay of the author's complete training schedule.
