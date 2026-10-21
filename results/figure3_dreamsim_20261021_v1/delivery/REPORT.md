# Fig.3 fixed-output DreamSim supplement

Completed ten existing native-float pairs: Pastries95 and Shopping cart75,
N1024, 10 dB, timing cases s10_i0095_r3 and s10_i0075_r3. Method order is
P100000, Swin200000, BPG, ARPC, NEST. The original manifest LPIPS-Alex strings
are preserved exactly. Each reconstruction archive and its dtype/shape-aware
array hash matches the original timing record. The two original source pixel
arrays match their registered PNGs. No reconstruction or noise was changed.

DreamSim uses the main evaluation's byte-pinned official ensemble, loader,
weights and dependency overlay. Native float RGB tensors are resized to
224x224 with bicubic antialiasing, without cropping or PNG quantization.
FP32, TF32/AMP off, batch1, six threads/two interop threads. The distance is
one minus cosine similarity of the ensemble embeddings. Both LPIPS-Alex and
DreamSim are lower-is-better. These are single-output whole-image scores,
not three-noise averages or common500 population statistics.

Actual new computation: one model load, two source embeddings, ten image
embeddings and ten distances. All23 recorded calls are COMPLETE. Child and
outer actual waits returned0, all recorded processes are gone, no scientific
failure or retry. New LPIPS/PSNR, channel, reconstruction and bootstrap calls=0.
The existing H800 machine's free GPU1 was used; this is not a latency result.

Read-only input checks first identified the existing NPZ key `image` and the
historical dtype+shape+bytes hashing convention. Those pre-registration checks
made zero model or scientific calls; the finalized registration pins both.
All previous main curves, sensitivity experiments and tables are unchanged.
