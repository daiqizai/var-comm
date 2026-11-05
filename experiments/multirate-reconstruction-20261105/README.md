# Multirate representation and reconstruction: running implementation

Branch `codex/multirate-reconstruction-20261105` contains the actual 639,592,611-parameter
main model and a four-GPU training run. This is research in progress, not a measured
rate–quality improvement. The previous four P2 arms remain safely paused and their
six pending arms are not launched. The old NeST manuscript is unchanged.

## Model and digital interface

`model.py` initializes the encoder, convolutional decoder and causal entropy model
from the verified public RDVQ checkpoint. It adds a 12-layer, width-768 encoder
refiner, a 24-layer, width-1024 conditional decoder refiner, and a stride-8 fine
representation. All 639.6M parameters, including the codebook, are trainable.
There are 1,360 discrete tokens at 256 and 5,440 at 512, in four scales and 60
causal slices. This is one deterministic reconstruction route, not parallel
AR/flow/diffusion projects. Decoder inputs contain only received quantized values,
their mask and the paid mode. Missing positions are erased before reconstruction.

Budget-conditioned encoding means descriptions at different budgets need not be
nested. The public 16-mode directory specifies 256/512 resolution and eight source
packet budgets: 512, 768, 1024, 1536, 2048, 3072, 4096, 8192 bits at 256; four times
these values at 512. `codec.py` charges a 17-byte magic/version/mode/endpoint/
stream-length/CRC wrapper. Native top-k rANS independently terminates all 60
prefix candidates; TX chooses the longest feasible candidate using actual bytes.
An empty 17-byte NULL packet produces gray. This is a source-packet protocol;
new radio protection/frame evaluation is pending and is not inferred from this check.

## Training and data

`train_v2.py` continues the same DDP model from `train.py` on available devices 1/2/4/5. Global batch is 32,
with microbatches 4 at 256 and 2 at 512. Other users' four occupied GPUs are untouched.
BF16 is used for E/D operations, with FP32 parameters, optimizer and entropy/CDF
arithmetic and TF32 disabled. Loss is MSE + LPIPS-VGG + soft bpp + 4 times excess
budget bpp + VQ/commitment. One in four updates trains the complete representation;
other updates use budget-conditioned prefixes. Estimated lengths guide training;
deployment feasibility always uses actual terminated bytes. Periodic actual-rate
calibration is not implemented in this first training version.

The implemented schedule is 100,000 updates at 256 followed by 50,000 at 512,
with a 500-step learning-rate warmup and cosine decay. These are targets, not
completed training. The first allocation is at most 12 wall hours / 48 GPU-device
hours, with a softer 11-hour training window and checkpoint grace. This finite
segment is not the total research budget. Unknown failures do not auto-retry.

The initial run began with 22,012 verified training images. COCO expansion targets
116,287 training IDs after excluding the existing development/calibration IDs.
Verified native pixels are deduplicated and checked against all held-out hashes.
Download failures are retained, not silently counted as successful data. New
immutable data snapshots are adopted at checkpoints after at least 1,000 new images.
The active corpus now contains all 116,238 verified unique usable images, with 34
transport gaps retained. The independent v2 training continuation admitted the
final sub-threshold remainder and saves data transitions with the checkpoint.

`prepare_openimages.py` prepares 100,000 deterministically selected CC-BY training
images from the [official Open Images index](https://storage.googleapis.com/openimages/web/download_v6.html)
and [CVDF image mirror](https://github.com/cvdfoundation/open-images-dataset).
The mirror supplies resized images, not original Flickr pixels. Attribution,
file/decoded-pixel hashes and failures are retained. This dataset is not yet
admitted into the active run. The 200k–1M unique-image objective remains incomplete;
no near-duplicate or public-pretraining-overlap guarantee is asserted.

## Checks actually passed

- 256 and 512 forward/backward and optimizer updates: finite losses and nonzero
  finite gradients in every module group; measured peak allocation 12.93/18.51 GB.
- Complete model/optimizer serialization and equality checks, including RNG state.
- A real 7,904-bit packet at the 8,192-bit mode, endpoint 43, decoded in a separate
  process using only checkpoint and wire bytes. Received tokens and pixels match
  the first receiver exactly. This is one engineering input, not broad quality evidence.
- CRC rejection and the paid packet capacity check. CPU wire-format checks are
  recorded separately from model qualification.
- Actual DDP optimizer updates and complete rolling checkpoint on PFS.

The first two qualification attempts failed before optimizer updates because the
upstream reranking used FP32 interpolants with BF16 indexed destinations. The final
implementation keeps the entire entropy path FP32. Original failures and bound
source snapshots are preserved. No upstream source is patched.

## Entry points and recovery

`admit_training.py` is the one-time admission/launch entry and refuses an existing
run. Do not rerun it to recover training. `collect_snapshot.py` is read-only.
`resume_training_v2.py` records the completed infrastructure transition at step
26,248 and preserves the first allocation deadline. The live run watches `L/STOP_MAIN` and saves at a safe optimizer boundary; do not
touch the old `STOP_P2`. Checkpoints include model, optimizer, all rank RNG states,
step, data cursor/manifest and schedule. Resume requires a fresh registered output
directory, verified checkpoint identity and the same world size/global batch.
World-size migration is not silently performed.

Large files remain under task PFS scratch and the dedicated checkpoint root.
Code, configurations, lightweight evidence and resource snapshots are published in
`results/multirate_reconstruction_20261105_v1/implementation_v1/`.

Original codec baselines and the plateau audit remain valid historical evidence.
New small-model control, conventional-training control, convergence, formal 512
training completion and actual rate–quality/PHY evaluation are pending. There is
no new quality claim from the training loss or these engineering checks.
