# Matched four-image uplink experiment

Four independently encoded 256×256 images occupy the same 12,288 complex resource elements. The two-antenna receiver returns four images. The external baseline uses the authors’ DeepJSCC-Q2 neural modules and progressive complex spreading projections, with a two-antenna, receiver-estimated-CSI front end. The local comparator is the frozen three-call SemSIC receiver from [the replica-routing experiment](../replica-routing-20261127/).

## Inputs and identities

The immutable protocol specifies the two power spreads, the training objective, the holdout, the checkpoint-selection rule and the primary estimands. The supplementary weak-only diagnostic has its own dated protocol. The four-user split of 500 images has 125 disjoint quartets. The 256 validation images and the 20,000 training images have separate pixel identities. The public release includes the 500 formal source identities and digests. The manifest of training and validation source pixels stays with the existing project dataset on PFS, with its digest recorded in the execution ledger.

Use the existing pinned project snapshots, frozen weights and visual environment from the November 24 and November 27 receiver studies. The three native checkpoint digests are verified before evaluation. The external codec starts from random weights. The public releases examined contained source code without a suitable pretrained four-user checkpoint. The small third-party modules and their licenses are fetched by the script from the pinned author and CompressAI commits. The vendor manifest records individual Git blob and SHA-256 digests. No third-party weights are published here.

## Execution

The physical study on PFS uses three separate directories. The code and small results are under `code/liulu/analysis/var_comm_pnoma_matched_20261127`, source snapshots and logs under `liberai-tmp/liulu/liulu-var-comm/pnoma_matched_20261127_v1`, and saved weights under `liberai-checkpoint/liulu/var_comm/pnoma_matched_20261127_v1`. The root and previous study paths are specified in `common.py`. The GPU owners reject duplicate runs. For an independent replication, set those paths to fresh physical directories with the pinned dependencies installed. The runtime was PyTorch 2.11.0+cu128. Training used mixed precision for the Q2 network, with FP32 parameter updates, and evaluation and timing used FP32 with TF32 disabled.

~~~sh
python3 -B fetch_sources.py "$SCRATCH/upstream"
python3 -B launch.py train --gpu 3
# When job_train/wait.json reports 0, launch the held-out comparison.
python3 -B launch.py evaluate --gpu 3
# Both following steps require the completed evaluation.
"$PY" -B analyze.py
python3 -B run_solo.py --gpu 3
"$PY" -B other_users.py
~~~

The baseline follows the authors’ one-, two- and four-user progressive training scheme using the Q2 network, independent learned transmit and receive projections, and per-user unit power. In this deployment the learned projection compresses each site into 12 complex coordinates per user and spreads them over a shared 48-coordinate space. Each user has a fixed port/signature. Four-by-four invertible pixel packing preserves every input pixel and aligns the official convolutional model with the waveform budget. The transmit-side SNR conditioning input is constant. The receive side estimates the channel and noise from the same in-band reference allocation as SemSIC. This spatially overloaded fixed-signature test evaluates simultaneous decoding when four users are active. It does not measure random activity detection against an unregistered population.

## Statistics and timing

The weak user is the slot with the lowest assigned received-power level, rotated by quartet and seed. Each of the three seeded channel/noise draws is shared across methods, as are images and absolute power. Each transmitter constructs its own waveform. The local receiver uses three stored checkpoints paired to those draws. The external codec has one training initialization. Quality estimates average draws within each group before sampling 125 groups with replacement 10,000 times. Timing uses a separate 32-group bootstrap and measures waveform-to-four-RGB receiver time in alternating order on the same A800. Host I/O, scoring and transmitter computation lie outside this boundary. The GPU was shared. These are machine-local processing times.

The `solo` diagnostic silences the other three transmitters, with the nominal weak signal, fade and absolute noise fixed. The two unmodified receivers use only their own observation. It uses the registered first draw and all 125 groups. `other_users.py` reports an exploratory arithmetic decomposition of the formal four-user mean after the primary results were known. It introduces no new image reconstructions.
