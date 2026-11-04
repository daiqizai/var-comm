# Multirate representation and reconstruction research

This is the user-requested research branch `codex/multirate-reconstruction-20261105`.
It starts from the verified pause commit on main. NeST-Com and the previous
RDVQ P0/P1 results remain historical evidence; the four P2 arms are safely
checkpointed and paused. The six remaining P2 arms must not be launched.

The new objective is a single trainable digital image codec that improves the
actual rate–quality curve through representation learning and reconstruction,
with a total model design target of 600M–1B parameters. This target is not an
implemented parameter count or evidence of improved reconstruction.

## What is already established

`audit_existing_plateau.py` checks the existing 500-source records and native
arrays. At 3072 and 4096 bits, the longest-feasible policy sends all 336 tokens,
and all received indices equal the transmitted indices. Both curve points use
the exact same reconstruction arrays as the full-representation reference.
There is no unreceived suffix at these points. These checks make no new model,
metric, bootstrap or PHY calls. Results are in
`results/multirate_reconstruction_20261105_v1/plateau_existing_records/`.

## New scope

- Jointly train encoder, quantization/codebook, entropy model and reconstructor.
- Increase useful transmitted representation capacity as well as model capacity.
- Permit a budget-conditioned encoder: descriptions at different budgets need
  not be nested prefixes. Charge every required mode/budget/resolution field.
- Retain actual entropy coding, independently runnable RX, complete overhead
  accounting and hard packet budgets. Estimated entropy is a training signal,
  not a substitute for transmitted bytes.
- Plan a substantial 256-pixel training stage and a formal 512-pixel stage.
  Expanded natural-image data are not yet verified; the previous 20k training
  IDs are not to be relabeled as a large-scale corpus.
- Use one main reconstruction route. Select its concrete architecture and
  initialization from the existing assets before binding a training job.
- Concentrate available training resources on the new main model. Retain the
  original codec and one small model with the same objective as early controls;
  add the large conventional-training control after the main route stabilizes.

See `DESIGN_zh.md` and `scope.json`. They distinguish intended design from
executed checks. No new training has been launched at this branch handoff.

## Execution boundaries

The old 336 GPU-hour first-batch ceiling does not cap this new project. New
training allocations must be explicit finite windows based on measured
throughput, effective image exposures and current device availability. A new
window must not silently reset a prior deadline or restart old P2.

Before a full run, check only implementation essentials: gradients and numerical
stability, actual independent coding, steady data supply, checkpoint/optimizer/
RNG/sampler restoration and measured memory/throughput. A preliminary paper-level
gain is not a requirement for allowing sufficient training. Scientific evaluation
uses fixed disjoint development, calibration and test identities.

All checkpoints stay under the dedicated PFS checkpoint root, large arrays and
downloads under task scratch. Only code, frozen configurations and lightweight
results enter Git. Unknown failures retain their diagnostics and partial state.

## Reproduce the completed read-only check

```bash
python audit_existing_plateau.py \
  --previous-root /mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/rdvq_prefix_20261104_v1 \
  --output /an/unused/output/directory
```

This is an audit of existing source results, not a new training/evaluation run.
