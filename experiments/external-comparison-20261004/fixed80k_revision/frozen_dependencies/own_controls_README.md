# Isolated own-method controls at N1024 and N2048

Protocol: `OWN-CONTROLS-N2048-20261004-R1`.

This extension creates N2048 QPSK whole-scale unconditional and entropy-order policies. It also exports exactly the frozen N1024 P1024, D_U_QPSK and entropy_policy reconstructions needed by the new comparison. It does not train models, read holdout, select on development, modify an old policy, or edit an old implementation.

## Scientific scope

- SNR: 1, 7, 13 dB. Original calibration seeds: 4101/4102/4103. Original development seeds: 2001/2002/2003.
- Calibration: original 1,000 sources; the original first 200 and seed 4101 form the screening subset. Development: original 100 sources, three noise realizations.
- Same frozen Encoder/F, VAR, Dc, LPIPS and DINO identities as the published M1 registration. No learned parameter changes.
- N2048 uses 68 QPSK header symbols and exact per-frame energy 4096. Header, separate prefix/partial CRC and tail, and rate matching use the byte-derived original PHY.
- Exactly 21 candidates: whole m4–m8 (5), entropy partial prefixes m4–m7 at the original four K values (16). No partial m8, 16QAM, raster/random/oracle search, or new candidate.
- Whole and entropy policies each use the original rank and shortlist rule: screen top two feasible plus best whole, followed by full calibration of the union. A single feasible action remains legal. Full calibration uses between 9,000 and 45,000 frames, usually 18,000–45,000; screening uses 12,600.
- `D_U_whole_N2048_v1` is a newly registered no-class-header q=0 whole policy. It must be labeled separately from historical N1024 D_U, whose protocol paid class bits even though unconditional generation ignored them.
- N2048 development produces 1,800 rows. N1024 exact replay produces 2,700 rows; its selected policies remain unchanged. P2048 is already available in the inherited historical caches and is not replayed here.

## Inputs and integrity checks

The exact original M1 source and the calibration/development registration copies must match the checked M1 publication. New PHY bytes must equal the original `partial_phy.py` after only its protocol ID, two N validation tuples, and registration budget list change. Loaded frozen model states must equal the original visual identity; original cached F/tokens/RGB population order and hashes are checked by the native loader and again against the published registration.

The N1024 replay additionally checks original native scalar parity, exact reconstructed float RGB, and reference RGB against the completed unified metric source checkpoints. Those checkpoints are authenticated by all 100 SHA256 entries in the published scoring inventory, not just their own internal checksum. The original published latent-field compatibility alias is used unchanged; parity tolerances are unchanged.

The new metric reference is the inherited P2048 `source_rgb`, matching new Swin/HiFi evaluation. Original M1 used uint8/255 RGB. Both references have identical original uint8 pixels but may differ slightly as float32. The lossless NPZ therefore preserves **both** references and explicitly keeps native old-target metrics separate. Do not reuse old metrics as if their target-float fingerprints matched.

## Runtime and commands

Use the original Linux VAR/modern unified-metrics Python environment with CUDA0 and the original numerical flags. Do not run these model stages in the new Swin Torch1.12 environment. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. The native frozen loaders enforce their usual GPU availability and thermal boundaries. Root controls GPU scheduling; no GPU work is launched by the CPU audit.

From the deployed runtime directory, CPU tests:

```sh
OWN_CONTROLS_TEST_ROOT="$ROOT" "$VAR_PYTHON" -B -m unittest own_controls_tests own_controls_phy_tests -v
```

CPU source admission (verifies published source and registration bytes without loading models):

```sh
"$VAR_PYTHON" -B -c 'import own_controls_common as c,sys; print(len(c.source_gate(sys.argv[1],c.HERE/"own_controls_protocol.json")))' "$ROOT"
```

Full native sequence after CPU admission, checked source publication, and an idle GPU:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$VAR_PYTHON" -B own_controls.py --root "$ROOT" --stage all
```

Individual stages are `qualify`, `screen`, `calibrate`, `development`, `export-n1024`. Qualification uses four original calibration sources and all 21 real physical round trips (84 cases), checks independent TX/RX entropy ordering and exact q=0 parity with old unconditional completion. It is an engineering gate, not a measured comparison.

Outputs are isolated under `outputs/EXTERNAL-COMPARISON-20261004/own_controls` and `results/external_comparison_20261004/own_controls`. Each completed source is sealed before progressing; restarting the same command checks and reuses it. SIGTERM/SIGINT request a safe pause (exit 75), and partially computed uncommitted sources are recomputed. An existing different orphan float cache is rejected. A failure receipt blocks automatic retry until reviewed. No Git commands run here.

## Metric handoff and remaining work

This package completes **calibration and reconstruction**. Development/export completion explicitly says `independent_metrics_pending=true`; it does not claim the whole external-comparison plan is complete. Use the unchanged unified metric suite against `comparison_reference`, retaining original metrics as native/original fields, and add the frozen ResNet50 top1 confidence, semantic error and confidently-wrong indicators. New metric values must not feed policy selection. The exact paired float caches, source identities, policy IDs, original parity records and published input bindings are ready for that scorer and final comparison/report merge.

Each `*/float_reconstructions/NNNN.npz` contains deduplicated `images` (float32 CHW 256), `native_reference`, and `comparison_reference`. Its sibling source checkpoint provides image slots, domain-separated RGB hashes, SHA256 container binding, and the ordered scientific rows. Hash domain: `float32:3,256,256:RGB` followed by a NUL byte and contiguous image bytes. Cached scalar metadata must never substitute for image validation.

Initial planning estimate is 4–12 GPU hours for these controls, not a measured promise. Per-source status records actual receiver event counts, elapsed time and a running ETA, which should replace this estimate after initial real sources. The extension does not contend with ongoing Swin training without explicit scheduling by the parent controller/operator.
