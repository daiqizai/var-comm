# V9 completion: bounded evidence and reader verification

Base: `dde679637b30f1166bc540d4c7b6fa91fe457e8c`. Results are in
`results/nestcom_v9_completion_20261101_v1/`; the V9 manuscript contains a
self-contained copy of the small reproduction files under `inputs/reproduction`.
All scientific stages have already completed. The production scripts are
provenance entries, **not commands to relaunch the closed study**.

## Reader commands (CPU, no weights required)

From a repository checkout with Python and NumPy:

```sh
python experiments/nestcom-v9-completion-20261101/verify_cdf_ac.py --repo . --output /tmp/nest-v9-cdf
python experiments/nestcom-v9-completion-20261101/verify_reference_vectors.py --vectors results/nestcom_v9_completion_20261101_v1/reproduction/reference_vectors --output /tmp/nest-v9-vectors.json
```

Output locations must be fresh. The first command actually encodes and
independently decodes six synthetic integer-CDF cases using the repository's
unchanged arithmetic coder. The second verifies the integrity of two exported
real-model executions, including exact TX/RX scale CDF hashes, bits and recovered
tokens; it does not claim to regenerate model probabilities. The full WHOLE
reference used one new independent receiver diagnostic. The boundary reference
reused a completed Kodak input. RX receives no TX CDF or hidden source tokens.

## Models and relocatable assets

`reproduction/model_manifest.json` records public upstream links, deployed file
identities, numerical routes and training recipes. NeST file and tensor-state
hashes are distinct. Public VAR/LlamaGen weights are not interchangeable with
project-adapted decoder or JSCC weights. Their exact checkpoints remain in PFS;
no public weight URL is claimed. Supply those matching files or reproduce the
documented training recipe (training was not run in V9).

Place model files in a directory by the names in `reader_asset_mapping`, or use a
JSON object mapping those names to your own paths:

```sh
export NESTCOM_ASSET_ROOT=/your/weights
# Alternatively: export NESTCOM_ASSETS_JSON=/your/asset_paths.json
python experiments/nestcom-v9-completion-20261101/verify_assets.py --manifest results/nestcom_v9_completion_20261101_v1/reproduction/model_manifest.json --output /tmp/nest-v9-assets.json
```

`--only model_100000.pt` checks one requested file. Explicit JSON paths take
precedence over the root; absent overrides use recorded project paths. This
entry checks hashes without loading, modifying or downloading weights. The
original inference engines remain unchanged; their registered execution
configurations and exact commands are in the receipts. Path relocation is
confined to the reader verification entry and isolated process bindings.

## Scientific configuration and results

- `reproduction_config.json`: CDF frequencies, arithmetic interval/termination,
  30 zero-extension reads, canonical re-encode, header/CRC/LDPC/QAM, gray rules,
  precision and metric identities.
- `candidate_pool.csv`: all 396 N512 candidates evaluated at three SNRs,
  actual N1024 inherited/m10 final pools, and five-MCS 7 dB pools. Token inherits
  targets/MCS and expands fallback endpoints, without new calibration. Separate
  equal-bit and matched-granularity ceilings are in the config.
- `phy_layouts.csv`: 24 actual constructor layouts, including all 12 MCSs at
  both budgets; BG, Z, filler, puncturing, interleaver and decoder settings.
- `paid_null.py`: additive extension of the current 3378-profile N1024
  catalogue. Five existing NULL IDs remain, seven occupy unused 3390–3396;
  ordinary records and BPG's reserved 3000–3011 are unchanged. `choose` exhausts
  valid longest-first streams, then returns a paid NULL. Missing/invalid inputs
  raise errors. `runtime` and `received_projection` connect the same PHY parser.
- `test_paid_null.py`: executed 33 checks, including 21 real header and 20 body
  decodes. The real constructor/regression result and catalogue hash are in the
  receipts; it is CPU engineering evidence, not image-quality evidence.
- `kodak/`: 864 logical records, 288 source rows, 60 means, 90 paired intervals,
  all five metrics, failure denominators, reuse identity and capacity records.
  The historical 10000×24 bank is pinned; events are averaged within each source.
- `execution_receipts/`: real process identity and actual exit records. Arrays,
  large catalogue, model weights and full logs remain in PFS.

The source/PHY/RX/metric preparation and summary scripts reproduce the execution
logic with original registered engines. They deliberately use immutable plans
and exclusive outputs; missing assets or unknown failures are not silently
retried. `export_reproduction.py` and `_v2.py` are preserved failed **metadata
export** attempts (missing keys, no scientific calls); `_v3.py` is the successful
export. No calibration, training, old branch or completed scientific call was
repeated to produce V9.
