# Final implementation addendum — active UM map

The full relocation selection, restoration mapping and projection receipts remain
bound. The two-source whole gate uses the UM runtime only, so its repeated active
file-byte verification excludes exactly the original path prefix
`/home/liulu/projects/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/`.
No other source root, similarly named prefix or Python standard-library file is
excluded. The complete mapping is still parsed and checked against the original
selection; the derived environment receipt records the excluded root, count and
bytes. The already completed full restoration is not redefined as partial.

The actual UM framework imports, native torch extension and lexical interpreter
must still occur in the active original-byte map with the same SHA and length.
UM personal native-library facts must also occur there. All four small generated
venv configurations/path files remain verified, as do the qualified host-native
facts and the two real-GPU dynamic-library observations described in v1. This
reduces unused LDPC I/O; it changes no scientific input, arithmetic, numerical
flag, budget or success criterion.

The final local check executed 22 tests: 21 passed and one Windows symlink test
was skipped because creating symlinks requires a privilege unavailable to that
process. Actual exit was 0, elapsed 0.703 seconds. No model/GPU/PHY execution has
occurred in this implementation task. The accompanying v2 receipt supersedes the
earlier local engineering receipt for final code hashes while retaining that
earlier record unchanged.

## Concrete staged commands for the owner

These are future commands, not records of completed work. Run only after the
actual frozen-runtime relocation/import completion exists, with the reviewed
scripts uploaded at the repository paths. Each command is a separate stage;
none schedules a successor. The registered absolute deadline does not extend on
failure, and a fresh output directory is required.

```bash
BASE=/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu
RT=$BASE/var_comm_runtime_20261010
PY=$RT/envs/engineering-py312-v1/bin/python
S=$BASE/code/VAR_COMM/experiments/entropy-partial-mismatch-20261010/scripts
OUT=$RT/qualification/whole_existing2_v1
DEADLINE=$($PY -B -c 'import time; print(time.time()+1800)')
RUNTIME_SHA=$($PY -B -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RT/receipts/frozen_runtime_relocation_v1/completion.json")
$PY -B "$S/leo_whole_input_map_v1.py" --source-receipt-sha256 db3b531f1b264f68b9e3f84c5232ab0ed47777fd981bd1b2941e28d3e22a2fd9 --model-receipt-sha256 94118d93a4baa02ab8599df70bc2e69cac6d8c2e34b5927f922718a62f17631b --candidate-receipt-sha256 6d30034fc7b37a3df8ff88a0196f35e71e0dde8451080cda9bdf1ab9ae6173ec --runtime-receipt-sha256 "$RUNTIME_SHA" --deadline-unix "$DEADLINE" --out "$OUT"
```

Read `map_receipt.json` and execute its exact `next_prepare_argv`; this only
verifies inputs and registers the gate. Read the returned actual request
descriptor, then explicitly run:

```bash
REQUEST=$OUT/registered_gate/request.json
REQUEST_SHA=$($PY -B -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$REQUEST")
$PY -B "$S/leo_whole_gate_v1.py" run --request "$REQUEST" --request-sha256 "$REQUEST_SHA"
```

The engineering interpreter is only the owner. Its child uses the exact lexical
frozen visual interpreter recorded by the completed relocation receipt. A busy
GPU or any preparation, numerical, resource or wait failure stops this attempt;
these commands do not authorize a retry or a changed threshold.
