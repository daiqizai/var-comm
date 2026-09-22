#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$EXPERIMENT/../.." && pwd)"
export PYTHONPATH="$EXPERIMENT/src:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$ROOT"
exec "$ROOT/experiments/backbone-eval-20260912/.venv/bin/python" -u -m latent_enhancement.pipeline
