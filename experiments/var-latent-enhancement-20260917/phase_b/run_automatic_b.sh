#!/usr/bin/env bash
set -euo pipefail
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="$(cd "$PHASE/.." && pwd)"
ROOT="$(cd "$EXPERIMENT/../.." && pwd)"
export PYTHONPATH="$PHASE/src:$EXPERIMENT/src:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$ROOT"
exec "$ROOT/experiments/backbone-eval-20260912/.venv/bin/python" -u -m latent_enhancement_b.pipeline
