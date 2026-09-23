#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E="$PWD/experiments/var-latent-enhancement-20260917"
export PYTHONPATH="$PWD/experiments/external-baseline-positioning-20260916/src:$PWD/experiments/token_channel_efficiency_20260923/src:$PWD/experiments/var-short-prefix-hybrid-20260923/src:$PWD/src:$E/src:$E/phase_b/src:$E/evaluation/src:$E/followup/src:$E/mechanisms/src:$E/research/src"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
exec "$PWD/experiments/external-baseline-positioning-20260916/.venv/bin/python" -u -m tools.replay_author_reference_timing --wait
