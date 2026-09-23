#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
E=experiments/var-latent-enhancement-20260917
export PYTHONPATH="$ROOT/experiments/var-short-prefix-hybrid-20260923/src:$ROOT/src:$ROOT/$E/src:$ROOT/$E/phase_b/src:$ROOT/$E/evaluation/src:$ROOT/$E/followup/src:$ROOT/$E/mechanisms/src:$ROOT/$E/research/src"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export VAR_COMM_DECODER_GATE="$ROOT/outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json"
exec python3 -u -m short_prefix.controller
