#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
mkdir -p outputs/logs
exec 9>outputs/logs/prefix_refinement.lock
if ! flock -n 9; then
    printf '%s\n' 'A prefix-refinement pipeline is already active; refusing a duplicate run.' >&2
    exit 1
fi

training=outputs/VAR-PREFIX-REFINEMENT-TRAIN-001
evaluation=outputs/VAR-PREFIX-REFINEMENT-EVAL-001
analysis=outputs/VAR-PREFIX-REFINEMENT-ANALYSIS-001

if [[ ! -f "$training/completion.json" ]]; then
    resume=()
    if [[ -f "$training/metadata.json" ]]; then
        resume=(--resume)
    fi
    python3 scripts/train_prefix_refinement.py --wait-for-gpu "${resume[@]}"
fi
if [[ ! -f "$evaluation/completion.json" ]]; then
    resume=()
    if [[ -f "$evaluation/metadata.json" ]]; then
        resume=(--resume)
    fi
    python3 scripts/evaluate_prefix_refinement.py "${resume[@]}"
fi
if [[ ! -f "$analysis/completion.json" ]]; then
    python3 scripts/analyze_prefix_refinement.py
fi
