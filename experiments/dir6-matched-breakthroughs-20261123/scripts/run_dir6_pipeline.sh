#!/usr/bin/env bash
# End-to-End 4x A800-80GB Orchestrator for Direction 6 (`codex/dir6-matched-breakthroughs-v1`):
#   - 27 Matched-Control Checkpoints (`9` model arms x `3` paired seeds: `1001, 2002, 3003`)
#   - 6 Sealed Held-Out 5-Metric Evaluations (`common500` & `kodak24` x `dir6a, dir6b, dir6c`)
#   - 3 Publication-Grade Figures (`300 DPI`) + Full Zero-Link Hygiene Verification
set -euo pipefail

PY="/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"
WORKTREE="/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1"
SCRIPTS="$WORKTREE/experiments/dir6-matched-breakthroughs-20261123/scripts"
LOG_DIR="/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir6_matched_breakthroughs_v1/logs"

mkdir -p "$LOG_DIR"
cd "$WORKTREE"

echo "=== [Stage 0] Precache Ground-Truth ImageNet-1k Labels & Verify Stems/Latents ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/train_dir6_matched.py" --track precache > "$LOG_DIR/stage0_precache.log" 2>&1
cat "$LOG_DIR/stage0_precache.log"

echo "=== [Wave 1] Launching 4-GPU Parallel Matched Training (Dir-6A + Dir-6B, 18 Checkpoints) ==="

# GPU 0: Dir-6A Burst_Symp_OTFS (3 seeds) + Dir-6B BayesVAR CBR=1/512 (3 seeds)
(
  set -euo pipefail
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6a --method burst_symp_otfs --seed "$sd" --steps 2000 --batch-size 24
  done
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6b_var --cbr-denom 512 --seed "$sd" --s1-steps 1200 --s2-steps 900
  done
) > "$LOG_DIR/wave1_gpu0.log" 2>&1 &
PID_W1_0=$!

# GPU 1: Dir-6A Burst_Symp_OFDM_2D (3 seeds, byte-identical!) + Dir-6B BayesVAR CBR=1/256 (3 seeds)
(
  set -euo pipefail
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6a --method burst_symp_ofdm_2d --seed "$sd" --steps 2000 --batch-size 24
  done
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6b_var --cbr-denom 256 --seed "$sd" --s1-steps 1200 --s2-steps 900
  done
) > "$LOG_DIR/wave1_gpu1.log" 2>&1 &
PID_W1_1=$!

# GPU 2: Dir-6B Swin_Holo_Perc CBR=1/512 & CBR=1/256 (6 runs, loss-matched!)
(
  set -euo pipefail
  for cbr in 512 256; do
    for sd in 1001 2002 3003; do
      CUDA_VISIBLE_DEVICES=2 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6b_swin --method swin_holo_perc --cbr-denom "$cbr" --seed "$sd" --s1-steps 900 --s2-steps 900
    done
  done
) > "$LOG_DIR/wave1_gpu2.log" 2>&1 &
PID_W1_2=$!

# GPU 3: Dir-6B Oracle Swin_Genie_Perc CBR=1/512 & CBR=1/256 (6 runs, loss-matched!)
(
  set -euo pipefail
  for cbr in 512 256; do
    for sd in 1001 2002 3003; do
      CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6b_swin --method swin_genie_perc --cbr-denom "$cbr" --seed "$sd" --s1-steps 900 --s2-steps 900
    done
  done
) > "$LOG_DIR/wave1_gpu3.log" 2>&1 &
PID_W1_3=$!

wait "$PID_W1_0" "$PID_W1_1" "$PID_W1_2" "$PID_W1_3"
echo "=== [Wave 1 Complete] All 18 Dir-6A & Dir-6B Checkpoints Trained Successfully! ==="

echo "=== [Wave 2] Launching 4-GPU Parallel Dir-6C Overloaded 4x2 Training + Dir-6A/6B Sealed Evaluations ==="

# GPU 0: Dir-6C SemSIC-4x2 (3 seeds, 1800 steps)
(
  set -euo pipefail
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6c --method semsic_3round --seed "$sd" --steps 1800 --batch-size 10
  done
) > "$LOG_DIR/wave2_gpu0.log" 2>&1 &
PID_W2_0=$!

# GPU 1: Dir-6C OwnDenoise-4x2 (3 seeds, 1800 steps, byte-identical!)
(
  set -euo pipefail
  for sd in 1001 2002 3003; do
    CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6c --method owndenoise_3round --seed "$sd" --steps 1800 --batch-size 10
  done
) > "$LOG_DIR/wave2_gpu1.log" 2>&1 &
PID_W2_1=$!

# GPU 2: Dir-6C Linear-1Stage (seeds 1001, 2002) + Sealed Evaluation Suite Dir-6A (`common500` & `kodak24`)
(
  set -euo pipefail
  for sd in 1001 2002; do
    CUDA_VISIBLE_DEVICES=2 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6c --method linear_1stage --seed "$sd" --steps 1800 --batch-size 10
  done
  CUDA_VISIBLE_DEVICES=2 "$PY" "$SCRIPTS/evaluate_dir6_matched.py" --suite dir6a --splits common500 kodak24 --seeds 1001 2002 3003
) > "$LOG_DIR/wave2_gpu2.log" 2>&1 &
PID_W2_2=$!

# GPU 3: Dir-6C Linear-1Stage (seed 3003) + Sealed Evaluation Suite Dir-6B (`common500` & `kodak24`)
(
  set -euo pipefail
  CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPTS/train_dir6_matched.py" --track dir6c --method linear_1stage --seed 3003 --steps 1800 --batch-size 10
  CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPTS/evaluate_dir6_matched.py" --suite dir6b --splits common500 kodak24 --seeds 1001 2002 3003
) > "$LOG_DIR/wave2_gpu3.log" 2>&1 &
PID_W2_3=$!

wait "$PID_W2_0" "$PID_W2_1" "$PID_W2_2" "$PID_W2_3"
echo "=== [Wave 2 Complete] All Dir-6C Checkpoints + Dir-6A/6B Sealed Evaluations Done! ==="

echo "=== [Stage 3] Sealed Evaluation Suite Dir-6C (`common500` 125 Disjoint 4-Tuples & `kodak24` 6 Disjoint 4-Tuples) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/evaluate_dir6_matched.py" --suite dir6c --splits common500 kodak24 --seeds 1001 2002 3003 > "$LOG_DIR/stage3_eval_dir6c.log" 2>&1
cat "$LOG_DIR/stage3_eval_dir6c.log"

echo "=== [Stage 4] Rendering Publication-Grade Figures (300 DPI) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPTS/render_dir6_figures.py" > "$LOG_DIR/stage4_figures.log" 2>&1
cat "$LOG_DIR/stage4_figures.log"

echo "=== [ALL STAGES COMPLETE] Direction 6 Pipeline Finished Cleanly! ==="
