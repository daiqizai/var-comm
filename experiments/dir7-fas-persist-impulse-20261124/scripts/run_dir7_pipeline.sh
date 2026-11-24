#!/usr/bin/env bash
set -euo pipefail

PY="/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"
EXP_DIR="/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1/experiments/dir7-fas-persist-impulse-20261124"
CKPT_ROOT="/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir7_fas_persist_impulse_v1"
TMP_ROOT="/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir7_fas_persist_impulse_v1"
LOG_DIR="$TMP_ROOT/logs"
RES_DIR="$EXP_DIR/results"
FIG_DIR="$EXP_DIR/figures"

mkdir -p "$CKPT_ROOT" "$TMP_ROOT/cache" "$LOG_DIR" "$RES_DIR" "$FIG_DIR"

echo "[Dir7-Pipeline] Phase 1: Balanced 4x A800 Training (Dir-7A & Dir-7C) + Dir-7B VAR Prefix & Outage Evaluation..."

(
  CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7a --models fas_kriging,fixed_port,random_hop --seeds 1001,2002 \
    --steps 2000 --batch_size 24 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP_DIR/scripts/evaluate_dir7_fpi.py" \
    --suite dir7b --splits common500,kodak24 --seeds 1001,2002,3003 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT" --out_dir "$RES_DIR"
) > "$LOG_DIR/gpu0_phase1.log" 2>&1 &
PID0=$!

(
  CUDA_VISIBLE_DEVICES=1 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7a --models fas_kriging,fixed_port,random_hop --seeds 3003 \
    --steps 2000 --batch_size 24 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
  CUDA_VISIBLE_DEVICES=1 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7c --models sem_impulse_2round,huber_owndenoise,classical_blanking --seeds 1001 \
    --steps 2000 --batch_size 20 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
) > "$LOG_DIR/gpu1_phase1.log" 2>&1 &
PID1=$!

(
  CUDA_VISIBLE_DEVICES=2 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7c --models standard_blind --seeds 1001 \
    --steps 2000 --batch_size 20 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
  CUDA_VISIBLE_DEVICES=2 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7c --models sem_impulse_2round,huber_owndenoise,classical_blanking,standard_blind --seeds 2002 \
    --steps 2000 --batch_size 20 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
) > "$LOG_DIR/gpu2_phase1.log" 2>&1 &
PID2=$!

(
  CUDA_VISIBLE_DEVICES=3 "$PY" "$EXP_DIR/scripts/train_dir7_fpi.py" \
    --track dir7c --models sem_impulse_2round,huber_owndenoise,classical_blanking,standard_blind --seeds 3003 \
    --steps 2000 --batch_size 20 --val_every 250 \
    --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT"
) > "$LOG_DIR/gpu3_phase1.log" 2>&1 &
PID3=$!

wait "$PID0"
wait "$PID1"
wait "$PID2"
wait "$PID3"
echo "[Dir7-Pipeline] Phase 1 complete!"

echo "[Dir7-Pipeline] Phase 2: Parallel Sealed Held-Out Evaluation (Dir-7A & Dir-7C on common500 & kodak24)..."
CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP_DIR/scripts/evaluate_dir7_fpi.py" \
  --suite dir7a --splits common500,kodak24 --seeds 1001,2002,3003 \
  --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT" --out_dir "$RES_DIR" \
  > "$LOG_DIR/eval_dir7a.log" 2>&1 &
EPID0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" "$EXP_DIR/scripts/evaluate_dir7_fpi.py" \
  --suite dir7c --splits common500,kodak24 --seeds 1001,2002,3003 \
  --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT" --out_dir "$RES_DIR" \
  > "$LOG_DIR/eval_dir7c.log" 2>&1 &
EPID1=$!

wait "$EPID0"
wait "$EPID1"
echo "[Dir7-Pipeline] Phase 2 complete!"

echo "[Dir7-Pipeline] Phase 3: Rendering 300-DPI Publication Figures..."
CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP_DIR/scripts/render_dir7_figures.py" \
  --eval_dir "$RES_DIR" --fig_dir "$FIG_DIR" \
  --ckpt_root "$CKPT_ROOT" --tmp_root "$TMP_ROOT" \
  > "$LOG_DIR/render_figures.log" 2>&1

echo "[Dir7-Pipeline] ALL PHASES COMPLETED SUCCESSFULLY!"
