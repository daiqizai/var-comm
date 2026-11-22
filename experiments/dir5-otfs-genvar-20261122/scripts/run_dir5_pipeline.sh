#!/usr/bin/env bash
set -euo pipefail

PY=/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python
REPO=/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1
CKPT=/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir5_otfs_genvar_v1
LOGS=/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/logs
TRAIN_OUT=$REPO/results/dir5_otfs_genvar_20261122/train
EVAL_OUT=$REPO/results/dir5_otfs_genvar_20261122/eval
FIG_OUT=$REPO/results/dir5_otfs_genvar_20261122/figures
SWIN_CKPT=/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model
V4_CKPT=/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_universal_blind_v4/u_sph_awgn/best_v4.pt

mkdir -p "$CKPT" "$LOGS" "$TRAIN_OUT" "$EVAL_OUT" "$FIG_OUT"
cd "$REPO"

echo "[$(date -Is)] Launching 4x A800 Parallel Training for Direction 5 (11 models)..."

# GPU 0: Dir-5A Main (symp_otfs) + Controlled Waveform Ablation (symp_ofdm_2d)
(
  CUDA_VISIBLE_DEVICES=0 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track otfs --method symp_otfs --run-tag symp_otfs --physical-gpu 0 \
    --total-steps 6300 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261121 \
    --ckpt-dir "$CKPT/symp_otfs" --out-dir "$TRAIN_OUT" > "$LOGS/train_symp_otfs.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track otfs --method symp_ofdm_2d --run-tag symp_ofdm_2d --physical-gpu 0 \
    --total-steps 6300 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261122 \
    --ckpt-dir "$CKPT/symp_ofdm_2d" --out-dir "$TRAIN_OUT" > "$LOGS/train_symp_ofdm_2d.log" 2>&1
) &
PID_G0=$!

# GPU 1: Dir-5A Neural Baseline (blackbox_otfs) + Dir-5B Oracle Genie-CSI Discriminative Baselines (1/128 & 1/256)
(
  CUDA_VISIBLE_DEVICES=1 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track otfs --method blackbox_otfs --run-tag blackbox_otfs --physical-gpu 1 \
    --total-steps 6300 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261123 \
    --ckpt-dir "$CKPT/blackbox_otfs" --out-dir "$TRAIN_OUT" > "$LOGS/train_blackbox_otfs.log" 2>&1

  CUDA_VISIBLE_DEVICES=1 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track swin_ul --method swin_genie_csi --cbr-denom 128 --run-tag swin_genie_csi_cbr128 --physical-gpu 1 \
    --total-steps 4900 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261124 \
    --ckpt-dir "$CKPT/swin_genie_csi_cbr128" --out-dir "$TRAIN_OUT" > "$LOGS/train_swin_genie_cbr128.log" 2>&1

  CUDA_VISIBLE_DEVICES=1 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track swin_ul --method swin_genie_csi --cbr-denom 256 --run-tag swin_genie_csi_cbr256 --physical-gpu 1 \
    --total-steps 4900 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261125 \
    --ckpt-dir "$CKPT/swin_genie_csi_cbr256" --out-dir "$TRAIN_OUT" > "$LOGS/train_swin_genie_cbr256.log" 2>&1
) &
PID_G1=$!

# GPU 2: Dir-5B CBR = 1/128 (K = 1,536) — GenHolo_VAR + BlackBox_VAR + Swin_Holo_Blind
(
  CUDA_VISIBLE_DEVICES=2 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track genvar --method genholo_var --cbr-denom 128 --run-tag genholo_var_cbr128 --physical-gpu 2 \
    --stage1-steps 6000 --stage2-steps 1500 --seed 20261126 \
    --ckpt-dir "$CKPT/genholo_var_cbr128" --out-dir "$TRAIN_OUT" > "$LOGS/train_genholo_var_cbr128.log" 2>&1

  CUDA_VISIBLE_DEVICES=2 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track genvar --method blackbox_var --cbr-denom 128 --run-tag blackbox_var_cbr128 --physical-gpu 2 \
    --stage1-steps 6000 --stage2-steps 1200 --seed 20261127 \
    --ckpt-dir "$CKPT/blackbox_var_cbr128" --out-dir "$TRAIN_OUT" > "$LOGS/train_blackbox_var_cbr128.log" 2>&1

  CUDA_VISIBLE_DEVICES=2 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track swin_ul --method swin_holo_blind --cbr-denom 128 --run-tag swin_holo_blind_cbr128 --physical-gpu 2 \
    --total-steps 4900 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261128 \
    --ckpt-dir "$CKPT/swin_holo_blind_cbr128" --out-dir "$TRAIN_OUT" > "$LOGS/train_swin_holo_cbr128.log" 2>&1
) &
PID_G2=$!

# GPU 3: Dir-5B CBR = 1/256 (K = 768 = 16x Compression!) — GenHolo_VAR + BlackBox_VAR + Swin_Holo_Blind
(
  CUDA_VISIBLE_DEVICES=3 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track genvar --method genholo_var --cbr-denom 256 --run-tag genholo_var_cbr256 --physical-gpu 3 \
    --stage1-steps 6000 --stage2-steps 1500 --seed 20261129 \
    --ckpt-dir "$CKPT/genholo_var_cbr256" --out-dir "$TRAIN_OUT" > "$LOGS/train_genholo_var_cbr256.log" 2>&1

  CUDA_VISIBLE_DEVICES=3 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track genvar --method blackbox_var --cbr-denom 256 --run-tag blackbox_var_cbr256 --physical-gpu 3 \
    --stage1-steps 6000 --stage2-steps 1200 --seed 20261130 \
    --ckpt-dir "$CKPT/blackbox_var_cbr256" --out-dir "$TRAIN_OUT" > "$LOGS/train_blackbox_var_cbr256.log" 2>&1

  CUDA_VISIBLE_DEVICES=3 $PY experiments/dir5-otfs-genvar-20261122/scripts/train_dir5_otfs_genvar.py \
    --track swin_ul --method swin_holo_blind --cbr-denom 256 --run-tag swin_holo_blind_cbr256 --physical-gpu 3 \
    --total-steps 4900 --eval-every 700 --batch-size 28 --lr 8e-4 --seed 20261131 \
    --ckpt-dir "$CKPT/swin_holo_blind_cbr256" --out-dir "$TRAIN_OUT" > "$LOGS/train_swin_holo_cbr256.log" 2>&1
) &
PID_G3=$!

wait "$PID_G0" "$PID_G1" "$PID_G2" "$PID_G3"
echo "[$(date -Is)] ALL 11 DIRECTION 5 MODELS TRAINED CLEANLY! Launching Sealed Held-Out Evaluation..."

CUDA_VISIBLE_DEVICES=0 $PY experiments/dir5-otfs-genvar-20261122/scripts/evaluate_dir5_otfs_genvar.py \
  --split common500 --backbone-ckpt "$SWIN_CKPT" --dir5-ckpt-root "$CKPT" --v4-usph-ckpt "$V4_CKPT" \
  --output-dir "$EVAL_OUT" --physical-gpu 0 > "$LOGS/eval_common500.log" 2>&1 &
PID_E0=$!

CUDA_VISIBLE_DEVICES=1 $PY experiments/dir5-otfs-genvar-20261122/scripts/evaluate_dir5_otfs_genvar.py \
  --split kodak24 --backbone-ckpt "$SWIN_CKPT" --dir5-ckpt-root "$CKPT" --v4-usph-ckpt "$V4_CKPT" \
  --output-dir "$EVAL_OUT" --physical-gpu 1 > "$LOGS/eval_kodak24.log" 2>&1 &
PID_E1=$!

wait "$PID_E0" "$PID_E1"
echo "[$(date -Is)] HELD-OUT EVALUATION COMPLETE! Rendering 3 Publication Figures..."

CUDA_VISIBLE_DEVICES=0 $PY experiments/dir5-otfs-genvar-20261122/scripts/render_dir5_figures.py \
  --eval-dir "$EVAL_OUT" --out-dir "$FIG_OUT" --backbone-ckpt "$SWIN_CKPT" \
  --dir5-ckpt-root "$CKPT" --v4-usph-ckpt "$V4_CKPT" --physical-gpu 0 > "$LOGS/render_figures.log" 2>&1

echo "[$(date -Is)] DIRECTION 5 FULL PIPELINE FINISHED 100% SUCCESSFULLY!"
