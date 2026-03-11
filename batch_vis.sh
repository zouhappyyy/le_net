#!/usr/bin/env bash
set -e

DATA_ROOT="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0"
PLANS_FILE="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl"
OUTPUT_FOLDER="/home/fangzheng/zoule/mednext/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv"
OUT_DIR="/home/fangzheng/zoule/mednext/fd_edge_vis_all312"
FOLD=1
PATCH_SIZE=64

mkdir -p "$OUT_DIR"

# 遍历所有 .npy，提取 case_id
for npy in "$DATA_ROOT"/*.npy; do
    fname=$(basename "$npy")
    case_id="${fname%.npy}"
    echo ">>> Visualizing case: $case_id"

    python visualize_fd_edge_and_ds.py run_model \
      --plans_file "$PLANS_FILE" \
      --fold "$FOLD" \
      --output_folder "$OUTPUT_FOLDER" \
      --case_id "$case_id" \
      --data_root "$DATA_ROOT" \
      --output_dir "$OUT_DIR" \
      --patch_size "$PATCH_SIZE" \
      --do_fd_vis
done