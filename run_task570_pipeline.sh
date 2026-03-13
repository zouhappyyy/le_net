#!/usr/bin/env bash

# 路径按需修改，这里使用你在 LeRead.md 中的约定
ROOT_DIR="/home/fangzheng/zoule/mednext"
IMAGES_DIR="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/imagesTr"
PRED_ROOT="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83"

cd "$ROOT_DIR" || exit 1

# 1) 使用三个模型对 Task570 做推理
#   直接调用 predict_simple 三次，分别对应三个 trainer + plans

# 模型 1: nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt, plans rwkv
python -m nnunet_mednext.inference.predict_simple \
  -i "$IMAGES_DIR" \
  -o "$PRED_ROOT/Double_CCA_UPSam_fd_RWKV/preds" \
  -t "Task530_EsoTJ_30pct" \
  -m "3d_fullres" \
  -tr "nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt" \
  -p "nnUNetPlansv2.1_trgSp_1x1x1_rwkv" \
  -f 1 \
  --chk "model_best" \
  --disable_tta

# 模型 2: nnUNetTrainerV2_MedNeXt_S_kernel3, plans rwkv
python -m nnunet_mednext.inference.predict_simple \
  -i "$IMAGES_DIR" \
  -o "$PRED_ROOT/MedNeXt_S_kernel3/preds" \
  -t "Task530_EsoTJ_30pct" \
  -m "3d_fullres" \
  -tr "nnUNetTrainerV2_MedNeXt_S_kernel3" \
  -p "nnUNetPlansv2.1_trgSp_1x1x1_rwkv" \
  -f 1 \
  --chk "model_best" \
  --disable_tta

# 模型 3: nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt, plans 无 rwkv
python -m nnunet_mednext.inference.predict_simple \
  -i "$IMAGES_DIR" \
  -o "$PRED_ROOT/Double_CCA_UPSam_fd_loss_RWKV/preds" \
  -t "Task530_EsoTJ_30pct" \
  -m "3d_fullres" \
  -tr "nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt" \
  -p "nnUNetPlansv2.1_trgSp_1x1x1" \
  -f 1 \
  --chk "model_best" \
  --disable_tta

# 2) 评估三个模型在 Task570 测试集上的 Dice
python eval_task570_three_models.py

