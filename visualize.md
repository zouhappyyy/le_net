python visualize_val_overlay.py task602_default
python visualize_task570_tp_fp_fn.py task570_default

 python visualize_model_stage_features.py \
   --checkpoint_path ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1 \
   --checkpoint_name model_best \
   --image ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw/ESO_TJ_60011222468.nii.gz \
   --out_dir ./stage_feature_vis \
   --axis z \
   --topk 6
   
   
   python visualize_model_stage_features.py \
   --checkpoint_path ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1 \
   --checkpoint_name model_best \
   --image ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw/ESO_TJ_60011222468.nii.gz \
   --out_dir ./stage_feature_vis \
   --axis z \
   --topk 6
   

   python visualize_model_stage_features.py \
   --checkpoint_path ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1 \
   --checkpoint_name model_best \
   --image /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task530_EsoTJ_30pct/imagesTr/ESO_TJ_60016836064_0000.nii.gz \
   --out_dir ./stage_feature_vis \
   --axis x \
   --topk 6