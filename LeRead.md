# 1) 预处理（生成 nnUNet 的预处理数据）
mednextv1_plan_and_preprocess -t 505 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1
mednextv1_plan_and_preprocess -t 601 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

# 2) 训练单个 fold（Small 模型，kernel=3，fold 0）
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1
mednextv1_train 3d_fullres nnUNetTrainerV2_MyMedNext 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1
mednextv1_train 3d_fullres nnUNetTrainerV2_Med_FDConv_Att 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1



mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 601 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > ./log/le/MedNeXt_S_kernel3_task601_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/MedNeXt_S_kernel3_b4_task530_fold1_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MyMedNext 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > ./log/le_fdconv_task505_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Med_FDConv_Att 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > ./log/le_fdconv_att_task505_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le_db_rwkv_task505_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le_rwkv_task530_fold1_train.log 2>&1 &


nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le/rwkv_up_task530_fold1_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le/rwkv_db_cca_up_task530_fold1_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le/rwkv_cca_up_fd_task530_fold1_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt 601 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > ./log/le/rwkv_db_cca_up_fd_loss_task601_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 530 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le_db_rwkv_task530_fold0_train.log 2>&1 &
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv > ./log/le_db_rwkv_task530_fold1_b4c16_train.log 2>&1 &

python ./nnunet_mednext/run/run_training_safe.py 3d_fullres nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv
cd /home/fangzheng/zoule/mednext
mednextv1_train 3d_fullres nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt 530 1 -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val


# 3) 训练所有 5 个 folds（示例循环）
for F in 0 1 2 3 4; do
  mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task505 $F -p nnUNetPlansv2.1_trgSp_1x1x1
done

# 4) 使用 UpKern 从已训练的 kernel=3 权重初始化 kernel=5 并训练（注意替换路径）
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task505 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights `SOME_PATH/nnUNet/3d_fullres/Task505/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_final_checkpoint.model` -resample_weights

# 5) 监控 GPU 使用情况及终止训练进程
nvidia-smi
nvidia-smi -q -d PIDS
fuser -v /dev/nvidia*
pkill -f mednextv1_train


   python visualize_fd_edge_and_ds.py \
     --plans_file /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl \
     --fold 1 \
     --output_folder /home/fangzheng/zoule/mednext/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv \
     --case_id ESO_TJ_2801087499 \
     --data_root /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
     --output_dir fd_edge_vis_flip

 python visualize_fdconv_highfreq.py \
   --data_root /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
   --case_id ESO_TJ_2801087499 \
   --channel 0 \
   --save_prefix fdconv_highfreq_real



python visualize_fd_edge_and_ds.py run_model \
  --plans_file /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl \
  --fold 1 \
  --output_folder /home/fangzheng/zoule/mednext/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv \
  --case_id ESO_TJ_60011222468 \
  --data_root /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
  --output_dir fd_edge_vis_flip \
  --patch_size 64 \
  --do_fd_vis



cd /home/fangzheng/zoule/mednext

chmod +x batch_vis.sh
./batch_vis.sh


python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_2/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_3/validation_raw_postprocessed/summary.json \

python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw_postprocessed/summary.json \


频带模式
python visualize_fd_edge_and_ds.py fd_bands \
  --image "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0/ESO_TJ_60017544096.npy" \
  --case_id "ESO_TJ_60017544096" \
  --output_dir "/home/fangzheng/zoule/mednext/fd_bands_vis" \
  --num_bins 3


python visualize_fd_edge_and_ds.py run_model \
  --plans_file "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl" \
  --fold 1 \
  --output_folder "/home/fangzheng/zoule/mednext/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv" \
  --case_id "ESO_TJ_60017544096" \
  --data_root "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0" \
  --output_dir "/home/fangzheng/zoule/mednext/fd_bands_vis" \
  --patch_size 64 \
  --do_fd_vis
