# 1) 预处理（生成 nnUNet 的预处理数据）
mednextv1_plan_and_preprocess -t 505 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

# 2) 训练单个 fold（Small 模型，kernel=3，fold 0）
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1
nohup mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 505 0 -p nnUNetPlansv2.1_trgSp_1x1x1 > ./log/MedNeXt_S_kernel3_task505_fold0_train.log 2>&1 &

# 3) 训练所有 5 个 folds（示例循环）
for F in 0 1 2 3 4; do
  mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task505 $F -p nnUNetPlansv2.1_trgSp_1x1x1
done

# 4) 使用 UpKern 从已训练的 kernel=3 权重初始化 kernel=5 并训练（注意替换路径）
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task505 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights `SOME_PATH/nnUNet/3d_fullres/Task505/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_final_checkpoint.model` -resample_weights
