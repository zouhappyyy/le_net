
#!/usr/bin/env bash
# 验证多个模型的 Linux 脚本
# 使用方法：
#   chmod +x val.sh
#   ./val.sh

#set -e  # 任意命令失败就退出脚本；如果想遇错继续，注释掉这一行

# 如果需要激活 conda 环境，请在这里加上（根据你机器的路径修改）：
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate zl_nnunetv1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    local desc="$1"
    shift
    log "开始：$desc"
    log "命令：$*"
    "$@"
    log "完成：$desc"
    echo "------------------------------------------------------------"
}


# 1. Double_CCA_UPSam_fd_RWKV_MedNeXt fold 1
run_cmd "Double_CCA_UPSam_fd_loss_RWKV_MedNeXt Task530 fold 1 验证" \
    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt 530 1 \
    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

# 2. Double_CCA_UPSam_fd_loss_RWKV_MedNeXt fold 2
run_cmd "Double_CCA_UPSam_fd_loss_RWKV_MedNeXt Task530 fold 2 验证" \
    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt 530 2 \
    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

# 3. Double_CCA_UPSam_fd_loss_RWKV_MedNeXt fold 3
run_cmd "Double_CCA_UPSam_fd_loss_RWKV_MedNeXt Task530 fold 3 验证" \
    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt 530 3 \
    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

run_cmd "Double_CCA_UPSam_fd_loss_RWKV_MedNeXt Task530 fold 3 验证" \
    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt 530 1 \
    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

## 4. Double_RWKV_MedNeXt fold 0  shibai
#run_cmd "Double_RWKV_MedNeXt Task530 fold 0 验证" \
#    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 530 0 \
#    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val
#
## 5. Double_RWKV_MedNeXt fold 1 shibai0.0
#run_cmd "Double_RWKV_MedNeXt Task530 fold 1 验证" \
#    mednextv1_train 3d_fullres nnUNetTrainerV2_Double_RWKV_MedNeXt 530 1 \
#    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

# 6. MedNeXt_S_kernel3 Task505 fold 1
run_cmd "MedNeXt_S_kernel3 Task505 fold 1 验证" \
    mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 530 1 \
    -p nnUNetPlansv2.1_trgSp_1x1x1_rwkv -val

log "全部验证命令执行完毕。"