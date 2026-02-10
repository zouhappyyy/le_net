import torch

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks import Double_UpSam_RWKV_MedNeXt as Double_UpSam_RWKV_MedNeXt_Orig
from nnunet_mednext.network_architecture.le_networks.RWKVSeq import RWKVSeq
import torch.nn as nn


class Double_UpSam_RWKV_MedNeXt(Double_UpSam_RWKV_MedNeXt_Orig):
    """Wrap Double_UpSam_RWKV_MedNeXt to be compatible with nnUNet SegmentationNetwork API."""

    def __init__(self, *args, **kwargs):
        """Signature follows MedNeXt / MyMedNext: in_channels, n_channels, n_classes, ..."""
        super().__init__(*args, **kwargs)
        # nnUNet evaluation / inference interface
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = self.inference_apply_nonlin if hasattr(self, "inference_apply_nonlin") else None
        self.input_shape_must_be_divisible_by = 2 ** 5
        self.num_classes = kwargs["n_classes"]

class nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer using Double_UpSam_RWKV_MedNeXt as the backbone.

    Decoder 的低分辨率两层使用 RWKV 语义上采样，高分辨率两层保持原 MedNeXt 上采样，
    在显存可控的前提下增强全局语义建模能力。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_network(self):
        # Some parent trainers may define use_amp; be defensive
        use_amp = getattr(self, "use_amp", False)

        self.network = Double_UpSam_RWKV_MedNeXt(
            in_channels=self.num_input_channels,
            n_channels=8,
            n_classes=self.num_classes,
            exp_r=2,
            kernel_size=3,
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            checkpoint_style="outside_block" if use_amp else None,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            norm_type="group",
            dim="3d",
            grn=True,
            fusion_mode="concat",
            rwkv_block_dec=self._build_rwkv_dec_block(),
        )



        # 设置推理时的非线性激活
        self.network.inference_apply_nonlin = self.inference_apply_nonlin
        self.batch_size = 1

        if torch.cuda.is_available():
            self.network.cuda()

    def _build_rwkv_dec_block(self):
        """构造一个 (B, N, C) -> (B, N, C) 的 RWKV 解码序列模块。

        这里基于 RWKVSeq，假设解码最低分辨率特征尺寸为 (D,H,W)=(4,16,16) 或其他已知值。
        请根据实际 patch 大小调整 patch_resolution。
        """
        # 这里先给一个合理的默认值，假设解码最低层特征为 4x16x16，总共 1024 tokens。
        # 通道 C 将由 RSU 在运行时保证匹配 RWKVSeq.n_embd。
        patch_resolution = (4, 16, 16)
        # n_embd 实际上在 RSU 中是对应 stage 的通道数，但 RWKVSeq 只关心 C 一致即可。
        # 这里使用占位通道数，真正的 C 在 forward 时由 RSU 校验。
        n_embd_placeholder = 8 * self.network.encoder.mednext_enc.n_channels  # 对应 8C stage 的通道数大致规模
        return RWKVSeq(n_embd=n_embd_placeholder, patch_resolution=patch_resolution)
