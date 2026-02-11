import torch

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks import Double_UpSam_RWKV_MedNeXt as Double_UpSam_RWKV_MedNeXt
import torch.nn as nn
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork


class Double_UpSam_RWKV_MedNeXt(Double_UpSam_RWKV_MedNeXt, SegmentationNetwork):
    """Wrap Double_UpSam_RWKV_MedNeXt to be compatible with nnUNet SegmentationNetwork API."""

    def __init__(self, *args, **kwargs):
        """Signature follows MedNeXt / MyMedNext: in_channels, n_channels, n_classes, ..."""
        super().__init__(*args, **kwargs)
        # nnUNet evaluation / inference interface
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2 ** 5
        self.num_classes = kwargs["n_classes"]

class nnUNetTrainerV2_Double_UpSam_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer using Double_UpSam_RWKV_MedNeXt as the backbone.

    Decoder 的低分辨率两层使用基于 SP_RWKV 的语义上采样，高分辨率两层保持原 MedNeXt 上采样，
    在显存可控的前提下增强全局语义建模能力。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_network(self):
        # Some parent trainers may define use_amp; be defensive
        use_amp = getattr(self, "use_amp", False)

        # 与原 MedNeXt/Double_RWKV_MedNeXt 保持一致：n_channels=16
        base_channels = 16

        self.network = Double_UpSam_RWKV_MedNeXt(
            in_channels=self.num_input_channels,
            n_channels=base_channels,
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
            # 不再在解码器阶段使用 RWKVSeq，令 RSU 内部回退到 SP_RWKV 的轻量 PropagationRWKV
            rwkv_block_dec=None,
        )

        if torch.cuda.is_available():
            self.network.cuda()

        # 显存友好，可以按需调整 batch_size
        self.batch_size = 1
