import torch
import torch.nn as nn

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
# 直接从具体网络文件导入类，避免通过 le_networks 包装成模块
from nnunet_mednext.network_architecture.le_networks.Double_CCA_UPSam_RWKV_MedNeXt import (
    Double_CCA_UPSam_RWKV_MedNeXt as Double_CCA_UPSam_RWKV_MedNeXt_Orig,
)
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork


class Double_CCA_UPSam_RWKV_MedNeXt(SegmentationNetwork):
    """Thin SegmentationNetwork wrapper around Double_CCA_UPSam_RWKV_MedNeXt_Orig.

    避免与原始网络类的 metaclass 冲突：
      - 这里继承 SegmentationNetwork；
      - 内部持有一个实际的 mednext-style 子网络 self.net；
      - forward 时直接调用 self.net(x)；
      - 并设置 nnUNet 所需的一些属性（conv_op, inference_apply_nonlin, input_shape_must_be_divisible_by, num_classes）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # 实际的 MedNeXt 派生网络
        self.net = Double_CCA_UPSam_RWKV_MedNeXt_Orig(*args, **kwargs)

        # nnUNet evaluation / inference interface
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2 ** 5
        self.num_classes = kwargs["n_classes"]

    def forward(self, x):
        return self.net(x)


class nnUNetTrainerV2_Double_CCA_UPSam_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer using Double_CCA_UPSam_RWKV_MedNeXt as the backbone.

    基于 Double_UpSam_RWKV_MedNeXt 的 trainer，仅在网络结构上增加 CCA+LRDU 解码器，
    训练流程（优化器、学习率等）沿用 MedNeXt/Double_RWKV_MedNeXt 版本。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_network(self):
        # Some parent trainers may define use_amp; be defensive
        use_amp = getattr(self, "use_amp", False)

        # 与原 MedNeXt/Double_RWKV_MedNeXt 保持一致：n_channels=16
        base_channels = 16

        self.network = Double_CCA_UPSam_RWKV_MedNeXt(
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
            rwkv_base_ch=None,
            rwkv_block_dec=None,
            cca_reduction=4,
        )

        if torch.cuda.is_available():
            self.network.cuda()

        # 显存友好，可以按需调整 batch_size
        self.batch_size = 4
