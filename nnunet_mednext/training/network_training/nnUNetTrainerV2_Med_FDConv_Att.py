from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_Optim_and_LR
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.le_networks.Med_FDConv_Att import MedNeXt_FDConv_Att as MedNeXt_FDConv_Att_Orig
import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class MedNeXt_FDConv_Att(MedNeXt_FDConv_Att_Orig, SegmentationNetwork):
    def __init__(self, *args, **kwargs):
        """
        参数签名保持和 MyMedNext_Orig 一致:
        in_channels, n_channels, n_classes, exp_r, kernel_size, deep_supervision, ...
        """
        super().__init__(*args, **kwargs)
        # nnUNet 评估/推理所需的接口
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        # 这里和 MedNeXt 包装类一致，从 kwargs 拿 n_classes
        self.num_classes = kwargs["n_classes"]


class nnUNetTrainerV2_Med_FDConv_Att(nnUNetTrainerV2_Optim_and_LR):
    """
    使用 MedNeXt_FDConv_Att 作为主干的 nnUNet Trainer。
    仅演示网络构造部分，其余训练逻辑沿用 nnUNetTrainerV2。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 统一训练轮数
        self.max_epochs = 300
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = 300

    def initialize_network(self):
        """
        根据 nnUNet 的规划信息构造 MedNeXt_FDConv_Att。
        """
        # 某些基类里可能没有 use_amp，用 getattr 提供默认值
        use_amp = getattr(self, "use_amp", False)

        self.network = MedNeXt_FDConv_Att(
            in_channels=self.num_input_channels,
            n_channels=16,
            n_classes=self.num_classes,
            exp_r=2,
            kernel_size=3,
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            checkpoint_style='outside_block' if use_amp else None,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            norm_type='group',
            dim='3d',
            grn=True
        )
        self.batch_size = 1

        if torch.cuda.is_available():
            self.network.cuda()