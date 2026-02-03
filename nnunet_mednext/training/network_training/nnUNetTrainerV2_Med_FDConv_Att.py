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
        # nnU-Net v2 常用属性名是 max_num_epochs，如果父类用这个，就一并设置
        # 视你的基类实现而定，可以同时设两个以防万一
        self.max_epochs = 300
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = 300

    def initialize_network(self):
        """
        根据 nnUNet 的规划信息构造 MedNeXt_FDConv_Att。
        """
        # 从规划中读取基础配置
        conv_kernel_sizes = self.conv_kernel_sizes  # list\[stages\]\->(kD,kH,kW)
        pool_op_kernel_sizes = self.pool_op_kernel_sizes
        # assert len(conv_kernel_sizes) == 9, "当前 MedNeXt_FDConv_Att 预期 9 个 stage"

        in_channels = self.num_input_channels
        n_classes = self.num_classes

        # 这里用第一个编码阶段的 kernel 作为 enc\_kernel\_size，第一个解码阶段的作为 dec\_kernel\_size
        # 如果你在原版 MedNeXt 中有更精细的映射方式，可以在此处按需修改
        enc_kernel_size = conv_kernel_sizes[0][0]  # 仅取 D 维，假设各向同性
        dec_kernel_size = conv_kernel_sizes[-1][0]

        # 膨胀比例数组，每个 stage 一个；可根据需要调整
        exp_r = [2] * 9

        # 每个 stage block 数；可从 self.net_num_pool 或其它配置推导，这里直接给默认值
        block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # 通道基数，可以和 nnUNet 的 base\_num\_features 对齐
        n_channels = self.base_num_features

        self.network = MedNeXt_FDConv_Att(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=None,  # 使用 enc\_kernel\_size/dec\_kernel\_size
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=self.deep_supervision,
            do_res=True,
            do_res_up_down=True,
            checkpoint_style='outside_block' if self.use_amp else None,
            block_counts=block_counts,
            norm_type='group',
            dim='3d',
            grn=True
        )

        # 将网络搬到正确的 device，并设置为 train 模式
        self.network.inference_apply_nonlin = self.inference_apply_nonlin
        self.network = self.network.to(self.device)