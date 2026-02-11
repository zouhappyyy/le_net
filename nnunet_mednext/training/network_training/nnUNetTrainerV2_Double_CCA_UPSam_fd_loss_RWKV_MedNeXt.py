import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks.Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import (
    Double_CCA_UPSam_fd_loss_RWKV_MedNeXt as Double_CCA_UPSam_fd_loss_RWKV_MedNeXt_Orig,
    extract_edge_gt,
)
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork


class Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(SegmentationNetwork):
    """SegmentationNetwork wrapper around Double_CCA_UPSam_fd_loss_RWKV_MedNeXt_Orig.

    forward 返回 (seg_outputs, edge_logit_f0, edge_logit_f1)。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.net = Double_CCA_UPSam_fd_loss_RWKV_MedNeXt_Orig(*args, **kwargs)

        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2 ** 5
        self.num_classes = kwargs["n_classes"]
        self.do_ds = getattr(self.net, "do_ds", False)

    def forward(self, x):
        return self.net(x, return_edge=True)


class nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer，基于 fd+多层边界分支 版本的 Double_CCA_UPSam_RWKV_MedNeXt。

    额外加入两层边界 BCE loss，权重通过 self.edge_loss_weight_f0 / f1 控制。
    """

    def __init__(self, *args, edge_loss_weight_f0: float = 0.4, edge_loss_weight_f1: float = 0.2, **kwargs):
        self.edge_loss_weight_f0 = edge_loss_weight_f0
        self.edge_loss_weight_f1 = edge_loss_weight_f1
        super().__init__(*args, **kwargs)

    def initialize_network(self):
        use_amp = getattr(self, "use_amp", False)
        base_channels = 16

        self.network = Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(
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

        self.batch_size = 2

    def _compute_edge_loss_single(self, edge_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """对单层边界 logit 计算 BCE loss。"""
        if target.dim() == 4:
            target = target.unsqueeze(1)
        elif target.dim() == 5 and target.size(1) != 1:
            # 假设为 one-hot 或多通道标签
            pass
        edge_gt = extract_edge_gt(target)
        return F.binary_cross_entropy_with_logits(edge_logit, edge_gt)

    def compute_loss(self, output, target):
        """重载 loss 计算，加入两层边界 loss。

        output: (seg_outputs, edge_logit_f0, edge_logit_f1)

        注意：seg_outputs 本身仍然是 nnU-Net 期望的 list/tuple 结构，
        会直接传递给父类的 compute_loss（通常封装为 MultipleOutputLoss2），
        以保证 deep supervision 正常工作；本方法只在此基础上额外叠加边界 loss。
        """
        # 拆分三元组输出
        seg_outputs, edge_logit_f0, edge_logit_f1 = output

        # 调用父类的 compute_loss 计算标准分割损失（含 deep supervision）
        seg_loss = super().compute_loss(seg_outputs, target)

        # 额外的两层边界 BCE 损失
        edge_loss0 = self._compute_edge_loss_single(edge_logit_f0, target)
        edge_loss1 = self._compute_edge_loss_single(edge_logit_f1, target)

        return seg_loss + self.edge_loss_weight_f0 * edge_loss0 + self.edge_loss_weight_f1 * edge_loss1
