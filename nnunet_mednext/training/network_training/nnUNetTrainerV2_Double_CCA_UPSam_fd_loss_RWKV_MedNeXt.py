import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks.Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import (
    Double_CCA_UPSam_fd_loss_RWKV_MedNeXt as Double_CCA_UPSam_fd_loss_RWKV_MedNeXt_Orig,
)
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.training.loss_functions.deep_supervision import (
    MultipleOutputWithEdgeLoss,
)


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

    通过 MultipleOutputWithEdgeLoss 将 seg deep supervision loss 与两层边界 BCE loss 统一封装。
    """

    def __init__(self, *args, edge_loss_weight_f0: float = 0.4, edge_loss_weight_f1: float = 0.2, **kwargs):
        self.edge_loss_weight_f0 = edge_loss_weight_f0
        self.edge_loss_weight_f1 = edge_loss_weight_f1
        # 调试计数器，仅用于在线评估时输出 shape 信息
        self._online_eval_debug_calls = 0
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

        base_loss = self.loss
        self.loss = MultipleOutputWithEdgeLoss(
            seg_loss=base_loss,
            edge_weight_f0=self.edge_loss_weight_f0,
            edge_weight_f1=self.edge_loss_weight_f1,
        )

    def run_online_evaluation(self, output, target):
        """自定义在线评估：仅使用最高分辨率分割输出与对应 target 计算简化 Dice。

        这样可以完全绕过父类 run_online_evaluation 中对输出结构和 batch 维的假设，
        同时保留在线 Dice 指标的统计功能。
        """
        # 解包网络输出：(seg_outputs, edge_logit_f0, edge_logit_f1)
        if isinstance(output, (tuple, list)) and len(output) == 3:
            seg_outputs, edge_logit_f0, edge_logit_f1 = output
            if isinstance(seg_outputs, (tuple, list)):
                logits = seg_outputs[0]
            else:
                logits = seg_outputs
        else:
            logits = output

        # target 可能是 [target_tensor]，统一展开
        if isinstance(target, (tuple, list)):
            target = target[0] if len(target) > 0 else target

        with torch.no_grad():
            # logits: [B, C, D, H, W]
            num_classes = logits.shape[1]
            output_softmax = softmax_helper(logits)
            output_seg = output_softmax.argmax(1)        # [B, D, H, W]

            # target: [B, 1, D, H, W] -> [B, D, H, W]
            if target.dim() == 5 and target.shape[1] == 1:
                target_lbl = target[:, 0].long()
            else:
                target_lbl = target.long()

            # 初始化累计容器（按 batch 汇总到 per-class tp/fp/fn）
            tp_hard = torch.zeros((num_classes - 1,), device=logits.device)
            fp_hard = torch.zeros_like(tp_hard)
            fn_hard = torch.zeros_like(tp_hard)

            for c in range(1, num_classes):
                pred_c = (output_seg == c)
                targ_c = (target_lbl == c)
                tp_hard[c - 1] = (pred_c & targ_c).sum().float()
                fp_hard[c - 1] = (pred_c & ~targ_c).sum().float()
                fn_hard[c - 1] = (~pred_c & targ_c).sum().float()

            # 父类 finish_online_evaluation 期望的是列表形式，这里保持相同风格
            if not hasattr(self, 'online_eval_tp'):
                self.online_eval_tp = []
                self.online_eval_fp = []
                self.online_eval_fn = []
                self.online_eval_foreground_dc = []

            self.online_eval_tp.append(list(tp_hard.detach().cpu().numpy()))
            self.online_eval_fp.append(list(fp_hard.detach().cpu().numpy()))
            self.online_eval_fn.append(list(fn_hard.detach().cpu().numpy()))
