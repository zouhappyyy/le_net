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
        """适配 fd+loss 网络的在线验证接口，并输出一次性调试信息。

        - 只用最高分辨率主输出(seg_outputs[0])进行评估；
        - 若 target 为 list/tuple，则取第一个元素；
        - 不依赖 deep supervision 的其余分支，避免 batch 维度不一致。
        """
        # 解包网络输出
        if isinstance(output, (tuple, list)) and len(output) == 3:
            seg_outputs, edge_logit_f0, edge_logit_f1 = output
            # seg_outputs 可能是 deep supervision list，取最高分辨率主输出
            if isinstance(seg_outputs, (tuple, list)):
                output_main = seg_outputs[0]
            else:
                output_main = seg_outputs
        else:
            output_main = output

        # target 可能是 [target_tensor]，统一展开
        if isinstance(target, (tuple, list)):
            target_main = target[0] if len(target) > 0 else target
        else:
            target_main = target

        # 调试输出：观察一次性 shape
        if self._online_eval_debug_calls < 3:
            try:
                self.print_to_log_file(
                    f"[DEBUG] run_online_evaluation call {self._online_eval_debug_calls}: "
                    f"output_main.shape={tuple(output_main.shape) if hasattr(output_main, 'shape') else type(output_main)}, "
                    f"target_main.shape={tuple(target_main.shape) if hasattr(target_main, 'shape') else type(target_main)}"
                )
            except Exception:
                pass
            self._online_eval_debug_calls += 1

        # 调用父类在线评估，仅使用主输出和对应 target
        return super().run_online_evaluation(output_main, target_main)
