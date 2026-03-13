import torch
import torch.nn as nn

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks.Double_CCA_UPSam_fd_RWKV_MedNeXt import (
    Double_CCA_UPSam_fd_RWKV_MedNeXt as Double_CCA_UPSam_fd_RWKV_MedNeXt_Orig,
)
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork


class Double_CCA_UPSam_fd_RWKV_MedNeXt(SegmentationNetwork):
    """Thin SegmentationNetwork wrapper around Double_CCA_UPSam_fd_RWKV_MedNeXt_Orig.

    Training uses the full deep supervision output list from the underlying
    network. During inference/validation, nnUNet calls `self(x)` inside
    `self.inference_apply_nonlin(self(x))`, which expects a 5D tensor, not a
    list, so we unwrap the main output when needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # 实际的 CCA + LRDU + FDConv MedNeXt 派生网络
        self.net = Double_CCA_UPSam_fd_RWKV_MedNeXt_Orig(*args, **kwargs)

        # nnUNet evaluation / inference interface
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2 ** 5
        # 从内部网络透传类别数和 deep supervision 开关
        self.num_classes = kwargs["n_classes"]
        self.do_ds = getattr(self.net, "do_ds", False)

    def forward(self, x):
        out = self.net(x)
        # 训练阶段：loss 期望看到完整的 DS 列表，这里直接返回原始输出
        if self.training:
            return out
        # 推理/验证阶段：nnUNet 通常设置 network.do_ds = False，但为安全起见，
        # 若仍返回 list/tuple，则只取主输出 (out[0]) 供 softmax 使用
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


class nnUNetTrainerV2_Double_CCA_UPSam_fd_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer using Double_CCA_UPSam_fd_RWKV_MedNeXt as the backbone.

    We enforce fp32 (fp16=False) via an explicit __init__ signature so that
    fp16 is only ever passed once down the inheritance chain.
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16: bool = False,
        sample_by_frequency: bool = False,
    ):
        # 不再通过 kwargs 注入 fp16，避免与 checkpoint init 中的位置参数冲突
        super().__init__(
            plans_file,
            fold,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
            sample_by_frequency=sample_by_frequency,
        )
        # unify max epochs as in your other custom trainers
        self.max_epochs = 300
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = 300

    def initialize_network(self):
        # Some parent trainers may define use_amp; be defensive
        use_amp = getattr(self, "use_amp", False)

        # 与原 MedNeXt/Double_RWKV_MedNeXt 保持一致: n_channels=16
        base_channels = 16

        self.network = Double_CCA_UPSam_fd_RWKV_MedNeXt(
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

        # 显存友好，可根据任务需要调整 batch_size
        self.batch_size = 4
