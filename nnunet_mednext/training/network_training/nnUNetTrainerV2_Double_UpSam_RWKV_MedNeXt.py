from nnunet_mednext.network_architecture.le_networks import Double_UpSam_RWKV_MedNeXt
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)

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

        # 这里使用与 Double_RWKV_MedNeXt 相同的默认配置
        self.network = Double_UpSam_RWKV_MedNeXt(
            in_channels=self.num_input_channels,
            n_channels=16,
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

        if self.cuda:
            self.network.cuda()

        self.network.inference_apply_nonlin = self.inference_apply_nonlin

    def _build_rwkv_dec_block(self):
        """构造一个 (B, N, C) -> (B, N, C) 的 RWKV 解码序列模块。

        这里为了保持通用性，默认直接返回 nn.Identity()，即只保留 RSU 的 γ/β 线性调制结构。
        如果你已经实现了专门的 3D RWKV/GLSP block，可在此处替换为真正的模块。
        """
        import torch.nn as nn

        return nn.Identity()
