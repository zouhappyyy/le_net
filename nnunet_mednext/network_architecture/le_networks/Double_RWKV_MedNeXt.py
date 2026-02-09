import torch
import torch.nn as nn

from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from nnunet_mednext.network_architecture.le_networks.GLSP3D_RWKV import RWKV_UNet_3D_Encoder


class FusionBlock3D(nn.Module):
    """Simple fusion of two feature maps with same spatial size.

    mode="concat": concat on channel dim then 1x1x1 conv back to out_channels
    mode="sum":    element-wise sum then optional 1x1x1 conv (identity if in_ch == out_ch)
    """

    def __init__(self, in_ch_med: int, in_ch_rwkv: int, out_ch: int, mode: str = "concat"):
        super().__init__()
        assert mode in ["concat", "sum"]
        self.mode = mode

        if mode == "concat":
            self.proj = nn.Sequential(
                nn.Conv3d(in_ch_med + in_ch_rwkv, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.SiLU(inplace=True),
            )
        else:  # sum
            # 先对 RWKV 分支做通道调整到 out_ch，再与 MedNeXt 分支相加
            self.align_med = (
                nn.Identity()
                if in_ch_med == out_ch
                else nn.Conv3d(in_ch_med, out_ch, kernel_size=1, bias=False)
            )
            self.align_rwkv = (
                nn.Identity()
                if in_ch_rwkv == out_ch
                else nn.Conv3d(in_ch_rwkv, out_ch, kernel_size=1, bias=False)
            )
            self.act = nn.SiLU(inplace=True)

    def forward(self, feat_med: torch.Tensor, feat_rwkv: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([feat_med, feat_rwkv], dim=1)
            return self.proj(x)
        else:  # sum
            x_med = self.align_med(feat_med)
            x_rwkv = self.align_rwkv(feat_rwkv)
            return self.act(x_med + x_rwkv)



class MedNeXt_EncoderOnly(MedNeXt_Orig):
    """Wrapper over original MedNeXt that exposes encoder multi-scale features.

    forward(x) -> list of feature maps: [enc0, enc1, enc2, enc3, bottleneck]
    Channel sizes: [C, 2C, 4C, 8C, 16C]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_encoder(self, x: torch.Tensor):
        # this follows MedNextV1.MedNeXt forward until bottleneck
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)

        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)

        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)

        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        bottleneck = self.bottleneck(x)

        return [x_res_0, x_res_1, x_res_2, x_res_3, bottleneck]

    def forward_from_fused_bottleneck(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """从已经融合后的 encoder 特征进入 bottleneck。

        参数:
            fused_feat: 空间尺寸与 enc\_block\_3 输出下采样后一致，通道与原来 bottleneck 输入一致的特征，
                        例如经过双分支融合后的 8C 特征。

        返回:
            bottleneck: 经过 MedNeXt 原始 bottleneck 模块后的特征 (16C)。
        """
        bottleneck = self.bottleneck(fused_feat)
        return bottleneck


class Double_RWKV_MedNeXt_Encoder(nn.Module):
    """双分支编码器：前两层仅 MedNeXt，第三/第四层与 RWKV 双分支 concat+1x1 卷积融合；
    bottleneck 只使用 MedNeXt 的 bottleneck 特征。
    """

    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        exp_r,
        kernel_size: int,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: str = None,
        block_counts=None,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
        fusion_mode: str = "concat",
        rwkv_base_ch: int = None,
    ):
        super().__init__()

        # 强制使用 concat 模式做融合（concat + 1x1x1 卷积）
        if fusion_mode != "concat":
            fusion_mode = "concat"

        if rwkv_base_ch is None:
            rwkv_base_ch = n_channels

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # MedNeXt encoder
        self.mednext_enc = MedNeXt_EncoderOnly(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=1,  # dummy, only encoder used
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            block_counts=block_counts,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        # RWKV encoder
        self.rwkv_enc = RWKV_UNet_3D_Encoder(in_ch=in_channels, base_ch=rwkv_base_ch)

        C = n_channels
        Cr = rwkv_base_ch

        # 第三、第四层用 concat+1x1x1 卷积融合（前两层不融合）
        self.fuse2 = FusionBlock3D(4 * C, 4 * Cr, 4 * C, mode=fusion_mode)
        self.fuse3 = FusionBlock3D(8 * C, 8 * Cr, 8 * C, mode=fusion_mode)

    def forward(self, x: torch.Tensor):
        # MedNeXt 多尺度特征：[C, 2C, 4C, 8C, 16C]
        feats_med = self.mednext_enc.forward_encoder(x)

        # 前两层：只使用 MedNeXt
        f0 = feats_med[0]  # C
        f1 = feats_med[1]  # 2C

        # 从 MedNeXt 的第三层特征（4C）开始送入两层 RWKV
        # RWKV 只输出 4Cr、8Cr 两个尺度
        rwkv_feats_2, rwkv_feats_3 = self.rwkv_enc.forward_from_feat(feats_med[2])

        # 第三、第四层：MedNeXt + RWKV，concat + 1x1x1 卷积融合
        f2 = self.fuse2(feats_med[2], rwkv_feats_2)  # 输出 4C
        f3 = self.fuse3(feats_med[3], rwkv_feats_3)  # 输出 8C

        # bottleneck：只使用 MedNeXt bottleneck（16C）
        f4 = feats_med[4]

        return [f0, f1, f2, f3, f4]



class Double_RWKV_MedNeXt_Encoder(nn.Module):
    """双分支编码器：前两层仅 MedNeXt，第三/第四层与 RWKV 双分支 concat+1x1 卷积融合；
    bottleneck 从融合后的 8C 特征进入 MedNeXt 原始 bottleneck。
    """

    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        exp_r,
        kernel_size: int,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: str = None,
        block_counts=None,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
        fusion_mode: str = "concat",
        rwkv_base_ch: int = None,
    ):
        super().__init__()

        if fusion_mode != "concat":
            fusion_mode = "concat"

        if rwkv_base_ch is None:
            rwkv_base_ch = n_channels

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # MedNeXt encoder
        self.mednext_enc = MedNeXt_EncoderOnly(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=1,  # dummy, only encoder used
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            block_counts=block_counts,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        # RWKV encoder（内部实现 forward_from_feat，只吃 4C 特征）
        self.rwkv_enc = RWKV_UNet_3D_Encoder(in_ch=in_channels, base_ch=rwkv_base_ch)

        C = n_channels
        Cr = rwkv_base_ch

        # 第三、第四层用 concat+1x1x1 卷积融合（前两层不融合）
        self.fuse2 = FusionBlock3D(4 * C, 4 * Cr, 4 * C, mode=fusion_mode)
        self.fuse3 = FusionBlock3D(8 * C, 8 * Cr, 8 * C, mode=fusion_mode)

    def forward(self, x: torch.Tensor):
        # MedNeXt 多尺度特征：[C, 2C, 4C, 8C, 16C]
        feats_med = self.mednext_enc.forward_encoder(x)

        # 前两层：只使用 MedNeXt
        f0 = feats_med[0]  # C
        f1 = feats_med[1]  # 2C

        # 从 MedNeXt 的第三层特征（4C）开始送入两层 RWKV
        rwkv_feats_2, rwkv_feats_3 = self.rwkv_enc.forward_from_feat(feats_med[2])

        # 第三、第四层：MedNeXt + RWKV，concat + 1x1x1 卷积融合
        f2 = self.fuse2(feats_med[2], rwkv_feats_2)  # 4C
        f3 = self.fuse3(feats_med[3], rwkv_feats_3)  # 8C

        # bottleneck：从融合后的 8C 特征进入 MedNeXt 原始 bottleneck，输出 16C
        f4 = self.mednext_enc.forward_from_fused_bottleneck(f3)

        return [f0, f1, f2, f3, f4]

