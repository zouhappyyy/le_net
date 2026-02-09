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


class Double_RWKV_MedNeXt_Encoder(nn.Module):
    """Double-branch encoder: RWKV_UNet_3D_Encoder + MedNeXt encoder.

    Inputs:
        x: (B, in_channels, D, H, W)

    Outputs:
        list of fused features [f0, f1, f2, f3, f4]
        where channel sizes follow MedNeXt convention but bottleneck is 8C.
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
        block_counts = None,
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
        fusion_mode: str = "concat",
        rwkv_base_ch: int = None,
    ):
        super().__init__()

        if rwkv_base_ch is None:
            rwkv_base_ch = n_channels

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # MedNeXt encoder
        self.mednext_enc = MedNeXt_EncoderOnly(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=1,  # dummy, we only use encoder
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

        # 四个尺度特征融合：1C, 2C, 4C, 8C
        self.fuse0 = FusionBlock3D(C, Cr, C, mode=fusion_mode)
        self.fuse1 = FusionBlock3D(2 * C, 2 * Cr, 2 * C, mode=fusion_mode)
        self.fuse2 = FusionBlock3D(4 * C, 4 * Cr, 4 * C, mode=fusion_mode)
        self.fuse3 = FusionBlock3D(8 * C, 8 * Cr, 8 * C, mode=fusion_mode)

        # 瓶颈层：Gated fusion，输入为 MedNeXt 与 RWKV 的最高层特征，输出 8C
        self.gated_bottleneck = nn.Sequential(
            FusionBlock3D(16 * C, 8 * Cr, 8 * C, mode=fusion_mode),
        )

    def forward(self, x: torch.Tensor):
        feats_med = self.mednext_enc.forward_encoder(x)
        feats_rwkv = self.rwkv_enc(x)

        f0 = self.fuse0(feats_med[0], feats_rwkv[0])
        f1 = self.fuse1(feats_med[1], feats_rwkv[1])
        f2 = self.fuse2(feats_med[2], feats_rwkv[2])
        f3 = self.fuse3(feats_med[3], feats_rwkv[3])

        # 瓶颈：8C 通道的 gated 融合输出
        f4 = self.gated_bottleneck[0](feats_med[4], feats_rwkv[4])

        return [f0, f1, f2, f3, f4]


class Double_RWKV_MedNeXt(MedNeXt_Orig):
    """Full segmentation network that uses Double_RWKV_MedNeXt_Encoder features
    and keeps MedNeXt decoder / output heads. Bottleneck feature has 8C channels.
    """

    def __init__(self, *args, fusion_mode: str = "concat", rwkv_base_ch: int = None, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels = kwargs.get("in_channels", args[0] if len(args) > 0 else 1)
        n_channels = kwargs.get("n_channels", 32)
        exp_r = kwargs.get("exp_r", 4)
        kernel_size = kwargs.get("kernel_size", 3)
        enc_kernel_size = kwargs.get("enc_kernel_size", None)
        dec_kernel_size = kwargs.get("dec_kernel_size", None)
        deep_supervision = kwargs.get("deep_supervision", False)
        do_res = kwargs.get("do_res", False)
        do_res_up_down = kwargs.get("do_res_up_down", False)
        checkpoint_style = kwargs.get("checkpoint_style", None)
        block_counts = kwargs.get("block_counts", [2, 2, 2, 2, 2, 2, 2, 2, 2])
        norm_type = kwargs.get("norm_type", "group")
        dim = kwargs.get("dim", "3d")
        grn = kwargs.get("grn", False)

        # 用双分支 encoder 替换原来的 MedNeXt encoder 结构
        self.double_encoder = Double_RWKV_MedNeXt_Encoder(
            in_channels=in_channels,
            n_channels=n_channels,
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
            fusion_mode=fusion_mode,
            rwkv_base_ch=rwkv_base_ch,
        )

    def forward(self, x):
        # 使用双分支 encoder 生成 multi-scale 特征，其中 f4 为 8C 通道的瓶颈输出
        f0, f1, f2, f3, bottleneck = self.double_encoder(x)

        x = bottleneck

        x_up_3 = self.up_3(x)
        dec_x = f3 + x_up_3
        x = self.dec_block_3(dec_x)

        x_up_2 = self.up_2(x)
        dec_x = f2 + x_up_2
        x = self.dec_block_2(dec_x)

        x_up_1 = self.up_1(x)
        dec_x = f1 + x_up_1
        x = self.dec_block_1(dec_x)

        x_up_0 = self.up_0(x)
        dec_x = f0 + x_up_0
        x = self.dec_block_0(dec_x)

        out_0 = self.out_0(x)

        if self.do_ds:
            x_ds_1 = self.out_1(f1)
            x_ds_2 = self.out_2(f2)
            x_ds_3 = self.out_3(f3)
            x_ds_4 = self.out_4(bottleneck)
            return [out_0, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return out_0

