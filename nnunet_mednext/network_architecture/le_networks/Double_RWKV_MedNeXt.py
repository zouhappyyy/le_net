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



from nnunet_mednext.network_architecture.mednextv1.blocks import (
    MedNeXtUpBlock,
    MedNeXtBlock,
    OutBlock,
)
from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import (
    Double_RWKV_MedNeXt_Encoder,
)


class Double_RWKV_MedNeXt(nn.Module):
    r"""完整的 Double\_RWKV\_MedNeXt:
    \- encoder 使用双分支 `Double_RWKV_MedNeXt_Encoder`，输出 \[C, 2C, 4C, 8C, 16C];
    \- decoder 结构与原始 MedNeXt 相同。
    """

    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r,
        kernel_size: int = 7,
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

        self.do_ds = deep_supervision
        assert dim in ["2d", "3d"]

        # 统一 kernel\_size 设置
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if block_counts is None:
            # 与 MedNeXt 默认一致
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # ---------- Encoder: 双分支 ----------
        self.encoder = Double_RWKV_MedNeXt_Encoder(
            in_channels=in_channels,
            n_channels=n_channels,
            exp_r=exp_r,
            kernel_size=enc_kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=False,          # encoder 不需要 DS
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

        # ---------- Decoder，与 MedNeXt 一致 ----------

        if dim == "2d":
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        C = n_channels
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # up\_3: 16C -> 8C
        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * C,
            out_channels=8 * C,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=8 * C,
                    out_channels=8 * C,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[5])
            ]
        )

        # up\_2: 8C -> 4C
        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * C,
            out_channels=4 * C,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=4 * C,
                    out_channels=4 * C,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[6])
            ]
        )

        # up\_1: 4C -> 2C
        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * C,
            out_channels=2 * C,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=2 * C,
                    out_channels=2 * C,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[7])
            ]
        )

        # up\_0: 2C -> C
        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * C,
            out_channels=C,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=C,
                    out_channels=C,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[8])
            ]
        )

        # 输出 head
        self.out_0 = OutBlock(in_channels=C, n_classes=n_classes, dim=dim)

        # deep supervision heads（与 MedNeXt 一致的位置）
        if self.do_ds:
            self.out_1 = OutBlock(in_channels=2 * C, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=4 * C, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=8 * C, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=16 * C, n_classes=n_classes, dim=dim)

    def forward(self, x: torch.Tensor):
        # encoder 返回 \[C, 2C, 4C, 8C, 16C]
        f0, f1, f2, f3, f4 = self.encoder(x)

        bottleneck = f4  # 16C

        if self.do_ds:
            x_ds_4 = self.out_4(bottleneck)

        # decoder 3: 16C -> 8C，跳接 f3
        x_up_3 = self.up_3(bottleneck)
        dec_x = f3 + x_up_3
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)

        # decoder 2: 8C -> 4C，跳接 f2
        x_up_2 = self.up_2(x)
        dec_x = f2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)

        # decoder 1: 4C -> 2C，跳接 f1
        x_up_1 = self.up_1(x)
        dec_x = f1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)

        # decoder 0: 2C -> C，跳接 f0
        x_up_0 = self.up_0(x)
        dec_x = f0 + x_up_0
        x = self.dec_block_0(dec_x)

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x