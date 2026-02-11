import torch
import torch.nn as nn

from .Double_RWKV_MedNeXt import Double_RWKV_MedNeXt_Encoder
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, OutBlock
from .LRDU import LRDU3D
from .ChannelCrossAttention3D import ChannelCrossAttention3D


class CCA_LRDU_Decoder3D(nn.Module):
    """3D decoder with LRDU3D upsampling and channel-wise cross-attention based skip fusion.

    用于 5 层 U-Net 结构的解码端，对应编码器输出通道 [C, 2C, 4C, 8C, 16C]：
      - 4 个阶段：16C->8C, 8C->4C, 4C->2C, 2C->C
      - 每个阶段：
          1) LRDU3D 上采样（通道不变，尺度×2）；
          2) 1x1x1 Conv 将上采样后特征通道调整到与 skip 一致；
          3) 使用 ChannelCrossAttention3D 以 skip 作为 F_global、上采样特征作为 F_local 做通道级 Cross-Attention；
          4) 再经过若干个 MedNeXtBlock 做局部解码。
    """

    def __init__(
        self,
        base_ch: int,
        exp_r_list,
        block_counts,
        dec_kernel_size: int,
        do_res: bool,
        norm_type: str,
        dim: str,
        grn: bool,
        cca_reduction: int = 4,
    ):
        super().__init__()

        assert dim == "3d", "CCA_LRDU_Decoder3D currently only supports 3D."

        C = base_ch
        lrdu_scale = 2

        # ---- Stage 3: 16C -> 8C, skip: 8C ----
        self.up_3 = LRDU3D(in_channels=16 * C, scale=lrdu_scale)
        self.proj_3 = nn.Conv3d(16 * C, 8 * C, kernel_size=1, bias=False)
        self.cca_3 = ChannelCrossAttention3D(in_channels=8 * C, reduction=cca_reduction)
        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=8 * C,
                    out_channels=8 * C,
                    exp_r=exp_r_list[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[5])
            ]
        )

        # ---- Stage 2: 8C -> 4C, skip: 4C ----
        self.up_2 = LRDU3D(in_channels=8 * C, scale=lrdu_scale)
        self.proj_2 = nn.Conv3d(8 * C, 4 * C, kernel_size=1, bias=False)
        self.cca_2 = ChannelCrossAttention3D(in_channels=4 * C, reduction=cca_reduction)
        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=4 * C,
                    out_channels=4 * C,
                    exp_r=exp_r_list[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[6])
            ]
        )

        # ---- Stage 1: 4C -> 2C, skip: 2C ----
        self.up_1 = LRDU3D(in_channels=4 * C, scale=lrdu_scale)
        self.proj_1 = nn.Conv3d(4 * C, 2 * C, kernel_size=1, bias=False)
        self.cca_1 = ChannelCrossAttention3D(in_channels=2 * C, reduction=cca_reduction)
        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=2 * C,
                    out_channels=2 * C,
                    exp_r=exp_r_list[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[7])
            ]
        )

        # ---- Stage 0: 2C -> C, skip: C ----
        self.up_0 = LRDU3D(in_channels=2 * C, scale=lrdu_scale)
        self.proj_0 = nn.Conv3d(2 * C, C, kernel_size=1, bias=False)
        self.cca_0 = ChannelCrossAttention3D(in_channels=C, reduction=cca_reduction)
        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=C,
                    out_channels=C,
                    exp_r=exp_r_list[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[8])
            ]
        )

    def forward(self, f0, f1, f2, f3, f4):
        """前向解码。

        Args:
            f0..f4: 编码器输出的 5 级特征，通道分别为 [C, 2C, 4C, 8C, 16C]
        Returns:
            x: 最后一级解码特征 (B, C, D, H, W)
            ds_feats: list, 供外部 deep supervision head 使用的多尺度特征 [x1, x2, x3, x4]
        """
        ds_feats = [None, None, None, None]

        # bottleneck
        x = f4  # 16C

        # Stage 3: 16C -> 8C
        x_up_3 = self.up_3(x)           # (B,16C)
        x_up_3 = self.proj_3(x_up_3)    # (B,8C)
        # CCA: skip 作为 global，引导 upsample 结果
        x = self.cca_3(F_global=f3, F_local=x_up_3)
        x = self.dec_block_3(x)
        ds_feats[3] = x  # 对应 8C 层

        # Stage 2: 8C -> 4C
        x_up_2 = self.up_2(x)
        x_up_2 = self.proj_2(x_up_2)
        x = self.cca_2(F_global=f2, F_local=x_up_2)
        x = self.dec_block_2(x)
        ds_feats[2] = x  # 4C 层

        # Stage 1: 4C -> 2C
        x_up_1 = self.up_1(x)
        x_up_1 = self.proj_1(x_up_1)
        x = self.cca_1(F_global=f1, F_local=x_up_1)
        x = self.dec_block_1(x)
        ds_feats[1] = x  # 2C 层

        # Stage 0: 2C -> C
        x_up_0 = self.up_0(x)
        x_up_0 = self.proj_0(x_up_0)
        x = self.cca_0(F_global=f0, F_local=x_up_0)
        x = self.dec_block_0(x)
        ds_feats[0] = x  # C 层

        return x, ds_feats


class Double_CCA_UPSam_RWKV_MedNeXt(nn.Module):
    """Double_CCA_UPSam_RWKV_MedNeXt

    要求：不修改已有模块，在 Double_RWKV_MedNeXt 双分支编码器 + LRDU 解码的基础上：
      - 新增一个带通道 Cross-Attention 的解码器 CCA_LRDU_Decoder3D；
      - 使用原来的 Double_RWKV_MedNeXt_Encoder 作为 encoder；
      - 形成一个新的完整网络类，接口与 Double_UpSam_RWKV_MedNeXt 保持基本一致。
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
        rwkv_block_dec: nn.Module = None,  # 保留参数以兼容已有调用，不在此类中使用
        cca_reduction: int = 4,
    ):
        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ["2d", "3d"], "Double_CCA_UPSam_RWKV_MedNeXt is designed mainly for 3D."
        if dim != "3d":
            raise NotImplementedError("Double_CCA_UPSam_RWKV_MedNeXt is implemented only for 3D (dim='3d').")

        # 统一 kernel_size 设置
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # ---------- Encoder: 复用原 Double_RWKV_MedNeXt_Encoder ----------
        self.encoder = Double_RWKV_MedNeXt_Encoder(
            in_channels=in_channels,
            n_channels=n_channels,
            exp_r=exp_r,
            kernel_size=enc_kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=False,
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

        C = n_channels
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # ---------- 新 Decoder: CCA_LRDU_Decoder3D ----------
        self.decoder = CCA_LRDU_Decoder3D(
            base_ch=C,
            exp_r_list=exp_r,
            block_counts=block_counts,
            dec_kernel_size=dec_kernel_size,
            do_res=do_res,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
            cca_reduction=cca_reduction,
        )

        # 输出 head
        self.out_0 = OutBlock(in_channels=C, n_classes=n_classes, dim=dim)

        if self.do_ds:
            self.out_1 = OutBlock(in_channels=2 * C, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=4 * C, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=8 * C, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=16 * C, n_classes=n_classes, dim=dim)
        else:
            self.out_1 = self.out_2 = self.out_3 = self.out_4 = None

    def forward(self, x: torch.Tensor):
        # 编码器输出 [C, 2C, 4C, 8C, 16C]
        f0, f1, f2, f3, f4 = self.encoder(x)

        # 解码器：返回最后一层 C 特征以及多尺度特征用于 DS
        x_dec, ds_feats = self.decoder(f0, f1, f2, f3, f4)

        x_main = self.out_0(x_dec)

        if not self.do_ds:
            return x_main

        # deep supervision 输出
        x_ds_0 = x_main
        x_ds_1 = self.out_1(ds_feats[1]) if self.out_1 is not None else None
        x_ds_2 = self.out_2(ds_feats[2]) if self.out_2 is not None else None
        x_ds_3 = self.out_3(ds_feats[3]) if self.out_3 is not None else None
        x_ds_4 = self.out_4(f4)          if self.out_4 is not None else None

        return [x_ds_0, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
