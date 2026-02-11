import torch
import torch.nn as nn

from .Double_RWKV_MedNeXt import Double_RWKV_MedNeXt_Encoder
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, OutBlock
from .LRDU import LRDU3D


class Double_UpSam_RWKV_MedNeXt(nn.Module):
    r"""Double_UpSam_RWKV_MedNeXt

    在 Double_RWKV_MedNeXt 的基础上，将 decoder 部分全部改为：
      - 上采样模块：使用 LRDU3D（所有 4 个解码阶段）；
      - 解码卷积：使用 MedNeXtBlock（所有 4 个解码阶段）；
    Encoder 结构保持不变，仍然使用 `Double_RWKV_MedNeXt_Encoder`，输出 [C, 2C, 4C, 8C, 16C]；
    网络构造参数与 Double_RWKV_MedNeXt 保持兼容。

    当前实现仅支持 3D (dim="3d")。
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
        rwkv_block_dec: nn.Module = None,  # 保留参数以保持与旧版本兼容，但在 LRDU 版本中不使用
    ):
        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ["2d", "3d"], "Double_UpSam_RWKV_MedNeXt is designed mainly for 3D."
        if dim != "3d":
            raise NotImplementedError(
                "Double_UpSam_RWKV_MedNeXt with LRDU-based decoder is currently implemented only for 3D (dim='3d')."
            )

        # 统一 kernel_size 设置
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if block_counts is None:
            # 与 MedNeXt 默认一致
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        # ---------- Encoder: 双分支（与 Double_RWKV_MedNeXt 相同） ----------
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

        C = n_channels
        # 将 exp_r 统一展开成列表，方便按层使用 MedNeXtBlock
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # ---------- Decoder: LRDU3D 上采样 + MedNeXtBlock 解码（4 个阶段一致） ----------
        lrdu_scale = 2

        # stage 3: 16C -> 8C，skip: 8C
        self.up_3 = LRDU3D(in_channels=16 * C, scale=lrdu_scale)
        self.proj_3 = nn.Conv3d(16 * C, 8 * C, kernel_size=1, bias=False)
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

        # stage 2: 8C -> 4C，skip: 4C
        self.up_2 = LRDU3D(in_channels=8 * C, scale=lrdu_scale)
        self.proj_2 = nn.Conv3d(8 * C, 4 * C, kernel_size=1, bias=False)
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

        # stage 1: 4C -> 2C，skip: 2C
        self.up_1 = LRDU3D(in_channels=4 * C, scale=lrdu_scale)
        self.proj_1 = nn.Conv3d(4 * C, 2 * C, kernel_size=1, bias=False)
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

        # stage 0: 2C -> C，skip: C
        self.up_0 = LRDU3D(in_channels=2 * C, scale=lrdu_scale)
        self.proj_0 = nn.Conv3d(2 * C, C, kernel_size=1, bias=False)
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

        # 输出 head，与 Double_RWKV_MedNeXt / MedNeXt 保持一致
        self.out_0 = OutBlock(in_channels=C, n_classes=n_classes, dim=dim)

        # deep supervision heads（与 MedNeXt / Double_RWKV_MedNeXt 一致的位置）
        if self.do_ds:
            self.out_1 = OutBlock(in_channels=2 * C, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=4 * C, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=8 * C, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=16 * C, n_classes=n_classes, dim=dim)
        else:
            self.out_1 = self.out_2 = self.out_3 = self.out_4 = None

    def forward(self, x: torch.Tensor):
        # encoder 返回 [C, 2C, 4C, 8C, 16C]
        f0, f1, f2, f3, f4 = self.encoder(x)

        bottleneck = f4  # 16C

        # 初始化 DS 变量，避免未赋值警告
        x_ds_1 = x_ds_2 = x_ds_3 = x_ds_4 = None

        if self.do_ds and self.out_4 is not None:
            x_ds_4 = self.out_4(bottleneck)

        # decoder 3: 16C -> 8C，LRDU3D 上采样 + skip f3 + MedNeXtBlock 解码
        x_up_3 = self.up_3(bottleneck)          # (B,16C, D/8,...)
        x_up_3 = self.proj_3(x_up_3)            # (B, 8C,  D/8,...)
        dec_x = f3 + x_up_3                     # (B, 8C,  D/8,...)
        x = self.dec_block_3(dec_x)
        if self.do_ds and self.out_3 is not None:
            x_ds_3 = self.out_3(x)

        # decoder 2: 8C -> 4C，LRDU3D 上采样 + skip f2 + MedNeXtBlock 解码
        x_up_2 = self.up_2(x)                   # (B, 8C,  D/4,...)
        x_up_2 = self.proj_2(x_up_2)            # (B, 4C,  D/4,...)
        dec_x = f2 + x_up_2                     # (B, 4C,  D/4,...)
        x = self.dec_block_2(dec_x)
        if self.do_ds and self.out_2 is not None:
            x_ds_2 = self.out_2(x)

        # decoder 1: 4C -> 2C，LRDU3D 上采样 + skip f1 + MedNeXtBlock 解码
        x_up_1 = self.up_1(x)                   # (B, 4C,  D/2,...)
        x_up_1 = self.proj_1(x_up_1)            # (B, 2C,  D/2,...)
        dec_x = f1 + x_up_1                     # (B, 2C,  D/2,...)
        x = self.dec_block_1(dec_x)
        if self.do_ds and self.out_1 is not None:
            x_ds_1 = self.out_1(x)

        # decoder 0: 2C -> C，LRDU3D 上采样 + skip f0 + MedNeXtBlock 解码
        x_up_0 = self.up_0(x)                   # (B, 2C,  D,...)
        x_up_0 = self.proj_0(x_up_0)            # (B, C,   D,...)
        dec_x = f0 + x_up_0                     # (B, C,   D,...)
        x = self.dec_block_0(dec_x)

        x = self.out_0(x)

        if self.do_ds:
            # 返回所有 DS 输出：主输出 + 4 个辅助输出（最深到较浅）
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x
