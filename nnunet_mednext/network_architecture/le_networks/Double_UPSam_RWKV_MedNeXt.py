import torch
import torch.nn as nn

from .Double_RWKV_MedNeXt import Double_RWKV_MedNeXt_Encoder
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtUpBlock, OutBlock
from .RSU import CrossStageRWKVSementicUpsampling3D


class Double_UpSam_RWKV_MedNeXt(nn.Module):
    r"""Double_UpSam_RWKV_MedNeXt

    在 Double_RWKV_MedNeXt 的基础上：
      - Encoder 结构保持不变，仍然使用 `Double_RWKV_MedNeXt_Encoder`，输出 [C, 2C, 4C, 8C, 16C];
      - Decoder 部分在每个上采样阶段引入 3D Cross-Stage RWKV Semantic Upsampling 模块，
        使用来自 encoder 的全局语义状态对上采样特征进行语义调制。

    适用于 3D MedNeXt 解码器（dim="3d"）。
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
        rwkv_block_dec: nn.Module = None,
    ):
        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ["2d", "3d"], "Double_UpSam_RWKV_MedNeXt is designed mainly for 3d, but 2d is allowed for consistency."

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

        # 原来这里强制要求 rwkv_block_dec 非空，现改为允许为 None：
        # 当为 None 时，CrossStageRWKVSementicUpsampling3D 会在内部回退到 SP_RWKV 的 PropagationRWKV。
        # 这使得解码器阶段可以只使用轻量 SP-RWKV，而不依赖 RWKVSeq。
        # if rwkv_block_dec is None:
        #     raise AssertionError("rwkv_block_dec must be provided for Double_UpSam_RWKV_MedNeXt")

        if dim == "2d":
            raise NotImplementedError("Double_UpSam_RWKV_MedNeXt is currently implemented for 3D (dim='3d') only.")

        C = n_channels
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # ---------- Decoder: 混合 MedNeXtUpBlock 与 3D RSU ----------
        # 约定：
        #   - decoder_3, decoder_2（低分辨率，通道 16C, 8C）使用 RSU + MedNeXtBlock；
        #   - decoder_1, decoder_0（高分辨率，通道 4C, 2C, C）使用原始 MedNeXtUpBlock 上采样。

        # 低分辨率 stage 3: 16C -> 8C，使用 RSU
        self.rsu_3 = CrossStageRWKVSementicUpsampling3D(
            in_channels=16 * C,
            scale_factor=2,
            rwkv_block=rwkv_block_dec,
            mode="trilinear",
            align_corners=False,
        )
        # 将 RSU 输出的 16C 通道映射到与 f3 对齐的 8C
        self.rsu_3_proj = nn.Conv3d(16 * C, 8 * C, kernel_size=1)
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

        # 低分辨率 stage 2: 8C -> 4C，使用 RSU
        self.rsu_2 = CrossStageRWKVSementicUpsampling3D(
            in_channels=8 * C,
            scale_factor=2,
            rwkv_block=rwkv_block_dec,
            mode="trilinear",
            align_corners=False,
        )
        # 将 RSU 输出的 8C 通道映射到与 f2 对齐的 4C
        self.rsu_2_proj = nn.Conv3d(8 * C, 4 * C, kernel_size=1)
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

        # 高分辨率 stage 1: 4C -> 2C，使用原始 MedNeXtUpBlock
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

        # 高分辨率 stage 0: 2C -> C，使用原始 MedNeXtUpBlock
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

        # deep supervision heads（与 MedNeXt / Double_RWKV_MedNeXt 一致的位置）
        if self.do_ds:
            self.out_1 = OutBlock(in_channels=2 * C, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=4 * C, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=8 * C, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=16 * C, n_classes=n_classes, dim=dim)
        else:
            self.out_1 = self.out_2 = self.out_3 = self.out_4 = None

    def _global_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """全局 3D 池化得到 (B, C) 语义向量，用作 CrossStage conditioning。"""
        B, C = feat.shape[0], feat.shape[1]
        # 对 (D,H,W) 做平均池化
        return feat.view(B, C, -1).mean(dim=-1)

    def forward(self, x: torch.Tensor):
        # encoder 返回 [C, 2C, 4C, 8C, 16C]
        f0, f1, f2, f3, f4 = self.encoder(x)

        bottleneck = f4  # 16C

        # 初始化 DS 变量，避免未赋值警告
        x_ds_1 = x_ds_2 = x_ds_3 = None

        # 从 bottleneck 中提取全局语义向量，供所有解码 stage 共享
        global_state = self._global_pool(bottleneck)  # (B, 16C) 但 CrossStageRSU3D 期望 (B, in_channels)
        # 对于不同 stage，我们只需保证传入的 global_state 在通道数上与对应 RSU 的 in_channels 一致。
        # 简化方案：使用 1x1x1 Conv 将 bottleneck 投影到各层需要的通道数，这里使用池化 + 线性层可以灵活实现。

        # decoder 3: 16C -> 8C，RSU + skip f3
        gs_3 = self._global_pool(bottleneck)  # (B,16C)
        x_up_3 = self.rsu_3(bottleneck, gs_3)  # (B,16C,...)
        x_up_3 = self.rsu_3_proj(x_up_3)       # (B,8C,...), 与 f3 对齐
        dec_x = f3 + x_up_3
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)

        # decoder 2: 8C -> 4C，RSU + skip f2
        gs_2 = self._global_pool(x)  # x 通道为 8C
        x_up_2 = self.rsu_2(x, gs_2)  # (B,8C,...)
        x_up_2 = self.rsu_2_proj(x_up_2)       # (B,4C,...), 与 f2 对齐
        dec_x = f2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)

        # decoder 1: 4C -> 2C，MedNeXtUpBlock + skip f1
        x_up_1 = self.up_1(x)
        dec_x = f1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)

        # decoder 0: 2C -> C，MedNeXtUpBlock + skip f0
        x_up_0 = self.up_0(x)
        dec_x = f0 + x_up_0
        x = self.dec_block_0(dec_x)

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, None]
        else:
            return x
