import torch
import torch.nn as nn

from .Double_CCA_UPSam_RWKV_MedNeXt import Double_CCA_UPSam_RWKV_MedNeXt
from .Double_RWKV_MedNeXt import Double_RWKV_MedNeXt_Encoder
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FDConv


class Double_CCA_UPSam_fd_RWKV_MedNeXt(Double_CCA_UPSam_RWKV_MedNeXt):
    """FDConv 变体的 Double_CCA_UPSam_RWKV_MedNeXt。

    仅在双分支编码器（Double_RWKV_MedNeXt_Encoder）的最上面两层中，
    将每个容器中的第二个 MedNeXtBlock.conv1 替换为 FDConv3d（强制 groups=1，非 depthwise）。
    其它结构与原网络完全一致。
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
        cca_reduction: int = 4,
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
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
            rwkv_block_dec=rwkv_block_dec,
            cca_reduction=cca_reduction,
        )

        # 仅在编码器 top-2 stages 内做 FDConv 替换
        if isinstance(self.encoder, Double_RWKV_MedNeXt_Encoder):
            self._replace_encoder_top2layers_second_conv1_with_fdconv()

    # ---- FDConv 构建 & 替换逻辑 ----

    def _build_fdconv_from_conv(self, conv: nn.Conv3d) -> FDConv:
        """根据已有 Conv3d 构建对应的 FDConv3d，强制 groups=1，并尽量拷贝权重和 bias。"""
        fdconv = FDConv(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,  # 强制非 depthwise
            bias=(conv.bias is not None),
            kernel_num=4,
            use_fdconv_if_c_gt=0,  # 总是启用频域分支
        )

        # 若 FDConv 内部仍保留原始 weight，则尝试从 Conv3d 拷贝参数
        if hasattr(fdconv, "weight") and fdconv.weight.shape == conv.weight.shape:
            fdconv.weight.data.copy_(conv.weight.data)

        if conv.bias is not None and getattr(fdconv, "bias", None) is not None:
            if fdconv.bias.shape == conv.bias.shape:
                fdconv.bias.data.copy_(conv.bias.data)

        return fdconv

    def _replace_second_conv1_within_container(self, container: nn.Module):
        """在一个容器内找到所有 MedNeXtBlock，只替换第二个的 conv1 为 FDConv。"""
        blocks = []
        for name, child in container.named_children():
            if isinstance(child, MedNeXtBlock):
                blocks.append((name, child))

        if len(blocks) < 2:
            return

        # 取第二个 MedNeXtBlock
        _, second_block = blocks[1]
        conv1 = getattr(second_block, "conv1", None)

        # 仅当 conv1 是 Conv3d 且 stride 为 (1,1,1) 时才替换
        if isinstance(conv1, nn.Conv3d) and tuple(conv1.stride) == (1, 1, 1):
            setattr(second_block, "conv1", self._build_fdconv_from_conv(conv1))

    def _replace_encoder_top2layers_second_conv1_with_fdconv(self):
        """仅在双分支编码器的上面两层中执行 FDConv 替换。

        Double_RWKV_MedNeXt_Encoder 内部使用 MedNeXt_EncoderOnly 作为 mednext_enc，
        其 enc_block_0 和 enc_block_1 分别对应最浅的两层 encoder 容器。
        """
        mednext_enc = getattr(self.encoder, "mednext_enc", None)
        if mednext_enc is None:
            return

        # enc_block_0: C 通道层
        if hasattr(mednext_enc, "enc_block_0"):
            self._replace_second_conv1_within_container(mednext_enc.enc_block_0)

        # enc_block_1: 2C 通道层
        if hasattr(mednext_enc, "enc_block_1"):
            self._replace_second_conv1_within_container(mednext_enc.enc_block_1)
