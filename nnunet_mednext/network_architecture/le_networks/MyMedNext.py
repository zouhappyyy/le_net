
import torch.nn as nn
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FDConv


class MyMedNext(MedNeXt):
    def __init__(self, in_channels=1, n_classes=2, **kwargs):
        super().__init__(in_channels=in_channels, n_classes=n_classes, **kwargs)
        # 只替换每个容器中的第二个 MedNeXtBlock.conv1
        self._replace_second_conv1_with_fdconv()

    def _build_fdconv_from_conv(self, conv: nn.Conv3d) -> FDConv:
        # 忽略原来的 groups，强制使用 groups=1
        fdconv = FDConv(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,                 # 强制非 depthwise
            bias=(conv.bias is not None),
            kernel_num=4,
            use_fdconv_if_c_gt=0,     # 总是启用 FD 分支
        )
        # 若 FDConv 仍有 weight，则尝试拷贝参数
        if hasattr(fdconv, "weight") and fdconv.weight.shape == conv.weight.shape:
            fdconv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None and getattr(fdconv, "bias", None) is not None:
            if fdconv.bias.shape == conv.bias.shape:
                fdconv.bias.data.copy_(conv.bias.data)
        return fdconv

    def _replace_second_conv1_within_container(self, container: nn.Module):
        # 在一个容器内找到所有 MedNeXtBlock，只替换第二个
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
            second_block.conv1 = self._build_fdconv_from_conv(conv1)

    def _replace_second_conv1_with_fdconv(self):
        # 遍历整个网络，查找包含多个 MedNeXtBlock 的容器
        for module in self.modules():
            # 常见情况：nn.Sequential / nn.ModuleList 作为 block 容器
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.Module)):
                self._replace_second_conv1_within_container(module)