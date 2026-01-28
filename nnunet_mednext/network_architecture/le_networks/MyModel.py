# python
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
import torch.nn as nn
import torch
from nnunet_mednext.utilities.nd_softmax import softmax_helper

class MySimpleNet(SegmentationNetwork):
    """
    最小可用示例网络：
    - 构造函数接收 n_classes（或 n_classes via kwargs）
    - 设置 SegmentationNetwork 需要的属性
    - 简单的前向卷积块（可根据需求替换为复杂结构）
    """
    def __init__(self, in_channels=1, n_classes=2, base_num_features=16, **kwargs):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**4
        self.num_classes = n_classes

        # 简单网络： conv -> relu -> conv -> 输出通道为 num_classes
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_num_features, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_num_features),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(base_num_features, base_num_features*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_num_features*2),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.final_conv = nn.Conv3d(base_num_features*2, self.num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_conv(x)
        return x
