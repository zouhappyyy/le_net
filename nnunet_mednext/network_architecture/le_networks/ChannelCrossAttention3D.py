import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelCrossAttention3D(nn.Module):
    """
    Channel-wise Cross-Attention for 3D features.
    用于双分支编码器：RWKV 全局分支 -> 局部分支通道级引导
    """

    def __init__(self, in_channels, reduction=4):
        """
        Args:
            in_channels (int): 输入通道数 C
            reduction (int): 通道压缩比例，用于生成注意力权重
        """
        super().__init__()
        hidden_channels = max(in_channels // reduction, 1)

        # 用于生成通道注意力的 MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, F_global, F_local):
        """
        Args:
            F_global: Tensor, 全局分支特征, shape = (B, C, D, H, W)
            F_local: Tensor, 局部分支特征, shape = (B, C, D, H, W)
        Returns:
            Tensor: 融合后的局部分支特征, shape = (B, C, D, H, W)
        """
        # 1. 空间全局平均池化
        f_global = F.adaptive_avg_pool3d(F_global, output_size=1)  # (B, C, 1, 1, 1)

        # 2. MLP 生成通道注意力
        alpha = self.mlp(f_global)  # (B, C, 1, 1, 1)

        # 3. 融合局部分支特征
        F_fused = alpha * F_local + F_local  # 通道级 Cross-Attention

        return F_fused
