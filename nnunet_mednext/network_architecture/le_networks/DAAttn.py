# 差异感知注意力模块（Difference-Aware Attention, DAAttn）
# 核心设计：针对双输入特征（如时序帧、双模态），通过“差异提取→注意力计算→加权增强”的流程，
# 基于双特征的差异信息生成注意力权重，针对性强化差异区域特征，提升双特征融合的针对性与辨识度

import torch
from torch import nn


class GDattention(nn.Module):
    """
    差异感知注意力模块（Difference-Aware Attention, DAAttn）
    功能：基于双特征差异生成注意力，加权增强差异区域，实现精准特征交互
    核心设计：
        - 差异驱动：以双特征绝对差异作为Value，聚焦融合关键点
        - 轻量投影：1×1卷积生成Q/K/V，无复杂维度变换，计算成本低
        - 简化注意力：通过点积求和生成单通道注意力权重，避免高维矩阵运算
        - 负向激活：Sigmoid(-attn)强化小差异区域（或抑制大差异区域，适配特定场景）
    Args:
        dim: 输入/输出通道数
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qk = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, t1, t2):
        B, C, H, W = t1.shape
        q = self.to_qk(t1) # B, C, H, W
        k = self.to_qk(t2)
        diff = torch.abs(t1 - t2)
        v = self.to_v(diff)

        _q = q.reshape(B, -1, H * W).transpose(-2, -1) # B, H * W, C
        _k = k.reshape(B, -1, H * W).transpose(-2, -1)
        _v = v.reshape(B, -1, H * W).transpose(-2, -1)

        attn = torch.sum((_q * _k) * self.scale, dim=-1, keepdim=True) # B, H * W, 1
        attn = self.sigmoid(-attn)
        res = (attn * _v) # B, H * W, C
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    f1 = torch.randn(1, 64, 32, 32).to(device)
    f2 = torch.randn(1, 64, 32, 32).to(device)
    model = GDattention(64).to(device)

    y = model(f1, f2)

    print("输入f1特征维度：", f1.shape)
    print("输入f2特征维度：", f2.shape)
    print("输出特征维度：", y.shape)