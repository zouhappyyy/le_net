import torch
import torch.nn as nn


class GDattentionPP(nn.Module):
    """
    GDattention++（可学习反向门控差异注意力）
    核心升级：
        - learnable alpha：控制反向门控强度
        - learnable beta ：控制门控偏置（阈值）
    门控形式：
        attn = sigmoid(-(alpha * attn + beta))
    """

    def __init__(self, dim, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_qk = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        # 🔥 可学习反向门控参数
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2):
        B, C, H, W = t1.shape

        q = self.to_qk(t1)
        k = self.to_qk(t2)

        diff = torch.abs(t1 - t2)
        v = self.to_v(diff)

        _q = q.reshape(B, C, H * W).transpose(1, 2)  # B, HW, C
        _k = k.reshape(B, C, H * W).transpose(1, 2)
        _v = v.reshape(B, C, H * W).transpose(1, 2)

        # pixel-wise similarity
        attn = torch.sum((_q * _k) * self.scale, dim=-1, keepdim=True)  # B, HW, 1

        # 🔥 可学习反向门控
        attn = self.sigmoid(-(self.alpha * attn + self.beta))

        res = attn * _v
        res = res.transpose(1, 2).reshape(B, C, H, W)

        return self.to_out(res)


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class SE_GDattentionPP(nn.Module):
    """
    SE + GDattention++
    """

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.gd_attn = GDattentionPP(dim)
        self.se = SEBlock(dim, reduction)

    def forward(self, t1, t2):
        x = self.gd_attn(t1, t2)
        x = self.se(x)
        return x
