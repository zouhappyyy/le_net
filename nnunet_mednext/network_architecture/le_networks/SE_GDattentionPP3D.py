import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // reduction, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class GDattentionPP3D(nn.Module):
    """
    3D GDattention++
    Difference-Aware, Voxel-wise, Learnable Reverse Gating
    """
    def __init__(self, dim, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_qk = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.to_v  = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta  = nn.Parameter(torch.tensor(init_beta))

        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2):
        """
        t1, t2: [B, C, D, H, W]
        """
        B, C, D, H, W = t1.shape

        q = self.to_qk(t1)
        k = self.to_qk(t2)

        diff = torch.abs(t1 - t2)
        v = self.to_v(diff)

        # flatten voxel dimension
        N = D * H * W
        _q = q.reshape(B, C, N).transpose(1, 2)  # [B, N, C]
        _k = k.reshape(B, C, N).transpose(1, 2)
        _v = v.reshape(B, C, N).transpose(1, 2)

        # voxel-wise similarity
        attn = torch.sum((_q * _k) * self.scale, dim=-1, keepdim=True)  # [B, N, 1]

        # learnable reverse gating
        attn = self.sigmoid(-(self.alpha * attn + self.beta))

        res = attn * _v
        res = res.transpose(1, 2).reshape(B, C, D, H, W)

        return self.to_out(res)


class SE_GDattentionPP3D(nn.Module):
    """
    3D SE–GDattention++
    """
    def __init__(self, dim, reduction=16):
        super().__init__()

        self.gd_attn = GDattentionPP3D(dim)
        self.se = SEBlock3D(dim, reduction)

    def forward(self, t1, t2):
        x = self.gd_attn(t1, t2)
        x = self.se(x)
        return x




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1 = torch.randn(1, 64, 32, 32, 32).to(device)
    f2 = torch.randn(1, 64, 32, 32, 32).to(device)

    model = SE_GDattentionPP3D(64).to(device)
    y = model(f1, f2)

    print(y.shape)  # [1, 64, 8, 32, 32]
