import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] +
              x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) * 0.5
        lh = (x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2]) * 0.5
        hl = (x[:, :, 0::2, 1::2] - x[:, :, 1::2, 1::2]) * 0.5
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2]) * 0.5
        return ll, lh, hl, hh

class PMD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, lh, hl, hh):
        x = torch.cat([lh, hl, hh], dim=1)
        return self.conv(x)

class DWT_PMD_RWKV(nn.Module):
    def __init__(self, dim, rwkv_block):
        super().__init__()
        self.dwt = HaarDWT()
        self.pmd = PMD(dim)
        self.rwkv = rwkv_block
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        edge_feat = self.pmd(lh, hl, hh)

        B, C, H, W = edge_feat.shape
        seq = edge_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        seq = self.rwkv(seq)
        edge_feat = seq.transpose(1, 2).reshape(B, C, H, W)

        return self.proj(edge_feat)

class BSBP_RWKV_Stage(nn.Module):
    def __init__(self, dim, rwkv_block):
        super().__init__()
        self.dwt_rwkv = DWT_PMD_RWKV(dim, rwkv_block)
        self.rk = RungeKuttaBlock(dim)

    def forward(self, x, prev_rk=None):
        edge_feat = self.dwt_rwkv(x)

        if prev_rk is not None:
            rk_input = torch.cat([edge_feat, prev_rk], dim=1)
            rk_input = F.conv2d(
                rk_input,
                weight=torch.eye(edge_feat.size(1), device=x.device)
                       .view(edge_feat.size(1), edge_feat.size(1), 1, 1)
            )
        else:
            rk_input = edge_feat

        rk_feat = self.rk(rk_input)
        return edge_feat, rk_feat

class BSBP_RWKV_Encoder(nn.Module):
    def __init__(self, in_ch=1, dims=[64, 128, 256, 512], rwkv_block=None):
        super().__init__()

        self.stem = nn.Conv2d(in_ch, dims[0], 3, padding=1)

        self.stages = nn.ModuleList()
        for d in dims:
            self.stages.append(BSBP_RWKV_Stage(d, rwkv_block))

        self.downs = nn.ModuleList([
            nn.Conv2d(dims[i], dims[i+1], 3, stride=2, padding=1)
            for i in range(len(dims)-1)
        ])

    def forward(self, x):
        x = self.stem(x)
        rk_prev = None
        features = []

        for i, stage in enumerate(self.stages):
            edge, rk = stage(x, rk_prev)
            features.append(rk)
            rk_prev = rk
            if i < len(self.downs):
                x = self.downs[i](rk)

        return features  # → 给 decoder
