import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b, c, d, h, w = x.size()
        r = self.scale
        out_c = c // (r ** 3)

        x = x.view(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, out_c, d * r, h * r, w * r)

        return x


class LRDU3D(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()

        self.scale = scale
        mid_channels = in_channels

        # 分离卷积 1x1x7
        self.conv1 = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 1, 7),
            padding=(0, 0, 3),
            groups=groups,
            bias=False
        )

        # 1x7x1
        self.conv2 = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=(1, 7, 1),
            padding=(0, 3, 0),
            groups=groups,
            bias=False
        )

        # 7x1x1
        self.conv3 = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=(7, 1, 1),
            padding=(3, 0, 0),
            groups=groups,
            bias=False
        )

        self.norm = nn.BatchNorm3d(mid_channels)
        self.act = nn.GELU()

        # 通道扩展
        self.channel_expand = nn.Conv3d(
            mid_channels,
            in_channels * scale ** 3,
            kernel_size=1,
            bias=False
        )

        self.pixel_shuffle = PixelShuffle3D(scale)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.norm(out)
        out = self.act(out)

        out = self.channel_expand(out)
        out = self.pixel_shuffle(out)

        # 残差上采样
        identity = F.interpolate(
            identity,
            scale_factor=self.scale,
            mode='trilinear',
            align_corners=False
        )

        out = out + identity

        return out

class LRDUDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, scale=2):
        super().__init__()

        self.up = LRDU3D(in_ch, scale=scale)

        self.fusion = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        return x
