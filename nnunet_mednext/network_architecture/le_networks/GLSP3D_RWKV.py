import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from torch.utils.cpp_extension import load
T_MAX = 4096
inplace = True
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        # Use xk, xv, xr to produce k, v, r
        if self.shift_pixel > 0:
            xx = q_shift_3d(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device
        sr, k, v = self.jit_func(x, patch_resolution)
        x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class SE3D(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25):
        super().__init__()
        rd_chs = int(in_chs * rd_ratio)
        self.fc1 = nn.Conv3d(in_chs, rd_chs, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(rd_chs, in_chs, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=(2, 3, 4), keepdim=True)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        return x * self.gate(s)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        # print("WKV assert failed: T =", T, "T_MAX =", T_MAX)
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)



def q_shift_3d(x, shift=1, gamma=1/6, size=None):
    """
    x: (B, N, C), N = D*H*W
    size: (D, H, W)
    """
    B, N, C = x.shape
    D, H, W = size
    x = x.transpose(1, 2).reshape(B, C, D, H, W)

    out = torch.zeros_like(x)
    g = int(C * gamma)

    out[:, 0:g, :, :, shift:] = x[:, 0:g, :, :, :-shift]         # W →
    out[:, g:2*g, :, :, :-shift] = x[:, g:2*g, :, :, shift:]     # ←
    out[:, 2*g:3*g, :, shift:, :] = x[:, 2*g:3*g, :, :-shift, :] # H ↓
    out[:, 3*g:4*g, :, :-shift, :] = x[:, 3*g:4*g, :, shift:, :] # ↑
    out[:, 4*g:5*g, shift:, :, :] = x[:, 4*g:5*g, :-shift, :, :] # D ↓
    out[:, 5*g:6*g, :-shift, :, :] = x[:, 5*g:6*g, shift:, :, :] # ↑
    out[:, 6*g:, ...] = x[:, 6*g:, ...]

    return out.flatten(2).transpose(1, 2)


class GLSP3D(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        has_skip=True,
        exp_ratio=1.0,
        dw_ks=3,
        stride=1,
        se_ratio=0.0,
        drop_path=0.,
        drop=0.,
        channel_gamma=1/6,
        shift_pixel=1
    ):
        super().__init__()

        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        dim_mid = int(dim_in * exp_ratio)

        # ---- Pointwise expand ----
        self.conv1 = nn.Conv3d(dim_in, dim_mid, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(dim_mid)

        # ---- RWKV Spatial Mix ----
        self.attn = VRWKV_SpatialMix(
            n_embd=dim_mid,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel
        )
        self.ln = nn.LayerNorm(dim_mid)

        # ---- Depthwise Conv3D ----
        self.dwconv = nn.Conv3d(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            padding=dw_ks // 2,
            groups=dim_mid,
            bias=False
        )
        self.norm2 = nn.BatchNorm3d(dim_mid)
        self.act = nn.SiLU(inplace=True)

        # ---- SE ----
        self.se = SE3D(dim_mid, se_ratio) if se_ratio > 0 else nn.Identity()

        # ---- Project ----
        self.proj = nn.Conv3d(dim_mid, dim_out, 1, bias=False)
        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.norm1(x)

        # RWKV over (D*H*W)
        B, C, D, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)          # (B, N, C)
        seq = seq + self.drop_path(self.ln(self.attn(seq, (D, H, W))))
        x = seq.transpose(1, 2).reshape(B, C, D, H, W)

        x = self.dwconv(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.se(x)

        x = self.proj(x)
        x = self.drop(x)

        if self.has_skip:
            x = shortcut + self.drop_path(x)

        return x


class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)



class RWKV_UNet_3D_Encoder(nn.Module):
    """Three-step 3D RWKV encoder branch for Double_RWKV_MedNeXt.

    Usage in Double_RWKV_MedNeXt:
      - 输入为 MedNeXt 第三层 encoder 特征 (4C 通道): feats_med[2]
      - 本分支经过两个 RWKV stage, 每个 stage 包含 GLSP3D + Downsample3D
      - 最终输出三个尺度的 RWKV 特征:
          * rwkv_feat_2: 4C, 与 MedNeXt feats_med[2] 同分辨率
          * rwkv_feat_3: 8C, 与 MedNeXt feats_med[3] 同分辨率
          * rwkv_feat_4: 16C, 与 MedNeXt bottleneck 输入同通道数与分辨率
    """

    def __init__(self, base_ch: int = 32, dim: str = "3d"):
        super().__init__()
        self.base_ch = base_ch

        if dim == "2d":
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d

        # MedNeXt 第三层特征通道为 4 * C，这里假设 base_ch == C
        in_ch_med2 = 4 * base_ch

        # 1x1x1 conv to align MedNeXt 4C feature to RWKV 4C
        self.in_proj = Conv(
            in_channels=in_ch_med2,
            out_channels=4 * base_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Stage 2: 4C -> 4C (GLSP3D) -> 8C (Downsample3D)
        self.stage2_block = GLSP3D(
            dim_in=4 * base_ch,
            dim_out=4 * base_ch,
            has_skip=True,
            exp_ratio=1.0,
            dw_ks=3,
            stride=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.stage2_down = Downsample3D(4 * base_ch, 8 * base_ch)

        # Stage 3: 8C -> 8C (GLSP3D) -> 16C (Downsample3D)
        self.stage3_block = GLSP3D(
            dim_in=8 * base_ch,
            dim_out=8 * base_ch,
            has_skip=True,
            exp_ratio=1.0,
            dw_ks=3,
            stride=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.stage3_down = Downsample3D(8 * base_ch, 16 * base_ch)

    def forward_from_feat(self, feat_med_2: torch.Tensor):
        """Forward starting from MedNeXt's third encoder feature (4C).

        Args:
            feat_med_2: tensor of shape (B, 4*C, D, H, W)

        Returns:
            rwkv_feat_2: (B, 4*C, D,   H,   W  )  # 对应 MedNeXt feats_med[2]
            rwkv_feat_3: (B, 8*C, D/2, H/2, W/2)  # 对应 MedNeXt feats_med[3]
            rwkv_feat_4: (B,16*C, D/4, H/4, W/4)  # 对应 MedNeXt bottleneck 输入
        """
        # 对齐输入通道到 4C
        x = self.in_proj(feat_med_2)          # (B, 4*C, ...)

        # Stage2: RWKV + Downsample
        x2 = self.stage2_block(x)            # (B, 4*C, D, H, W)
        rwkv_feat_2 = x2
        x3_in = self.stage2_down(x2)         # (B, 8*C, D/2, H/2, W/2)

        # Stage3: RWKV + Downsample
        x3 = self.stage3_block(x3_in)        # (B, 8*C, D/2, H/2, W/2)
        rwkv_feat_3 = x3
        x4 = self.stage3_down(x3)            # (B, 16*C, D/4, H/4, W/4)
        rwkv_feat_4 = x4

        return rwkv_feat_2, rwkv_feat_3, rwkv_feat_4
