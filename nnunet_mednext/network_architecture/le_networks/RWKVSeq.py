import torch
import torch.nn as nn

from .GLSP3D_RWKV import VRWKV_SpatialMix


class RWKVSeq(nn.Module):
    """Lightweight RWKV-based sequence module for RSU.

    输入/输出: (B, N, C)
    内部使用 VRWKV_SpatialMix 在给定的 3D patch 尺寸 (D, H, W) 上做 RWKV 空间混合。

    用法:
      - 在构造时指定 n_embd=C 和 patch_resolution=(D, H, W);
      - forward(tokens) 会自动将 (B,N,C) reshape 为 (B,C,D,H,W)，调用 VRWKV_SpatialMix，再还原回序列。

    注意: patch_resolution 应该与解码对应 stage 的特征尺寸匹配。
    """

    def __init__(self, n_embd: int, patch_resolution: tuple[int, int, int], channel_gamma: float = 1/6, shift_pixel: int = 1):
        super().__init__()
        self.n_embd = n_embd
        self.D, self.H, self.W = patch_resolution

        self.attn = VRWKV_SpatialMix(
            n_embd=n_embd,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
        )
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, C) -> (B, N, C) with RWKV spatial mix over (D,H,W)."""
        assert tokens.dim() == 3, f"Expected (B,N,C), got {tokens.shape}"
        B, N, C = tokens.shape
        assert C == self.n_embd, f"Channel mismatch: got C={C}, expected {self.n_embd}"
        assert N == self.D * self.H * self.W, (
            f"Token length N={N} does not match patch_resolution D*H*W={self.D*self.H*self.W}"
        )

        x = tokens.transpose(1, 2).view(B, C, self.D, self.H, self.W)  # (B,C,D,H,W)
        # 展平为序列并应用 VRWKV_SpatialMix
        seq = x.flatten(2).transpose(1, 2)  # (B,N,C)
        seq = self.attn(seq, (self.D, self.H, self.W))  # (B,N,C)
        seq = self.ln(seq)
        return seq
