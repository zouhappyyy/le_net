import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionedRWKV(nn.Module):
    """
    RWKV conditioned on encoder global semantic state.
    """
    def __init__(self, dim, rwkv_block):
        super().__init__()
        self.rwkv = rwkv_block

        # FiLM-style conditioning
        self.gamma_proj = nn.Linear(dim, dim)
        self.beta_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, global_state):
        """\
        tokens: (B, N, C)
        global_state: (B, C)
        """
        B, N, C = tokens.shape

        gamma = self.gamma_proj(global_state).unsqueeze(1)  # (B, 1, C)
        beta = self.beta_proj(global_state).unsqueeze(1)    # (B, 1, C)

        conditioned = tokens * (1.0 + gamma) + beta
        conditioned = self.norm(conditioned)

        return self.rwkv(conditioned)


class CrossStageRWKVSementicUpsampling(nn.Module):
    """
    RSU with Cross-Stage RWKV Conditioning (2D version).
    """

    def __init__(
        self,
        in_channels,
        scale_factor,
        rwkv_block
    ):
        super().__init__()

        self.scale_factor = scale_factor

        self.cond_rwkv = ConditionedRWKV(
            dim=in_channels,
            rwkv_block=rwkv_block
        )

        self.gamma_proj = nn.Linear(in_channels, in_channels)
        self.beta_proj = nn.Linear(in_channels, in_channels)

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, global_state):
        """\
        x: (B, C, H, W)
        global_state: (B, C) from encoder
        """
        B, C, H, W = x.shape

        # 1. semantic-neutral upsampling
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False
        )

        Hp, Wp = x_up.shape[2], x_up.shape[3]

        # 2. flatten to tokens
        tokens = x_up.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.norm(tokens)

        # 3. cross-stage conditioned RWKV
        context = self.cond_rwkv(tokens, global_state)

        # 4. semantic modulation
        gamma = self.gamma_proj(context)
        beta = self.beta_proj(context)

        gamma = gamma.transpose(1, 2).view(B, C, Hp, Wp)
        beta = beta.transpose(1, 2).view(B, C, Hp, Wp)

        out = x_up * (1.0 + torch.tanh(gamma)) + beta
        return out


class ConditionedRWKV3D(nn.Module):
    """\
    3D variant of RWKV conditioning on encoder global semantic state.

    与 2D ConditionedRWKV 逻辑相同，只是语义上明确用于 3D 解码场景，
    假设传入/传出的 tokens 仍为 (B, N, C)，global_state 为 (B, C)。
    """

    def __init__(self, dim: int, rwkv_block: nn.Module):
        super().__init__()
        self.rwkv = rwkv_block

        # FiLM-style conditioning
        self.gamma_proj = nn.Linear(dim, dim)
        self.beta_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """\
        tokens: (B, N, C) flattened from (D, H, W)
        global_state: (B, C) encoder global semantic state (e.g., pooled 3D feature)
        """
        B, N, C = tokens.shape
        assert global_state.shape == (B, C), f"global_state shape {global_state.shape} incompatible with tokens {tokens.shape}"

        gamma = self.gamma_proj(global_state).unsqueeze(1)  # (B, 1, C)
        beta = self.beta_proj(global_state).unsqueeze(1)    # (B, 1, C)

        conditioned = tokens * (1.0 + gamma) + beta
        conditioned = self.norm(conditioned)

        return self.rwkv(conditioned)


class CrossStageRWKVSementicUpsampling3D(nn.Module):
    """\
    3D RSU with Cross-Stage RWKV Conditioning, for MedNeXt3D decoder upsampling.

    设计目标:
      - 输入/输出特征为 (B, C, D, H, W);
      - 使用 3D 插值对体素空间 (D, H, W) 做语义中性上采样;
      - 将上采样特征展平为序列 (B, N, C), 调用跨 stage 条件化 RWKV 进行全局建模;
      - 从 RWKV 上下文中预测体素级 gamma/beta, 对上采样结果做语义调制。

    参数:
      in_channels: 解码层特征通道数 C;
      scale_factor: 体素空间的上采样倍率 (int 或 tuple of 3);
      rwkv_block: 接收 (B, N, C) -> (B, N, C) 的 RWKV 模块实例。
    """

    def __init__(
        self,
        in_channels: int,
        scale_factor,
        rwkv_block: nn.Module,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        # 条件化 RWKV (3D 语义, 但接口仍是 (B,N,C))
        self.cond_rwkv = ConditionedRWKV3D(
            dim=in_channels,
            rwkv_block=rwkv_block,
        )

        # 从 RWKV 上下文预测体素级调制参数
        self.gamma_proj = nn.Linear(in_channels, in_channels)
        self.beta_proj = nn.Linear(in_channels, in_channels)

        # 对序列通道维归一化
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """\
        x: (B, C, D, H, W) -- decoder feature map before upsampling
        global_state: (B, C) -- encoder global semantic state (3D-pooled)
        return: (B, C, D', H', W') -- upsampled and RWKV-guided feature map
        """
        assert x.dim() == 5, f"Expected 5D input (B,C,D,H,W), got {x.shape}"
        B, C, D, H, W = x.shape
        assert global_state.shape == (B, C), f"global_state shape {global_state.shape} incompatible with x {x.shape}"

        # 1. semantic-neutral 3D upsampling
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )  # (B, C, D', H', W')

        Dp, Hp, Wp = x_up.shape[2], x_up.shape[3], x_up.shape[4]

        # 2. flatten to tokens (B, N, C), N = D'*H'*W'
        tokens = x_up.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        tokens = self.norm(tokens)

        # 3. cross-stage conditioned RWKV in 3D context
        context = self.cond_rwkv(tokens, global_state)  # (B, N, C)

        # 4. semantic modulation parameters from RWKV context
        gamma = self.gamma_proj(context)  # (B, N, C)
        beta = self.beta_proj(context)    # (B, N, C)

        # 5. reshape back to 3D volumes
        gamma = gamma.transpose(1, 2).view(B, C, Dp, Hp, Wp)
        beta = beta.transpose(1, 2).view(B, C, Dp, Hp, Wp)

        # 6. RWKV-guided semantic modulation of upsampled feature
        out = x_up * (1.0 + torch.tanh(gamma)) + beta
        return out
