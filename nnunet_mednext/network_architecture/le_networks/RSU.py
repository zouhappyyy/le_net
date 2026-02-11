import torch
import torch.nn as nn
import torch.nn.functional as F

from .SP_RWKV import SemanticConditioning, PropagationRWKV


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

    This implementation now reuses the generic semantic conditioning + RWKV
    propagation logic from SP_RWKV (SemanticConditioning + PropagationRWKV),
    keeping the public API identical while centralizing the behaviour.
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

        # Reuse SP_RWKV's conditioning and lightweight RWKV-style propagation.
        # We still accept an external rwkv_block for backward compatibility;
        # if provided, it wraps PropagationRWKV, otherwise we use PropagationRWKV
        # directly.
        self.semantic_condition = SemanticConditioning(in_channels)

        # Prefer external rwkv_block if it matches (B,N,C)->(B,N,C), else fallback.
        self.use_external_rwkv = rwkv_block is not None
        if self.use_external_rwkv:
            self.rwkv = rwkv_block
        else:
            self.rwkv = PropagationRWKV(in_channels)

        self.gamma_proj = nn.Linear(in_channels, in_channels)
        self.beta_proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """\
        x: (B, C, D, H, W), global_state: (B, C) -> (B, C, D', H', W')
        """
        assert x.dim() == 5, f"Expected 5D input (B,C,D,H,W), got {x.shape}"
        B, C, D, H, W = x.shape
        assert global_state.shape == (B, C), (
            f"global_state shape {global_state.shape} incompatible with x {x.shape}"
        )

        # 1. 3D upsampling (semantic-neutral)
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )  # (B, C, D', H', W')

        Dp, Hp, Wp = x_up.shape[2], x_up.shape[3], x_up.shape[4]

        # 2. flatten to tokens and apply global semantic conditioning
        tokens = x_up.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        tokens = self.semantic_condition(tokens, global_state)
        tokens = self.norm(tokens)

        # 3. RWKV-style propagation (either external rwkv_block or local PropagationRWKV)
        context = self.rwkv(tokens)  # (B, N, C)

        # 4. semantic modulation parameters from RWKV context
        gamma = self.gamma_proj(context)  # (B, N, C)
        beta = self.beta_proj(context)   # (B, N, C)

        # 5. reshape back to 3D volumes
        gamma = gamma.transpose(1, 2).view(B, C, Dp, Hp, Wp)
        beta = beta.transpose(1, 2).view(B, C, Dp, Hp, Wp)

        # 6. semantic FiLM modulation of upsampled feature
        out = x_up * (1.0 + torch.tanh(gamma)) + beta
        return out
