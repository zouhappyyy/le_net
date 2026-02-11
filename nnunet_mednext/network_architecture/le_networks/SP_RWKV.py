import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticConditioning(nn.Module):
    """Generate FiLM-style modulation parameters from encoder global state."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, tokens, global_state):
        """tokens: (B, N, C), global_state: (B, C)"""
        gamma = self.gamma(global_state).unsqueeze(1)  # (B,1,C)
        beta = self.beta(global_state).unsqueeze(1)    # (B,1,C)
        return tokens * (1 + gamma) + beta


class PropagationRWKV(nn.Module):
    """Lightweight RWKV variant for semantic propagation.

    No global perception, only directional recurrence.
    Works on generic sequences of shape (B, N, C), thus can be reused
    for both 2D (N = H*W) and 3D (N = D*H*W) flattenings.
    """

    def __init__(self, dim):
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, N, C)"""
        B, N, C = x.shape

        x = self.norm(x)
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))

        state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outputs = []

        decay = torch.exp(-torch.relu(self.time_decay))

        for t in range(N):
            state = state * decay + k[:, t, :]
            out = r[:, t, :] * (state + v[:, t, :])
            outputs.append(out.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.output(y)
        return y


class SemanticPropagationUpsample(nn.Module):
    """2D SP-RWKV based semantic propagation upsampling.

    This module performs:
    1) plain bilinear upsampling on (B, C, H, W)
    2) flatten to tokens (B, N, C)
    3) global semantic conditioning
    4) one or more RWKV-based semantic propagation + FiLM-style modulation
    """

    def __init__(self, dim, scale_factor=2, num_blocks: int = 2):
        """Args:
        dim: channel dimension C
        scale_factor: upsampling factor for H and W
        num_blocks: how many propagation+modulation blocks to apply in sequence.
                    Default 2 per user request ("add one more semantic propagation+modulation block").
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.num_blocks = max(1, int(num_blocks))

        self.condition = SemanticConditioning(dim)

        # We stack `num_blocks` propagation blocks with their own gamma/beta projections.
        self.propagations = nn.ModuleList([PropagationRWKV(dim) for _ in range(self.num_blocks)])
        self.gamma_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_blocks)])
        self.beta_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_blocks)])

    def _single_block(self, x_spatial, tokens, block_idx: int):
        """Apply one semantic propagation + FiLM modulation block.

        x_spatial: current spatial feature map after previous block, shape (B, C, H, W)
        tokens: tokens corresponding to x_spatial, shape (B, N, C)
        block_idx: index into propagation / proj lists
        """
        B, C, H, W = x_spatial.shape
        tokens = self.propagations[block_idx](tokens)  # (B, N, C)
        gamma = self.gamma_projs[block_idx](tokens)
        beta = self.beta_projs[block_idx](tokens)

        gamma = gamma.transpose(1, 2).view(B, C, H, W)
        beta = beta.transpose(1, 2).view(B, C, H, W)

        out = x_spatial * (1 + torch.tanh(gamma)) + beta
        return out

    def forward(self, x, global_state):
        """Args:
        x: (B, C, H, W)
        global_state: (B, C)
        """
        B, C, H, W = x.shape

        # 1) plain interpolation (no semantics yet)
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )

        _, _, Hp, Wp = x_up.shape

        # 2) flatten to tokens
        tokens = x_up.flatten(2).transpose(1, 2)  # (B, N, C)

        # 3) global semantic conditioning (applied once and shared as input for all blocks)
        tokens = self.condition(tokens, global_state)

        # 4) stacked semantic propagation + FiLM modulation blocks
        out = x_up
        current_tokens = tokens
        for i in range(self.num_blocks):
            # Re-flatten current spatial map to keep tokens consistent with spatial features
            if i > 0:
                current_tokens = out.flatten(2).transpose(1, 2)
            out = self._single_block(out, current_tokens, i)

        return out


class SemanticPropagationUpsample3D(nn.Module):
    """3D SP-RWKV based semantic propagation upsampling.

    Direct 3D extension of SemanticPropagationUpsample using trilinear
    interpolation and D*H*W flattening.
    """

    def __init__(self, dim, scale_factor=2, num_blocks: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_blocks = max(1, int(num_blocks))

        self.condition = SemanticConditioning(dim)
        self.propagations = nn.ModuleList([PropagationRWKV(dim) for _ in range(self.num_blocks)])
        self.gamma_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_blocks)])
        self.beta_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.num_blocks)])

    def _single_block(self, x_spatial, tokens, block_idx: int):
        """One semantic propagation + FiLM modulation block for 3D.

        x_spatial: (B, C, D, H, W)
        tokens: (B, N, C), where N = D*H*W
        """
        B, C, D, H, W = x_spatial.shape
        tokens = self.propagations[block_idx](tokens)  # (B, N, C)
        gamma = self.gamma_projs[block_idx](tokens)
        beta = self.beta_projs[block_idx](tokens)

        gamma = gamma.transpose(1, 2).view(B, C, D, H, W)
        beta = beta.transpose(1, 2).view(B, C, D, H, W)

        out = x_spatial * (1 + torch.tanh(gamma)) + beta
        return out

    def forward(self, x, global_state):
        """Args:
        x: (B, C, D, H, W)
        global_state: (B, C)
        """
        B, C, D, H, W = x.shape

        # 1) 3D interpolation
        x_up = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="trilinear",
            align_corners=False,
        )

        _, _, Dp, Hp, Wp = x_up.shape

        # 2) flatten to tokens: (B, C, D',H',W') -> (B, N, C)
        tokens = x_up.flatten(2).transpose(1, 2)

        # 3) global semantic conditioning
        tokens = self.condition(tokens, global_state)

        # 4) stacked semantic propagation + FiLM modulation blocks
        out = x_up
        current_tokens = tokens
        for i in range(self.num_blocks):
            if i > 0:
                current_tokens = out.flatten(2).transpose(1, 2)
            out = self._single_block(out, current_tokens, i)

        return out
