"""
BindCORE: COformational Representation Ensemble for LIP prediction
===================================================================
ProteinMultiScaleTransformer — the main model.

Integrates:
  - Sequence embeddings + sinusoidal positional encodings
  - Per-residue local features  (x_local)
  - Per-protein scalar features (x_scalar)
  - Pairwise residue features   (x_pairwise)

through a series of CNN-biased Transformer blocks.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bindcore.config import ProteinModelConfig
from bindcore.modeling.compute_default_matrices import (
    subtract_random_coil_pairwise_baseline,
)

# ---------------------------------------------------------------------------
# 1.  Small reusable primitives
# ---------------------------------------------------------------------------


class MLP2(nn.Module):
    """Two-layer MLP: in_dim → hidden_dim → out_dim with ReLU + optional dropout."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN: E → 2E → E (as used inside each Transformer block).
    Wraps with LayerNorm + residual.
    """

    def __init__(self, embed_dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = embed_dim * expansion
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, E]  →  [B, L, E]"""
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# 2.  Input embedding components
# ---------------------------------------------------------------------------


class SequenceEmbedding(nn.Module):
    """
    Learned token embedding + sinusoidal positional encoding.
    Returns [B, L, E].
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int,
        dropout: float = 0.1,
        use_pos_embedding: bool = False,
    ):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute sinusoidal positional encoding (not learned)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, E]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] (int)  →  [B, L, E]"""
        L = tokens.size(1)
        x = self.token_emb(tokens)
        if self.use_pos_embedding:
            x += self.pe[:, :L].expand(tokens.size(0), -1, -1)
        return self.dropout(x)


class LocalFeatureProjector(nn.Module):
    """
    Projects per-residue local features to embedding space with learned scaling.
    x_local [B, nb_local, L]  →  [B, L, E]
    """

    def __init__(
        self,
        nb_local: int,
        embed_dim: int,
        hidden_dim: int,
        means: torch.Tensor,
        stds: torch.Tensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Initialize the normalization layer with local feature stats
        self.scaler = LearnedScalarNorm(
            nb_local, initial_means=means, initial_stds=stds
        )

        # 2. Project the normalized features
        self.mlp = MLP2(nb_local, hidden_dim, embed_dim, dropout)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        """x_local: [B, nb_local, L]  →  [B, L, E]"""
        # Step 1: Move features to the last dimension [B, L, nb_local]
        x = x_local.permute(0, 2, 1)

        # Step 2: Apply learned scaling per feature
        x_scaled = self.scaler(x)

        # Step 3: Project to embedding space
        return self.mlp(x_scaled)  # [B, L, E]


class LearnedScalarNorm(nn.Module):
    def __init__(
        self,
        nb_scalar: int,
        initial_means: torch.Tensor = None,
        initial_stds: torch.Tensor = None,
    ):
        super().__init__()
        # Initialize shift (mu) and scale (sigma)
        # If no stats provided, start at 0 and 1
        if initial_means is None:
            initial_means = torch.zeros(nb_scalar)
        if initial_stds is None:
            initial_stds = torch.ones(nb_scalar)

        # We use nn.Parameter so the optimizer can update them
        # self.mean = nn.Parameter(initial_means)
        # self.log_std = nn.Parameter(torch.log(initial_stds + 1e-6))
        # Register as buffers, not parameters
        self.register_buffer("mean", initial_means)
        self.register_buffer("log_std", torch.log(initial_stds + 1e-6))

    def forward(self, x):
        # We use exp(log_std) to ensure the standard deviation stays positive
        return (x - self.mean) / (torch.exp(self.log_std) + 1e-6)


class ScalarFeatureProjector(nn.Module):
    def __init__(
        self,
        nb_scalar: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        means: torch.Tensor,
        stds: torch.Tensor,
    ):
        super().__init__()
        # 1. The Scaling Layer (Independent scaling per feature)
        self.scaler = LearnedScalarNorm(
            nb_scalar, initial_means=means, initial_stds=stds
        )

        # 2. The Projection Layer
        self.mlp = MLP2(nb_scalar, hidden_dim, embed_dim, dropout=dropout)

    def forward(self, x_scalar: torch.Tensor, L: int) -> torch.Tensor:
        # Step 1: Scale raw values
        x_scaled = self.scaler(x_scalar)
        # Step 2: Project
        protein_repr = self.mlp(x_scaled)
        return protein_repr.unsqueeze(1).expand(-1, L, -1)


class ProteinLengthProjector(nn.Module):
    """Projects the mask-derived protein length as a global feature."""

    def __init__(
        self, embed_dim: int, hidden_dim: int, max_seq_len: int, dropout: float
    ):
        super().__init__()
        self.max_seq_len = max(1, max_seq_len)
        self.mlp = MLP2(1, hidden_dim, embed_dim, dropout=dropout)

    def normalize(self, protein_lengths: torch.Tensor) -> torch.Tensor:
        protein_lengths = protein_lengths.to(dtype=torch.float32)
        max_len = float(self.max_seq_len)
        return torch.log1p(protein_lengths).div(math.log1p(max_len))

    def forward(self, protein_lengths: torch.Tensor, L: int) -> torch.Tensor:
        x = self.normalize(protein_lengths).unsqueeze(-1)
        protein_repr = self.mlp(x)
        return protein_repr.unsqueeze(1).expand(-1, L, -1)


class PairwiseContextProjector(nn.Module):
    """
    Converts pairwise features into per-residue context vectors using
    multi-scale pooling (local, distant, global).

    For each residue i:
      - short_ctx : mean over ±short_r sequence neighbours
      - long_ctx  : mean over residues beyond ±short_r
      - global_ctx: mean over all residues

    The three contexts are concatenated → [B, L, 3C] then projected to [B, L, E].

    This is robust to any sequence length and captures both local flexibility
    and long-range allosteric signals (important for DCCM features).
    """

    def __init__(
        self,
        nb_pairwise: int,
        embed_dim: int,
        dropout: float = 0.1,
        short_r: int = 10,  # radius for local context (±short_r residues)
    ):
        super().__init__()
        self.short_r = short_r
        # 3C: short + long + global
        in_dim = nb_pairwise * 3
        self.mlp = MLP2(in_dim, embed_dim, embed_dim, dropout)

    def forward(
        self, x_pairwise: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x_pairwise : [B, C, L, L]
        batch_mask : [B, L]  (True = real token, False = padding)
        →  [B, L, E]
        """
        B, C, L, _ = x_pairwise.shape
        idx = torch.arange(L, device=x_pairwise.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [L, L]
        local_mask = (dist <= self.short_r).float()  # [L, L]
        distant_mask = 1.0 - local_mask  # [L, L]

        # a pair (i, j) is valid only if both tokens are real (not padding)
        pair_mask = (
            batch_mask.unsqueeze(1) & batch_mask.unsqueeze(2)
        ).float()  # [B, L, L]

        local_mask = local_mask.unsqueeze(0) * pair_mask  # [B, L, L]
        distant_mask = distant_mask.unsqueeze(0) * pair_mask  # [B, L, L]
        global_mask = pair_mask  # [B, L, L]

        n_local = local_mask.sum(dim=-1).clamp(min=1)  # [B, L]
        n_distant = distant_mask.sum(dim=-1).clamp(min=1)  # [B, L]
        n_global = global_mask.sum(dim=-1).clamp(min=1)  # [B, L]

        local_mask = local_mask.unsqueeze(1)  # [B, 1, L, L]
        distant_mask = distant_mask.unsqueeze(1)  # [B, 1, L, L]
        global_mask = global_mask.unsqueeze(1)  # [B, 1, L, L]

        short_ctx = (x_pairwise * local_mask).sum(dim=-1) / n_local.unsqueeze(
            1
        )  # [B, C, L]
        long_ctx = (x_pairwise * distant_mask).sum(dim=-1) / n_distant.unsqueeze(
            1
        )  # [B, C, L]
        global_ctx = (x_pairwise * global_mask).sum(dim=-1) / n_global.unsqueeze(
            1
        )  # [B, C, L]

        ctx = torch.cat(
            [
                short_ctx.permute(0, 2, 1),  # [B, L, C]
                long_ctx.permute(0, 2, 1),  # [B, L, C]
                global_ctx.permute(0, 2, 1),  # [B, L, C]
            ],
            dim=-1,
        )  # [B, L, 3C]

        out = self.mlp(ctx)  # [B, L, E]
        out = out * batch_mask.unsqueeze(-1).to(out.dtype)  # zero-out padded query rows
        return out


# ---------------------------------------------------------------------------
# 3.  Transformer block sub-components
# ---------------------------------------------------------------------------


def _valid_group_count(num_channels: int, max_groups: int = 32) -> int:
    """Largest valid group count not exceeding max_groups."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class MaskedPairwiseGroupNorm(nn.Module):
    """
    GroupNorm for pairwise [B, C, L, L] tensors that ignores padded pairs.

    Standard GroupNorm normalizes over all spatial positions, so zero-padded
    regions change the statistics for real residue pairs whenever the batch
    max length changes.  This layer keeps the same per-channel affine
    parameters but computes mean/variance only over valid (i, j) residue pairs.
    """

    def __init__(self, num_channels: int, max_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = _valid_group_count(num_channels, max_groups)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(
        self,
        x: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pair_mask is None:
            return F.group_norm(
                x,
                self.num_groups,
                self.weight,
                self.bias,
                self.eps,
            )

        B, C, L, _ = x.shape
        G = self.num_groups
        channels_per_group = C // G

        mask = pair_mask.to(device=x.device, dtype=x.dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask[:, :1].unsqueeze(1)  # [B, 1, 1, L, L]

        x_grouped = x.view(B, G, channels_per_group, L, L)
        valid_pairs = mask.sum(dim=(-1, -2), keepdim=True)
        denom = (valid_pairs * channels_per_group).clamp(min=1.0)

        mean = (x_grouped * mask).sum(dim=(2, 3, 4), keepdim=True) / denom
        centered = x_grouped - mean
        var = (centered.square() * mask).sum(dim=(2, 3, 4), keepdim=True) / denom

        x_norm = centered * torch.rsqrt(var + self.eps)
        x_norm = x_norm.view(B, C, L, L)
        return x_norm * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)


class MaskedPairwiseConvBranch(nn.Module):
    """Conv -> masked pairwise norm -> GELU branch used by PairwiseCNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = MaskedPairwiseGroupNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x), pair_mask))


class PairwiseCNN(nn.Module):
    """
    Pairwise feature extractor for protein contact/distance maps.

    Input : [B, C_in, L, L]
    Output: [B, num_heads, L, L]

    Pipeline (if dilations are provided):
      1) Depthwise conv — independent spatial processing per input channel
      2) Parallel dilated conv branches
      3) Concatenate + GroupNorm + GELU
      4) 1x1 conv to num_heads

    If dilations=[], the spatial CNN is skipped and a 1x1 projection is applied.
    """

    def __init__(
        self,
        nb_pairwise: int,
        cnn_channels: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.dilations = dilations

        # If empty dilations sequence is passed, skip the CNN and just project.
        if not self.dilations:
            self.to_heads = nn.Conv2d(nb_pairwise, num_heads, kernel_size=1, bias=True)
            return

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve L×L shape.")

        pad = kernel_size // 2

        # Stage 1: independent spatial processing per channel
        self.depthwise = nn.Conv2d(
            nb_pairwise,
            nb_pairwise,
            kernel_size=kernel_size,
            padding=pad,
            groups=nb_pairwise,
            bias=False,
        )
        self.depthwise_norm = MaskedPairwiseGroupNorm(nb_pairwise)
        self.depthwise_act = nn.GELU()

        # Stage 2: dilated branches for multi-scale mixing
        branches = []
        for d in self.dilations:
            branches.append(
                MaskedPairwiseConvBranch(
                    nb_pairwise,
                    cnn_channels,
                    kernel_size=kernel_size,
                    padding=d * pad,
                    dilation=d,
                )
            )
        self.branches = nn.ModuleList(branches)

        merged_channels = cnn_channels * len(self.dilations)
        self.post_norm = MaskedPairwiseGroupNorm(merged_channels)
        self.post_act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Final projection to attention heads
        self.to_heads = nn.Conv2d(merged_channels, num_heads, kernel_size=1, bias=True)

    def forward(
        self, x_pairwise: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x_pairwise: [B, nb_pairwise, L, L]
        batch_mask: [B, L]  (True = real token, False = padding)
        →  [B, num_heads, L, L]
        """
        B, _, L, _ = x_pairwise.shape

        pair_mask = batch_mask.unsqueeze(1) & batch_mask.unsqueeze(2)  # [B, L, L]
        pair_mask_c = pair_mask.unsqueeze(
            1
        ).float()  # [B, 1, L, L], broadcasts over channel dim

        # Mask input so padded positions don't bleed into the conv stack
        x_pairwise = x_pairwise * pair_mask_c

        # Bypass CNN logic if dilations is empty
        if not self.dilations:
            out = self.to_heads(x_pairwise)
            return out * pair_mask_c  # to_heads may project/bias, re-mask to be safe

        x = self.depthwise_act(
            self.depthwise_norm(self.depthwise(x_pairwise), pair_mask)
        )
        x = x * pair_mask_c  # depthwise_norm bias can reintroduce nonzero padding

        feats = [branch(x, pair_mask) for branch in self.branches]
        feats = [
            f * pair_mask_c for f in feats
        ]  # each dilated branch re-masked independently
        x = torch.cat(feats, dim=1)

        x = self.post_act(self.post_norm(x, pair_mask))
        x = x * pair_mask_c  # post_norm bias, same issue as above

        x = self.dropout(x)
        x = self.to_heads(x)
        return x * pair_mask_c


class BiasedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with an additive per-head pairwise bias.

    Each head has a learnable scalar gate (alpha_h) controlling how much the
    pairwise bias contributes:
        logits_h = QKᵀ / √d + alpha_h · bias_h

    x:    [B, L, E]
    bias: [B, num_heads, L, L]
    mask: [B, L]  (1 = real residue, 0 = padding)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        activate_bias: bool = True,
        activate_classical_attention: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # One learnable gate per head
        self.bias_gate = nn.Parameter(torch.zeros(num_heads))
        self.activate_bias = activate_bias
        self.activate_classical_attention = activate_classical_attention

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, E = x.shape
        H, D = self.num_heads, self.head_dim

        residual = x
        x = self.norm(x)

        def _proj_reshape(proj, t):
            return proj(t).reshape(B, L, H, D).transpose(1, 2)

        attn_logits = torch.zeros((B, H, L, L), device=x.device, dtype=x.dtype)
        V = _proj_reshape(self.v_proj, x)
        if self.activate_classical_attention:
            Q = _proj_reshape(self.q_proj, x)
            K = _proj_reshape(self.k_proj, x)

            attn_logits += torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.activate_bias:
            attn_logits += torch.sigmoid(self.bias_gate).view(1, H, 1, 1) * bias

        if mask is not None:
            # Mask padding key positions: [B, 1, 1, L]
            key_mask = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
            attn_logits = attn_logits + key_mask

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Zero out nan from fully-padded rows and padding query positions
        if mask is not None:
            query_mask = mask.float().unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            attn_weights = attn_weights.nan_to_num(0.0) * query_mask

        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, L, E)

        # Zero padding positions in output before residual
        if mask is not None:
            out = out * mask.float().unsqueeze(-1)  # [B, L, 1]

        out = residual + self.out_proj(out)
        if mask is not None:
            out = out * mask.float().unsqueeze(-1).to(out.dtype)
        return out


class TransformerBlock(nn.Module):
    """
    One full Transformer block:
        1. PairwiseUpdateBlock   — updates x_pairwise from current x (new)
        2. PairwiseCNN           — refines x_pairwise, produces per-head attention bias
        3. BiasedMHA             — self-attention with the pairwise bias
        4. FeedForwardNetwork    — position-wise FFN (E → 2E → E)
    """

    def __init__(self, cfg: ProteinModelConfig):
        super().__init__()
        self.activate_pairwise_bias = cfg.activate_pairwise_bias
        self.update_pairwise = True

        if self.update_pairwise:
            self.pairwise_update = PairwiseUpdateBlock(
                embed_dim=cfg.embed_dim,
                nb_pairwise=cfg.nb_pairwise,
                dropout=cfg.dropout,
            )
        self.pairwise_cnn = PairwiseCNN(
            nb_pairwise=cfg.nb_pairwise,
            cnn_channels=cfg.pairwise_cnn_channels,
            num_heads=cfg.num_heads,
            kernel_size=cfg.pairwise_cnn_kernel,
            dilations=cfg.dilatations_cnn,
            dropout=cfg.dropout,
        )
        self.attention = BiasedMultiHeadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            activate_bias=cfg.activate_pairwise_bias,
            activate_classical_attention=cfg.activate_classical_attention,
        )
        self.ffn = FeedForwardNetwork(
            embed_dim=cfg.embed_dim,
            expansion=cfg.ffn_expansion,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, L, E]
        x_pairwise: torch.Tensor,  # [B, C, L, L]
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x, x_pairwise) — both updated.
        The updated x_pairwise is passed to the next block instead of
        reusing the same initial pairwise representation every time.
        """
        # 1. Update pairwise from current sequence representation
        if self.update_pairwise:
            x_pairwise = self.pairwise_update(x_pairwise, x, mask)

        # 2. Compute attention bias from (updated) pairwise features
        if self.activate_pairwise_bias:
            attn_bias = self.pairwise_cnn(x_pairwise, mask)  # [B, H, L, L]
        else:
            attn_bias = None

        # 3. Sequence update
        x = self.attention(x, attn_bias, mask)
        x = self.ffn(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x, x_pairwise


# ---------------------------------------------------------------------------
# 4.  Pooling + classification head
# ---------------------------------------------------------------------------


class ClassificationHead(nn.Module):
    """Maps per-residue representations to class logits."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseUpdateBlock(nn.Module):
    """
    Updates the pairwise representation [B, C, L, L] using the current
    sequence embedding [B, L, E] via an outer product, mixed with a
    residual CNN refinement of the pairwise features themselves.

    Inspired by AlphaFold2's outer-product mean update in the Evoformer.
    """

    def __init__(self, embed_dim: int, nb_pairwise: int, dropout: float = 0.1):
        super().__init__()
        self.nb_pairwise = nb_pairwise

        # Project sequence embedding to a low-dim space before outer product
        # to keep the parameter count small
        self.low_dim = max(4, nb_pairwise)
        self.seq_to_low = nn.Linear(embed_dim, self.low_dim)

        # Outer product gives [B, L, L, low_dim^2], project back to nb_pairwise
        self.outer_proj = nn.Linear(self.low_dim, nb_pairwise)

        # Lightweight CNN to refine pairwise features with local spatial context
        self.cnn_depthwise = nn.Conv2d(
            nb_pairwise,
            nb_pairwise,
            kernel_size=3,
            padding=1,
            groups=nb_pairwise,
            bias=False,
        )
        self.cnn_norm = MaskedPairwiseGroupNorm(nb_pairwise)
        self.cnn_act = nn.GELU()
        self.cnn_pointwise = nn.Conv2d(
            nb_pairwise,
            nb_pairwise,
            kernel_size=1,
            bias=True,
        )
        self.cnn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(nb_pairwise)

    def forward(
        self,
        x_pairwise: torch.Tensor,  # [B, C, L, L]
        x: torch.Tensor,  # [B, L, E]
        batch_mask: torch.Tensor,  # [B, L]  (True = real token, False = padding)
    ) -> torch.Tensor:
        """Returns updated x_pairwise: [B, C, L, L]"""
        B, C, L, _ = x_pairwise.shape

        # pair_mask[b, i, j] = True only if both i and j are real tokens
        pair_mask = batch_mask.unsqueeze(1) & batch_mask.unsqueeze(2)  # [B, L, L]
        pair_mask_f = pair_mask.unsqueeze(
            -1
        ).float()  # [B, L, L, 1] for the [B,L,L,C] tensors
        pair_mask_c = pair_mask.unsqueeze(
            1
        ).float()  # [B, 1, L, L] for the [B,C,L,L] tensors

        # ── 0. Make sure incoming pairwise features have no garbage in padded slots ──
        x_pairwise = x_pairwise * pair_mask_c

        # ── 1. Outer product update from sequence embedding ──────────────
        # Zero out padded token embeddings first: since outer is elementwise
        # (a_i * a_j), zeroing a_i guarantees outer[i, j] == outer[j, i] == 0
        # for any padded i, regardless of a_j.
        a = self.seq_to_low(x)
        a = a * batch_mask.unsqueeze(-1).to(a.dtype)  # [B, L, low_dim]

        outer = a.unsqueeze(2) * a.unsqueeze(1)  # [B, L, L, low_dim]
        outer = self.outer_proj(outer)  # [B, L, L, C]

        # Symmetrize: pairwise features should be symmetric for contact/distance
        outer = (outer + outer.transpose(1, 2)) / 2  # [B, L, L, C]

        # LayerNorm has a learnable bias, so even zeroed-out padded entries
        # can come out nonzero after norm — re-mask afterward.
        outer = self.norm(outer) * pair_mask_f
        outer = outer.permute(0, 3, 1, 2)  # [B, C, L, L]

        # ── 2. CNN refinement of existing pairwise features ──────────────
        x_pairwise = x_pairwise + outer  # residual from sequence
        x_pairwise = x_pairwise * pair_mask_c  # re-mask before convolving

        cnn_out = self.cnn_depthwise(x_pairwise)
        cnn_out = self.cnn_act(self.cnn_norm(cnn_out, pair_mask))
        cnn_out = cnn_out * pair_mask_c
        cnn_out = self.cnn_pointwise(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = (
            cnn_out * pair_mask_c
        )  # conv receptive field can leak padding into real tokens
        x_pairwise = x_pairwise + cnn_out  # residual spatial refine

        return x_pairwise


# ---------------------------------------------------------------------------
# 5.  Main model
# ---------------------------------------------------------------------------


class ProteinMultiScaleTransformer(nn.Module):
    """
    BindCORE: multi-scale protein representation model for binding prediction.
    """

    def __init__(
        self,
        cfg: ProteinModelConfig,
        stats,
        pairwise_features: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.E = cfg.embed_dim
        self.use_scalar_features = cfg.use_scalar_features
        self.use_local_features = cfg.use_local_features
        self.use_pairwise_features = cfg.use_pairwise_features
        self.use_token_embedding = cfg.use_token_embedding
        self.use_plm_embedding = cfg.use_plm_embedding
        self.share_block_weights = (
            cfg.share_block_weights
        )  # Universal Transformer style
        self.pairwise_features = list(pairwise_features or [])

        # ── Input embeddings ──────────────────────────────────────────────
        self.seq_emb = SequenceEmbedding(
            cfg.vocab_size,
            self.E,
            cfg.max_seq_len,
            cfg.dropout,
            cfg.use_positional_embeddings,
        )
        if self.use_plm_embedding:
            self.plm_proj = nn.Sequential(
                nn.Linear(cfg.plm_dim, self.E),
                nn.Dropout(0.6),
                nn.GELU(),
                nn.Linear(self.E, self.E),
                nn.Dropout(cfg.dropout),
            )

        self.scalar_proj = ScalarFeatureProjector(
            cfg.nb_scalar,
            self.E,
            cfg.scalar_mlp_hidden,
            cfg.dropout,
            stats["scalar"]["means"],
            stats["scalar"]["stds"],
        )
        self.length_proj = ProteinLengthProjector(
            self.E,
            cfg.scalar_mlp_hidden,
            cfg.max_seq_len,
            cfg.dropout,
        )
        self.local_proj = LocalFeatureProjector(
            cfg.nb_local,
            self.E,
            cfg.local_mlp_hidden,
            stats["local"]["means"],
            stats["local"]["stds"],
            cfg.dropout,
        )
        self.pairwise_init_proj = PairwiseContextProjector(
            cfg.nb_pairwise,
            self.E,
            cfg.dropout,
        )
        self.pair_wise_scaler = LearnedScalarNorm(
            cfg.nb_pairwise,
            initial_means=stats["pairwise"]["means"],
            initial_stds=stats["pairwise"]["stds"],
        )
        self.embed_norm = nn.LayerNorm(self.E)

        # ── Transformer blocks ────────────────────────────────────────────
        # If share_block_weights is True: create ONE block and reuse it
        # cfg.num_blocks times during the forward pass (Universal Transformer).
        # Parameter count drops from num_blocks × block_params to 1 × block_params,
        # while depth (number of refinement passes) is preserved.
        if self.share_block_weights:
            self.shared_block = TransformerBlock(cfg)
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(cfg) for _ in range(cfg.num_blocks)]
            )
        self.num_blocks = cfg.num_blocks

        # ── Classification head ───────────────────────────────────────────
        self.head = ClassificationHead(self.E, cfg.num_classes, cfg.dropout)

    def pair_dropout(
        self, x_pairwise: torch.Tensor, mask: torch.Tensor, rate: float, training: bool
    ) -> torch.Tensor:
        if not training or rate <= 0:
            return x_pairwise
        keep = (torch.rand(mask.shape, device=mask.device) > rate).float()  # [B, L]
        pair_keep = (keep.unsqueeze(1) * keep.unsqueeze(2)).unsqueeze(1)  # [B, 1, L, L]
        return x_pairwise * pair_keep / ((1 - rate) ** 2 + 1e-8)

    def forward(
        self,
        tokens: torch.Tensor,
        x_scalar: torch.Tensor,
        x_local: torch.Tensor,
        x_pairwise: torch.Tensor,
        mask: torch.Tensor,
        plm_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = tokens.shape
        if mask is None:
            mask_bool = torch.ones((B, L), dtype=torch.bool, device=tokens.device)
            protein_lengths = torch.full(
                (B,),
                L,
                dtype=torch.float32,
                device=tokens.device,
            )
        else:
            mask_bool = mask.to(device=tokens.device).bool()
            protein_lengths = mask_bool.to(dtype=torch.float32).sum(dim=1)
        mask_float = mask_bool.to(dtype=x_pairwise.dtype)

        pairwise_mask = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(
            1
        ).unsqueeze(2)
        x_pairwise = x_pairwise * pairwise_mask

        # x_pairwise = subtract_random_coil_pairwise_baseline(
        #     x_pairwise, self.pairwise_features, mask_bool
        # ) # Finnally don't seem to help

        x_pairwise_permute = x_pairwise.permute(0, 2, 3, 1)  # [B, L, L, C]
        x_pairwise_permute_scaled = self.pair_wise_scaler(x_pairwise_permute)
        x_pairwise_scaled = x_pairwise_permute_scaled.permute(
            0, 3, 1, 2
        )  # [B, C, L, L]
        x_pairwise_scaled = self.pair_dropout(
            x_pairwise_scaled,
            mask_bool,
            self.cfg.pair_dropout_rate,
            self.training,
        )

        # 1. Build initial [B, L, E] embedding
        x = torch.zeros((B, L, self.E), device=tokens.device)

        if self.use_token_embedding:
            x = self.seq_emb(tokens)

        if self.use_scalar_features:
            x = x + self.scalar_proj(x_scalar, L)
        x = x + self.length_proj(protein_lengths, L)

        if self.use_local_features:
            x = x + self.local_proj(x_local)

        if self.use_pairwise_features:
            x = x + self.pairwise_init_proj(x_pairwise_scaled, mask_bool)

        if self.use_plm_embedding:
            x = x + self.plm_proj(plm_pad)

        x = self.embed_norm(x)
        x = x * mask_bool.unsqueeze(-1).to(dtype=x.dtype)

        # 2. Transformer blocks — x_pairwise evolves across blocks
        if self.share_block_weights:
            for _ in range(self.num_blocks):
                x, x_pairwise_scaled = self.shared_block(
                    x, x_pairwise_scaled, mask_bool
                )
        else:
            for block in self.blocks:
                x, x_pairwise_scaled = block(x, x_pairwise_scaled, mask_bool)

        # 3. Classification head
        logits = self.head(x)
        return logits * mask_bool.unsqueeze(-1).to(dtype=logits.dtype)
