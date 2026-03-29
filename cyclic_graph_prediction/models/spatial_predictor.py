"""
Spatially-structured inter-node predictors (Experiment 8).

A/B test: does the inductive bias of the inter-node projection matter?

The V1→V2 projection in cortex is retinotopic — nearby V1 neurons project
to nearby V2 neurons. This module provides four predictor architectures
with increasing spatial structure:

1. MLPPredictor: pool patches → MLP → target (destroys spatial info)
   This is the existing LatentPredictor wrapped for the spatial API.

2. ConvPredictor: reshape patches to 2D grid → Conv layers → target grid
   Preserves spatial locality (like CNNs / Tegmark's topological nets).

3. TopographicPredictor: each source patch predicts its corresponding
   target patch via a local receptive field. Directly models retinotopy.

4. CrossAttentionPredictor: source patches attend to target positions.
   Learns spatial relationships from data (like ViT but for inter-node).

All share the same interface: (B, num_patches, dim) → (B, num_patches, dim)
so they're drop-in replaceable in the graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPredictorBase(nn.Module):
    """Base interface for spatially-aware predictors.

    All spatial predictors operate on patch-level features:
        Input:  [B, N, D_src]  (source node's patch tokens)
        Output: [B, N, D_tgt]  (predicted target node's patch tokens)

    This is fundamentally different from LatentPredictor which operates
    on pooled [B, D] vectors.
    """

    def forward(
        self, source_patches: torch.Tensor
    ) -> torch.Tensor:
        """Predict target patch features from source patch features."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. MLP — pools away spatial structure (baseline / control)
# ---------------------------------------------------------------------------

class MLPSpatialPredictor(SpatialPredictorBase):
    """Per-patch MLP with no spatial interaction between patches.

    Each patch is independently projected. No information flows between
    spatial positions — the predictor treats patches as a bag.
    This is the "no spatial inductive bias" baseline.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, source_patches: torch.Tensor) -> torch.Tensor:
        # [B, N, D_src] → [B, N, D_tgt], applied independently per patch
        return self.mlp(source_patches)


# ---------------------------------------------------------------------------
# 2. Convolutional — preserves spatial locality (CNN / Tegmark-style)
# ---------------------------------------------------------------------------

class ConvSpatialPredictor(SpatialPredictorBase):
    """Convolutional predictor operating on the 2D patch grid.

    Reshapes patch tokens to a 2D feature map, applies Conv2d layers,
    then reshapes back. This bakes in spatial locality: each predicted
    patch depends only on a local neighborhood of source patches.

    This is the core "does convolution matter" test — the V1→V2
    projection as a learned convolutional map.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        num_layers: int = 2,
        grid_size: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size  # inferred from input if None

        layers = []
        in_c = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            ])
            in_c = hidden_dim
        layers.append(
            nn.Conv2d(in_c, output_dim, kernel_size, padding=kernel_size // 2)
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, source_patches: torch.Tensor) -> torch.Tensor:
        B, N, D = source_patches.shape
        H = self.grid_size or int(math.sqrt(N))
        W = N // H

        # [B, N, D] → [B, D, H, W]
        x = source_patches.transpose(1, 2).reshape(B, D, H, W)
        x = self.conv(x)
        # [B, D_out, H, W] → [B, N, D_out]
        return x.reshape(B, -1, N).transpose(1, 2)


# ---------------------------------------------------------------------------
# 3. Topographic — explicit retinotopic mapping
# ---------------------------------------------------------------------------

class TopographicPredictor(SpatialPredictorBase):
    """Topographic predictor with explicit retinotopic receptive fields.

    Each target patch position is predicted from a local neighborhood
    of source patches centered at the corresponding position.
    This directly models the biological V1→V2 topographic projection:
    source patch (i,j) contributes most strongly to target patch (i,j)
    and its immediate neighbors.

    Uses unfold/fold operations for efficient local windowing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        receptive_field: int = 3,
        hidden_dim: int = 256,
        grid_size: int | None = None,
    ):
        super().__init__()
        self.receptive_field = receptive_field
        self.grid_size = grid_size
        self.padding = receptive_field // 2

        # Local projection: maps (rf * rf * input_dim) → output_dim
        # Each target patch sees rf×rf source patches
        local_input_dim = receptive_field * receptive_field * input_dim
        self.local_proj = nn.Sequential(
            nn.Linear(local_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, source_patches: torch.Tensor) -> torch.Tensor:
        B, N, D = source_patches.shape
        H = self.grid_size or int(math.sqrt(N))
        W = N // H
        rf = self.receptive_field

        # [B, N, D] → [B, D, H, W]
        x = source_patches.transpose(1, 2).reshape(B, D, H, W)

        # Pad and unfold to extract local windows
        x_padded = F.pad(x, [self.padding] * 4, mode="reflect")
        # unfold extracts sliding windows: [B, D, H, W] → [B, D*rf*rf, H*W]
        x_unfolded = x_padded.unfold(2, rf, 1).unfold(3, rf, 1)
        # x_unfolded: [B, D, H, W, rf, rf]
        x_unfolded = x_unfolded.contiguous().reshape(B, D, H * W, rf * rf)
        # → [B, H*W, D*rf*rf]
        x_unfolded = x_unfolded.permute(0, 2, 1, 3).reshape(B, H * W, D * rf * rf)

        # Local projection per position
        return self.local_proj(x_unfolded)


# ---------------------------------------------------------------------------
# 4. Cross-attention — learns spatial relationships (ViT-style)
# ---------------------------------------------------------------------------

class CrossAttentionPredictor(SpatialPredictorBase):
    """Cross-attention predictor: target positions attend to source patches.

    No built-in spatial bias — learns which source locations are relevant
    for each target location via attention. Positional embeddings provide
    the spatial signal, but the model must learn to use them.

    This is the "ViT-style" control: maximal flexibility, minimal
    inductive bias.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        max_patches: int = 196,
    ):
        super().__init__()

        # Learnable positional embeddings for source and target
        self.src_pos_embed = nn.Parameter(torch.randn(1, max_patches, input_dim) * 0.02)
        self.tgt_pos_embed = nn.Parameter(torch.randn(1, max_patches, input_dim) * 0.02)

        # Project input_dim to a common dim if needed
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )
            self.layers.append(nn.LayerNorm(input_dim))

        # Final projection to output dim
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, source_patches: torch.Tensor) -> torch.Tensor:
        B, N, D = source_patches.shape

        # Add positional embeddings
        src = self.input_proj(source_patches) + self.src_pos_embed[:, :N]
        tgt = self.tgt_pos_embed[:, :N].expand(B, -1, -1)

        # Cross-attention: target queries attend to source keys/values
        for i in range(0, len(self.layers), 2):
            attn_layer = self.layers[i]
            norm_layer = self.layers[i + 1]
            attended, _ = attn_layer(tgt, src, src)
            tgt = norm_layer(tgt + attended)

        return self.output_proj(tgt)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

PREDICTOR_REGISTRY: dict[str, type[SpatialPredictorBase]] = {
    "mlp": MLPSpatialPredictor,
    "conv": ConvSpatialPredictor,
    "topographic": TopographicPredictor,
    "cross_attention": CrossAttentionPredictor,
}


def build_spatial_predictor(
    predictor_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> SpatialPredictorBase:
    """Factory for spatial predictor variants.

    Args:
        predictor_type: one of "mlp", "conv", "topographic", "cross_attention"
        input_dim: source node's per-patch feature dim
        output_dim: target node's per-patch feature dim
        **kwargs: predictor-specific args (kernel_size, receptive_field, etc.)
    """
    if predictor_type not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown predictor type '{predictor_type}'. "
            f"Available: {list(PREDICTOR_REGISTRY.keys())}"
        )
    cls = PREDICTOR_REGISTRY[predictor_type]
    return cls(input_dim=input_dim, output_dim=output_dim, **kwargs)
