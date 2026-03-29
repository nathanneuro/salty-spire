"""
Pixel reconstruction decoder for V1-like nodes (cortical cascade Stage 1).

V1 trains with a pixel reconstruction objective under masking, analogous
to SALT's Stage 1 teacher training. This grounds the hierarchy in
sensory input before higher areas learn via latent prediction.
"""

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """Lightweight decoder that reconstructs masked patches from encoder features.

    Used for the V1 node's pixel reconstruction objective (SALT Stage 1 analog).
    Takes patch-level encoder features and decodes to pixel space.
    """

    def __init__(
        self,
        encoder_dim: int,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        pixel_dim = patch_size * patch_size * num_channels

        layers = []
        in_d = encoder_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, pixel_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode patch features to pixel values.

        Args:
            features: [B, num_patches, encoder_dim] from encoder.forward_features()

        Returns:
            pixels: [B, num_patches, patch_size * patch_size * num_channels]
        """
        return self.decoder(features)


class PixelReconstructionLoss(nn.Module):
    """Masked pixel reconstruction loss for V1 pre-training.

    Only computes loss on masked (invisible) patches, forcing the
    encoder to learn predictive representations from visible context.
    """

    def __init__(self, norm_pix_loss: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(
        self,
        predicted_pixels: torch.Tensor,
        target_pixels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss on masked patches.

        Args:
            predicted_pixels: [B, num_patches, pixel_dim] from decoder
            target_pixels: [B, num_patches, pixel_dim] ground truth
            mask: [B, num_patches] bool, True = masked (reconstruct these)

        Returns:
            Scalar loss
        """
        if self.norm_pix_loss:
            # Normalize target per-patch (as in MAE)
            mean = target_pixels.mean(dim=-1, keepdim=True)
            var = target_pixels.var(dim=-1, keepdim=True)
            target_pixels = (target_pixels - mean) / (var + 1e-6).sqrt()

        loss = (predicted_pixels - target_pixels) ** 2
        loss = loss.mean(dim=-1)  # [B, num_patches]

        # Only loss on masked patches
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss
