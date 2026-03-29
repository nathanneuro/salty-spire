"""
Graph node: a neural network encoder that produces latent representations.

Each node in the prediction graph is an encoder (ViT or ResNet) that receives
some view of the input and produces latent activations. Nodes predict their
neighbors' latents via attached predictor heads.
"""

import torch
import torch.nn as nn
import timm


class GraphNode(nn.Module):
    """A single node in the cyclic prediction graph.

    Each node has:
    - An encoder backbone that maps input patches to latent representations
    - A mask defining which input patches this node observes
    - A frozen/unfrozen state for schedule-based training
    """

    def __init__(
        self,
        node_id: int,
        encoder_name: str = "vit_small_patch16_224",
        latent_dim: int = 384,
        pretrained: bool = False,
    ):
        super().__init__()
        self.node_id = node_id
        self.latent_dim = latent_dim

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )

        # Project to common latent dim if encoder output differs
        encoder_dim = self.encoder.num_features
        if encoder_dim != latent_dim:
            self.projector = nn.Sequential(
                nn.Linear(encoder_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.projector = nn.Identity()

        self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input and project to latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Latent representation [B, latent_dim]
        """
        features = self.encoder(x)
        return self.projector(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate features (patch tokens) for richer prediction targets.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Patch-level features [B, num_patches, encoder_dim]
        """
        return self.encoder.forward_features(x)

    def freeze(self):
        """Freeze all parameters (node becomes a fixed target)."""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (node becomes trainable)."""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def __repr__(self):
        status = "frozen" if self._frozen else "trainable"
        return f"GraphNode(id={self.node_id}, dim={self.latent_dim}, {status})"
