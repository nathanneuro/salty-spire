"""
Spatial prediction graph: operates on patch-level features instead of pooled vectors.

This is the patch-level counterpart to PredictionGraph. Instead of
encoding to [B, D] and predicting globally, each node produces
[B, num_patches, D] and predictors map between spatial feature maps.

This preserves the spatial structure needed to test whether convolutional
vs. attention vs. topographic inter-node projections matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .node import GraphNode
from .spatial_predictor import SpatialPredictorBase, build_spatial_predictor


class SpatialPredictionGraph(nn.Module):
    """Prediction graph operating on patch-level (spatial) features.

    Like PredictionGraph, but:
    - encode_all returns [B, num_patches, D] per node (not [B, D])
    - Predictors are SpatialPredictorBase instances that map between patch grids
    - Loss is computed per-patch then averaged
    """

    def __init__(
        self,
        nodes: list[GraphNode],
        edges: list[tuple[int, int]],
        predictor_type: str = "conv",
        predictor_kwargs: dict | None = None,
    ):
        super().__init__()
        self.node_ids = [n.node_id for n in nodes]
        self.nodes = nn.ModuleDict({str(n.node_id): n for n in nodes})
        self.edges = edges

        predictor_kwargs = predictor_kwargs or {}

        # Create spatial predictors for each edge
        self.predictors = nn.ModuleDict()
        for src_id, tgt_id in edges:
            src_node = self.nodes[str(src_id)]
            tgt_node = self.nodes[str(tgt_id)]
            key = f"{src_id}_to_{tgt_id}"
            self.predictors[key] = build_spatial_predictor(
                predictor_type=predictor_type,
                input_dim=src_node.encoder.num_features,
                output_dim=tgt_node.encoder.num_features,
                **predictor_kwargs,
            )

    def get_node(self, node_id: int) -> GraphNode:
        return self.nodes[str(node_id)]

    def get_predictor(self, src_id: int, tgt_id: int) -> SpatialPredictorBase:
        return self.predictors[f"{src_id}_to_{tgt_id}"]

    def encode_all_spatial(
        self, inputs: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Encode inputs through all nodes, returning patch-level features.

        Returns:
            {node_id: [B, num_patches, encoder_dim]}
        """
        latents = {}
        for node_id in self.node_ids:
            node = self.get_node(node_id)
            features = node.forward_features(inputs[node_id])
            # timm ViTs return [B, num_patches+1, D] (with CLS token)
            # or [B, num_patches, D] depending on model. Handle both:
            if hasattr(node.encoder, "num_prefix_tokens"):
                prefix = node.encoder.num_prefix_tokens
                if prefix > 0:
                    features = features[:, prefix:]  # strip CLS
            latents[node_id] = features
        return latents

    def encode_all(
        self, inputs: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Encode and pool to [B, D] for compatibility with linear probing."""
        spatial = self.encode_all_spatial(inputs)
        return {nid: feat.mean(dim=1) for nid, feat in spatial.items()}

    def compute_edge_losses(
        self,
        spatial_latents: dict[int, torch.Tensor],
        detach_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute per-patch prediction loss for each edge.

        Args:
            spatial_latents: {node_id: [B, N, D]} from encode_all_spatial
            detach_targets: stop gradient on targets

        Returns:
            {edge_key: scalar loss}
        """
        edge_losses = {}
        for src_id, tgt_id in self.edges:
            src_patches = spatial_latents[src_id]
            tgt_patches = spatial_latents[tgt_id]
            if detach_targets:
                tgt_patches = tgt_patches.detach()

            predictor = self.get_predictor(src_id, tgt_id)
            predicted = predictor(src_patches)

            # Per-patch MSE, averaged over patches and batch
            loss = nn.functional.mse_loss(predicted, tgt_patches)
            edge_losses[f"{src_id}_to_{tgt_id}"] = loss

        return edge_losses

    @staticmethod
    def make_ring(
        num_nodes: int,
        encoder_name: str = "vit_small_patch16_224",
        latent_dim: int = 384,
        predictor_type: str = "conv",
        bidirectional: bool = False,
        **predictor_kwargs,
    ) -> "SpatialPredictionGraph":
        """Create a ring topology with spatial predictors."""
        nodes = [
            GraphNode(i, encoder_name=encoder_name, latent_dim=latent_dim)
            for i in range(num_nodes)
        ]
        edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        if bidirectional:
            edges += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
        return SpatialPredictionGraph(
            nodes, edges,
            predictor_type=predictor_type,
            predictor_kwargs=predictor_kwargs,
        )

    def __repr__(self):
        pred_type = "unknown"
        for key, pred in self.predictors.items():
            pred_type = type(pred).__name__
            break
        edge_str = ", ".join(f"{s}->{t}" for s, t in self.edges)
        return (
            f"SpatialPredictionGraph(nodes={self.node_ids}, "
            f"predictor={pred_type}, edges=[{edge_str}])"
        )
