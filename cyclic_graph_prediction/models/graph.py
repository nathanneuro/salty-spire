"""
Prediction graph: the full cyclic graph of nodes and prediction edges.

Manages the topology (which node predicts which), the predictor heads on
each edge, and message-passing at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .node import GraphNode
from .predictor import LatentPredictor, PrecisionWeightedLoss


class PredictionGraph(nn.Module):
    """A directed cyclic graph of neural network nodes with mutual latent prediction.

    The graph owns:
    - A set of GraphNode encoders
    - A LatentPredictor for each directed edge
    - The loss computation (optionally precision-weighted)
    """

    def __init__(
        self,
        nodes: list[GraphNode],
        edges: list[tuple[int, int]],
        predictor_hidden_dim: int = 512,
        predictor_layers: int = 2,
        use_precision: bool = False,
    ):
        super().__init__()
        self.node_ids = [n.node_id for n in nodes]
        self.nodes = nn.ModuleDict({str(n.node_id): n for n in nodes})
        self.edges = edges

        # Create a predictor for each directed edge
        self.predictors = nn.ModuleDict()
        for src_id, tgt_id in edges:
            src_node = self.nodes[str(src_id)]
            tgt_node = self.nodes[str(tgt_id)]
            key = f"{src_id}_to_{tgt_id}"
            self.predictors[key] = LatentPredictor(
                input_dim=src_node.latent_dim,
                output_dim=tgt_node.latent_dim,
                hidden_dim=predictor_hidden_dim,
                num_layers=predictor_layers,
                use_precision=use_precision,
            )

        self.loss_fn = PrecisionWeightedLoss()

    def get_node(self, node_id: int) -> GraphNode:
        return self.nodes[str(node_id)]

    def get_predictor(self, src_id: int, tgt_id: int) -> LatentPredictor:
        return self.predictors[f"{src_id}_to_{tgt_id}"]

    def encode_all(
        self, inputs: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Encode inputs through all nodes.

        Args:
            inputs: {node_id: input_tensor} for each node

        Returns:
            {node_id: latent_tensor} for each node
        """
        latents = {}
        for node_id in self.node_ids:
            node = self.get_node(node_id)
            latents[node_id] = node(inputs[node_id])
        return latents

    def compute_edge_losses(
        self,
        latents: dict[int, torch.Tensor],
        detach_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute prediction loss for each edge.

        Args:
            latents: pre-computed latents from encode_all
            detach_targets: if True, stop gradient on target latents
                (should be True when target node is the "frozen" one in the schedule)

        Returns:
            {edge_key: loss_value} for each edge
        """
        edge_losses = {}
        for src_id, tgt_id in self.edges:
            src_latent = latents[src_id]
            tgt_latent = latents[tgt_id]
            if detach_targets:
                tgt_latent = tgt_latent.detach()

            predictor = self.get_predictor(src_id, tgt_id)
            predicted, precision = predictor(src_latent)
            loss = self.loss_fn(predicted, tgt_latent, precision)
            edge_losses[f"{src_id}_to_{tgt_id}"] = loss

        return edge_losses

    def message_pass(
        self,
        inputs: dict[int, torch.Tensor],
        num_rounds: int = 1,
    ) -> dict[int, torch.Tensor]:
        """Recurrent inference via iterative message passing (Experiment 4).

        Each round: every node updates its representation by averaging its
        own encoding with predictions from its incoming neighbors.

        Args:
            inputs: raw inputs per node
            num_rounds: number of message-passing iterations

        Returns:
            Refined latents after k rounds
        """
        # Initial encoding
        latents = self.encode_all(inputs)

        # Build adjacency: for each node, which edges point to it?
        incoming = {nid: [] for nid in self.node_ids}
        for src_id, tgt_id in self.edges:
            incoming[tgt_id].append(src_id)

        for _ in range(num_rounds):
            new_latents = {}
            for tgt_id in self.node_ids:
                # Collect predictions from all incoming neighbors
                preds = []
                for src_id in incoming[tgt_id]:
                    predictor = self.get_predictor(src_id, tgt_id)
                    pred, _ = predictor(latents[src_id].detach())
                    preds.append(pred)

                if preds:
                    # Average own latent with neighbor predictions
                    neighbor_mean = torch.stack(preds).mean(dim=0)
                    new_latents[tgt_id] = 0.5 * latents[tgt_id] + 0.5 * neighbor_mean
                else:
                    new_latents[tgt_id] = latents[tgt_id]

            latents = new_latents

        return latents

    @staticmethod
    def make_ring(
        num_nodes: int,
        encoder_name: str = "vit_small_patch16_224",
        latent_dim: int = 384,
        bidirectional: bool = False,
        **kwargs,
    ) -> "PredictionGraph":
        """Convenience: create a ring topology.

        Args:
            num_nodes: number of nodes in the ring
            encoder_name: timm model name for each node
            latent_dim: shared latent dimension
            bidirectional: if True, add edges in both directions
        """
        nodes = [
            GraphNode(i, encoder_name=encoder_name, latent_dim=latent_dim)
            for i in range(num_nodes)
        ]
        # Clockwise edges
        edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        if bidirectional:
            edges += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
        return PredictionGraph(nodes, edges, **kwargs)

    def __repr__(self):
        edge_str = ", ".join(f"{s}->{t}" for s, t in self.edges)
        return (
            f"PredictionGraph(nodes={self.node_ids}, edges=[{edge_str}])"
        )
