"""
Stream-specific probing for ventral/dorsal specialization (Experiment 7).

Tests the central prediction: does graph topology alone cause ventral-stream
nodes to specialize for object identity and dorsal-stream nodes for motion?

Metrics:
- Identity probe: classify object category from frozen node features
- Motion probe: classify spatial transform type from frozen node features
- Selectivity index: (identity_acc - motion_acc) / (identity_acc + motion_acc)
  Positive = identity-biased (ventral), Negative = motion-biased (dorsal)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.graph import PredictionGraph
from ..trainers.dual_stream import DUAL_STREAM_REGIONS, VENTRAL_NODE_IDS, DORSAL_NODE_IDS

logger = logging.getLogger(__name__)


class StreamProbe(nn.Module):
    """Linear probe for a single task (identity or motion)."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _train_probe(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: torch.Tensor,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 0.01,
) -> float:
    """Train a linear probe and return validation accuracy."""
    dim = features_train.shape[1]
    probe = StreamProbe(dim, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    features_train = features_train.to(device)
    labels_train = labels_train.to(device)
    features_val = features_val.to(device)
    labels_val = labels_val.to(device)

    batch_size = 256
    n = features_train.shape[0]

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(features_train[idx])
            loss = criterion(logits, labels_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(features_val)
        acc = (logits.argmax(1) == labels_val).float().mean().item()
    return acc


@torch.no_grad()
def _extract_node_features(
    graph: PredictionGraph,
    loader: DataLoader,
    node_id: int,
    input_builder,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from a single node, plus identity and motion labels.

    Args:
        graph: the trained prediction graph
        loader: DataLoader yielding dicts with frame1, frame2, identity_label, motion_label
        node_id: which node to extract from
        input_builder: DualStreamInputBuilder to create per-node inputs
        device: device string

    Returns:
        features: [N, latent_dim]
        identity_labels: [N]
        motion_labels: [N]
    """
    graph.eval()
    all_features = []
    all_identity = []
    all_motion = []

    for batch in loader:
        frame1 = batch["frame1"].to(device)
        frame2 = batch["frame2"].to(device)
        identity_labels = batch["identity_label"]
        motion_labels = batch["motion_label"]

        inputs = input_builder.build_inputs(frame1, frame2)
        node_input = inputs[node_id]
        latent = graph.get_node(node_id)(node_input)

        all_features.append(latent.cpu())
        all_identity.append(identity_labels)
        all_motion.append(motion_labels)

    return (
        torch.cat(all_features),
        torch.cat(all_identity),
        torch.cat(all_motion),
    )


def evaluate_stream_specialization(
    graph: PredictionGraph,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_builder,
    num_identity_classes: int,
    num_motion_classes: int,
    device: str = "cuda",
    probe_epochs: int = 50,
) -> dict[str, float]:
    """Full ventral/dorsal specialization evaluation.

    For each node, trains both an identity probe and a motion probe,
    then computes a selectivity index.

    Returns:
        Dict with per-node identity_acc, motion_acc, selectivity_index,
        plus stream-level aggregates.
    """
    results = {}

    region_names = {v["node_id"]: k for k, v in DUAL_STREAM_REGIONS.items()}

    for node_id in graph.node_ids:
        name = region_names.get(node_id, f"node_{node_id}")
        logger.info(f"Probing node {node_id} ({name})...")

        # Extract features
        train_feats, train_id, train_motion = _extract_node_features(
            graph, train_loader, node_id, input_builder, device,
        )
        val_feats, val_id, val_motion = _extract_node_features(
            graph, val_loader, node_id, input_builder, device,
        )

        # Identity probe
        id_acc = _train_probe(
            train_feats, train_id, val_feats, val_id,
            num_identity_classes, device, probe_epochs,
        )
        results[f"identity_acc/{name}"] = id_acc

        # Motion probe
        mot_acc = _train_probe(
            train_feats, train_motion, val_feats, val_motion,
            num_motion_classes, device, probe_epochs,
        )
        results[f"motion_acc/{name}"] = mot_acc

        # Selectivity index: positive = identity-biased, negative = motion-biased
        denom = id_acc + mot_acc
        if denom > 0:
            selectivity = (id_acc - mot_acc) / denom
        else:
            selectivity = 0.0
        results[f"selectivity/{name}"] = selectivity

        logger.info(
            f"  {name}: identity={id_acc:.4f}, motion={mot_acc:.4f}, "
            f"selectivity={selectivity:+.4f}"
        )

    # Stream-level aggregates
    ventral_id = [results[f"identity_acc/{region_names[n]}"] for n in VENTRAL_NODE_IDS if n in graph.node_ids]
    ventral_mot = [results[f"motion_acc/{region_names[n]}"] for n in VENTRAL_NODE_IDS if n in graph.node_ids]
    dorsal_id = [results[f"identity_acc/{region_names[n]}"] for n in DORSAL_NODE_IDS if n in graph.node_ids]
    dorsal_mot = [results[f"motion_acc/{region_names[n]}"] for n in DORSAL_NODE_IDS if n in graph.node_ids]

    if ventral_id:
        results["stream/ventral_identity_acc"] = sum(ventral_id) / len(ventral_id)
        results["stream/ventral_motion_acc"] = sum(ventral_mot) / len(ventral_mot)
    if dorsal_id:
        results["stream/dorsal_identity_acc"] = sum(dorsal_id) / len(dorsal_id)
        results["stream/dorsal_motion_acc"] = sum(dorsal_mot) / len(dorsal_mot)

    # The key test: does ventral beat dorsal on identity, and vice versa for motion?
    if ventral_id and dorsal_id:
        results["stream/ventral_identity_advantage"] = (
            results["stream/ventral_identity_acc"] - results["stream/dorsal_identity_acc"]
        )
        results["stream/dorsal_motion_advantage"] = (
            results["stream/dorsal_motion_acc"] - results["stream/ventral_motion_acc"]
        )
        # Double dissociation: both advantages positive = specialization
        results["stream/double_dissociation"] = (
            results["stream/ventral_identity_advantage"] > 0
            and results["stream/dorsal_motion_advantage"] > 0
        )

    return results
