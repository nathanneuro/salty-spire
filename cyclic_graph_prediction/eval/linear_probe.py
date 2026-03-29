"""
Linear probing for evaluating learned representations (Experiments 1-5).

Trains a linear classifier on frozen representations to measure quality.
Supports probing individual nodes and concatenated graph representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.graph import PredictionGraph


class LinearProbe(nn.Module):
    """Linear classifier on top of frozen representations."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@torch.no_grad()
def extract_features(
    graph: PredictionGraph,
    loader: DataLoader,
    device: str = "cuda",
    node_ids: list[int] | None = None,
    message_passing_rounds: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract concatenated features from graph nodes.

    Args:
        graph: trained prediction graph
        loader: data loader
        device: device
        node_ids: which nodes to concatenate (None = all)
        message_passing_rounds: if > 0, use iterative message passing (Exp 4)

    Returns:
        features: [N, concat_dim]
        labels: [N]
    """
    graph.eval()
    if node_ids is None:
        node_ids = graph.node_ids

    all_features = []
    all_labels = []

    for image, masked_views, labels in loader:
        inputs = {nid: view.to(device) for nid, view in masked_views.items()}

        if message_passing_rounds > 0:
            latents = graph.message_pass(inputs, num_rounds=message_passing_rounds)
        else:
            latents = graph.encode_all(inputs)

        # Concatenate selected nodes
        concat = torch.cat([latents[nid] for nid in node_ids], dim=-1)
        all_features.append(concat.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def evaluate_linear_probe(
    graph: PredictionGraph,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    node_ids: list[int] | None = None,
    message_passing_rounds: int = 0,
    epochs: int = 50,
    lr: float = 0.01,
) -> dict[str, float]:
    """Train and evaluate a linear probe on graph representations.

    Returns:
        Dict with train_acc, val_acc, and per-node val_acc.
    """
    # Extract features
    train_feats, train_labels = extract_features(
        graph, train_loader, device, node_ids, message_passing_rounds
    )
    val_feats, val_labels = extract_features(
        graph, val_loader, device, node_ids, message_passing_rounds
    )

    feat_dim = train_feats.shape[1]
    probe = LinearProbe(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop on extracted features
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    batch_size = 256
    n = train_feats.shape[0]

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = probe(train_feats[idx])
            loss = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_feats)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()

        train_logits = probe(train_feats)
        train_preds = train_logits.argmax(dim=1)
        train_acc = (train_preds == train_labels).float().mean().item()

    results = {"train_acc": train_acc, "val_acc": val_acc}

    # Per-node probing
    if node_ids is None:
        node_ids = graph.node_ids
    for nid in node_ids:
        node_val_feats, node_val_labels = extract_features(
            graph, val_loader, device, [nid], message_passing_rounds
        )
        node_train_feats, node_train_labels = extract_features(
            graph, train_loader, device, [nid], message_passing_rounds
        )
        node_dim = node_val_feats.shape[1]
        node_probe = LinearProbe(node_dim, num_classes).to(device)
        node_opt = torch.optim.SGD(node_probe.parameters(), lr=lr, momentum=0.9)

        node_train_feats = node_train_feats.to(device)
        node_train_labels = node_train_labels.to(device)
        node_val_feats = node_val_feats.to(device)
        node_val_labels = node_val_labels.to(device)

        for epoch in range(epochs):
            node_probe.train()
            perm = torch.randperm(node_train_feats.shape[0], device=device)
            for i in range(0, node_train_feats.shape[0], batch_size):
                idx = perm[i : i + batch_size]
                logits = node_probe(node_train_feats[idx])
                loss = criterion(logits, node_train_labels[idx])
                node_opt.zero_grad()
                loss.backward()
                node_opt.step()

        node_probe.eval()
        with torch.no_grad():
            node_logits = node_probe(node_val_feats)
            node_acc = (node_logits.argmax(1) == node_val_labels).float().mean().item()
        results[f"val_acc/node_{nid}"] = node_acc

    return results
