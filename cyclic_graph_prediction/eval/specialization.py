"""
Emergent specialization analysis (Experiment 3).

Measures whether different nodes in the graph learn to represent different
features despite identical architectures and inputs, driven only by
graph topology asymmetry.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import DataLoader

from ..models.graph import PredictionGraph


@torch.no_grad()
def measure_specialization(
    graph: PredictionGraph,
    loader: DataLoader,
    device: str = "cuda",
    num_samples: int = 2048,
) -> dict[str, float]:
    """Analyze representational specialization across graph nodes.

    Metrics:
    - CKA similarity between all pairs of nodes (low = specialized)
    - Per-node feature variance distribution (concentrated = specialized)
    - Representation redundancy (how much info is shared vs unique)

    Returns:
        Dict of specialization metrics.
    """
    graph.eval()

    # Collect latents from all nodes
    node_latents = {nid: [] for nid in graph.node_ids}
    count = 0

    for image, masked_views, labels in loader:
        if count >= num_samples:
            break
        inputs = {nid: view.to(device) for nid, view in masked_views.items()}
        latents = graph.encode_all(inputs)
        for nid, lat in latents.items():
            node_latents[nid].append(lat.cpu())
        count += image.shape[0]

    for nid in graph.node_ids:
        node_latents[nid] = torch.cat(node_latents[nid])[:num_samples]

    results = {}

    # Pairwise CKA (linear) between node representations
    cka_values = []
    ids = graph.node_ids
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            X = node_latents[ids[i]].float()
            Y = node_latents[ids[j]].float()
            cka = _linear_cka(X, Y)
            results[f"cka/node_{ids[i]}_vs_{ids[j]}"] = cka
            cka_values.append(cka)

    results["cka/mean_pairwise"] = float(np.mean(cka_values))

    # Per-node effective dimensionality (via PCA variance explained)
    for nid in graph.node_ids:
        X = node_latents[nid].float()
        X = X - X.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(X)
        var_explained = (s ** 2) / (s ** 2).sum()
        # How many components for 90% variance
        cumvar = var_explained.cumsum(0)
        dim90 = (cumvar < 0.9).sum().item() + 1
        results[f"effective_dim_90/node_{nid}"] = dim90

    return results


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear Centered Kernel Alignment between two representation matrices.

    CKA measures similarity of representations invariant to rotation and scaling.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = _hsic(X, Y)
    hsic_xx = _hsic(X, X)
    hsic_yy = _hsic(Y, Y)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def _hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion with linear kernel."""
    n = X.shape[0]
    K = X @ X.T
    L = Y @ Y.T
    # Center the kernels
    H = torch.eye(n, device=X.device) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return (Kc * Lc).sum() / ((n - 1) ** 2)
