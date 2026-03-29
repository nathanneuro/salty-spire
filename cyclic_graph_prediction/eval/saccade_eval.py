"""
Saccade-specific evaluation (Experiment 9).

Tests:
1. Does the integration node improve with more fixations? (saccade integration)
2. Does the Where_Next node learn to predict upcoming foveal content?
3. Does the peripheral node capture scene gist better than the foveal node?
4. Does combining foveal + peripheral outperform either alone? (like the brain)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.graph import PredictionGraph

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_saccade_integration(
    graph: PredictionGraph,
    loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    probe_epochs: int = 50,
) -> dict[str, float]:
    """Evaluate how representation quality improves with more fixations.

    For each number of fixations k=1..num_fixations:
    - Accumulate the integration node's latent across k fixations (mean pooling)
    - Linear probe on the accumulated representation
    - Compare: does more fixations = better accuracy?

    Also probes individual nodes to test specialization:
    - Peripheral: should be best at coarse/scene-level classification
    - Foveal: should be best at fine-grained classification
    - Integration: should be best overall
    """
    graph.eval()
    results = {}

    # Collect per-fixation latents for each node
    fixation_latents = defaultdict(lambda: defaultdict(list))
    all_labels = []

    for batch in loader:
        labels = batch["identity_label"]
        all_labels.append(labels)
        num_fix = batch["foveal_crops"].shape[1]

        for t in range(num_fix):
            peripheral = batch["peripheral"].to(device)
            foveal = batch["foveal_crops"][:, t].to(device)
            foveated = batch["foveated_images"][:, t].to(device)

            inputs = {0: peripheral, 1: foveal, 2: foveated, 3: foveated, 4: foveated, 5: foveated}
            latents = graph.encode_all(inputs)

            for nid, lat in latents.items():
                fixation_latents[t][nid].append(lat.cpu())

    all_labels = torch.cat(all_labels)
    num_fix = len(fixation_latents)

    # Concatenate per-fixation latents
    for t in range(num_fix):
        for nid in fixation_latents[t]:
            fixation_latents[t][nid] = torch.cat(fixation_latents[t][nid])

    region_names = {0: "peripheral", 1: "foveal", 2: "ventral_gist",
                    3: "ventral_detail", 4: "integration", 5: "where_next"}

    # 1. Per-node probe at first fixation
    for nid in graph.node_ids:
        name = region_names.get(nid, f"node_{nid}")
        if 0 in fixation_latents and nid in fixation_latents[0]:
            feats = fixation_latents[0][nid]
            acc = _quick_linear_probe(feats, all_labels, num_classes, device, probe_epochs)
            results[f"single_fixation/{name}"] = acc
            logger.info(f"  {name} (1 fixation): {acc:.4f}")

    # 2. Integration node with cumulative fixations
    if 4 in fixation_latents.get(0, {}):
        logger.info("\nIntegration node accuracy vs. number of fixations:")
        for k in range(1, num_fix + 1):
            # Mean-pool integration latent across first k fixations
            accumulated = torch.stack(
                [fixation_latents[t][4] for t in range(k)]
            ).mean(dim=0)
            acc = _quick_linear_probe(accumulated, all_labels, num_classes, device, probe_epochs)
            results[f"integration_{k}_fixations"] = acc
            logger.info(f"  k={k}: {acc:.4f}")

    # 3. Foveal + peripheral combination vs each alone
    if 0 in fixation_latents.get(0, {}) and 1 in fixation_latents.get(0, {}):
        peripheral_feats = fixation_latents[0][0]
        foveal_feats = fixation_latents[0][1]
        combined = torch.cat([peripheral_feats, foveal_feats], dim=-1)
        acc_combined = _quick_linear_probe(combined, all_labels, num_classes, device, probe_epochs)
        results["combined_foveal_peripheral"] = acc_combined
        logger.info(f"\n  Foveal+Peripheral combined: {acc_combined:.4f}")

    return results


def _quick_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: str,
    epochs: int = 50,
) -> float:
    """Train and eval a linear probe. Returns val accuracy (80/20 split)."""
    n = features.shape[0]
    split = int(n * 0.8)
    perm = torch.randperm(n)

    train_feats = features[perm[:split]].to(device)
    train_labels = labels[perm[:split]].to(device)
    val_feats = features[perm[split:]].to(device)
    val_labels = labels[perm[split:]].to(device)

    probe = nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    batch_size = 256
    for _ in range(epochs):
        probe.train()
        idx = torch.randperm(split, device=device)
        for i in range(0, split, batch_size):
            batch_idx = idx[i : i + batch_size]
            logits = probe(train_feats[batch_idx])
            loss = criterion(logits, train_labels[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_feats)
        acc = (val_logits.argmax(1) == val_labels).float().mean().item()
    return acc
