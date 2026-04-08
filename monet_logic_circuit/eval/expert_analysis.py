"""Expert population analysis: activation frequencies, clustering, diagnostics."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def compute_activation_frequencies(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
) -> dict[str, float]:
    """Count how often the router selects each expert across the dataset.

    Hooks into the router/gating modules to record selection decisions.

    Args:
        model: Monet model with router modules.
        dataloader: DataLoader yielding batches with 'input_ids'.
        device: Device for inference.

    Returns:
        Dict mapping expert name (e.g., 'layer3_expert7') -> frequency (0-1).
    """
    from monet_logic_circuit.models.monet_loader import get_router_modules

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Track per-layer expert selection counts
    selection_counts: dict[str, np.ndarray] = {}
    total_tokens = 0
    handles = []

    routers = get_router_modules(model)

    for router_name, router_module in routers:
        # Parse layer index from router name
        layer_idx = _extract_layer_idx(router_name)

        def make_hook(lidx):
            def hook(module, input, output):
                # Router output is typically (routing_weights, selected_experts)
                # or just the gating logits. Handle both patterns.
                if isinstance(output, tuple) and len(output) >= 2:
                    selected = output[1]  # Expert indices
                else:
                    selected = output.argmax(dim=-1) if output.dim() > 1 else output

                key = f"layer{lidx}"
                if key not in selection_counts:
                    # Will be initialized on first call when we know num_experts
                    num_experts = output.shape[-1] if output.dim() > 1 else int(selected.max()) + 1
                    selection_counts[key] = np.zeros(num_experts)

                for idx in selected.flatten().cpu().numpy():
                    selection_counts[key][idx] += 1

            return hook

        handle = router_module.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing activation frequencies"):
            input_ids = batch["input_ids"].to(device)
            model(input_ids)
            total_tokens += input_ids.numel()

    for h in handles:
        h.remove()

    # Normalize to frequencies
    frequencies = {}
    for layer_key, counts in selection_counts.items():
        for expert_idx, count in enumerate(counts):
            name = f"{layer_key}_expert{expert_idx}"
            frequencies[name] = float(count / total_tokens) if total_tokens > 0 else 0.0

    return frequencies


def cluster_experts(
    expert_features: np.ndarray,
    num_clusters: int | str = "auto",
    max_clusters: int = 20,
) -> tuple[np.ndarray, int]:
    """Cluster experts by their feature vectors (input/output statistics).

    Args:
        expert_features: (num_experts, feature_dim) array of per-expert features.
        num_clusters: Number of clusters, or 'auto' for elbow method.
        max_clusters: Maximum clusters to try for elbow method.

    Returns:
        Tuple of (cluster_labels, optimal_k).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(expert_features) < 2:
        return np.zeros(len(expert_features), dtype=int), 1

    # Normalize features
    mean = expert_features.mean(axis=0)
    std = expert_features.std(axis=0) + 1e-10
    normalized = (expert_features - mean) / std

    distances = pdist(normalized, metric="euclidean")
    Z = linkage(distances, method="ward")

    if num_clusters == "auto":
        # Elbow method on within-cluster variance
        max_k = min(max_clusters, len(expert_features) - 1)
        inertias = []
        for k in range(1, max_k + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            inertia = _compute_inertia(normalized, labels)
            inertias.append(inertia)

        num_clusters = _find_elbow(inertias) + 1  # 1-indexed

    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
    return labels - 1, num_clusters  # 0-indexed


def analyze_expert_population(
    experts,
    trace_store,
    frequencies: Optional[dict[str, float]] = None,
) -> dict:
    """Run full expert population analysis.

    Args:
        experts: ExpertPopulation instance.
        trace_store: ExpertTraceStore with cached calibration traces.
        frequencies: Optional pre-computed activation frequencies.

    Returns:
        Dict with analysis results: stats summary, cluster assignments, etc.
    """
    from monet_logic_circuit.models.expert_wrapper import ExpertPopulation

    # Compute per-expert statistics from traces
    feature_list = []
    for expert in experts:
        if trace_store.has_traces(expert.name):
            inputs, outputs = trace_store.load_traces(expert.name)
            expert.compute_input_stats(inputs)
            expert.compute_output_stats(outputs)

            if frequencies and expert.name in frequencies:
                expert.stats.activation_frequency = frequencies[expert.name]

            # Build feature vector for clustering
            features = np.concatenate([
                expert.stats.input_mean[:10] if expert.stats.input_mean is not None else np.zeros(10),
                expert.stats.output_mean[:10] if expert.stats.output_mean is not None else np.zeros(10),
                [expert.stats.input_effective_rank],
                [expert.stats.activation_frequency],
            ])
            feature_list.append(features)

    # Cluster
    if feature_list:
        feature_array = np.stack(feature_list)
        labels, k = cluster_experts(feature_array)
        for i, expert in enumerate(experts):
            if i < len(labels):
                expert.stats.cluster_id = int(labels[i])
    else:
        k = 0

    summary = experts.get_stats_summary()
    summary["num_clusters"] = k

    return summary


def _extract_layer_idx(name: str) -> int:
    """Extract layer index from a module name like 'model.layers.3.gate'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "layer", "blocks", "block") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return 0


def _compute_inertia(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute within-cluster sum of squared distances."""
    inertia = 0.0
    for k in np.unique(labels):
        cluster = data[labels == k]
        center = cluster.mean(axis=0)
        inertia += ((cluster - center) ** 2).sum()
    return inertia


def _find_elbow(inertias: list[float]) -> int:
    """Find elbow point in an inertia curve using max distance to line."""
    if len(inertias) <= 2:
        return 0

    n = len(inertias)
    coords = np.array([(i, inertias[i]) for i in range(n)])

    # Line from first to last point
    line_vec = coords[-1] - coords[0]
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    # Distance from each point to the line
    distances = []
    for i in range(n):
        vec = coords[i] - coords[0]
        proj = np.dot(vec, line_unit)
        proj_point = coords[0] + proj * line_unit
        dist = np.linalg.norm(coords[i] - proj_point)
        distances.append(dist)

    return int(np.argmax(distances))
