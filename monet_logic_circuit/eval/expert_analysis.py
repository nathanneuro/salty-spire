"""Half-expert population analysis: activation frequencies, clustering, diagnostics.

Monet has a product-key decomposition with 2N half-experts per layer (two
axes of N each). Activation frequencies are reported per (layer, axis,
half_expert_idx), matching the HalfExpertWrapper naming scheme.
"""

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
    """Count how often each product-key router selects each half-expert.

    Hooks into the router/gating modules to record selection decisions.
    Monet has two routers per layer (one per product-key axis); the axis
    for each router is inferred from its name.

    Args:
        model: Monet model with router modules.
        dataloader: DataLoader yielding batches with 'input_ids'.
        device: Device for inference.

    Returns:
        Dict mapping half-expert name (e.g., 'layer3_axis0_he17') -> frequency (0-1).
    """
    from monet_logic_circuit.models.monet_loader import get_router_modules

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Track per-(layer, axis) half-expert selection counts
    selection_counts: dict[tuple[int, int], np.ndarray] = {}
    total_tokens = 0
    handles = []

    routers = get_router_modules(model)

    for router_name, router_module in routers:
        layer_idx = _extract_layer_idx(router_name)
        axis = _extract_axis(router_name)

        def make_hook(lidx, ax):
            def hook(module, input, output):
                # Router output is typically (routing_weights, selected_indices)
                # or just the gating logits. Handle both patterns.
                if isinstance(output, tuple) and len(output) >= 2:
                    selected = output[1]
                else:
                    selected = output.argmax(dim=-1) if output.dim() > 1 else output

                key = (lidx, ax)
                if key not in selection_counts:
                    num_half_experts = (
                        output.shape[-1] if output.dim() > 1
                        else int(selected.max()) + 1
                    )
                    selection_counts[key] = np.zeros(num_half_experts)

                for idx in selected.flatten().cpu().numpy():
                    selection_counts[key][idx] += 1

            return hook

        handle = router_module.register_forward_hook(make_hook(layer_idx, axis))
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
    for (layer_idx, axis), counts in selection_counts.items():
        for he_idx, count in enumerate(counts):
            name = f"layer{layer_idx}_axis{axis}_he{he_idx}"
            frequencies[name] = float(count / total_tokens) if total_tokens > 0 else 0.0

    return frequencies


def cluster_half_experts(
    half_expert_features: np.ndarray,
    num_clusters: int | str = "auto",
    max_clusters: int = 20,
) -> tuple[np.ndarray, int]:
    """Cluster half-experts by their feature vectors (input/output statistics).

    Args:
        half_expert_features: (num_half_experts, feature_dim) feature array.
        num_clusters: Number of clusters, or 'auto' for elbow method.
        max_clusters: Maximum clusters to try for elbow method.

    Returns:
        Tuple of (cluster_labels, optimal_k).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(half_expert_features) < 2:
        return np.zeros(len(half_expert_features), dtype=int), 1

    # Normalize features
    mean = half_expert_features.mean(axis=0)
    std = half_expert_features.std(axis=0) + 1e-10
    normalized = (half_expert_features - mean) / std

    distances = pdist(normalized, metric="euclidean")
    Z = linkage(distances, method="ward")

    if num_clusters == "auto":
        # Elbow method on within-cluster variance
        max_k = min(max_clusters, len(half_expert_features) - 1)
        inertias = []
        for k in range(1, max_k + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            inertia = _compute_inertia(normalized, labels)
            inertias.append(inertia)

        num_clusters = _find_elbow(inertias) + 1  # 1-indexed

    labels = fcluster(Z, t=num_clusters, criterion="maxclust")
    return labels - 1, num_clusters  # 0-indexed


def analyze_expert_population(
    half_experts,
    trace_store,
    frequencies: Optional[dict[str, float]] = None,
) -> dict:
    """Run full half-expert population analysis.

    Args:
        half_experts: HalfExpertPopulation instance.
        trace_store: ExpertTraceStore with cached calibration traces.
        frequencies: Optional pre-computed activation frequencies.

    Returns:
        Dict with analysis results: stats summary, cluster assignments, etc.
    """
    # Compute per-half-expert statistics from traces
    feature_list = []
    for he in half_experts:
        if trace_store.has_traces(he.name):
            inputs, outputs = trace_store.load_traces(he.name)
            he.compute_input_stats(inputs)
            he.compute_output_stats(outputs)

            if frequencies and he.name in frequencies:
                he.stats.activation_frequency = frequencies[he.name]

            # Build feature vector for clustering
            features = np.concatenate([
                he.stats.input_mean[:10] if he.stats.input_mean is not None else np.zeros(10),
                he.stats.output_mean[:10] if he.stats.output_mean is not None else np.zeros(10),
                [he.stats.input_effective_rank],
                [he.stats.activation_frequency],
            ])
            feature_list.append(features)

    # Cluster
    if feature_list:
        feature_array = np.stack(feature_list)
        labels, k = cluster_half_experts(feature_array)
        for i, he in enumerate(half_experts):
            if i < len(labels):
                he.stats.cluster_id = int(labels[i])
    else:
        k = 0

    summary = half_experts.get_stats_summary()
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


def _extract_axis(name: str) -> int:
    """Extract product-key axis (0 or 1) from a router module name.

    Monet's two per-layer routers are typically distinguished by an
    ``axis``/``dim`` index or a ``h`` vs ``v`` suffix. Defaults to 0 when
    no signal is available.
    """
    lowered = name.lower()
    for token in ("axis1", "axis_1", "dim1", "dim_1", "_v_", ".v.", "right", "top"):
        if token in lowered:
            return 1
    for token in ("axis0", "axis_0", "dim0", "dim_0", "_h_", ".h.", "left", "bottom"):
        if token in lowered:
            return 0
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
