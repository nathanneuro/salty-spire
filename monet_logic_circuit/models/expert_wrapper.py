"""Wrappers for individual Monet half-experts to provide a uniform interface
for tracing, quantization, and conversion.

Monet's product-key decomposition means each layer has 2N half-experts
(N = 512 per product-key axis in the released checkpoints), not N^2
effective experts. Conversion operates at the half-expert granularity:
each half-expert is a small independent function of a slice of the
residual stream (VD) and is converted to its own logic circuit.
Effective-expert outputs are formed by composing half-experts according
to the decomposition rule (additive sum over axes for VD).

See docs/model_selection.md for the rationale.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import numpy as np


# Axis identifies which of the two product-key dimensions a half-expert
# lives on. For VD these correspond to the left and right input slices;
# for HD they correspond to the bottom and top compositional halves.
Axis = Literal[0, 1]


@dataclass
class HalfExpertStats:
    """Per-half-expert statistics collected during analysis."""

    layer_idx: int = 0
    axis: Axis = 0
    half_expert_idx: int = 0  # 0..N-1 along the given axis
    name: str = ""

    # Activation frequency (fraction of tokens routed to this half-expert
    # by the corresponding product-key router).
    activation_frequency: float = 0.0

    # Input distribution (over the slice this half-expert actually sees).
    input_mean: Optional[np.ndarray] = None
    input_var: Optional[np.ndarray] = None
    input_effective_rank: float = 0.0

    # Output distribution
    output_mean: Optional[np.ndarray] = None
    output_var: Optional[np.ndarray] = None

    # Quantization diagnostics
    reconstruction_error: float = 0.0       # vs float half-expert
    effective_output_cardinality: int = 0   # distinct outputs on calibration set

    # Cluster assignment
    cluster_id: int = -1


class HalfExpertWrapper(nn.Module):
    """Wraps a single Monet half-expert for uniform I/O tracing and replacement.

    Provides hooks to intercept inputs/outputs for calibration trace collection,
    and a replacement interface for swapping in quantized or circuit-based
    implementations. A half-expert is identified by the triple
    (layer_idx, axis, half_expert_idx).
    """

    def __init__(
        self,
        half_expert_module: nn.Module,
        layer_idx: int,
        axis: Axis,
        half_expert_idx: int,
    ):
        super().__init__()
        self.half_expert = half_expert_module
        self.layer_idx = layer_idx
        self.axis = axis
        self.half_expert_idx = half_expert_idx
        self.name = f"layer{layer_idx}_axis{axis}_he{half_expert_idx}"

        self.stats = HalfExpertStats(
            layer_idx=layer_idx,
            axis=axis,
            half_expert_idx=half_expert_idx,
            name=self.name,
        )

        # Trace buffers (populated during calibration)
        self._collecting_traces = False
        self._input_traces: list[torch.Tensor] = []
        self._output_traces: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._collecting_traces:
            self._input_traces.append(x.detach().cpu())

        out = self.half_expert(x)

        if self._collecting_traces:
            self._output_traces.append(out.detach().cpu())

        return out

    def start_tracing(self):
        self._collecting_traces = True
        self._input_traces.clear()
        self._output_traces.clear()

    def stop_tracing(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stop tracing and return concatenated (inputs, outputs)."""
        self._collecting_traces = False
        inputs = torch.cat(self._input_traces, dim=0) if self._input_traces else torch.empty(0)
        outputs = torch.cat(self._output_traces, dim=0) if self._output_traces else torch.empty(0)
        self._input_traces.clear()
        self._output_traces.clear()
        return inputs, outputs

    def compute_input_stats(self, inputs: torch.Tensor):
        """Compute and store input distribution statistics."""
        flat = inputs.reshape(-1, inputs.shape[-1]).float()
        self.stats.input_mean = flat.mean(dim=0).numpy()
        self.stats.input_var = flat.var(dim=0).numpy()

        # Effective rank via singular values
        try:
            _, s, _ = torch.linalg.svd(flat[:min(len(flat), 4096)], full_matrices=False)
            s_norm = s / s.sum()
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            self.stats.input_effective_rank = float(torch.exp(entropy))
        except Exception:
            self.stats.input_effective_rank = float(flat.shape[-1])

    def compute_output_stats(self, outputs: torch.Tensor):
        """Compute and store output distribution statistics."""
        flat = outputs.reshape(-1, outputs.shape[-1]).float()
        self.stats.output_mean = flat.mean(dim=0).numpy()
        self.stats.output_var = flat.var(dim=0).numpy()

    def compute_reconstruction_error(
        self, reference_outputs: torch.Tensor, actual_outputs: torch.Tensor
    ) -> float:
        """Compute normalized MSE between reference and actual outputs."""
        ref = reference_outputs.float()
        act = actual_outputs.float()
        mse = ((ref - act) ** 2).mean()
        ref_var = ref.var()
        nmse = float(mse / (ref_var + 1e-10))
        self.stats.reconstruction_error = nmse
        return nmse

    def compute_effective_output_cardinality(
        self, outputs: torch.Tensor, tolerance: float = 1e-4
    ) -> int:
        """Count effectively distinct output patterns.

        Quantizes outputs to a grid defined by tolerance, then counts unique
        patterns. Low cardinality means the half-expert has collapsed to few modes.
        """
        flat = outputs.reshape(-1, outputs.shape[-1]).float()
        # Discretize to tolerance grid
        quantized = (flat / tolerance).round()
        # Count unique rows
        unique = torch.unique(quantized, dim=0)
        cardinality = len(unique)
        self.stats.effective_output_cardinality = cardinality
        return cardinality


class HalfExpertPopulation:
    """Manages the full set of half-experts across all layers and both axes."""

    def __init__(self, half_experts: list[HalfExpertWrapper]):
        self.half_experts = half_experts
        self._by_name = {h.name: h for h in half_experts}

    def __len__(self) -> int:
        return len(self.half_experts)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self.half_experts[key]

    def __iter__(self):
        return iter(self.half_experts)

    def get_layer(self, layer_idx: int) -> list[HalfExpertWrapper]:
        return [h for h in self.half_experts if h.layer_idx == layer_idx]

    def get_axis(self, layer_idx: int, axis: Axis) -> list[HalfExpertWrapper]:
        return [
            h for h in self.half_experts
            if h.layer_idx == layer_idx and h.axis == axis
        ]

    def sort_by_reconstruction_error(self, ascending: bool = True) -> list[HalfExpertWrapper]:
        return sorted(
            self.half_experts,
            key=lambda h: h.stats.reconstruction_error,
            reverse=not ascending,
        )

    def sort_by_activation_frequency(self, ascending: bool = False) -> list[HalfExpertWrapper]:
        return sorted(
            self.half_experts,
            key=lambda h: h.stats.activation_frequency,
            reverse=not ascending,
        )

    def get_stats_summary(self) -> dict:
        """Return summary statistics across the half-expert population."""
        errors = [h.stats.reconstruction_error for h in self.half_experts]
        freqs = [h.stats.activation_frequency for h in self.half_experts]
        cards = [
            h.stats.effective_output_cardinality
            for h in self.half_experts
            if h.stats.effective_output_cardinality > 0
        ]
        layer_axis_counts: dict[tuple[int, int], int] = {}
        for h in self.half_experts:
            key = (h.layer_idx, int(h.axis))
            layer_axis_counts[key] = layer_axis_counts.get(key, 0) + 1

        return {
            "num_half_experts": len(self.half_experts),
            "num_layer_axis_groups": len(layer_axis_counts),
            "reconstruction_error": {
                "mean": float(np.mean(errors)) if errors else 0,
                "std": float(np.std(errors)) if errors else 0,
                "max": float(np.max(errors)) if errors else 0,
                "p90": float(np.percentile(errors, 90)) if errors else 0,
                "p99": float(np.percentile(errors, 99)) if errors else 0,
            },
            "activation_frequency": {
                "mean": float(np.mean(freqs)) if freqs else 0,
                "std": float(np.std(freqs)) if freqs else 0,
                "min": float(np.min(freqs)) if freqs else 0,
                "max": float(np.max(freqs)) if freqs else 0,
            },
            "effective_output_cardinality": {
                "mean": float(np.mean(cards)) if cards else 0,
                "median": float(np.median(cards)) if cards else 0,
                "min": int(np.min(cards)) if cards else 0,
                "max": int(np.max(cards)) if cards else 0,
            },
        }
