"""Wrappers for individual Monet experts to provide a uniform interface
for tracing, quantization, and conversion."""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


@dataclass
class ExpertStats:
    """Per-expert statistics collected during analysis."""

    layer_idx: int = 0
    expert_idx: int = 0
    name: str = ""

    # Activation frequency (fraction of tokens routed to this expert)
    activation_frequency: float = 0.0

    # Input distribution
    input_mean: Optional[np.ndarray] = None
    input_var: Optional[np.ndarray] = None
    input_effective_rank: float = 0.0

    # Output distribution
    output_mean: Optional[np.ndarray] = None
    output_var: Optional[np.ndarray] = None

    # Quantization diagnostics
    reconstruction_error: float = 0.0  # vs float expert
    effective_output_cardinality: int = 0  # distinct outputs on calibration set

    # Cluster assignment
    cluster_id: int = -1


class ExpertWrapper(nn.Module):
    """Wraps a single Monet expert FFN for uniform I/O tracing and replacement.

    Provides hooks to intercept inputs/outputs for calibration trace collection,
    and a replacement interface for swapping in quantized or circuit-based
    implementations.
    """

    def __init__(self, expert_module: nn.Module, layer_idx: int, expert_idx: int):
        super().__init__()
        self.expert = expert_module
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.name = f"layer{layer_idx}_expert{expert_idx}"

        self.stats = ExpertStats(
            layer_idx=layer_idx, expert_idx=expert_idx, name=self.name
        )

        # Trace buffers (populated during calibration)
        self._collecting_traces = False
        self._input_traces: list[torch.Tensor] = []
        self._output_traces: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._collecting_traces:
            self._input_traces.append(x.detach().cpu())

        out = self.expert(x)

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
        patterns. Low cardinality means the expert has collapsed to few modes.
        """
        flat = outputs.reshape(-1, outputs.shape[-1]).float()
        # Discretize to tolerance grid
        quantized = (flat / tolerance).round()
        # Count unique rows
        unique = torch.unique(quantized, dim=0)
        cardinality = len(unique)
        self.stats.effective_output_cardinality = cardinality
        return cardinality


class ExpertPopulation:
    """Manages the full set of experts across all layers."""

    def __init__(self, experts: list[ExpertWrapper]):
        self.experts = experts
        self._by_name = {e.name: e for e in experts}

    def __len__(self) -> int:
        return len(self.experts)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self.experts[key]

    def __iter__(self):
        return iter(self.experts)

    def get_layer(self, layer_idx: int) -> list[ExpertWrapper]:
        return [e for e in self.experts if e.layer_idx == layer_idx]

    def sort_by_reconstruction_error(self, ascending: bool = True) -> list[ExpertWrapper]:
        return sorted(
            self.experts,
            key=lambda e: e.stats.reconstruction_error,
            reverse=not ascending,
        )

    def sort_by_activation_frequency(self, ascending: bool = False) -> list[ExpertWrapper]:
        return sorted(
            self.experts,
            key=lambda e: e.stats.activation_frequency,
            reverse=not ascending,
        )

    def get_stats_summary(self) -> dict:
        """Return summary statistics across the expert population."""
        errors = [e.stats.reconstruction_error for e in self.experts]
        freqs = [e.stats.activation_frequency for e in self.experts]
        cards = [
            e.stats.effective_output_cardinality
            for e in self.experts
            if e.stats.effective_output_cardinality > 0
        ]
        return {
            "num_experts": len(self.experts),
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
