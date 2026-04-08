"""Aggressive quantization (2-bit / 1.58-bit ternary) with optional QAT.

This is where the interesting tradeoffs live. The ternary variant is
particularly important because ternary weights + binary activations
= a discrete function structurally close to a logic circuit.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizationSweepConfig:
    """Configuration for a quantization sweep across bit widths and schemes."""

    bits_options: list[float] = field(default_factory=lambda: [1.0, 1.58, 2.0])
    scope_options: list[str] = field(
        default_factory=lambda: ["weights_only", "weights_and_activations"]
    )
    scale_options: list[str] = field(
        default_factory=lambda: ["per_tensor", "per_channel", "per_expert"]
    )

    def num_configs(self) -> int:
        return len(self.bits_options) * len(self.scope_options) * len(self.scale_options)

    def iter_configs(self):
        for bits in self.bits_options:
            for scope in self.scope_options:
                for scale in self.scale_options:
                    yield {"bits": bits, "scope": scope, "scale_granularity": scale}


class TernaryQuantizer(nn.Module):
    """BitNet-style ternary quantization: weights in {-1, 0, +1}.

    Implements quantization-aware training via straight-through estimator.
    Each expert gets its own learned scale factor.
    """

    def __init__(self, weight: torch.Tensor, scale_per_expert: bool = True):
        super().__init__()
        self.register_buffer("shape", torch.tensor(weight.shape))

        # Learned scale factor (one per output channel if per-expert)
        if scale_per_expert:
            self.scale = nn.Parameter(weight.abs().mean(dim=-1, keepdim=True))
        else:
            self.scale = nn.Parameter(torch.tensor(weight.abs().mean()))

        # Store float weights for STE during training
        self.weight = nn.Parameter(weight.clone())

    def quantize(self) -> torch.Tensor:
        """Quantize weights to ternary {-scale, 0, +scale}."""
        # Threshold at scale * 0.5 for ternary buckets
        threshold = self.scale * 0.5
        ternary = torch.zeros_like(self.weight)
        ternary[self.weight > threshold] = 1.0
        ternary[self.weight < -threshold] = -1.0
        return ternary * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with STE: use quantized weights but pass gradients through."""
        q_weight = self.quantize()
        if self.training:
            # Straight-through estimator
            q_weight = self.weight + (q_weight - self.weight).detach()
        return F.linear(x, q_weight)

    def get_ternary_weights(self) -> torch.Tensor:
        """Return the raw ternary values {-1, 0, +1} without scale."""
        threshold = self.scale * 0.5
        ternary = torch.zeros_like(self.weight)
        ternary[self.weight > threshold] = 1.0
        ternary[self.weight < -threshold] = -1.0
        return ternary


class BinaryQuantizer(nn.Module):
    """1-bit quantization: weights in {-1, +1}."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(weight.abs().mean(dim=-1, keepdim=True))
        self.weight = nn.Parameter(weight.clone())

    def quantize(self) -> torch.Tensor:
        return torch.sign(self.weight) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantize()
        if self.training:
            q_weight = self.weight + (q_weight - self.weight).detach()
        return F.linear(x, q_weight)


class TwoBitQuantizer(nn.Module):
    """2-bit quantization: weights in {-1, -1/3, +1/3, +1} * scale."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(weight.abs().mean(dim=-1, keepdim=True))
        self.weight = nn.Parameter(weight.clone())

    def quantize(self) -> torch.Tensor:
        normalized = self.weight / (self.scale + 1e-10)
        # Quantize to 4 levels
        boundaries = torch.tensor([-2 / 3, 0.0, 2 / 3], device=self.weight.device)
        levels = torch.tensor([-1.0, -1 / 3, 1 / 3, 1.0], device=self.weight.device)
        indices = torch.bucketize(normalized, boundaries)
        return levels[indices] * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantize()
        if self.training:
            q_weight = self.weight + (q_weight - self.weight).detach()
        return F.linear(x, q_weight)


def apply_aggressive_quantization(
    model: nn.Module,
    bits: float = 1.58,
    scope: str = "weights_only",
    scale_granularity: str = "per_expert",
) -> nn.Module:
    """Replace expert FFN linear layers with aggressively quantized versions.

    Args:
        model: Monet model to quantize.
        bits: Bit width (1.0, 1.58, or 2.0).
        scope: 'weights_only' or 'weights_and_activations'.
        scale_granularity: 'per_tensor', 'per_channel', or 'per_expert'.

    Returns:
        Model with quantized expert layers (modified in place).
    """
    from monet_logic_circuit.models.monet_loader import get_expert_modules

    quantizer_cls = _get_quantizer_class(bits)
    experts = get_expert_modules(model)

    for name, expert_module in experts:
        _replace_linear_layers(expert_module, quantizer_cls, scale_granularity)

    return model


def run_quantization_sweep(
    model: nn.Module,
    tokenizer,
    sweep_config: QuantizationSweepConfig,
    calibration_data,
    eval_fn,
    fine_tune_fn=None,
) -> list[dict]:
    """Run a sweep over quantization configurations.

    Args:
        model: Base model (will be deepcopied for each config).
        tokenizer: Model tokenizer.
        sweep_config: Sweep configuration.
        calibration_data: Calibration dataset.
        eval_fn: Function(model) -> dict of eval metrics.
        fine_tune_fn: Optional function(model, calibration_data) -> model
            for quantization-aware fine-tuning.

    Returns:
        List of dicts, one per config, with config params and eval results.
    """
    import copy

    results = []
    for config in sweep_config.iter_configs():
        model_copy = copy.deepcopy(model)

        # Apply quantization
        quantized = apply_aggressive_quantization(
            model_copy,
            bits=config["bits"],
            scope=config["scope"],
            scale_granularity=config["scale_granularity"],
        )

        # Optional QAT fine-tuning
        if fine_tune_fn is not None:
            quantized = fine_tune_fn(quantized, calibration_data)

        # Evaluate
        metrics = eval_fn(quantized)

        result = {**config, **metrics}
        results.append(result)

        del model_copy, quantized

    return results


def compute_effective_output_cardinality(
    model: nn.Module,
    trace_store,
    tolerance: float = 1e-4,
) -> dict[str, int]:
    """Measure effective number of distinct outputs per expert after quantization.

    Low cardinality indicates the expert has collapsed to few modes and is
    a candidate for extreme compression or removal.
    """
    from monet_logic_circuit.models.monet_loader import get_expert_modules

    cardinalities = {}
    experts = dict(get_expert_modules(model))

    for expert_name in trace_store.list_experts():
        if expert_name not in experts:
            continue

        inputs, _ = trace_store.load_traces(expert_name)
        expert = experts[expert_name]

        with torch.no_grad():
            outputs = expert(inputs.to(next(expert.parameters()).device)).cpu()

        flat = outputs.reshape(-1, outputs.shape[-1]).float()
        quantized = (flat / tolerance).round()
        unique = torch.unique(quantized, dim=0)
        cardinalities[expert_name] = len(unique)

    return cardinalities


def _get_quantizer_class(bits: float):
    """Return the appropriate quantizer class for the given bit width."""
    if bits <= 1.0:
        return BinaryQuantizer
    elif bits <= 1.58:
        return TernaryQuantizer
    elif bits <= 2.0:
        return TwoBitQuantizer
    else:
        raise ValueError(f"Bits {bits} too high for aggressive quantization. Use gentle quantization instead.")


def _replace_linear_layers(
    module: nn.Module, quantizer_cls, scale_granularity: str
):
    """Recursively replace nn.Linear layers in a module with quantized versions."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            per_expert = scale_granularity == "per_expert"
            quantized = quantizer_cls(
                child.weight.data,
                **({"scale_per_expert": per_expert} if quantizer_cls == TernaryQuantizer else {}),
            )
            setattr(module, name, quantized)
        else:
            _replace_linear_layers(child, quantizer_cls, scale_granularity)
