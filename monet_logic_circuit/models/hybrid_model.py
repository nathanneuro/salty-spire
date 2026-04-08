"""Hybrid Monet model with logic-circuit experts replacing float experts.

Used in Steps 3c and 3d: the router and attention remain in float,
while experts dispatch to either logic-circuit implementations or
quantized-float fallbacks.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ExpertImplementation:
    """Tracks which implementation backs each expert slot."""

    layer_idx: int
    expert_idx: int
    method: str  # "exact_circuit", "learned_circuit", "quantized_float"
    circuit_size: int = 0  # Number of gates (0 for float fallback)
    reconstruction_error: float = 0.0


class InputBinarizer(nn.Module):
    """Binarizes the residual stream input to expert circuits.

    Supports strict binary (sign), stochastic binarization, and learned
    per-channel thresholds. Multi-bit mode quantizes to n bits instead
    of strict binary.
    """

    def __init__(self, hidden_dim: int, method: str = "sign", bits: int = 1):
        super().__init__()
        self.method = method
        self.bits = bits

        if method == "learned_threshold":
            self.threshold = nn.Parameter(torch.zeros(hidden_dim))
        if bits > 1:
            # Multi-bit: learned quantization boundaries
            self.boundaries = nn.Parameter(
                torch.linspace(-1, 1, 2**bits - 1).unsqueeze(0).expand(hidden_dim, -1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 1:
            if self.method == "sign":
                return (x > 0).float() * 2 - 1  # {-1, +1}
            elif self.method == "stochastic":
                prob = torch.sigmoid(x)
                if self.training:
                    binary = torch.bernoulli(prob)
                else:
                    binary = (prob > 0.5).float()
                return binary * 2 - 1
            elif self.method == "learned_threshold":
                return (x > self.threshold).float() * 2 - 1
        else:
            # Multi-bit quantization
            # x: (..., hidden_dim), boundaries: (hidden_dim, 2^bits - 1)
            expanded = x.unsqueeze(-1)  # (..., hidden_dim, 1)
            level = (expanded > self.boundaries).sum(dim=-1).float()
            # Normalize to [-1, 1]
            max_level = 2**self.bits - 1
            return level / max_level * 2 - 1

        raise ValueError(f"Unknown binarization method: {self.method}")


class OutputDecoder(nn.Module):
    """Per-expert small MLP that decodes circuit bit-outputs back to float."""

    def __init__(self, circuit_output_dim: int, hidden_dim: int, decoder_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(circuit_output_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogicCircuitExpert(nn.Module):
    """Wraps a compiled logic circuit as a PyTorch module for inference.

    The circuit itself is a bit-operation graph evaluated via integer
    tensor ops. This module handles binarization of input, circuit
    evaluation, and decoding of output.
    """

    def __init__(
        self,
        circuit,  # Compiled circuit object from conversion module
        input_binarizer: InputBinarizer,
        output_decoder: OutputDecoder,
    ):
        super().__init__()
        self.circuit = circuit
        self.binarizer = input_binarizer
        self.decoder = output_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_in = self.binarizer(x)
        circuit_out = self.circuit(binary_in)
        return self.decoder(circuit_out)


class HybridMonetModel(nn.Module):
    """Full Monet model with per-expert implementation dispatch.

    Router and attention run in float. Each expert slot is backed by
    either a LogicCircuitExpert or the original quantized-float expert,
    decided per-expert based on conversion quality.
    """

    def __init__(
        self,
        base_model: nn.Module,
        expert_impls: dict[str, nn.Module],
        impl_metadata: list[ExpertImplementation],
    ):
        super().__init__()
        self.base_model = base_model
        self.expert_impls = nn.ModuleDict(expert_impls)
        self.impl_metadata = impl_metadata

    def get_conversion_stats(self) -> dict:
        """Report conversion method breakdown."""
        methods = [m.method for m in self.impl_metadata]
        total = len(methods)
        return {
            "total_experts": total,
            "exact_circuit": methods.count("exact_circuit"),
            "learned_circuit": methods.count("learned_circuit"),
            "quantized_float": methods.count("quantized_float"),
            "circuit_fraction": (
                (methods.count("exact_circuit") + methods.count("learned_circuit"))
                / total
            )
            if total > 0
            else 0,
        }

    def forward(self, *args, **kwargs):
        """Forward pass through the hybrid model.

        The actual dispatch happens inside the MoE layer, which has been
        patched to use self.expert_impls instead of the original experts.
        This forward just calls the base model, which has been monkey-patched.
        """
        return self.base_model(*args, **kwargs)


def build_hybrid_model(
    base_model: nn.Module,
    circuits: dict[str, object],
    quantized_fallbacks: dict[str, nn.Module],
    hidden_dim: int,
    binarizer_config: dict,
    decoder_hidden: int = 64,
) -> HybridMonetModel:
    """Build a hybrid model from a base model, circuits, and fallbacks.

    Args:
        base_model: Original Monet model (will be modified in place).
        circuits: Dict mapping expert name -> compiled circuit object.
        quantized_fallbacks: Dict mapping expert name -> quantized float expert.
        hidden_dim: Model hidden dimension.
        binarizer_config: Config for InputBinarizer (method, bits).
        decoder_hidden: Hidden dim for output decoders.

    Returns:
        HybridMonetModel with per-expert dispatch.
    """
    expert_impls = {}
    metadata = []

    all_expert_names = set(circuits.keys()) | set(quantized_fallbacks.keys())

    for name in sorted(all_expert_names):
        parts = name.replace("layer", "").replace("expert", "").split("_")
        layer_idx = int(parts[0]) if len(parts) >= 1 else 0
        expert_idx = int(parts[1]) if len(parts) >= 2 else 0

        if name in circuits:
            circuit = circuits[name]
            binarizer = InputBinarizer(
                hidden_dim,
                method=binarizer_config.get("method", "sign"),
                bits=binarizer_config.get("bits", 1),
            )
            circuit_output_dim = getattr(circuit, "output_dim", hidden_dim)
            decoder = OutputDecoder(circuit_output_dim, hidden_dim, decoder_hidden)
            impl = LogicCircuitExpert(circuit, binarizer, decoder)
            expert_impls[name] = impl

            method = "exact_circuit"  # Could check circuit provenance
            meta = ExpertImplementation(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                method=method,
                circuit_size=getattr(circuit, "num_gates", 0),
            )
        else:
            expert_impls[name] = quantized_fallbacks[name]
            meta = ExpertImplementation(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                method="quantized_float",
            )

        metadata.append(meta)

    return HybridMonetModel(base_model, expert_impls, metadata)
