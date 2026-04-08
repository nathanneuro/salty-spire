"""Hybrid Monet model with logic-circuit half-experts replacing float half-experts.

Used in Steps 3c and 3d: routers and attention remain in float, while each
half-expert slot is backed by either a LogicCircuitHalfExpert or the original
quantized-float half-expert, decided per-half-expert based on conversion
quality. Effective-expert outputs are formed by composing half-experts
according to the decomposition rule (VD: additive over axes).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


Axis = Literal[0, 1]


@dataclass
class HalfExpertImplementation:
    """Tracks which implementation backs each half-expert slot."""

    layer_idx: int
    axis: Axis
    half_expert_idx: int
    method: str  # "exact_circuit", "learned_circuit", "quantized_float"
    circuit_size: int = 0  # Number of gates (0 for float fallback)
    reconstruction_error: float = 0.0

    @property
    def name(self) -> str:
        return f"layer{self.layer_idx}_axis{self.axis}_he{self.half_expert_idx}"


class InputBinarizer(nn.Module):
    """Binarizes the residual-stream slice feeding a half-expert circuit.

    Supports strict binary (sign), stochastic binarization, and learned
    per-channel thresholds. Multi-bit mode quantizes to n bits instead
    of strict binary. For VD, ``hidden_dim`` is the size of the disjoint
    input slice that this half-expert actually sees, not the full residual.
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
            expanded = x.unsqueeze(-1)  # (..., hidden_dim, 1)
            level = (expanded > self.boundaries).sum(dim=-1).float()
            max_level = 2**self.bits - 1
            return level / max_level * 2 - 1

        raise ValueError(f"Unknown binarization method: {self.method}")


class OutputDecoder(nn.Module):
    """Per-half-expert small MLP that decodes circuit bit-outputs back to float."""

    def __init__(self, circuit_output_dim: int, output_dim: int, decoder_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(circuit_output_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogicCircuitHalfExpert(nn.Module):
    """Wraps a compiled logic circuit as a PyTorch module for inference.

    The circuit itself is a bit-operation graph evaluated via integer
    tensor ops. This module handles binarization of the half-expert's
    input slice, circuit evaluation, and decoding of the output back to
    the float subspace that the effective-expert composition expects.
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
    """Full Monet model with per-half-expert implementation dispatch.

    Routers and attention run in float. Each half-expert slot is backed by
    either a LogicCircuitHalfExpert or the original quantized-float
    half-expert, decided per-half-expert based on conversion quality.
    Effective-expert outputs are recovered by composing half-expert outputs
    under the decomposition rule (VD: sum over axes).
    """

    def __init__(
        self,
        base_model: nn.Module,
        half_expert_impls: dict[str, nn.Module],
        impl_metadata: list[HalfExpertImplementation],
    ):
        super().__init__()
        self.base_model = base_model
        self.half_expert_impls = nn.ModuleDict(half_expert_impls)
        self.impl_metadata = impl_metadata

    def get_conversion_stats(self) -> dict:
        """Report conversion method breakdown at the half-expert granularity."""
        methods = [m.method for m in self.impl_metadata]
        total = len(methods)
        return {
            "total_half_experts": total,
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
        patched to use ``self.half_expert_impls`` instead of the original
        half-experts. This forward just calls the base model.
        """
        return self.base_model(*args, **kwargs)


def _parse_half_expert_name(name: str) -> tuple[int, Axis, int]:
    """Parse a half-expert name of the form 'layerL_axisA_heH' into components.

    Raises ValueError if the name does not match the expected format. This
    format is the canonical one produced by HalfExpertWrapper.
    """
    # e.g. "layer3_axis0_he17"
    try:
        layer_part, axis_part, he_part = name.split("_", 2)
        assert layer_part.startswith("layer")
        assert axis_part.startswith("axis")
        assert he_part.startswith("he")
        layer_idx = int(layer_part[len("layer"):])
        axis = int(axis_part[len("axis"):])
        half_expert_idx = int(he_part[len("he"):])
    except (ValueError, AssertionError) as e:
        raise ValueError(
            f"Half-expert name {name!r} does not match "
            f"'layerL_axisA_heH' format"
        ) from e
    if axis not in (0, 1):
        raise ValueError(f"Half-expert axis must be 0 or 1, got {axis}")
    return layer_idx, axis, half_expert_idx  # type: ignore[return-value]


def build_hybrid_model(
    base_model: nn.Module,
    circuits: dict[str, object],
    quantized_fallbacks: dict[str, nn.Module],
    half_expert_input_dim: int,
    half_expert_output_dim: int,
    binarizer_config: dict,
    decoder_hidden: int = 64,
) -> HybridMonetModel:
    """Build a hybrid model from a base model, circuits, and fallbacks.

    Args:
        base_model: Original Monet model (will be modified in place).
        circuits: Dict mapping half-expert name -> compiled circuit object.
        quantized_fallbacks: Dict mapping half-expert name -> quantized float half-expert.
        half_expert_input_dim: Size of the input slice each half-expert sees.
        half_expert_output_dim: Size of a half-expert's output (d_expert for VD).
        binarizer_config: Config for InputBinarizer (method, bits).
        decoder_hidden: Hidden dim for output decoders.

    Returns:
        HybridMonetModel with per-half-expert dispatch.
    """
    half_expert_impls: dict[str, nn.Module] = {}
    metadata: list[HalfExpertImplementation] = []

    all_names = set(circuits.keys()) | set(quantized_fallbacks.keys())

    for name in sorted(all_names):
        layer_idx, axis, half_expert_idx = _parse_half_expert_name(name)

        if name in circuits:
            circuit = circuits[name]
            binarizer = InputBinarizer(
                half_expert_input_dim,
                method=binarizer_config.get("method", "sign"),
                bits=binarizer_config.get("bits", 1),
            )
            circuit_output_dim = getattr(circuit, "output_dim", half_expert_output_dim)
            decoder = OutputDecoder(
                circuit_output_dim, half_expert_output_dim, decoder_hidden
            )
            impl: nn.Module = LogicCircuitHalfExpert(circuit, binarizer, decoder)
            half_expert_impls[name] = impl

            meta = HalfExpertImplementation(
                layer_idx=layer_idx,
                axis=axis,
                half_expert_idx=half_expert_idx,
                method="exact_circuit",  # Could check circuit provenance
                circuit_size=getattr(circuit, "num_gates", 0),
            )
        else:
            half_expert_impls[name] = quantized_fallbacks[name]
            meta = HalfExpertImplementation(
                layer_idx=layer_idx,
                axis=axis,
                half_expert_idx=half_expert_idx,
                method="quantized_float",
            )

        metadata.append(meta)

    return HybridMonetModel(base_model, half_expert_impls, metadata)
