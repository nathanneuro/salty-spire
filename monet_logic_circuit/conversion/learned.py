"""Learned converter: trains a model to map expert weights -> approximate logic circuits.

For experts where exact conversion (Aytekin) is too expensive, the learned
converter produces approximate circuits that trade exact equivalence for
dramatically smaller circuit sizes.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monet_logic_circuit.conversion.circuit import LogicCircuit, GateType


# Gate type indices for the converter's output logits
GATE_TYPES = [
    GateType.AND, GateType.OR, GateType.XOR, GateType.NAND,
    GateType.NOR, GateType.XNOR, GateType.BUF, GateType.NOT,
]
NUM_GATE_TYPES = len(GATE_TYPES)


@dataclass
class ConverterConfig:
    """Configuration for the circuit converter architecture."""

    # Circuit topology
    circuit_depth: int = 8
    circuit_width: int = 256
    wiring_sparsity: float = 0.3

    # Converter model
    encoder_hidden: int = 512
    encoder_layers: int = 4

    # Training
    supervised_weight: float = 1.0
    distillation_weight: float = 1.0
    size_penalty: float = 0.01
    learning_rate: float = 3e-4
    temperature: float = 1.0
    min_temperature: float = 0.1


class CircuitConverter(nn.Module):
    """Neural network that maps expert parameters to logic circuit gate choices.

    Input: flattened expert weight tensor + input distribution statistics.
    Output: per-gate type logits for each gate in a fixed-topology circuit.

    The topology (wiring) is fixed and random-sparse; only the gate
    function at each node is learned.
    """

    def __init__(self, config: ConverterConfig, expert_param_dim: int, input_stat_dim: int):
        super().__init__()
        self.config = config

        total_gates = config.circuit_depth * config.circuit_width
        input_dim = expert_param_dim + input_stat_dim

        # Encoder: expert params + stats -> latent
        layers = []
        d = input_dim
        for _ in range(config.encoder_layers):
            layers.extend([
                nn.Linear(d, config.encoder_hidden),
                nn.GELU(),
                nn.LayerNorm(config.encoder_hidden),
            ])
            d = config.encoder_hidden
        self.encoder = nn.Sequential(*layers)

        # Gate predictor: latent -> per-gate type logits
        self.gate_predictor = nn.Linear(config.encoder_hidden, total_gates * NUM_GATE_TYPES)

        # Fixed random sparse wiring (not learned)
        self.register_buffer(
            "wiring_mask",
            self._generate_sparse_wiring(config.circuit_depth, config.circuit_width, config.wiring_sparsity),
        )
        self.register_buffer(
            "wiring_indices",
            self._generate_wiring_indices(config.circuit_depth, config.circuit_width),
        )

        self.total_gates = total_gates
        self.temperature = config.temperature

    def forward(self, expert_params: torch.Tensor, input_stats: torch.Tensor) -> dict:
        """Predict gate types for the fixed-topology circuit.

        Args:
            expert_params: (batch, expert_param_dim) flattened expert weights.
            input_stats: (batch, input_stat_dim) expert input distribution stats.

        Returns:
            Dict with:
                'gate_logits': (batch, total_gates, NUM_GATE_TYPES)
                'gate_probs': soft gate selection probabilities
                'gate_choices': hard gate selections (argmax)
        """
        x = torch.cat([expert_params, input_stats], dim=-1)
        latent = self.encoder(x)
        logits = self.gate_predictor(latent)
        logits = logits.view(-1, self.total_gates, NUM_GATE_TYPES)

        probs = F.softmax(logits / self.temperature, dim=-1)
        choices = logits.argmax(dim=-1)

        return {
            "gate_logits": logits,
            "gate_probs": probs,
            "gate_choices": choices,
        }

    def to_circuit(
        self, expert_params: torch.Tensor, input_stats: torch.Tensor,
        input_dim: int, output_dim: int,
    ) -> LogicCircuit:
        """Convert predicted gate choices to an actual LogicCircuit.

        Args:
            expert_params: (expert_param_dim,) single expert's params.
            input_stats: (input_stat_dim,) single expert's input stats.
            input_dim: Circuit input dimension.
            output_dim: Circuit output dimension.

        Returns:
            LogicCircuit with the predicted gate types.
        """
        self.eval()
        with torch.no_grad():
            result = self(expert_params.unsqueeze(0), input_stats.unsqueeze(0))
            choices = result["gate_choices"][0]  # (total_gates,)

        circuit = LogicCircuit(input_dim, output_dim)
        wiring = self.wiring_indices.cpu()

        for gate_idx in range(self.total_gates):
            gate_type = GATE_TYPES[choices[gate_idx].item()]
            inp_ids = wiring[gate_idx].tolist()

            # Filter to valid inputs based on gate type
            if gate_type.num_inputs == 1:
                inp_ids = [inp_ids[0]]

            circuit.add_gate(gate_type, inp_ids)

        # Last circuit_width gates are outputs
        output_start = self.total_gates - self.config.circuit_width
        output_ids = list(range(output_start, output_start + output_dim))
        circuit.set_outputs(output_ids)

        return circuit

    def _generate_sparse_wiring(self, depth: int, width: int, sparsity: float) -> torch.Tensor:
        """Generate random sparse wiring mask."""
        total = depth * width
        mask = torch.rand(total, 2) > sparsity  # Keep connections above threshold
        return mask

    def _generate_wiring_indices(self, depth: int, width: int) -> torch.Tensor:
        """Generate random wiring indices (which gates feed into which)."""
        total = depth * width
        indices = torch.zeros(total, 2, dtype=torch.long)

        for layer in range(depth):
            layer_start = layer * width
            for j in range(width):
                gate_idx = layer_start + j
                if layer == 0:
                    # First layer connects to primary inputs (negative IDs)
                    indices[gate_idx, 0] = -(torch.randint(0, width, (1,)).item() + 1)
                    indices[gate_idx, 1] = -(torch.randint(0, width, (1,)).item() + 1)
                else:
                    # Subsequent layers connect to previous layer
                    prev_start = (layer - 1) * width
                    indices[gate_idx, 0] = prev_start + torch.randint(0, width, (1,)).item()
                    indices[gate_idx, 1] = prev_start + torch.randint(0, width, (1,)).item()

        return indices


class ConverterTrainer:
    """Training loop for the circuit converter.

    Combines supervised loss (match known-good circuits from Step 3a)
    with distillation loss (match expert outputs on calibration data).
    """

    def __init__(
        self,
        converter: CircuitConverter,
        config: ConverterConfig,
        device: str = "cuda",
    ):
        self.converter = converter.to(device)
        self.config = config
        self.device = torch.device(device)

        self.optimizer = torch.optim.AdamW(
            converter.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def train_step(
        self,
        expert_params: torch.Tensor,
        input_stats: torch.Tensor,
        reference_outputs: Optional[torch.Tensor] = None,
        calibration_inputs: Optional[torch.Tensor] = None,
        calibration_targets: Optional[torch.Tensor] = None,
        supervised_targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Single training step.

        Args:
            expert_params: (batch, param_dim) expert weight vectors.
            input_stats: (batch, stat_dim) input distribution features.
            reference_outputs: Optional supervised gate-choice targets from 3a.
            calibration_inputs: Optional calibration inputs for distillation.
            calibration_targets: Optional expert outputs for distillation.
            supervised_targets: Optional per-gate target labels from 3a.

        Returns:
            Dict of loss components.
        """
        self.converter.train()
        result = self.converter(expert_params.to(self.device), input_stats.to(self.device))

        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Supervised loss: match known-good gate choices from exact conversion
        if supervised_targets is not None:
            sup_loss = F.cross_entropy(
                result["gate_logits"].reshape(-1, NUM_GATE_TYPES),
                supervised_targets.reshape(-1).to(self.device),
            )
            losses["supervised"] = sup_loss.item()
            total_loss = total_loss + self.config.supervised_weight * sup_loss

        # Distillation loss: circuit output matches expert output on calibration data
        if calibration_inputs is not None and calibration_targets is not None:
            # Use soft gate probs (STE) to evaluate circuit differentiably
            dist_loss = self._distillation_loss(
                result["gate_probs"],
                calibration_inputs.to(self.device),
                calibration_targets.to(self.device),
            )
            losses["distillation"] = dist_loss.item()
            total_loss = total_loss + self.config.distillation_weight * dist_loss

        # Circuit size penalty: encourage using BUF/NOT (fewer transistors)
        size_loss = self._size_penalty(result["gate_probs"])
        losses["size_penalty"] = size_loss.item()
        total_loss = total_loss + self.config.size_penalty * size_loss

        losses["total"] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return losses

    def _distillation_loss(
        self, gate_probs: torch.Tensor,
        calibration_inputs: torch.Tensor,
        calibration_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss via differentiable circuit evaluation."""
        # This would evaluate the circuit with soft gate selections
        # Using straight-through estimator for discrete gate choices
        # Placeholder: MSE between predicted and target outputs
        return F.mse_loss(
            gate_probs.mean(dim=-1),  # Simplified
            calibration_targets[:, :gate_probs.shape[1]].float(),
        )

    def _size_penalty(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Encourage simpler gates (BUF, NOT have fewer transistors)."""
        # BUF and NOT are indices 6 and 7 in GATE_TYPES
        simple_gate_prob = gate_probs[:, :, 6:].sum(dim=-1).mean()
        return -simple_gate_prob  # Negative because we want to maximize simple gate usage

    def update_temperature(self, step: int, total_steps: int):
        """Cosine anneal the temperature for gate selection sharpening."""
        import math
        progress = step / total_steps
        temp = self.config.min_temperature + 0.5 * (
            self.config.temperature - self.config.min_temperature
        ) * (1 + math.cos(math.pi * progress))
        self.converter.temperature = temp
