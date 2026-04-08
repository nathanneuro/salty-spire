"""Logic circuit representation and evaluation.

Circuits are directed acyclic graphs of logic gates operating on binary
inputs. This is the shared representation used by both exact conversion
(Step 3a) and the learned converter (Step 3b).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class GateType(Enum):
    """Supported logic gate types."""

    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"
    BUF = "BUF"    # Buffer (identity)
    NOT = "NOT"    # Inverter

    @property
    def num_inputs(self) -> int:
        if self in (GateType.BUF, GateType.NOT):
            return 1
        return 2

    def evaluate(self, a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate gate on binary {0, 1} tensors."""
        if self == GateType.BUF:
            return a
        if self == GateType.NOT:
            return 1 - a
        assert b is not None
        if self == GateType.AND:
            return a * b
        if self == GateType.OR:
            return torch.clamp(a + b, max=1)
        if self == GateType.XOR:
            return (a + b) % 2
        if self == GateType.NAND:
            return 1 - a * b
        if self == GateType.NOR:
            return 1 - torch.clamp(a + b, max=1)
        if self == GateType.XNOR:
            return 1 - (a + b) % 2
        raise ValueError(f"Unknown gate type: {self}")


@dataclass
class Gate:
    """A single gate in a logic circuit."""

    gate_id: int
    gate_type: GateType
    input_ids: list[int]  # IDs of input gates or primary inputs
    # For primary inputs, input_ids references the input bit index (negative IDs)

    def __repr__(self):
        return f"Gate({self.gate_id}, {self.gate_type.value}, inputs={self.input_ids})"


class LogicCircuit:
    """A logic circuit as a DAG of gates.

    Primary inputs are referenced by negative IDs: input bit i is -(i+1).
    Gate IDs are non-negative integers.

    The circuit is evaluated in topological order. Output bits are the
    gates listed in output_gate_ids.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gates: Optional[list[Gate]] = None,
        output_gate_ids: Optional[list[int]] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gates = gates or []
        self.output_gate_ids = output_gate_ids or []
        self._topo_order: Optional[list[int]] = None

    @property
    def num_gates(self) -> int:
        return len(self.gates)

    def add_gate(self, gate_type: GateType, input_ids: list[int]) -> int:
        """Add a gate and return its ID."""
        gate_id = len(self.gates)
        self.gates.append(Gate(gate_id, gate_type, input_ids))
        self._topo_order = None  # Invalidate cache
        return gate_id

    def set_outputs(self, gate_ids: list[int]):
        """Set which gates produce the circuit outputs."""
        assert len(gate_ids) == self.output_dim
        self.output_gate_ids = gate_ids

    def _compute_topo_order(self) -> list[int]:
        """Compute topological ordering of gates."""
        if self._topo_order is not None:
            return self._topo_order

        visited = set()
        order = []

        def visit(gate_id: int):
            if gate_id < 0 or gate_id in visited:
                return
            visited.add(gate_id)
            gate = self.gates[gate_id]
            for inp_id in gate.input_ids:
                visit(inp_id)
            order.append(gate_id)

        for gate in self.gates:
            visit(gate.gate_id)

        self._topo_order = order
        return order

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit on binary input.

        Args:
            x: (..., input_dim) tensor with values in {0, 1} or {-1, +1}.

        Returns:
            (..., output_dim) tensor of binary outputs.
        """
        # Normalize to {0, 1}
        if x.min() < 0:
            x = (x + 1) / 2

        batch_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.input_dim)
        batch_size = flat_x.shape[0]

        # Wire values: negative IDs map to inputs, non-negative to gates
        values = {}
        for i in range(self.input_dim):
            values[-(i + 1)] = flat_x[:, i]

        # Evaluate in topological order
        for gate_id in self._compute_topo_order():
            gate = self.gates[gate_id]
            a = values[gate.input_ids[0]]
            b = values[gate.input_ids[1]] if len(gate.input_ids) > 1 else None
            values[gate_id] = gate.gate_type.evaluate(a, b)

        # Collect outputs
        outputs = torch.stack(
            [values[gid] for gid in self.output_gate_ids], dim=-1
        )
        return outputs.reshape(*batch_shape, self.output_dim)

    def to_dict(self) -> dict:
        """Serialize circuit to a JSON-compatible dict."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_gates": self.num_gates,
            "gates": [
                {
                    "id": g.gate_id,
                    "type": g.gate_type.value,
                    "inputs": g.input_ids,
                }
                for g in self.gates
            ],
            "output_gate_ids": self.output_gate_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogicCircuit":
        """Deserialize circuit from a dict."""
        circuit = cls(d["input_dim"], d["output_dim"])
        for g in d["gates"]:
            gate = Gate(g["id"], GateType(g["type"]), g["inputs"])
            circuit.gates.append(gate)
        circuit.output_gate_ids = d["output_gate_ids"]
        return circuit

    def gate_type_counts(self) -> dict[str, int]:
        """Count gates by type."""
        counts = {}
        for gate in self.gates:
            name = gate.gate_type.value
            counts[name] = counts.get(name, 0) + 1
        return counts
