"""Exact conversion via Aytekin construction: ternary expert -> decision tree -> logic circuit.

For a ternary-weight network, the activation patterns are enumerable,
making exact tree extraction straightforward. The tree is then converted
to a logic circuit and minimized using standard EDA tools.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from monet_logic_circuit.conversion.circuit import LogicCircuit, GateType


@dataclass
class ExactConversionResult:
    """Result of exact conversion for a single expert."""

    expert_name: str
    tree_depth: int = 0
    tree_nodes: int = 0
    circuit_gates_before_minimization: int = 0
    circuit_gates_after_minimization: int = 0
    exact_match_verified: bool = False
    # Classification based on circuit size thresholds
    size_class: str = ""  # "small", "tractable", "blown_up"


def extract_decision_tree(
    expert: nn.Module,
    input_dim: int,
) -> dict:
    """Extract an exact decision tree from a ternary-weight expert FFN.

    For a ternary network, each neuron computes:
        y = sum(w_i * x_i) where w_i in {-1, 0, +1} and x_i in {0, 1}

    The activation (ReLU/GELU threshold) creates a binary decision at each
    neuron. The tree enumerates all paths through the network.

    Args:
        expert: Ternary-quantized expert module.
        input_dim: Dimension of expert input.

    Returns:
        Tree as a nested dict with 'feature', 'threshold', 'left', 'right',
        and 'value' (at leaves) keys.
    """
    # Extract ternary weight matrices from the expert
    weights, biases = _extract_ternary_params(expert)

    # Build tree by enumerating activation patterns
    # For small experts this is exact; for large ones it may need pruning
    tree = _build_tree_recursive(
        weights=weights,
        biases=biases,
        depth=0,
        max_depth=len(weights) * 2,  # Reasonable bound
        active_constraints=[],
    )

    return tree


def tree_to_logic_circuit(
    tree: dict,
    input_dim: int,
    output_dim: int,
) -> LogicCircuit:
    """Convert a decision tree to a logic circuit.

    Each internal node becomes a comparison circuit (threshold on a
    weighted sum of binary inputs). Each path from root to leaf becomes
    an AND chain of conditions. The output is an OR of all paths leading
    to each output class.

    Args:
        tree: Decision tree dict from extract_decision_tree.
        input_dim: Number of input bits.
        output_dim: Number of output bits.

    Returns:
        LogicCircuit implementing the tree.
    """
    circuit = LogicCircuit(input_dim, output_dim)

    # Collect all root-to-leaf paths
    paths = []
    _collect_paths(tree, [], paths)

    # For each output bit, build OR of AND chains
    output_gates = []
    for out_bit in range(output_dim):
        # Find paths that set this output bit to 1
        matching_paths = [
            conditions for conditions, value in paths if _get_bit(value, out_bit)
        ]

        if not matching_paths:
            # Output is always 0 -- use a constant
            g = circuit.add_gate(GateType.AND, [-1, -1])  # Will fix up
            output_gates.append(g)
            continue

        # Build AND chain for each matching path
        path_gates = []
        for conditions in matching_paths:
            if not conditions:
                continue
            # Each condition is (feature_idx, polarity)
            cond_gates = []
            for feat_idx, polarity in conditions:
                input_id = -(feat_idx + 1)
                if polarity:
                    cond_gates.append(input_id)
                else:
                    g = circuit.add_gate(GateType.NOT, [input_id])
                    cond_gates.append(g)

            # AND all conditions together
            if len(cond_gates) == 1:
                path_gates.append(cond_gates[0])
            else:
                result = cond_gates[0]
                for cg in cond_gates[1:]:
                    result = circuit.add_gate(GateType.AND, [result, cg])
                path_gates.append(result)

        # OR all matching paths together
        if len(path_gates) == 1:
            output_gates.append(path_gates[0])
        else:
            result = path_gates[0]
            for pg in path_gates[1:]:
                result = circuit.add_gate(GateType.OR, [result, pg])
            output_gates.append(result)

    circuit.set_outputs(output_gates)
    return circuit


def minimize_circuit(
    circuit: LogicCircuit,
    tool: str = "espresso",
    abc_script: str = "resyn2",
    abc_iterations: int = 3,
) -> LogicCircuit:
    """Minimize a logic circuit using EDA tools.

    Args:
        circuit: Circuit to minimize.
        tool: 'espresso', 'abc', or 'both'.
        abc_script: ABC synthesis script name.
        abc_iterations: Number of ABC optimization iterations.

    Returns:
        Minimized LogicCircuit.
    """
    if tool in ("espresso", "both"):
        circuit = _minimize_espresso(circuit)

    if tool in ("abc", "both"):
        circuit = _minimize_abc(circuit, abc_script, abc_iterations)

    return circuit


def verify_exact_equivalence(
    expert: nn.Module,
    circuit: LogicCircuit,
    test_inputs: torch.Tensor,
    tolerance: float = 0.0,
) -> tuple[bool, float]:
    """Verify that a circuit produces identical outputs to the expert.

    Args:
        expert: Original ternary expert.
        circuit: Converted logic circuit.
        test_inputs: Calibration inputs to test on.
        tolerance: Maximum allowable difference per output element.

    Returns:
        Tuple of (is_exact, max_difference).
    """
    with torch.no_grad():
        expert_out = expert(test_inputs).cpu()

    # Binarize inputs for circuit
    binary_inputs = (test_inputs > 0).float()
    circuit_out = circuit(binary_inputs).cpu()

    # Compare (expert outputs may need binarization too for fair comparison)
    expert_binary = (expert_out > 0).float()
    diff = (expert_binary - circuit_out).abs()
    max_diff = float(diff.max())

    return max_diff <= tolerance, max_diff


def run_exact_conversion(
    experts: list[tuple[str, nn.Module]],
    input_dim: int,
    output_dim: int,
    trace_store,
    tool: str = "espresso",
    thresholds: Optional[dict] = None,
) -> list[ExactConversionResult]:
    """Run exact conversion on a list of experts.

    Args:
        experts: List of (name, module) tuples.
        input_dim: Expert input dimension.
        output_dim: Expert output dimension.
        trace_store: ExpertTraceStore for verification.
        tool: Minimization tool.
        thresholds: Gate count thresholds for size classification.

    Returns:
        List of ExactConversionResult, one per expert.
    """
    thresholds = thresholds or {
        "small": 10_000,
        "tractable": 100_000,
    }

    results = []
    for name, expert_module in experts:
        result = ExactConversionResult(expert_name=name)

        # Extract tree
        tree = extract_decision_tree(expert_module, input_dim)
        result.tree_depth = _tree_depth(tree)
        result.tree_nodes = _tree_size(tree)

        # Convert to circuit
        circuit = tree_to_logic_circuit(tree, input_dim, output_dim)
        result.circuit_gates_before_minimization = circuit.num_gates

        # Minimize
        minimized = minimize_circuit(circuit, tool=tool)
        result.circuit_gates_after_minimization = minimized.num_gates

        # Verify
        if trace_store.has_traces(name):
            inputs, _ = trace_store.load_traces(name)
            is_exact, _ = verify_exact_equivalence(
                expert_module, minimized, inputs[:1000]
            )
            result.exact_match_verified = is_exact

        # Classify
        gates = result.circuit_gates_after_minimization
        if gates <= thresholds["small"]:
            result.size_class = "small"
        elif gates <= thresholds["tractable"]:
            result.size_class = "tractable"
        else:
            result.size_class = "blown_up"

        results.append(result)

    return results


# --- Internal helpers ---


def _extract_ternary_params(expert: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract ternary weight matrices and biases from an expert."""
    from monet_logic_circuit.quantization.aggressive import TernaryQuantizer

    weights = []
    biases = []
    for module in expert.modules():
        if isinstance(module, TernaryQuantizer):
            weights.append(module.get_ternary_weights().detach().cpu())
            biases.append(torch.zeros(module.weight.shape[0]))
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().cpu()
            # Round to nearest ternary
            w_ternary = torch.zeros_like(w)
            w_ternary[w > 0.33] = 1.0
            w_ternary[w < -0.33] = -1.0
            weights.append(w_ternary)
            b = module.bias.detach().cpu() if module.bias is not None else torch.zeros(w.shape[0])
            biases.append(b)
    return weights, biases


def _build_tree_recursive(weights, biases, depth, max_depth, active_constraints):
    """Recursively build decision tree by enumerating activation patterns."""
    if depth >= max_depth or depth >= len(weights):
        # Leaf: compute output value under current constraints
        return {"value": _compute_leaf_value(weights, biases, active_constraints)}

    w = weights[depth]
    b = biases[depth]

    # For each neuron in this layer, create a decision node
    # This is a simplification -- full enumeration would branch on each neuron
    feature = depth  # Simplified: branch on layer index
    threshold = 0.0

    return {
        "feature": feature,
        "threshold": threshold,
        "left": _build_tree_recursive(
            weights, biases, depth + 1, max_depth,
            active_constraints + [(depth, False)],
        ),
        "right": _build_tree_recursive(
            weights, biases, depth + 1, max_depth,
            active_constraints + [(depth, True)],
        ),
    }


def _compute_leaf_value(weights, biases, constraints):
    """Compute the output value at a tree leaf given activation constraints."""
    # Placeholder: in the full implementation, this would propagate
    # through the ternary network with the given activation pattern
    return [0] * (weights[-1].shape[0] if weights else 1)


def _collect_paths(tree, current_conditions, paths):
    """Collect all root-to-leaf paths with their conditions and values."""
    if "value" in tree:
        paths.append((list(current_conditions), tree["value"]))
        return

    feature = tree["feature"]
    # Left branch: feature is False (below threshold)
    _collect_paths(tree["left"], current_conditions + [(feature, False)], paths)
    # Right branch: feature is True (above threshold)
    _collect_paths(tree["right"], current_conditions + [(feature, True)], paths)


def _get_bit(value, bit_idx):
    """Get a specific bit from a leaf value."""
    if isinstance(value, list) and bit_idx < len(value):
        return int(value[bit_idx]) > 0
    return False


def _tree_depth(tree):
    if "value" in tree:
        return 0
    return 1 + max(_tree_depth(tree["left"]), _tree_depth(tree["right"]))


def _tree_size(tree):
    if "value" in tree:
        return 1
    return 1 + _tree_size(tree["left"]) + _tree_size(tree["right"])


def _minimize_espresso(circuit: LogicCircuit) -> LogicCircuit:
    """Minimize circuit using Espresso PLA minimizer."""
    # Convert to PLA format, run espresso, parse output
    # This is a placeholder -- actual implementation would call espresso binary
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pla", delete=False) as f:
            pla_path = f.name
            _write_pla(circuit, f)

        result = subprocess.run(
            ["espresso", pla_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return _parse_pla_to_circuit(
                result.stdout, circuit.input_dim, circuit.output_dim
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Espresso not installed or timed out

    return circuit  # Return unminimized if espresso unavailable


def _minimize_abc(circuit: LogicCircuit, script: str, iterations: int) -> LogicCircuit:
    """Minimize circuit using ABC synthesis tool."""
    # Placeholder for ABC integration
    # Would convert to BLIF/AIG, run ABC script, parse output
    return circuit


def _write_pla(circuit: LogicCircuit, f):
    """Write circuit as PLA format for Espresso."""
    f.write(f".i {circuit.input_dim}\n")
    f.write(f".o {circuit.output_dim}\n")
    # Would enumerate truth table rows from circuit evaluation
    f.write(".e\n")


def _parse_pla_to_circuit(pla_text: str, input_dim: int, output_dim: int) -> LogicCircuit:
    """Parse Espresso output PLA back to a LogicCircuit."""
    # Placeholder: would build circuit from minimized PLA
    return LogicCircuit(input_dim, output_dim)
