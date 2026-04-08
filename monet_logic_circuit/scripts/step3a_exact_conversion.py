"""Step 3a: Exact conversion via Aytekin construction.

Deterministic pipeline: ternary half-expert -> decision tree -> logic
circuit -> minimized circuit. Tests whether this works at Monet's
half-expert scale (2N per layer; N = 512).

Usage:
    python -m monet_logic_circuit.scripts.step3a_exact_conversion --config configs/step3a_exact_conversion.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch

from monet_logic_circuit.models.monet_loader import (
    load_monet_model,
    get_half_expert_modules,
)
from monet_logic_circuit.data.expert_traces import ExpertTraceStore
from monet_logic_circuit.conversion.exact import (
    run_exact_conversion,
    ExactConversionResult,
)


def main():
    parser = argparse.ArgumentParser(description="Step 3a: Exact conversion")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ternary model from Step 2
    print("Loading ternary model...")
    ternary_path = Path(config["model"]["ternary_checkpoint"])
    model, model_config = load_monet_model(
        str(ternary_path),
        device=config["model"]["device"],
        allowed_decompositions=config["model"].get("allowed_decompositions"),
    )

    # Load reconstruction errors from Step 2 to select easy half-experts
    he_cfg = config["conversion"]["half_expert_subset"]
    recon_dir = Path(he_cfg["reconstruction_error_dir"])
    recon_path = recon_dir / "results.json"
    reconstruction_errors = {}
    if recon_path.exists():
        with open(recon_path) as f:
            step2_results = json.load(f)
            reconstruction_errors = step2_results.get("ternary_cardinality", {})

    # Select half-expert subset
    all_half_experts = get_half_expert_modules(model)
    num_to_select = he_cfg["num_half_experts"]
    selection = he_cfg["selection"]
    balance_axes = he_cfg.get("balance_axes", True)

    if selection == "easy_first" and reconstruction_errors:
        # Sort by reconstruction error, take easiest
        sorted_he = sorted(
            all_half_experts,
            key=lambda x: reconstruction_errors.get(x[0], float("inf")),
        )
    else:
        sorted_he = list(all_half_experts)

    if balance_axes:
        # Interleave by axis to keep both product-key axes represented.
        # Half-expert name convention: layerL_axisA_heH
        def axis_of(name: str) -> int:
            for tok in name.split("_"):
                if tok.startswith("axis"):
                    return int(tok[len("axis"):])
            return 0

        axis0 = [he for he in sorted_he if axis_of(he[0]) == 0]
        axis1 = [he for he in sorted_he if axis_of(he[0]) == 1]
        interleaved = []
        for a, b in zip(axis0, axis1):
            interleaved.extend([a, b])
        interleaved.extend(axis0[len(axis1):])
        interleaved.extend(axis1[len(axis0):])
        selected = interleaved[:num_to_select]
    else:
        selected = sorted_he[:num_to_select]

    print(f"Selected {len(selected)} half-experts for exact conversion")

    # Load trace store for verification
    trace_store = ExpertTraceStore(
        config["conversion"]["verification"]["calibration_traces_dir"]
    )

    # Run exact conversion
    print("Running exact conversion pipeline...")
    thresholds = {
        "small": config["thresholds"]["small_circuit_max_gates"],
        "tractable": config["thresholds"]["tractable_circuit_max_gates"],
    }

    # VD half-experts see a disjoint slice of the residual stream (d_model/2)
    # and output into a d_expert subspace.
    half_expert_input_dim = model_config.hidden_dim // 2
    half_expert_output_dim = model_config.expert_dim

    conversion_results = run_exact_conversion(
        selected,
        input_dim=half_expert_input_dim,
        output_dim=half_expert_output_dim,
        trace_store=trace_store,
        tool=config["conversion"]["logic_minimization"]["tool"],
        thresholds=thresholds,
    )

    # Analyze results
    size_classes = {"small": 0, "tractable": 0, "blown_up": 0}
    gate_counts = []
    verified_count = 0

    for r in conversion_results:
        size_classes[r.size_class] += 1
        gate_counts.append(r.circuit_gates_after_minimization)
        if r.exact_match_verified:
            verified_count += 1

    total = len(conversion_results)
    print(f"\nConversion results ({total} half-experts):")
    print(f"  Small (<{thresholds['small']} gates): {size_classes['small']}")
    print(f"  Tractable (<{thresholds['tractable']} gates): {size_classes['tractable']}")
    print(f"  Blown up: {size_classes['blown_up']}")
    print(f"  Verified exact: {verified_count}/{total}")
    if gate_counts:
        print(f"  Gate count: mean={np.mean(gate_counts):.0f}, "
              f"median={np.median(gate_counts):.0f}, "
              f"max={np.max(gate_counts)}")

    # Determine outcome
    small_frac = size_classes["small"] / total if total > 0 else 0
    if small_frac > 0.5:
        outcome = "dream_case"
        print("\n  OUTCOME: Dream case -- most half-experts have small circuits.")
        print("  Scale to all half-experts and proceed to 3c.")
    elif size_classes["blown_up"] / total < 0.5:
        outcome = "tractable"
        print("\n  OUTCOME: Tractable -- proceed to 3b for learned converter.")
    else:
        outcome = "blown_up"
        print("\n  OUTCOME: Blown up -- learned converter (3b) is necessary.")

    # Save results
    results = {
        "num_half_experts_converted": total,
        "size_distribution": size_classes,
        "gate_count_stats": {
            "mean": float(np.mean(gate_counts)) if gate_counts else 0,
            "median": float(np.median(gate_counts)) if gate_counts else 0,
            "min": int(np.min(gate_counts)) if gate_counts else 0,
            "max": int(np.max(gate_counts)) if gate_counts else 0,
            "p90": float(np.percentile(gate_counts, 90)) if gate_counts else 0,
        },
        "verified_exact_count": verified_count,
        "outcome": outcome,
        "per_half_expert": [
            {
                "name": r.expert_name,
                "tree_depth": r.tree_depth,
                "tree_nodes": r.tree_nodes,
                "gates_before": r.circuit_gates_before_minimization,
                "gates_after": r.circuit_gates_after_minimization,
                "size_class": r.size_class,
                "verified": r.exact_match_verified,
            }
            for r in conversion_results
        ],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save individual circuits for use by 3b
    circuits_dir = output_dir / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for r in conversion_results:
        if r.size_class in ("small", "tractable"):
            # Circuit data would be saved here
            pass

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
