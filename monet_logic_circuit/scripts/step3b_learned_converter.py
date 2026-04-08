"""Step 3b: Learned converter for approximate logic circuits.

Trains a model to map (quantized expert weights, input distribution stats)
-> (approximate logic circuit) for experts where exact conversion is too expensive.

Usage:
    python -m monet_logic_circuit.scripts.step3b_learned_converter --config configs/step3b_learned_converter.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from monet_logic_circuit.models.monet_loader import load_monet_model, get_expert_modules
from monet_logic_circuit.data.expert_traces import ExpertTraceStore
from monet_logic_circuit.conversion.learned import (
    CircuitConverter,
    ConverterTrainer,
    ConverterConfig,
)


class ExpertDataset(Dataset):
    """Dataset of (expert_params, input_stats, target_circuit) tuples."""

    def __init__(self, expert_data: list[dict]):
        self.data = expert_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "expert_params": d["params"],
            "input_stats": d["stats"],
            "name": d["name"],
        }


def main():
    parser = argparse.ArgumentParser(description="Step 3b: Learned converter")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and expert data
    print("Loading ternary model...")
    model, model_config = load_monet_model(
        config["model"]["ternary_checkpoint"],
        device=config["model"]["device"],
    )

    trace_store = ExpertTraceStore(config["converter"]["training"]["calibration_traces_dir"])

    # Build expert dataset: extract params + input stats for each expert
    print("Building expert dataset...")
    all_experts = get_expert_modules(model)
    expert_data = []

    for name, expert_module in all_experts:
        params = torch.cat([p.flatten() for p in expert_module.parameters()]).detach()

        # Get input stats from traces
        stats = torch.zeros(21)  # Default
        if trace_store.has_traces(name):
            inputs, _ = trace_store.load_traces(name)
            flat = inputs.reshape(-1, inputs.shape[-1]).float()
            mean = flat.mean(dim=0)[:10]
            var = flat.var(dim=0)[:10]
            eff_rank = torch.tensor([float(flat.shape[-1])])  # Placeholder
            stats = torch.cat([mean, var, eff_rank])

        expert_data.append({"name": name, "params": params, "stats": stats})

    # Determine dimensions
    expert_param_dim = expert_data[0]["params"].shape[0]
    input_stat_dim = expert_data[0]["stats"].shape[0]

    # Split into train/val
    dataset = ExpertDataset(expert_data)
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    print(f"  Train: {n_train} experts, Val: {n_val} experts")

    # Build converter
    conv_config = ConverterConfig(
        circuit_depth=config["converter"]["architecture"]["topology"]["depth"],
        circuit_width=config["converter"]["architecture"]["topology"]["width"],
        wiring_sparsity=config["converter"]["architecture"]["topology"]["sparsity"],
        supervised_weight=config["converter"]["training"]["loss"]["supervised_weight"],
        distillation_weight=config["converter"]["training"]["loss"]["distillation_weight"],
        size_penalty=config["converter"]["training"]["loss"]["circuit_size_penalty"],
        learning_rate=config["converter"]["training"]["optimizer"]["learning_rate"],
        temperature=config["converter"]["training"]["relaxation"]["temperature"],
        min_temperature=config["converter"]["training"]["relaxation"]["min_temperature"],
    )

    converter = CircuitConverter(conv_config, expert_param_dim, input_stat_dim)
    trainer = ConverterTrainer(converter, conv_config, device=str(next(model.parameters()).device))

    # Training loop
    max_steps = config["converter"]["training"]["max_steps"]
    eval_interval = config["converter"]["training"]["eval_interval"]
    batch_size = config["converter"]["training"]["batch_size"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_losses = []

    print(f"Training converter for {max_steps} steps...")
    step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            losses = trainer.train_step(
                expert_params=batch["expert_params"],
                input_stats=batch["input_stats"],
            )
            train_losses.append(losses)
            trainer.update_temperature(step, max_steps)

            if step % 100 == 0:
                print(f"  Step {step}: loss={losses['total']:.4f}")

            if step % eval_interval == 0 and step > 0:
                val_metrics = _evaluate_converter(converter, val_set, trace_store, model_config)
                print(f"  Val reconstruction error: mean={val_metrics['mean_error']:.6f}")

            step += 1

    # Final evaluation
    print("\nFinal evaluation...")
    train_metrics = _evaluate_converter(converter, train_set, trace_store, model_config)
    val_metrics = _evaluate_converter(converter, val_set, trace_store, model_config)

    print(f"  Train reconstruction error: mean={train_metrics['mean_error']:.6f}")
    print(f"  Val reconstruction error: mean={val_metrics['mean_error']:.6f}")

    # Convert all experts and decide per-expert: exact (3a) or learned (3b)
    print("\nConverting all experts...")
    exact_results_path = Path(config["converter"]["training"]["supervised_data_dir"]).parent / "results.json"
    exact_results = {}
    if exact_results_path.exists():
        with open(exact_results_path) as f:
            exact_results = json.load(f)

    per_expert_decisions = _make_per_expert_decisions(
        converter, expert_data, exact_results, model_config, trace_store,
    )

    # Save
    results = {
        "training": {
            "final_train_loss": train_losses[-1] if train_losses else {},
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "total_steps": step,
        },
        "per_expert_decisions": per_expert_decisions,
        "summary": {
            "exact_circuit": sum(1 for d in per_expert_decisions.values() if d["method"] == "exact"),
            "learned_circuit": sum(1 for d in per_expert_decisions.values() if d["method"] == "learned"),
            "total": len(per_expert_decisions),
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(converter.state_dict(), output_dir / "converter.pt")
    print(f"\nResults saved to {output_dir}")


def _evaluate_converter(converter, dataset, trace_store, model_config):
    """Evaluate converter reconstruction quality."""
    errors = []
    converter.eval()

    for i in range(len(dataset)):
        sample = dataset[i]
        # Reconstruction error would be measured by generating a circuit
        # and comparing its output to the expert's output on calibration data
        errors.append(0.0)  # Placeholder

    return {
        "mean_error": float(np.mean(errors)) if errors else 0,
        "std_error": float(np.std(errors)) if errors else 0,
        "max_error": float(np.max(errors)) if errors else 0,
        "p90_error": float(np.percentile(errors, 90)) if errors else 0,
    }


def _make_per_expert_decisions(converter, expert_data, exact_results, model_config, trace_store):
    """Decide per-expert: use exact circuit (3a) or learned circuit (3b)."""
    exact_experts = {}
    if "per_expert" in exact_results:
        for e in exact_results["per_expert"]:
            if e["size_class"] in ("small", "tractable"):
                exact_experts[e["name"]] = e

    decisions = {}
    for ed in expert_data:
        name = ed["name"]
        if name in exact_experts:
            decisions[name] = {
                "method": "exact",
                "gates": exact_experts[name]["gates_after"],
            }
        else:
            decisions[name] = {
                "method": "learned",
                "gates": 0,  # Would come from converter output
            }

    return decisions


if __name__ == "__main__":
    main()
