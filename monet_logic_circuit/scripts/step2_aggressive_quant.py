"""Step 2: Aggressive quantization baseline (2-bit / 1.58-bit ternary).

Pushes quantization as far as it can go. The ternary variant is
particularly important as the substrate for logic circuit conversion.

Usage:
    python -m monet_logic_circuit.scripts.step2_aggressive_quant --config configs/step2_aggressive_quant.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch

from monet_logic_circuit.models.monet_loader import load_monet_model
from monet_logic_circuit.data.calibration import load_calibration_cache, load_calibration_data
from monet_logic_circuit.data.expert_traces import ExpertTraceStore
from monet_logic_circuit.eval.perplexity import compute_perplexity
from monet_logic_circuit.eval.downstream import (
    run_downstream_eval,
    compute_quality_delta,
    format_results_table,
)
from monet_logic_circuit.eval.profiling import profile_model
from monet_logic_circuit.quantization.aggressive import (
    apply_aggressive_quantization,
    QuantizationSweepConfig,
    run_quantization_sweep,
    compute_effective_output_cardinality,
)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Aggressive quantization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-sweep", action="store_true", help="Skip sweep, use config defaults")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baselines
    baseline_dir = Path(config["baseline"]["reference_dir"])
    with open(baseline_dir / "reference_numbers.json") as f:
        baseline_results = json.load(f)

    # Load model
    print("Loading model...")
    model, model_config = load_monet_model(
        config["model"]["checkpoint"],
        device=config["model"]["device"],
    )
    tokenizer = _load_tokenizer(config["model"]["checkpoint"])
    device = next(model.parameters()).device

    # Load calibration data
    cal_cache = baseline_dir / "calibration_cache.pt"
    if cal_cache.exists():
        cal_data = load_calibration_cache(cal_cache)
    else:
        cal_data = load_calibration_data(
            dataset_name=config["calibration"]["dataset"],
            split=config["calibration"]["split"],
            num_tokens=config["calibration"]["num_samples"] * config["calibration"]["seq_length"],
            tokenizer=tokenizer,
        )

    trace_store = ExpertTraceStore(baseline_dir / "traces")
    results = {}

    if not args.skip_sweep:
        # Run quantization sweep
        print("Running quantization sweep...")
        sweep_config = QuantizationSweepConfig(
            bits_options=config["quantization"]["sweep"]["bits"],
            scope_options=config["quantization"]["sweep"]["scope"],
            scale_options=config["quantization"]["sweep"]["scale_granularity"],
        )
        print(f"  {sweep_config.num_configs()} configurations to evaluate")

        def eval_fn(m):
            ppl = compute_perplexity(m, cal_data.data, device=device)
            ds = run_downstream_eval(
                m, tokenizer,
                tasks=config["eval"]["downstream"]["tasks"],
                device=str(device),
                limit=200,  # Fast eval for sweep
            )
            return {"perplexity": ppl["perplexity"], "loss_nats": ppl["loss_nats"], "downstream": ds}

        sweep_results = run_quantization_sweep(
            model, tokenizer, sweep_config, cal_data, eval_fn
        )
        results["sweep"] = sweep_results

        # Find best config by quality-per-bit
        best_idx = _find_best_config(sweep_results)
        best_config = sweep_results[best_idx]
        print(f"  Best config: {best_config['bits']}-bit, {best_config['scope']}, {best_config['scale_granularity']}")
        results["best_config"] = best_config
    else:
        best_config = {"bits": 1.58, "scope": "weights_only", "scale_granularity": "per_expert"}

    # Apply best quantization config
    import copy
    print(f"\nApplying best config: {best_config['bits']}-bit quantization...")
    best_model = copy.deepcopy(model)
    apply_aggressive_quantization(
        best_model,
        bits=best_config["bits"],
        scope=best_config["scope"],
        scale_granularity=best_config["scale_granularity"],
    )

    # Also build the ternary variant (always, regardless of best config)
    print("Building ternary (1.58-bit) variant...")
    ternary_model = copy.deepcopy(model)
    apply_aggressive_quantization(
        ternary_model,
        bits=1.58,
        scope="weights_only",
        scale_granularity="per_expert",
    )

    # QAT fine-tuning if enabled
    if config["fine_tuning"]["enabled"]:
        print("Running quantization-aware fine-tuning...")
        best_model = _run_qat(best_model, config["fine_tuning"], tokenizer, device)
        ternary_model = _run_qat(ternary_model, config["fine_tuning"], tokenizer, device)

    # Evaluate both variants
    for name, quant_model in [("best", best_model), ("ternary", ternary_model)]:
        print(f"\nEvaluating {name} variant...")

        ppl = compute_perplexity(
            quant_model, cal_data.data,
            stride=config["eval"]["perplexity"]["stride"],
            max_length=config["eval"]["perplexity"]["max_length"],
            device=device,
        )
        results[f"{name}_perplexity"] = ppl

        downstream = run_downstream_eval(
            quant_model, tokenizer,
            tasks=config["eval"]["downstream"]["tasks"],
            device=str(device),
        )
        results[f"{name}_downstream"] = downstream

        if "downstream" in baseline_results:
            deltas = compute_quality_delta(baseline_results["downstream"], downstream)
            results[f"{name}_downstream_delta"] = deltas
            print(format_results_table(downstream, baseline_results.get("downstream")))

        # Effective output cardinality
        print(f"  Computing output cardinality for {name}...")
        cardinalities = compute_effective_output_cardinality(quant_model, trace_store)
        results[f"{name}_cardinality"] = cardinalities
        if cardinalities:
            cards = list(cardinalities.values())
            print(f"  Mean cardinality: {np.mean(cards):.0f}")
            print(f"  Min cardinality: {np.min(cards)}")
            low_card = sum(1 for c in cards if c <= 5)
            print(f"  Experts with <=5 distinct outputs: {low_card}/{len(cards)}")

    # Profile
    if config["profiling"]["enabled"]:
        print("\nProfiling ternary model...")
        profile = profile_model(ternary_model, tokenizer, device=str(device))
        results["ternary_profiling"] = profile.to_dict()

    # Go/no-go
    go_no_go = _check_go_no_go(results, baseline_results, config.get("go_no_go", {}))
    results["go_no_go"] = go_no_go
    if not go_no_go["pass"]:
        print(f"\n  WARNING: Go/no-go FAILED: {go_no_go['reason']}")
    else:
        print("\n  Go/no-go: PASS")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save model checkpoints
    if config["outputs"].get("save_ternary_model"):
        ternary_dir = output_dir / "ternary_model"
        ternary_dir.mkdir(exist_ok=True)
        torch.save(ternary_model.state_dict(), ternary_dir / "model.pt")

    print(f"\nResults saved to {output_dir}")


def _find_best_config(sweep_results: list[dict]) -> int:
    """Find config with best quality-per-bit tradeoff."""
    scores = []
    for r in sweep_results:
        ppl = r.get("perplexity", float("inf"))
        bits = r.get("bits", 4)
        # Lower perplexity and lower bits are both good
        score = -ppl / bits  # Negative ppl / bits = higher is better
        scores.append(score)
    return int(np.argmax(scores))


def _run_qat(model, ft_config, tokenizer, device):
    """Run quantization-aware fine-tuning."""
    from torch.utils.data import DataLoader

    model.train()
    # Freeze non-expert parameters if configured
    if ft_config.get("freeze_non_expert", True):
        for name, param in model.named_parameters():
            if not any(kw in name for kw in ["expert", "moe"]):
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ft_config["learning_rate"],
    )

    # Training loop placeholder
    max_steps = ft_config.get("max_steps", 5000)
    print(f"  QAT: {max_steps} steps (placeholder -- needs training data integration)")

    model.eval()
    return model


def _check_go_no_go(results, baseline, thresholds):
    max_ppl = thresholds.get("max_perplexity_delta_nats", 0.5)
    max_ds = thresholds.get("max_downstream_loss_pct", 5.0)

    # Check ternary variant (the one we care about for circuit conversion)
    ppl_delta = 0
    if "ternary_perplexity" in results and "perplexity" in baseline:
        ppl_delta = results["ternary_perplexity"]["loss_nats"] - baseline["perplexity"]["loss_nats"]

    downstream_loss = 0
    if "ternary_downstream_delta" in results:
        for task, metrics in results["ternary_downstream_delta"].items():
            for metric, delta in metrics.items():
                if "acc" in metric.lower():
                    downstream_loss = max(downstream_loss, -delta * 100)

    if ppl_delta > max_ppl:
        return {"pass": False, "reason": f"Ternary perplexity delta {ppl_delta:.4f} > {max_ppl}"}
    if downstream_loss > max_ds:
        return {"pass": False, "reason": f"Ternary downstream loss {downstream_loss:.1f}% > {max_ds}%"}

    return {"pass": True, "reason": "Within thresholds"}


def _load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


if __name__ == "__main__":
    main()
