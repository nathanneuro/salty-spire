"""Step 1: Gentle quantization baseline (4-bit / 8-bit).

Establishes how much quality loss is attributable to quantization alone
using conservative off-the-shelf techniques. This is a sanity check --
the numbers should match literature expectations for similar models.

Usage:
    python -m monet_logic_circuit.scripts.step1_gentle_quant --config configs/step1_gentle_quant.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import torch

from monet_logic_circuit.models.monet_loader import load_monet_model
from monet_logic_circuit.data.calibration import load_calibration_data, load_calibration_cache
from monet_logic_circuit.data.expert_traces import ExpertTraceStore
from monet_logic_circuit.eval.perplexity import compute_perplexity
from monet_logic_circuit.eval.downstream import (
    run_downstream_eval,
    compute_quality_delta,
    format_results_table,
)
from monet_logic_circuit.eval.profiling import profile_model
from monet_logic_circuit.quantization.gentle import (
    apply_gentle_quantization,
    measure_per_expert_reconstruction,
)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Gentle quantization baseline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline reference numbers
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
    print("Loading calibration data...")
    cal_cache = baseline_dir / "calibration_cache.pt"
    if cal_cache.exists():
        cal_data = load_calibration_cache(cal_cache)
    else:
        cal_data = load_calibration_data(
            dataset_name=config["calibration"]["dataset"],
            split=config["calibration"]["split"],
            num_tokens=config["calibration"]["num_samples"] * config["calibration"]["seq_length"],
            seq_length=config["calibration"]["seq_length"],
            seed=config["calibration"]["seed"],
            tokenizer=tokenizer,
        )

    # Apply gentle quantization
    print(f"Applying {config['quantization']['method']} {config['quantization']['bits']}-bit quantization...")
    method_config = config["quantization"].get(config["quantization"]["method"], {})

    quantized_model = apply_gentle_quantization(
        model,
        tokenizer,
        method=config["quantization"]["method"],
        bits=config["quantization"]["bits"],
        calibration_data=cal_data,
        config=method_config,
        targets=config["quantization"]["targets"],
    )

    results = {}

    # Evaluate perplexity
    if config["eval"]["perplexity"]["enabled"]:
        print("Computing perplexity...")
        ppl = compute_perplexity(
            quantized_model, cal_data.data,
            stride=config["eval"]["perplexity"]["stride"],
            max_length=config["eval"]["perplexity"]["max_length"],
            device=device,
        )
        results["perplexity"] = ppl
        ppl_delta = ppl["loss_nats"] - baseline_results.get("perplexity", {}).get("loss_nats", 0)
        print(f"  Perplexity: {ppl['perplexity']:.2f} (delta nats: {ppl_delta:+.4f})")

    # Downstream
    if config["eval"]["downstream"]["enabled"]:
        print("Running downstream evaluation...")
        downstream = run_downstream_eval(
            quantized_model, tokenizer,
            tasks=config["eval"]["downstream"]["tasks"],
            num_fewshot=config["eval"]["downstream"]["num_fewshot"],
            device=str(device),
        )
        results["downstream"] = downstream

        if "downstream" in baseline_results:
            deltas = compute_quality_delta(baseline_results["downstream"], downstream)
            results["downstream_delta"] = deltas
            print(format_results_table(downstream, baseline_results.get("downstream")))

    # Per-expert reconstruction error
    if config["expert_analysis"]["per_expert_reconstruction_error"]:
        print("Measuring per-expert reconstruction error...")
        trace_store = ExpertTraceStore(baseline_dir / "traces")
        errors = measure_per_expert_reconstruction(model, quantized_model, trace_store)
        results["per_expert_reconstruction_error"] = errors
        if errors:
            import numpy as np
            err_vals = list(errors.values())
            print(f"  Mean NMSE: {np.mean(err_vals):.6f}")
            print(f"  P90 NMSE: {np.percentile(err_vals, 90):.6f}")
            print(f"  Max NMSE: {np.max(err_vals):.6f}")

    # Profiling
    if config["profiling"]["enabled"]:
        print("Profiling quantized model...")
        profile = profile_model(
            quantized_model, tokenizer,
            num_batches=config["profiling"]["num_batches"],
            device=str(device),
        )
        results["profiling"] = profile.to_dict()

    # Go/no-go check
    go_no_go = _check_go_no_go(results, baseline_results, config.get("go_no_go", {}))
    results["go_no_go"] = go_no_go
    if not go_no_go["pass"]:
        print(f"\n  WARNING: Go/no-go check FAILED: {go_no_go['reason']}")
        print("  Investigate before proceeding to Step 2.")
    else:
        print("\n  Go/no-go: PASS")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


def _check_go_no_go(results: dict, baseline: dict, thresholds: dict) -> dict:
    """Check go/no-go criteria for Step 1."""
    max_ppl_delta = thresholds.get("max_perplexity_delta_nats", 0.1)
    max_downstream_loss = thresholds.get("max_downstream_loss_pct", 2.0)

    ppl_delta = 0
    if "perplexity" in results and "perplexity" in baseline:
        ppl_delta = results["perplexity"]["loss_nats"] - baseline["perplexity"]["loss_nats"]

    downstream_loss = 0
    if "downstream_delta" in results:
        for task, metrics in results["downstream_delta"].items():
            for metric, delta in metrics.items():
                if "acc" in metric.lower():
                    downstream_loss = max(downstream_loss, -delta * 100)

    if ppl_delta > max_ppl_delta:
        return {"pass": False, "reason": f"Perplexity delta {ppl_delta:.4f} > {max_ppl_delta}"}
    if downstream_loss > max_downstream_loss:
        return {"pass": False, "reason": f"Downstream loss {downstream_loss:.1f}% > {max_downstream_loss}%"}

    return {"pass": True, "reason": "Within thresholds", "ppl_delta": ppl_delta, "downstream_loss_pct": downstream_loss}


def _load_tokenizer(checkpoint: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


if __name__ == "__main__":
    main()
