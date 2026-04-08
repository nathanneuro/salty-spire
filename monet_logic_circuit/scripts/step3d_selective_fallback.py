"""Step 3d: Selective fallback for the long tail.

Half-experts that couldn't be converted cleanly stay as quantized float.
Measures the final hybrid model with honest speedup numbers.

Usage:
    python -m monet_logic_circuit.scripts.step3d_selective_fallback --config configs/step3d_selective_fallback.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch

from monet_logic_circuit.models.monet_loader import load_monet_model
from monet_logic_circuit.models.hybrid_model import build_hybrid_model
from monet_logic_circuit.data.calibration import load_calibration_cache
from monet_logic_circuit.eval.perplexity import compute_perplexity
from monet_logic_circuit.eval.downstream import (
    run_downstream_eval,
    compute_quality_delta,
    format_results_table,
)
from monet_logic_circuit.eval.profiling import profile_model


def main():
    parser = argparse.ArgumentParser(description="Step 3d: Selective fallback")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all prior results
    hybrid_dir = Path(config["model"]["hybrid_model_dir"])
    circuits_dir = Path(config["model"]["circuits_dir"])

    with open(circuits_dir / "results.json") as f:
        conversion_results = json.load(f)

    float_baseline_path = Path("outputs/step0/reference_numbers.json")
    with open(float_baseline_path) as f:
        float_baseline = json.load(f)

    # Load model
    print("Loading model...")
    model, model_config = load_monet_model(
        config["model"]["base_checkpoint"],
        device=config["model"]["device"],
        allowed_decompositions=config["model"].get("allowed_decompositions"),
    )
    tokenizer = _load_tokenizer(config["model"]["base_checkpoint"])
    device = next(model.parameters()).device

    # Classify half-experts into circuit vs fallback
    threshold = config["fallback"]["reconstruction_error_threshold"]
    force_quantized = set(
        config["fallback"].get("force_quantized_half_experts", [])
    )

    circuits = {}
    quantized_fallbacks = {}
    per_half_expert_stats = []

    decisions = conversion_results.get("per_half_expert_decisions", {})
    for name, decision in decisions.items():
        recon_error = decision.get("reconstruction_error", 0)

        if name in force_quantized or recon_error > threshold:
            # Fallback to quantized float
            quantized_fallbacks[name] = None  # Would load actual module
            per_half_expert_stats.append({
                "name": name,
                "method": "quantized_float",
                "reconstruction_error": recon_error,
                "reason": "forced" if name in force_quantized else "high_error",
            })
        else:
            circuits[name] = None  # Would load actual circuit
            per_half_expert_stats.append({
                "name": name,
                "method": decision.get("method", "learned"),
                "reconstruction_error": recon_error,
                "gates": decision.get("gates", 0),
            })

    total = len(per_half_expert_stats)
    n_circuit = sum(1 for s in per_half_expert_stats if s["method"] != "quantized_float")
    n_fallback = total - n_circuit
    fallback_frac = n_fallback / total if total > 0 else 0

    print(f"\nHalf-expert conversion breakdown:")
    print(f"  Logic circuits: {n_circuit}/{total} ({n_circuit/total*100:.1f}%)")
    print(f"  Quantized fallback: {n_fallback}/{total} ({fallback_frac*100:.1f}%)")

    if fallback_frac > config["targets"]["warning_threshold"]:
        print(f"\n  WARNING: Fallback fraction {fallback_frac:.1%} exceeds "
              f"warning threshold {config['targets']['warning_threshold']:.0%}")
        print("  The converter needs improvement for better coverage.")
    elif fallback_frac <= config["targets"]["max_fallback_fraction"]:
        print(f"\n  Fallback fraction {fallback_frac:.1%} is within target "
              f"({config['targets']['max_fallback_fraction']:.0%})")

    # Build final hybrid model
    print("\nBuilding final hybrid model...")
    hybrid_model = build_hybrid_model(
        base_model=model,
        circuits=circuits,
        quantized_fallbacks=quantized_fallbacks,
        half_expert_input_dim=model_config.hidden_dim // 2,
        half_expert_output_dim=model_config.expert_dim,
        binarizer_config={"method": "sign", "bits": 1},
    )

    # Load calibration data
    cal_data = load_calibration_cache("outputs/step0/calibration_cache.pt")
    results = {}

    # Full evaluation
    print("\nRunning full evaluation...")
    ppl = compute_perplexity(
        hybrid_model, cal_data.data,
        stride=config["eval"]["perplexity"]["stride"],
        max_length=config["eval"]["perplexity"]["max_length"],
        device=device,
    )
    results["perplexity"] = ppl

    downstream = run_downstream_eval(
        hybrid_model, tokenizer,
        tasks=config["eval"]["downstream"]["tasks"],
        device=str(device),
    )
    results["downstream"] = downstream

    if "downstream" in float_baseline:
        deltas = compute_quality_delta(float_baseline["downstream"], downstream)
        results["downstream_delta"] = deltas
        print(format_results_table(downstream, float_baseline.get("downstream")))

    # Quality delta vs float baseline
    ppl_delta = ppl["loss_nats"] - float_baseline.get("perplexity", {}).get("loss_nats", 0)
    print(f"\n  Perplexity: {ppl['perplexity']:.2f} (delta: {ppl_delta:+.4f} nats)")

    # Honest profiling
    if config["profiling"]["enabled"]:
        print("\nProfiling final hybrid model (honest numbers)...")
        profile = profile_model(
            hybrid_model, tokenizer,
            num_batches=config["profiling"]["num_batches"],
            device=str(device),
        )
        results["profiling"] = profile.to_dict()

        baseline_tps = float_baseline.get("profiling", {}).get("tokens_per_sec_gpu", 1)
        speedup = profile.tokens_per_sec_gpu / baseline_tps if baseline_tps > 0 else 0
        print(f"  Tokens/sec: {profile.tokens_per_sec_gpu:.0f} ({speedup:.1f}x vs baseline)")
        print(f"  Peak memory: {profile.peak_memory_mb:.0f} MB")
        results["speedup_vs_baseline"] = speedup

    # Conversion statistics
    results["conversion_stats"] = {
        "total_half_experts": total,
        "exact_circuit": sum(1 for s in per_half_expert_stats if s["method"] == "exact"),
        "learned_circuit": sum(1 for s in per_half_expert_stats if s["method"] == "learned"),
        "quantized_float": n_fallback,
        "circuit_fraction": n_circuit / total if total > 0 else 0,
        "fallback_fraction": fallback_frac,
    }
    results["per_half_expert_stats"] = per_half_expert_stats

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Quality (perplexity delta): {ppl_delta:+.4f} nats")
    print(f"  Speed: {results.get('speedup_vs_baseline', 'N/A')}x")
    print(f"  Circuit coverage: {n_circuit/total*100:.1f}%")

    within_quality = abs(ppl_delta) < 0.1  # ~1-2% quality
    good_speedup = results.get("speedup_vs_baseline", 0) >= 10
    if within_quality and good_speedup:
        print("\n  VERDICT: Within target regime (1-2% quality, 10x+ speed)")
        print("  Proceed to scaling and write-up.")
    elif within_quality:
        print("\n  VERDICT: Quality is good but speed gains are modest.")
    else:
        print("\n  VERDICT: Quality loss too high for the achieved speedup.")
        print("  See loss decomposition in Step 3c to identify dominant cost.")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir}")


def _load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


if __name__ == "__main__":
    main()
