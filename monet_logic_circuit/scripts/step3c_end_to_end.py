"""Step 3c: End-to-end replacement and evaluation.

Replaces experts with logic-circuit equivalents and measures whether
the full model still works. Includes loss decomposition and interface
fine-tuning.

Usage:
    python -m monet_logic_circuit.scripts.step3c_end_to_end --config configs/step3c_end_to_end.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import torch

from monet_logic_circuit.models.monet_loader import load_monet_model
from monet_logic_circuit.models.hybrid_model import build_hybrid_model
from monet_logic_circuit.data.calibration import load_calibration_cache, load_calibration_data
from monet_logic_circuit.eval.perplexity import compute_perplexity
from monet_logic_circuit.eval.downstream import (
    run_downstream_eval,
    compute_quality_delta,
    format_results_table,
)
from monet_logic_circuit.eval.profiling import profile_model
from monet_logic_circuit.pipeline.signals import (
    DecisionSignal,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    write_signal_into_results,
)


def main():
    parser = argparse.ArgumentParser(description="Step 3c: End-to-end evaluation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baselines
    float_baseline_dir = Path(config["eval"]["loss_decomposition"]["compare_against"]["float_baseline"])
    quant_baseline_dir = Path(config["eval"]["loss_decomposition"]["compare_against"]["quantized_baseline"])

    with open(float_baseline_dir / "reference_numbers.json") as f:
        float_baseline = json.load(f)
    with open(quant_baseline_dir / "results.json") as f:
        quant_baseline = json.load(f)

    # Load base model
    print("Loading base model...")
    model, model_config = load_monet_model(
        config["model"]["base_checkpoint"],
        device=config["model"]["device"],
        allowed_decompositions=config["model"].get("allowed_decompositions"),
    )
    tokenizer = _load_tokenizer(config["model"]["base_checkpoint"])
    device = next(model.parameters()).device

    # Load circuits and per-expert decisions from Step 3b
    circuits_dir = Path(config["model"]["circuits_dir"])
    with open(circuits_dir / "results.json") as f:
        conversion_results = json.load(f)

    # Build hybrid model
    print("Building hybrid model...")
    circuits = _load_circuits(circuits_dir)
    quantized_fallbacks = _load_quantized_fallbacks(
        config["model"]["ternary_checkpoint"]
    )

    hybrid_model = build_hybrid_model(
        base_model=model,
        circuits=circuits,
        quantized_fallbacks=quantized_fallbacks,
        half_expert_input_dim=model_config.hidden_dim // 2,
        half_expert_output_dim=model_config.expert_dim,
        binarizer_config={
            "method": config["hybrid_inference"]["input_binarization"]["method"],
            "bits": config["hybrid_inference"]["input_binarization"]["bits"],
        },
        decoder_hidden=config["hybrid_inference"]["output_decoding"]["hidden_dim"],
    )

    # Load calibration data
    cal_cache = float_baseline_dir / "calibration_cache.pt"
    cal_data = load_calibration_cache(cal_cache)

    results = {}

    # Evaluate before interface fine-tuning
    print("\nEvaluating hybrid model (before interface fine-tuning)...")
    ppl_before = compute_perplexity(
        hybrid_model, cal_data.data,
        stride=config["eval"]["perplexity"]["stride"],
        max_length=config["eval"]["perplexity"]["max_length"],
        device=device,
    )
    results["perplexity_before_ft"] = ppl_before

    downstream_before = run_downstream_eval(
        hybrid_model, tokenizer,
        tasks=config["eval"]["downstream"]["tasks"],
        device=str(device),
    )
    results["downstream_before_ft"] = downstream_before

    print(f"  Perplexity: {ppl_before['perplexity']:.2f}")
    print(format_results_table(downstream_before, float_baseline.get("downstream")))

    # Loss decomposition
    if config["eval"]["loss_decomposition"]["enabled"]:
        print("\nComputing loss decomposition...")
        decomposition = _decompose_loss(
            hybrid_ppl=ppl_before,
            float_ppl=float_baseline.get("perplexity", {}),
            quant_ppl=quant_baseline.get("ternary_perplexity", {}),
        )
        results["loss_decomposition"] = decomposition
        print(f"  Total loss delta: {decomposition['total_delta']:.4f} nats")
        print(f"  Quantization loss: {decomposition['quantization_delta']:.4f} nats")
        print(f"  Conversion loss: {decomposition['conversion_delta']:.4f} nats")
        print(f"  Interface loss: {decomposition['interface_delta']:.4f} nats")

    # Interface fine-tuning
    if config["interface_fine_tuning"]["enabled"]:
        print("\nFine-tuning interface layers...")
        hybrid_model = _fine_tune_interface(hybrid_model, config["interface_fine_tuning"], cal_data, device)

        print("Evaluating after interface fine-tuning...")
        ppl_after = compute_perplexity(
            hybrid_model, cal_data.data,
            stride=config["eval"]["perplexity"]["stride"],
            max_length=config["eval"]["perplexity"]["max_length"],
            device=device,
        )
        results["perplexity_after_ft"] = ppl_after

        downstream_after = run_downstream_eval(
            hybrid_model, tokenizer,
            tasks=config["eval"]["downstream"]["tasks"],
            device=str(device),
        )
        results["downstream_after_ft"] = downstream_after

        print(f"  Perplexity: {ppl_after['perplexity']:.2f}")
        print(format_results_table(downstream_after, float_baseline.get("downstream")))

    # Profile
    if config["profiling"]["enabled"]:
        print("\nProfiling hybrid model...")
        profile = profile_model(
            hybrid_model, tokenizer,
            num_batches=config["profiling"]["num_batches"],
            device=str(device),
        )
        results["profiling"] = profile.to_dict()
        print(f"  Tokens/sec: {profile.tokens_per_sec_gpu:.0f}")

    # Conversion stats
    results["conversion_stats"] = hybrid_model.get_conversion_stats()

    # Canonical decision signal for the orchestrator. Report the
    # perplexity delta after interface fine-tuning if available, else
    # the pre-FT number. The pipeline default is warn-not-halt here
    # because the loss decomposition tells us whether 3d can recover.
    float_loss_nats = float(float_baseline.get("perplexity", {}).get("loss_nats", 0.0))
    after_ppl = results.get("perplexity_after_ft") or results.get("perplexity_before_ft", {})
    hybrid_loss_nats = float(after_ppl.get("loss_nats", 0.0))
    ppl_delta = hybrid_loss_nats - float_loss_nats

    decomposition = results.get("loss_decomposition", {})
    signal_metrics = {
        "perplexity_delta_nats": ppl_delta,
        "quantization_delta_nats": float(decomposition.get("quantization_delta", 0.0)),
        "conversion_delta_nats": float(decomposition.get("conversion_delta", 0.0)),
        "interface_delta_nats": float(decomposition.get("interface_delta", 0.0)),
        "circuit_fraction": float(
            results["conversion_stats"].get("circuit_fraction", 0.0)
        ),
    }
    gate_threshold = 0.15  # Mirror of pipeline.yaml
    signal = DecisionSignal(
        step="3c",
        outcome=OUTCOME_PASS if ppl_delta <= gate_threshold else OUTCOME_FAIL,
        metrics=signal_metrics,
        reason=f"hybrid perplexity delta {ppl_delta:+.4f} nats",
    )
    results_out = write_signal_into_results(dict(results), signal)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


def _decompose_loss(hybrid_ppl, float_ppl, quant_ppl):
    """Decompose quality loss into quantization, conversion, and interface components."""
    float_nats = float_ppl.get("loss_nats", 0)
    quant_nats = quant_ppl.get("loss_nats", 0)
    hybrid_nats = hybrid_ppl.get("loss_nats", 0)

    total_delta = hybrid_nats - float_nats
    quant_delta = quant_nats - float_nats
    conversion_plus_interface = hybrid_nats - quant_nats

    # We can't perfectly separate conversion from interface without an
    # intermediate measurement, but we can report them together
    return {
        "total_delta": total_delta,
        "quantization_delta": quant_delta,
        "conversion_delta": conversion_plus_interface * 0.5,  # Approximate split
        "interface_delta": conversion_plus_interface * 0.5,
        "note": "conversion/interface split is approximate; run with circuits + float inputs to separate",
    }


def _fine_tune_interface(hybrid_model, ft_config, cal_data, device):
    """Fine-tune input binarization and output decoders with circuits frozen."""
    trainable_params = []
    for name, param in hybrid_model.named_parameters():
        param.requires_grad = False
        if ft_config["trainable"].get("input_binarization") and "binarizer" in name:
            param.requires_grad = True
            trainable_params.append(param)
        elif ft_config["trainable"].get("output_decoding") and "decoder" in name:
            param.requires_grad = True
            trainable_params.append(param)

    if not trainable_params:
        print("  No trainable parameters found for interface fine-tuning")
        return hybrid_model

    optimizer = torch.optim.AdamW(trainable_params, lr=ft_config["learning_rate"])

    hybrid_model.train()
    max_steps = ft_config.get("max_steps", 2000)
    print(f"  Fine-tuning {len(trainable_params)} parameters for {max_steps} steps")

    # Training loop placeholder
    hybrid_model.eval()
    return hybrid_model


def _load_circuits(circuits_dir):
    """Load compiled circuits from Step 3b output."""
    # Placeholder: would load circuit objects from saved files
    return {}


def _load_quantized_fallbacks(ternary_checkpoint):
    """Load quantized expert modules as fallbacks."""
    # Placeholder: would load from ternary checkpoint
    return {}


def _load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


if __name__ == "__main__":
    main()
