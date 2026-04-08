"""Step 0: Baseline evaluation of unmodified pretrained Monet.

Establishes ground truth numbers that every subsequent step is measured against:
- Perplexity on held-out data
- Downstream task accuracy
- Inference profiling (speed, memory, per-component FLOP breakdown)
- Half-expert population characterization (activation frequency, input/output stats, clustering)
- Cached per-half-expert (input, output) calibration traces

Usage:
    python -m monet_logic_circuit.scripts.step0_baseline --config configs/step0_baseline.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import torch

from monet_logic_circuit.models.monet_loader import (
    load_monet_model,
    get_half_expert_modules,
)
from monet_logic_circuit.models.expert_wrapper import (
    HalfExpertPopulation,
    HalfExpertWrapper,
)
from monet_logic_circuit.data.calibration import (
    load_calibration_data,
    save_calibration_cache,
)
from monet_logic_circuit.data.expert_traces import ExpertTraceStore
from monet_logic_circuit.eval.perplexity import compute_perplexity
from monet_logic_circuit.eval.downstream import run_downstream_eval, format_results_table
from monet_logic_circuit.eval.profiling import profile_model, estimate_flops_per_token
from monet_logic_circuit.eval.expert_analysis import (
    compute_activation_frequencies,
    analyze_expert_population,
)
from monet_logic_circuit.pipeline.signals import (
    DecisionSignal,
    OUTCOME_PASS,
    write_signal_into_results,
)


def main():
    parser = argparse.ArgumentParser(description="Step 0: Baseline Monet evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model, model_config = load_monet_model(
        config["model"]["checkpoint"],
        device=config["model"]["device"],
        allowed_decompositions=config["model"].get("allowed_decompositions"),
    )
    tokenizer = _load_tokenizer(config["model"]["checkpoint"])
    device = next(model.parameters()).device

    # Load calibration data
    print("Loading calibration data...")
    cal_data = load_calibration_data(
        dataset_name=config["data"]["calibration"]["dataset"],
        split=config["data"]["calibration"]["split"],
        num_tokens=config["data"]["calibration"]["num_tokens"],
        seed=config["data"]["calibration"]["seed"],
        tokenizer=tokenizer,
    )
    save_calibration_cache(cal_data, output_dir / "calibration_cache.pt")

    results = {}

    # Perplexity
    if config["eval"]["perplexity"]["enabled"]:
        print("Computing perplexity...")
        eval_data = load_calibration_data(
            dataset_name=config["data"]["eval"]["dataset"],
            split=config["data"]["eval"]["split"],
            num_tokens=config["data"]["eval"]["num_tokens"],
            tokenizer=tokenizer,
        )
        ppl_results = compute_perplexity(
            model, eval_data.data,
            stride=config["eval"]["perplexity"]["stride"],
            max_length=config["eval"]["perplexity"]["max_length"],
            device=device,
        )
        results["perplexity"] = ppl_results
        print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
        print(f"  Loss (nats): {ppl_results['loss_nats']:.4f}")

    # Downstream tasks
    if config["eval"]["downstream"]["enabled"]:
        print("Running downstream evaluation...")
        downstream_results = run_downstream_eval(
            model, tokenizer,
            tasks=config["eval"]["downstream"]["tasks"],
            num_fewshot=config["eval"]["downstream"]["num_fewshot"],
            batch_size=config["eval"]["downstream"]["batch_size"],
            device=str(device),
        )
        results["downstream"] = downstream_results
        print(format_results_table(downstream_results))

    # Profiling
    if config["profiling"]["enabled"]:
        print("Profiling inference...")
        profile = profile_model(
            model, tokenizer,
            num_batches=config["profiling"]["num_batches"],
            batch_size=config["profiling"]["batch_size"],
            sequence_length=config["profiling"]["sequence_length"],
            measure_components=config["profiling"]["measure_components"],
            device=str(device),
        )
        results["profiling"] = profile.to_dict()

        flops = estimate_flops_per_token(
            model_config.num_layers, model_config.hidden_dim,
            model_config.expert_dim, model_config.num_attention_heads,
            model_config.vocab_size, model_config.top_k,
            config["profiling"]["sequence_length"],
        )
        results["flops_per_token"] = flops

        print(f"  Expert FFN FLOP fraction: {profile.expert_flop_fraction():.1%}")
        print(f"  Tokens/sec (GPU): {profile.tokens_per_sec_gpu:.0f}")
        print(f"  Peak memory: {profile.peak_memory_mb:.0f} MB")

    # Half-expert population analysis
    if config["expert_analysis"]["enabled"]:
        print("Analyzing half-expert population...")

        # Wrap half-experts. Naming convention (layerL_axisA_heH) is set
        # by HalfExpertWrapper; loaders produce per-layer per-axis modules
        # in a canonical order so we can index them deterministically.
        raw_half_experts = get_half_expert_modules(model)
        n_per_axis = model_config.num_half_experts_per_axis
        per_layer = 2 * n_per_axis  # two axes
        wrapped = []
        for i, (name, mod) in enumerate(raw_half_experts):
            layer_idx = i // per_layer
            within_layer = i % per_layer
            axis = 0 if within_layer < n_per_axis else 1
            half_expert_idx = within_layer % n_per_axis
            wrapped.append(
                HalfExpertWrapper(mod, layer_idx, axis, half_expert_idx)
            )
        population = HalfExpertPopulation(wrapped)

        # Collect traces
        trace_store = ExpertTraceStore(config["traces"]["output_dir"])
        if config["traces"]["save"]:
            print("  Collecting half-expert traces...")
            cal_loader = torch.utils.data.DataLoader(cal_data, batch_size=1)

            for half_expert in population:
                half_expert.start_tracing()

            # Run calibration data through model
            with torch.no_grad():
                for batch in cal_loader:
                    model(batch["input_ids"].to(device))

            for half_expert in population:
                inputs, outputs = half_expert.stop_tracing()
                if len(inputs) > 0:
                    trace_store.save_traces(half_expert.name, inputs, outputs)

            print(f"  Saved traces for {len(trace_store.list_experts())} half-experts")

        # Activation frequencies
        if config["expert_analysis"]["activation_frequency"]:
            print("  Computing activation frequencies...")
            cal_loader = torch.utils.data.DataLoader(cal_data, batch_size=4)
            freqs = compute_activation_frequencies(model, cal_loader, device=str(device))
        else:
            freqs = None

        # Full analysis
        analysis = analyze_expert_population(population, trace_store, freqs)
        results["expert_analysis"] = analysis
        print(f"  Half-experts: {analysis['num_half_experts']}")
        print(f"  Clusters: {analysis['num_clusters']}")

    # Save all results
    with open(output_dir / "reference_numbers.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also write a results.json with a canonical decision signal for the
    # orchestrator. Step 0 always passes; downstream steps gate against
    # the metrics here, not against step 0 itself.
    baseline_metrics = {
        "perplexity": float(results.get("perplexity", {}).get("perplexity", 0.0)),
        "loss_nats": float(results.get("perplexity", {}).get("loss_nats", 0.0)),
    }
    signal = DecisionSignal(
        step="0",
        outcome=OUTCOME_PASS,
        metrics=baseline_metrics,
        reason="baseline reference established",
    )
    results_out = write_signal_into_results(dict(results), signal)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir}")


def _load_tokenizer(checkpoint: str):
    """Load tokenizer from checkpoint or hub."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


if __name__ == "__main__":
    main()
