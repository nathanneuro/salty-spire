# Monet -> Logic Circuit Conversion

Research project investigating conversion of Monet mixture-of-experts language model experts into logic circuits for efficient inference.

## Overview

The core hypothesis: Monet's expert FFNs, once aggressively quantized (especially to ternary weights), become discrete functions that are structurally close to logic circuits. Converting them to actual logic circuits enables bit-operation-based inference that is dramatically faster than float arithmetic.

## Pipeline

| Step | Goal | Key Metric |
|------|------|------------|
| 0 | Baseline pretrained Monet | Perplexity, downstream accuracy, speed |
| 1 | Gentle quantization (8/4-bit) | Quality delta vs Step 0 |
| 2 | Aggressive quantization (2/1.58-bit) | Quality delta, expert output cardinality |
| 3a | Exact conversion (Aytekin trees) | Circuit size per expert |
| 3b | Learned converter | Reconstruction error distribution |
| 3c | End-to-end replacement | Full model quality + speed |
| 3d | Selective fallback | Hybrid model final numbers |

## Project Structure

```
monet_logic_circuit/
├── configs/           # YAML experiment configs (one per step)
├── scripts/           # Runnable experiment scripts
├── models/            # Monet model loading, expert wrappers
├── data/              # Calibration dataset, expert trace I/O
├── quantization/      # Gentle + aggressive quantization
├── conversion/        # Exact (Aytekin) + learned circuit conversion
└── eval/              # Perplexity, downstream, profiling, expert analysis
```

## Usage

```bash
# Step 0: Baseline evaluation
python -m monet_logic_circuit.scripts.step0_baseline --config configs/step0_baseline.yaml

# Step 1: Gentle quantization
python -m monet_logic_circuit.scripts.step1_gentle_quant --config configs/step1_gentle_quant.yaml

# Step 2: Aggressive quantization
python -m monet_logic_circuit.scripts.step2_aggressive_quant --config configs/step2_aggressive_quant.yaml

# Step 3a: Exact conversion
python -m monet_logic_circuit.scripts.step3a_exact_conversion --config configs/step3a_exact_conversion.yaml

# Step 3b: Learned converter
python -m monet_logic_circuit.scripts.step3b_learned_converter --config configs/step3b_learned_converter.yaml

# Step 3c: End-to-end evaluation
python -m monet_logic_circuit.scripts.step3c_end_to_end --config configs/step3c_end_to_end.yaml

# Step 3d: Selective fallback
python -m monet_logic_circuit.scripts.step3d_selective_fallback --config configs/step3d_selective_fallback.yaml
```

## Go/No-Go Criteria

- **After Step 1:** If gentle quantization loses >1-2% downstream or >0.1 nats perplexity, investigate Monet-specific quantization sensitivity before proceeding.
- **After Step 2:** If aggressive quantization loses >5% downstream or >0.5 nats perplexity even with fine-tuning, reassess the project ceiling.
- **After Step 3:** Target regime is "within 1-2% of baseline quality at 10x+ speed improvement."
