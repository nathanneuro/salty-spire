# Monet -> Logic Circuit Conversion

Research project investigating conversion of Monet mixture-of-experts language
model experts into logic circuits for efficient inference.

## Overview

The core hypothesis: Monet's expert FFNs, once aggressively quantized
(especially to ternary weights), become discrete functions that are
structurally close to logic circuits. Converting them to actual logic
circuits enables bit-operation-based inference that is dramatically faster
than float arithmetic.

## What we're actually converting

Monet's headline "262,144 experts per layer" is a **product-key
decomposition**: each layer stores `2N = 1024` **half-experts**
(`N = 512` per axis), and the effective expert is the Cartesian product of
one half-expert from each axis. There are `N^2 = 262_144` effective
experts, but only `2N = 1024` genuinely independent functions to convert
per layer.

At 6–12 layers per model, that's ~10k circuits to build per model — well
inside the range where per-half-expert distillation is feasible even
without the Step 3b learned converter. The conversion pipeline operates at
the **half-expert granularity** throughout.

Each half-expert has a very small effective dimension: `d_expert = 12` at
850M, `16` at 1.4B, `24` at 4.1B. This is tiny compared to a normal FFN
hidden size and is the main reason we expect Aytekin-style exact tree
extraction to be tractable here.

### VD only

The paper ships two decomposition variants: **VD (vertical)** and **HD
(horizontal)**. VD outperforms HD in the paper, every specialized Monet
variant (chat, code, vision) is VD, and VD half-experts are genuinely
independent functions of disjoint input slices — which makes them cleanly
convertible in isolation. HD half-experts are dynamically composed, making
conversion substantially more entangled. **We target VD throughout and
treat HD as a late-stage comparison point only.**

See [`docs/model_selection.md`](docs/model_selection.md) for full
architecture notes and the rationale behind these choices.

## Pipeline

| Step | Goal | Key Metric |
|------|------|------------|
| 0 | Baseline pretrained Monet | Perplexity, downstream accuracy, speed |
| 1 | Gentle quantization (8/4-bit) | Quality delta vs Step 0 |
| 2 | Aggressive quantization (2/1.58-bit) | Quality delta, half-expert output cardinality |
| 3a | Exact conversion (Aytekin trees) | Circuit size per half-expert |
| 3b | Learned converter | Reconstruction error distribution |
| 3c | End-to-end replacement | Full model quality + speed |
| 3d | Selective fallback | Hybrid model final numbers |

## Model progression

| Stage | Checkpoint | Rationale |
|-------|-----------|-----------|
| Primary development | `MonetLLM/monet-vd-850M-100BT-hf` | Smallest VD, `d_expert = 12`, fastest iteration |
| Scaling | `MonetLLM/monet-vd-1.4B-100BT-hf` | Paper's reference scale, `d_expert = 16` |
| Full-scale validation | `MonetLLM/monet-vd-4.1B-100BT-hf` | `d_expert = 24`, stresses the approach |
| Specialized demos | `monet-vd-1.4B-100BT-chat-hf`, `codemonet-vd-1.4B-100BT-hf` | "Preserves capabilities?" story |
| Late-stage comparison only | `monet-hd-*` checkpoints | "Generalizes beyond VD?" |

All development-default configs point at `MonetLLM/monet-vd-850M-100BT-hf`.
To switch scale, override `model.checkpoint` on the command line or in a
config override file.

## Project Structure

```
monet_logic_circuit/
├── configs/           # YAML experiment configs (one per step)
├── docs/              # Design notes (model selection, architecture)
├── scripts/           # Runnable experiment scripts
├── models/            # Monet model loading, half-expert wrappers, registry
├── data/              # Calibration dataset, half-expert trace I/O
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

- **After Step 1:** If gentle quantization loses >1-2% downstream or
  >0.1 nats perplexity, investigate Monet-specific quantization sensitivity
  before proceeding.
- **After Step 2:** If aggressive quantization loses >5% downstream or
  >0.5 nats perplexity even with fine-tuning, reassess the project
  ceiling.
- **After Step 3:** Target regime is "within 1-2% of baseline quality at
  10x+ speed improvement."
- **After Step 3a on 850M:** If most half-experts produce circuits above
  the `tractable` threshold (100k gates), the Aytekin approach has blown
  up on the smallest model and scaling is not worth attempting. Route all
  effort to Step 3b.
- **850M -> 1.4B scaling break:** Investigate `d_expert` sensitivity
  before moving to 4.1B.
