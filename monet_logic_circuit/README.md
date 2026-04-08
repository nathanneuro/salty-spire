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
├── configs/           # YAML experiment configs (one per step) + pipeline.yaml
├── docs/              # Design notes (model selection, architecture)
├── scripts/           # Runnable experiment scripts + run_pipeline.py
├── models/            # Monet model loading, half-expert wrappers, registry
├── data/              # Calibration dataset, half-expert trace I/O
├── quantization/      # Gentle + aggressive quantization
├── conversion/        # Exact (Aytekin) + learned circuit conversion
├── pipeline/          # Decision signals, gates, orchestrator
└── eval/              # Perplexity, downstream, profiling, expert analysis
```

## Usage

### Automated pipeline (recommended)

The orchestrator runs every step in sequence, gates each step's
``decision_signal`` against the thresholds in
[`configs/pipeline.yaml`](configs/pipeline.yaml), and either proceeds,
takes a branch (Step 3a's `dream_case` skips Step 3b), or halts. The
pipeline trace is written to ``outputs/pipeline_trace.json``.

```bash
# Run the full pipeline at the default scale (vd-850M)
python -m monet_logic_circuit.scripts.run_pipeline \
    --pipeline monet_logic_circuit/configs/pipeline.yaml

# Preview the planned execution without running anything
python -m monet_logic_circuit.scripts.run_pipeline \
    --pipeline monet_logic_circuit/configs/pipeline.yaml \
    --dry-run

# Resume after a crash
python -m monet_logic_circuit.scripts.run_pipeline \
    --pipeline monet_logic_circuit/configs/pipeline.yaml \
    --from-step 3b

# Run the pipeline at every scale in scale_progression, advancing only
# when the previous scale's final verdict passes
python -m monet_logic_circuit.scripts.run_pipeline \
    --pipeline monet_logic_circuit/configs/pipeline.yaml \
    --scale-automatically
```

The full set of pre-committed thresholds and branching rules lives in
``configs/pipeline.yaml``. Edit them in one place; every step script
emits a canonical ``decision_signal`` block in its ``results.json`` so
the orchestrator can read it back.

### Running individual steps

Each step can still be invoked directly. The orchestrator just calls
these scripts as subprocesses, so the standalone CLIs are unchanged.

```bash
python -m monet_logic_circuit.scripts.step0_baseline          --config monet_logic_circuit/configs/step0_baseline.yaml
python -m monet_logic_circuit.scripts.step1_gentle_quant      --config monet_logic_circuit/configs/step1_gentle_quant.yaml
python -m monet_logic_circuit.scripts.step2_aggressive_quant  --config monet_logic_circuit/configs/step2_aggressive_quant.yaml
python -m monet_logic_circuit.scripts.step3a_exact_conversion --config monet_logic_circuit/configs/step3a_exact_conversion.yaml
python -m monet_logic_circuit.scripts.step3b_learned_converter --config monet_logic_circuit/configs/step3b_learned_converter.yaml
python -m monet_logic_circuit.scripts.step3c_end_to_end       --config monet_logic_circuit/configs/step3c_end_to_end.yaml
python -m monet_logic_circuit.scripts.step3d_selective_fallback --config monet_logic_circuit/configs/step3d_selective_fallback.yaml
```

### Decision signal contract

Every step writes a ``decision_signal`` block into its ``results.json``:

```json
{
  "decision_signal": {
    "step": "3a",
    "outcome": "tractable",
    "metrics": {"small_frac": 0.31, "blown_up_frac": 0.18, "mean_gates": 52000},
    "reason": "small=31, tractable=51, blown_up=18"
  }
}
```

Outcome codes are ``pass``, ``fail``, ``dream_case``, ``tractable``,
``blown_up``. Most steps emit ``pass``/``fail``; Step 3a uses the
qualitative regimes for branching. Gates in pipeline.yaml threshold
against the numeric values in ``metrics``.

## Go/No-Go Criteria

These thresholds are pre-committed in
[`configs/pipeline.yaml`](configs/pipeline.yaml) and enforced
automatically by the orchestrator. Editing the gates there changes the
go/no-go behavior of the automated pipeline runs.

- **After Step 1:** If gentle quantization loses >2% downstream or
  >0.1 nats perplexity, halt — Monet has unusual quantization sensitivity
  worth investigating before moving on.
- **After Step 2:** If aggressive quantization loses >5% downstream,
  >0.5 nats perplexity, or collapses mean output cardinality below 4,
  halt — the ternary substrate isn't usable for Steps 3a/3b.
- **After Step 3a:** Branch on outcome.
  - `dream_case` (most half-experts have small circuits): skip Step 3b
    and go straight to end-to-end evaluation.
  - `tractable`: run Step 3b to amortize the long tail.
  - `blown_up`: still run Step 3b — the learned converter is now the
    main hope.
- **After Step 3b:** If validation reconstruction NMSE > 0.05, halt —
  the converter isn't accurate enough for selective fallback to recover
  the long tail.
- **After Step 3c:** Warn (don't halt) if the hybrid model's perplexity
  delta exceeds 0.15 nats. The loss decomposition tells us whether 3d
  fallback can recover it; halting here would prevent 3d from running
  the diagnostic.
- **Final verdict (after Step 3d):** Within 0.1 nats perplexity delta
  AND ≥10x speedup. This is the "would we ship it?" check.
- **850M → 1.4B scaling break:** Investigate `d_expert` sensitivity
  before moving to 4.1B. With ``--scale-automatically``, the orchestrator
  halts at the first scale whose final verdict fails.
