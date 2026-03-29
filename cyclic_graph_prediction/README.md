# Cyclic Graph Prediction

**Mutual latent prediction on cyclic neural network graphs.**

Extending SALT's frozen-teacher paradigm to multi-node cyclic topologies to test whether mutual latent prediction on a structured graph is a sufficient objective for self-organizing hierarchical representation learning.

## Experiments

| # | Name | Question |
|---|------|----------|
| 1 | Propagation Schedules | Which update dynamics (simultaneous, round-robin, async Gibbs, wave) produce stable + high-quality representations on a cyclic graph? |
| 2 | Weak-Node-Strong-Graph | Does SALT's "weak teacher → strong student" finding generalize to heterogeneous-capacity graphs? |
| 3 | Emergent Specialization | Can graph topology alone drive functional specialization in identical-architecture nodes? |
| 4 | Recurrent Inference | Does iterative message passing at test time improve representations, especially for degraded inputs? |
| 5 | Precision Weighting | Do learnable edge precisions discover attention-like modulation and correlate with edge reliability? |

## Quick Start

```bash
pip install -r requirements.txt

# Experiment 1: compare schedules on a 4-node ring with CIFAR-100
python -m cyclic_graph_prediction.scripts.run_experiment1 \
    --schedule round_robin --max_steps 50000 --device cuda

# Experiment 4: test-time message passing
python -m cyclic_graph_prediction.scripts.run_experiment4_inference \
    --checkpoint ./results/exp1/checkpoints/graph_final.pt \
    --max_rounds 10
```

## Structure

```
cyclic_graph_prediction/
├── models/
│   ├── node.py          # GraphNode: single encoder in the graph
│   ├── predictor.py     # LatentPredictor: edge prediction heads + precision weighting
│   └── graph.py         # PredictionGraph: full cyclic topology + message passing
├── data/
│   ├── masking.py       # Patch mask generation for multi-node input partitioning
│   └── datasets.py      # Dataset wrappers producing per-node masked views
├── trainers/
│   ├── schedules.py     # Update schedules: simultaneous, round-robin, async Gibbs, wave
│   └── trainer.py       # Main training loop with RankMe collapse detection
├── eval/
│   ├── linear_probe.py  # Linear probing (per-node and concatenated)
│   └── specialization.py # CKA-based specialization analysis
├── configs/             # YAML configs for each experiment
└── scripts/             # Entry-point scripts
```
