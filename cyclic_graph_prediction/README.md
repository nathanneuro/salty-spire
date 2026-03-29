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
| 6 | Cortical Cascade | Does SALT's frozen-teacher advantage compound across a V1->V2->V4->Parietal hierarchy? Does order matter? |

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

# Experiment 6: cortical cascade (V1 -> V2 -> V4 -> Parietal)
python -m cyclic_graph_prediction.scripts.run_experiment6 \
    --steps_per_stage 10000 --feedback_steps 5000 --device cuda

# Experiment 6 ablation: reverse cascade (Parietal first)
python -m cyclic_graph_prediction.scripts.run_experiment6 --reverse --device cuda
```

## Structure

```
cyclic_graph_prediction/
├── models/
│   ├── node.py          # GraphNode: single encoder in the graph
│   ├── predictor.py     # LatentPredictor: edge prediction heads + precision weighting
│   ├── pixel_decoder.py # PixelDecoder: masked pixel reconstruction for V1 stage
│   └── graph.py         # PredictionGraph: full cyclic topology + message passing
├── data/
│   ├── masking.py       # Patch mask generation for multi-node input partitioning
│   └── datasets.py      # Dataset wrappers producing per-node masked views
├── trainers/
│   ├── schedules.py     # Update schedules: simultaneous, round-robin, async Gibbs, wave
│   ├── trainer.py       # Main training loop with RankMe collapse detection
│   └── cortical_cascade.py # Sequential V1->V2->V4->Parietal cascade trainer
├── eval/
│   ├── linear_probe.py  # Linear probing (per-node and concatenated)
│   └── specialization.py # CKA-based specialization analysis
├── configs/             # YAML configs for each experiment
└── scripts/             # Entry-point scripts
```
