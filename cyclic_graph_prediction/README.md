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
| 7 | Ventral/Dorsal Streams | Does branching topology drive ventral nodes toward identity and dorsal nodes toward motion (double dissociation)? |
| 8 | Spatial Inductive Bias | A/B: MLP vs Conv vs Topographic vs Cross-attention inter-node projections — do convolutions matter? |

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

# Experiment 7: ventral/dorsal stream specialization (7-node dual stream)
python -m cyclic_graph_prediction.scripts.run_experiment7 --device cuda

# Experiment 7 control: biased input (ventral=appearance, dorsal=motion)
python -m cyclic_graph_prediction.scripts.run_experiment7 \
    --input_strategy per_stream --device cuda

# Experiment 8: spatial inductive bias A/B test
python -m cyclic_graph_prediction.scripts.run_experiment8 --run_all --device cuda
python -m cyclic_graph_prediction.scripts.run_experiment8 --predictor conv_k3 --device cuda
```

## Structure

```
cyclic_graph_prediction/
├── models/
│   ├── node.py          # GraphNode: single encoder in the graph
│   ├── predictor.py         # LatentPredictor: edge prediction heads + precision weighting
│   ├── pixel_decoder.py     # PixelDecoder: masked pixel reconstruction for V1 stage
│   ├── spatial_predictor.py # Conv/Topographic/CrossAttn patch-level predictors
│   ├── spatial_graph.py     # SpatialPredictionGraph: patch-level inter-node prediction
│   └── graph.py             # PredictionGraph: full cyclic topology + message passing
├── data/
│   ├── masking.py       # Patch mask generation for multi-node input partitioning
│   ├── datasets.py      # Dataset wrappers producing per-node masked views
│   └── temporal.py      # Frame pair generation with synthetic motion transforms
├── trainers/
│   ├── schedules.py     # Update schedules: simultaneous, round-robin, async Gibbs, wave
│   ├── trainer.py       # Main training loop with RankMe collapse detection
│   ├── cortical_cascade.py # Sequential V1->V2->V4->Parietal cascade trainer
│   ├── dual_stream.py    # 7-node ventral/dorsal branching hierarchy
│   └── spatial_trainer.py # Trainer for patch-level spatial prediction graphs
├── eval/
│   ├── linear_probe.py  # Linear probing (per-node and concatenated)
│   ├── specialization.py # CKA-based specialization analysis
│   └── stream_probing.py # Ventral/dorsal identity-vs-motion probing
├── configs/             # YAML configs for each experiment
└── scripts/             # Entry-point scripts
```
