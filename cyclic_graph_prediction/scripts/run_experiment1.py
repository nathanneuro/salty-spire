"""
Run Experiment 1: Propagation schedules on a 4-node ring.

Compares simultaneous, round-robin, async Gibbs, and wave propagation
schedules on CIFAR-100. This is the foundational experiment — results
determine which schedule to use for Experiments 2-5.

Usage:
    python -m cyclic_graph_prediction.scripts.run_experiment1 \
        --schedule round_robin --steps_per_phase 1000 --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.models.graph import PredictionGraph
from cyclic_graph_prediction.data.datasets import MaskedMultiViewDataset
from cyclic_graph_prediction.trainers.schedules import (
    SimultaneousSchedule,
    RoundRobinSchedule,
    AsyncGibbsSchedule,
    WavePropagationSchedule,
)
from cyclic_graph_prediction.trainers.trainer import GraphTrainer
from cyclic_graph_prediction.eval.linear_probe import evaluate_linear_probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def collate_masked_views(batch):
    """Custom collate for MaskedMultiViewDataset."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[2] for b in batch])
    # Merge dicts: {node_id: stacked tensors}
    node_ids = batch[0][1].keys()
    masked_views = {
        nid: torch.stack([b[1][nid] for b in batch]) for nid in node_ids
    }
    return images, masked_views, labels


def build_schedule(args, graph):
    if args.schedule == "simultaneous":
        return SimultaneousSchedule(graph.node_ids)
    elif args.schedule == "round_robin":
        return RoundRobinSchedule(graph.node_ids, steps_per_phase=args.steps_per_phase)
    elif args.schedule == "async_gibbs":
        return AsyncGibbsSchedule(graph.node_ids, update_prob=args.update_prob)
    elif args.schedule == "wave_propagation":
        return WavePropagationSchedule(
            graph.node_ids,
            graph.edges,
            seed_node=0,
            warmup_steps_per_hop=args.warmup_steps_per_hop,
        )
    else:
        raise ValueError(f"Unknown schedule: {args.schedule}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, default="round_robin",
                        choices=["simultaneous", "round_robin", "async_gibbs", "wave_propagation"])
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--steps_per_phase", type=int, default=1000)
    parser.add_argument("--update_prob", type=float, default=0.5)
    parser.add_argument("--warmup_steps_per_hop", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build graph
    logger.info(f"Building {args.num_nodes}-node ring with {args.encoder}")
    graph = PredictionGraph.make_ring(
        num_nodes=args.num_nodes,
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
    )
    logger.info(f"Graph: {graph}")

    # Build schedule
    schedule = build_schedule(args, graph)
    logger.info(f"Schedule: {schedule.name()}")

    # Data
    train_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes,
        dataset_name=args.dataset,
        train=True,
    )
    val_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes,
        dataset_name=args.dataset,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_masked_views, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_masked_views, pin_memory=True,
    )

    # Train
    trainer = GraphTrainer(
        graph=graph,
        schedule=schedule,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_steps=args.max_steps,
        device=args.device,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    metrics = trainer.train()

    # Linear probe evaluation
    logger.info("Running linear probe evaluation...")
    num_classes = 100 if args.dataset == "cifar100" else 10
    probe_results = evaluate_linear_probe(
        graph=graph,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=args.device,
    )
    logger.info(f"Linear probe results: {probe_results}")

    # Save results
    results = {
        "schedule": schedule.name(),
        "args": vars(args),
        "training_metrics": metrics,
        "probe_results": probe_results,
    }
    results_path = output_dir / f"results_{schedule.name()}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
