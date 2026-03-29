"""
Run Experiment 8: Spatial inductive bias A/B test.

Compares MLP vs Conv vs Topographic vs Cross-attention predictors
on the same 4-node ring graph to test whether convolutional spatial
structure in the inter-node projection matters.

Usage:
    # Run one condition:
    python -m cyclic_graph_prediction.scripts.run_experiment8 \
        --predictor conv --kernel_size 3 --device cuda

    # Run all conditions (sequentially):
    python -m cyclic_graph_prediction.scripts.run_experiment8 \
        --run_all --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.models.spatial_graph import SpatialPredictionGraph
from cyclic_graph_prediction.models.spatial_predictor import PREDICTOR_REGISTRY
from cyclic_graph_prediction.data.datasets import MaskedMultiViewDataset
from cyclic_graph_prediction.trainers.schedules import RoundRobinSchedule
from cyclic_graph_prediction.trainers.spatial_trainer import SpatialGraphTrainer
from cyclic_graph_prediction.eval.linear_probe import evaluate_linear_probe
from cyclic_graph_prediction.scripts.run_experiment1 import collate_masked_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# A/B conditions with matched param budgets where possible
CONDITIONS = {
    "mlp_per_patch": {
        "predictor_type": "mlp",
        "predictor_kwargs": {"hidden_dim": 512, "num_layers": 2},
    },
    "conv_k3": {
        "predictor_type": "conv",
        "predictor_kwargs": {"hidden_dim": 256, "kernel_size": 3, "num_layers": 2},
    },
    "conv_k5": {
        "predictor_type": "conv",
        "predictor_kwargs": {"hidden_dim": 256, "kernel_size": 5, "num_layers": 2},
    },
    "topographic_rf3": {
        "predictor_type": "topographic",
        "predictor_kwargs": {"receptive_field": 3, "hidden_dim": 256},
    },
    "topographic_rf5": {
        "predictor_type": "topographic",
        "predictor_kwargs": {"receptive_field": 5, "hidden_dim": 256},
    },
    "cross_attention": {
        "predictor_type": "cross_attention",
        "predictor_kwargs": {"num_heads": 8, "num_layers": 2},
    },
}


def count_params(model: torch.nn.Module) -> dict[str, int]:
    """Count total and predictor-only parameters."""
    total = sum(p.numel() for p in model.parameters())
    predictor_params = sum(
        p.numel() for name, p in model.named_parameters() if "predictors" in name
    )
    encoder_params = total - predictor_params
    return {
        "total_params": total,
        "predictor_params": predictor_params,
        "encoder_params": encoder_params,
    }


def run_condition(
    condition_name: str,
    condition_cfg: dict,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict:
    """Run a single A/B condition."""
    logger.info(f"\n{'='*60}")
    logger.info(f"CONDITION: {condition_name}")
    logger.info(f"Predictor: {condition_cfg['predictor_type']}")
    logger.info(f"{'='*60}")

    # Build spatial graph
    graph = SpatialPredictionGraph.make_ring(
        num_nodes=args.num_nodes,
        encoder_name=args.encoder,
        predictor_type=condition_cfg["predictor_type"],
        **condition_cfg["predictor_kwargs"],
    )
    logger.info(f"Graph: {graph}")

    param_counts = count_params(graph)
    logger.info(f"Parameters: {param_counts}")

    # Schedule
    schedule = RoundRobinSchedule(graph.node_ids, steps_per_phase=args.steps_per_phase)

    # Train
    trainer = SpatialGraphTrainer(
        graph=graph,
        schedule=schedule,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_steps=args.max_steps,
        device=args.device,
        checkpoint_dir=str(Path(args.output_dir) / "checkpoints" / condition_name),
    )
    training_metrics = trainer.train()

    # Evaluate
    logger.info(f"Evaluating {condition_name}...")
    num_classes = 100 if args.dataset == "cifar100" else 10
    probe_results = evaluate_linear_probe(
        graph=graph,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=args.device,
    )
    logger.info(f"Probe results: {probe_results}")

    return {
        "condition": condition_name,
        "predictor_type": condition_cfg["predictor_type"],
        "predictor_kwargs": condition_cfg["predictor_kwargs"],
        "param_counts": param_counts,
        "training_metrics": training_metrics,
        "probe_results": probe_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", type=str, default=None,
                        choices=list(CONDITIONS.keys()),
                        help="Run a single condition")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all A/B conditions sequentially")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--steps_per_phase", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp8")
    # Extra kwargs for single-condition mode
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--receptive_field", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data (shared across all conditions)
    train_dataset = MaskedMultiViewDataset(
        root=args.data_root, num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes, dataset_name=args.dataset, train=True,
    )
    val_dataset = MaskedMultiViewDataset(
        root=args.data_root, num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes, dataset_name=args.dataset, train=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_masked_views, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_masked_views, pin_memory=True,
    )

    # Determine which conditions to run
    if args.run_all:
        conditions_to_run = CONDITIONS
    elif args.predictor:
        conditions_to_run = {args.predictor: CONDITIONS[args.predictor]}
    else:
        parser.error("Specify --predictor <name> or --run_all")

    all_results = {}
    for name, cfg in conditions_to_run.items():
        result = run_condition(name, cfg, args, train_loader, val_loader)
        all_results[name] = result

        # Save incrementally
        results_path = output_dir / "results_all.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Spatial Inductive Bias A/B Test")
    logger.info(f"{'='*60}")
    logger.info(f"{'Condition':<25} {'Val Acc':>8} {'Pred Params':>12} {'Acc/MParam':>10}")
    logger.info("-" * 60)
    for name, result in sorted(
        all_results.items(),
        key=lambda x: x[1]["probe_results"].get("val_acc", 0),
        reverse=True,
    ):
        val_acc = result["probe_results"].get("val_acc", 0)
        pred_params = result["param_counts"]["predictor_params"]
        efficiency = val_acc / (pred_params / 1e6) if pred_params > 0 else 0
        logger.info(f"{name:<25} {val_acc:>7.4f} {pred_params:>12,} {efficiency:>9.2f}")

    logger.info(f"\nFull results saved to {output_dir / 'results_all.json'}")


if __name__ == "__main__":
    main()
