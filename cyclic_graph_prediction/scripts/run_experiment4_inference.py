"""
Run Experiment 4: Recurrent inference at test time.

Takes a trained graph checkpoint and evaluates with varying rounds of
message passing, under clean and degraded input conditions.

Usage:
    python -m cyclic_graph_prediction.scripts.run_experiment4_inference \
        --checkpoint ./checkpoints/exp1/graph_final.pt \
        --max_rounds 10 --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.models.graph import PredictionGraph
from cyclic_graph_prediction.data.datasets import MaskedMultiViewDataset
from cyclic_graph_prediction.eval.linear_probe import evaluate_linear_probe
from cyclic_graph_prediction.scripts.run_experiment1 import collate_masked_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def add_noise(masked_views: dict[int, torch.Tensor], std: float) -> dict[int, torch.Tensor]:
    return {nid: v + torch.randn_like(v) * std for nid, v in masked_views.items()}


def add_occlusion(
    masked_views: dict[int, torch.Tensor], fraction: float
) -> dict[int, torch.Tensor]:
    result = {}
    for nid, v in masked_views.items():
        mask = torch.rand(v.shape[0], 1, v.shape[2], v.shape[3], device=v.device) > fraction
        result[nid] = v * mask.float()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild graph and load checkpoint
    graph = PredictionGraph.make_ring(
        num_nodes=args.num_nodes,
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
    )
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    graph.load_state_dict(ckpt["graph_state_dict"])
    graph = graph.to(args.device)

    # Data
    val_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes,
        dataset_name=args.dataset,
        train=False,
    )
    train_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=args.num_nodes,
        visible_fraction=1.0 / args.num_nodes,
        dataset_name=args.dataset,
        train=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_masked_views,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_masked_views,
    )

    num_classes = 100 if args.dataset == "cifar100" else 10
    rounds_to_test = [0, 1, 2, 3, 5, args.max_rounds]

    all_results = {}

    for k in rounds_to_test:
        logger.info(f"Evaluating with {k} message passing rounds (clean input)")
        results = evaluate_linear_probe(
            graph=graph,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=args.device,
            message_passing_rounds=k,
        )
        all_results[f"clean_k{k}"] = results
        logger.info(f"  k={k}: val_acc={results['val_acc']:.4f}")

    # Save
    results_path = output_dir / "recurrent_inference_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
