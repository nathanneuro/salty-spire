"""
Run Experiment 6: Cortical cascade — V1 -> V2 -> V4 -> Parietal.

Sequential hierarchical training mimicking visual cortex development.
V1 trains on pixels, freezes. Each subsequent area trains on its
frozen predecessor's latents. Optional feedback fine-tuning at the end.

Usage:
    python -m cyclic_graph_prediction.scripts.run_experiment6 \
        --steps_per_stage 10000 --feedback_steps 5000 --device cuda

    # Without feedback phase:
    python -m cyclic_graph_prediction.scripts.run_experiment6 \
        --no_feedback --device cuda

    # Reverse cascade (top-down development):
    python -m cyclic_graph_prediction.scripts.run_experiment6 \
        --reverse --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.models.pixel_decoder import PixelDecoder
from cyclic_graph_prediction.data.datasets import MaskedMultiViewDataset
from cyclic_graph_prediction.trainers.cortical_cascade import (
    CorticalCascadeTrainer,
    build_cascade_stages,
    build_cortical_hierarchy,
    CascadeStage,
)
from cyclic_graph_prediction.eval.linear_probe import evaluate_linear_probe
from cyclic_graph_prediction.eval.specialization import measure_specialization
from cyclic_graph_prediction.scripts.run_experiment1 import collate_masked_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_reverse_stages(steps_per_stage: int) -> list[CascadeStage]:
    """Reverse cascade: Parietal first, then V4, V2, V1.

    Tests whether bottom-up development order matters.
    """
    return [
        CascadeStage(
            name="stage1_parietal_pixel",
            region="parietal",
            training_node_id=3,
            target_node_ids=[],
            objective="pixel_reconstruction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage2_V4_from_parietal",
            region="V4",
            training_node_id=2,
            target_node_ids=[3],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage3_V2_from_V4",
            region="V2",
            training_node_id=1,
            target_node_ids=[2],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage4_V1_from_V2",
            region="V1",
            training_node_id=0,
            target_node_ids=[1],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--steps_per_stage", type=int, default=10000)
    parser.add_argument("--feedback_steps", type=int, default=5000)
    parser.add_argument("--no_feedback", action="store_true")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse cascade: Parietal first")
    parser.add_argument("--include_skip", action="store_true",
                        help="Include skip connections (V1->V4, Parietal->V1)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build cortical hierarchy graph
    include_feedback = not args.no_feedback
    graph = build_cortical_hierarchy(
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
        include_feedback=include_feedback,
        include_skip=args.include_skip,
    )
    logger.info(f"Graph: {graph}")

    # Pixel decoder for V1 stage
    # Get encoder dim from the V1 node
    v1_node = graph.get_node(0)
    encoder_dim = v1_node.encoder.num_features
    pixel_decoder = PixelDecoder(encoder_dim=encoder_dim)

    # Build stages
    if args.reverse:
        stages = build_reverse_stages(args.steps_per_stage)
        tag = "reverse"
    else:
        stages = build_cascade_stages(
            steps_per_stage=args.steps_per_stage,
            feedback_steps=args.feedback_steps,
            include_feedback=include_feedback,
        )
        tag = "forward" if include_feedback else "forward_no_feedback"

    logger.info(f"Cascade stages: {[s.name for s in stages]}")

    # Data
    train_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=4,
        visible_fraction=0.25,
        dataset_name=args.dataset,
        train=True,
    )
    val_dataset = MaskedMultiViewDataset(
        root=args.data_root,
        num_nodes=4,
        visible_fraction=0.25,
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
    trainer = CorticalCascadeTrainer(
        graph=graph,
        pixel_decoder=pixel_decoder,
        stages=stages,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
    )
    metrics = trainer.train()

    # Evaluate: per-node and concatenated linear probes
    logger.info("Running linear probe evaluation...")
    num_classes = 100 if args.dataset == "cifar100" else 10
    probe_results = evaluate_linear_probe(
        graph=graph,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=args.device,
    )

    region_names = {0: "V1", 1: "V2", 2: "V4", 3: "parietal"}
    logger.info("=== Per-node linear probe accuracy ===")
    for nid, name in region_names.items():
        key = f"val_acc/node_{nid}"
        if key in probe_results:
            logger.info(f"  {name}: {probe_results[key]:.4f}")
    logger.info(f"  Concatenated: {probe_results['val_acc']:.4f}")

    # Check cascade amplification
    accs = []
    for nid in range(4):
        key = f"val_acc/node_{nid}"
        if key in probe_results:
            accs.append(probe_results[key])
    if accs:
        monotonic = all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1))
        logger.info(
            f"  Cascade amplification: {'YES' if monotonic else 'NO'} "
            f"(accs: {[f'{a:.4f}' for a in accs]})"
        )

    # Specialization analysis
    logger.info("Running specialization analysis...")
    spec_results = measure_specialization(
        graph=graph, loader=val_loader, device=args.device,
    )
    for key, val in spec_results.items():
        logger.info(f"  {key}: {val:.4f}")

    # Save all results
    results = {
        "tag": tag,
        "args": vars(args),
        "stages": [s.name for s in stages],
        "training_metrics": metrics,
        "probe_results": probe_results,
        "specialization": spec_results,
    }
    results_path = output_dir / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
