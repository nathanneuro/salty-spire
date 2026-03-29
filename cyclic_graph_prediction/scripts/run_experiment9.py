"""
Run Experiment 9: Foveated vision with saccadic eye movements.

Tests whether a prediction graph can learn to integrate information
across saccadic fixations, combining sharp foveal detail with blurry
peripheral gist.

Usage:
    python -m cyclic_graph_prediction.scripts.run_experiment9 --device cuda

    # Different saccade policies:
    python -m cyclic_graph_prediction.scripts.run_experiment9 \
        --saccade_policy scanpath --num_fixations 4 --device cuda

    python -m cyclic_graph_prediction.scripts.run_experiment9 \
        --saccade_policy random --num_fixations 4 --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.data.foveated import FoveatedSaccadeDataset
from cyclic_graph_prediction.trainers.saccade_trainer import (
    build_saccade_graph,
    build_saccade_stages,
    SaccadeTrainer,
)
from cyclic_graph_prediction.models.pixel_decoder import PixelDecoder
from cyclic_graph_prediction.eval.saccade_eval import evaluate_saccade_integration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def saccade_collate(batch: list[dict]) -> dict:
    """Collate for FoveatedSaccadeDataset."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "peripheral": torch.stack([b["peripheral"] for b in batch]),
        "foveal_crops": torch.stack([b["foveal_crops"] for b in batch]),
        "foveated_images": torch.stack([b["foveated_images"] for b in batch]),
        "fixation_coords": torch.stack([b["fixation_coords"] for b in batch]),
        "saccade_directions": torch.stack([b["saccade_directions"] for b in batch]),
        "identity_label": torch.tensor([b["identity_label"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--fovea_size", type=int, default=64)
    parser.add_argument("--blur_sigma", type=float, default=8.0)
    parser.add_argument("--num_fixations", type=int, default=4)
    parser.add_argument("--saccade_policy", type=str, default="saliency",
                        choices=["random", "center_bias", "scanpath",
                                 "saliency", "object_center", "information_gain",
                                 "attention_map"])
    parser.add_argument("--steps_per_stage", type=int, default=10000)
    parser.add_argument("--saccade_pred_steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp9")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build graph
    graph = build_saccade_graph(
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
    )
    logger.info(f"Graph: {graph}")

    # Pixel decoders (for potential pixel recon stage)
    v1_dim = graph.get_node(0).encoder.num_features
    pixel_decoder_foveal = PixelDecoder(encoder_dim=v1_dim)
    pixel_decoder_peripheral = PixelDecoder(encoder_dim=v1_dim)

    # Build stages
    stages = build_saccade_stages(
        steps_per_stage=args.steps_per_stage,
        saccade_prediction_steps=args.saccade_pred_steps,
    )
    logger.info(f"Stages: {[s.name for s in stages]}")

    # Data
    train_dataset = FoveatedSaccadeDataset(
        root=args.data_root,
        image_size=224,
        fovea_size=args.fovea_size,
        peripheral_blur_sigma=args.blur_sigma,
        num_fixations=args.num_fixations,
        saccade_policy=args.saccade_policy,
        dataset_name=args.dataset,
        train=True,
    )
    val_dataset = FoveatedSaccadeDataset(
        root=args.data_root,
        image_size=224,
        fovea_size=args.fovea_size,
        peripheral_blur_sigma=args.blur_sigma,
        num_fixations=args.num_fixations,
        saccade_policy=args.saccade_policy,
        dataset_name=args.dataset,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=saccade_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=saccade_collate, pin_memory=True,
    )

    # Train
    trainer = SaccadeTrainer(
        graph=graph,
        pixel_decoder_foveal=pixel_decoder_foveal,
        pixel_decoder_peripheral=pixel_decoder_peripheral,
        stages=stages,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
    )
    training_metrics = trainer.train()

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("SACCADE INTEGRATION EVALUATION")
    logger.info("=" * 60)

    num_classes = train_dataset.num_classes
    eval_results = evaluate_saccade_integration(
        graph=graph,
        loader=val_loader,
        num_classes=num_classes,
        device=args.device,
    )

    # Save
    results = {
        "saccade_policy": args.saccade_policy,
        "num_fixations": args.num_fixations,
        "fovea_size": args.fovea_size,
        "blur_sigma": args.blur_sigma,
        "args": vars(args),
        "stages": [s.name for s in stages],
        "training_metrics": training_metrics,
        "eval_results": eval_results,
    }
    tag = f"{args.saccade_policy}_fix{args.num_fixations}_fov{args.fovea_size}"
    results_path = output_dir / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
