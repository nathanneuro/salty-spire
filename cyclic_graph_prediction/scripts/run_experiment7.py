"""
Run Experiment 7: Ventral/Dorsal stream specialization.

Tests whether branching graph topology drives ventral nodes toward identity
and dorsal nodes toward motion, using frame pairs on CIFAR-100.

Usage:
    # Pure topology test (all nodes see same 6-channel frame pair input):
    python -m cyclic_graph_prediction.scripts.run_experiment7 --device cuda

    # Biased control (ventral=appearance, dorsal=motion):
    python -m cyclic_graph_prediction.scripts.run_experiment7 \
        --input_strategy per_stream --device cuda

    # Without cross-stream bridge:
    python -m cyclic_graph_prediction.scripts.run_experiment7 \
        --no_cross_stream --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclic_graph_prediction.data.temporal import (
    TemporalPairDataset,
    DualStreamInputBuilder,
    NUM_MOTION_CLASSES,
)
from cyclic_graph_prediction.trainers.dual_stream import (
    build_dual_stream_graph,
    build_dual_stream_cascade_stages,
    VENTRAL_NODE_IDS,
    DORSAL_NODE_IDS,
    DUAL_STREAM_REGIONS,
)
from cyclic_graph_prediction.trainers.cortical_cascade import (
    CorticalCascadeTrainer,
    CascadeStage,
)
from cyclic_graph_prediction.models.pixel_decoder import PixelDecoder
from cyclic_graph_prediction.eval.stream_probing import evaluate_stream_specialization
from cyclic_graph_prediction.eval.specialization import measure_specialization

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def temporal_collate(batch: list[dict]) -> dict:
    """Collate function for TemporalPairDataset."""
    return {
        "frame1": torch.stack([b["frame1"] for b in batch]),
        "frame2": torch.stack([b["frame2"] for b in batch]),
        "identity_label": torch.tensor([b["identity_label"] for b in batch]),
        "motion_label": torch.tensor([b["motion_label"] for b in batch]),
    }


def temporal_to_cascade_collate(input_builder, num_nodes: int):
    """Wraps temporal collate to produce the (image, masked_views, label) format
    expected by CorticalCascadeTrainer.

    For the cascade trainer, we map:
    - image = frame1
    - masked_views = {node_id: input_tensor} from the input_builder
    - label = identity_label
    """
    def collate_fn(batch: list[dict]):
        frame1 = torch.stack([b["frame1"] for b in batch])
        frame2 = torch.stack([b["frame2"] for b in batch])
        labels = torch.tensor([b["identity_label"] for b in batch])

        inputs = input_builder.build_inputs(frame1, frame2)
        return frame1, inputs, labels

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vit_small_patch16_224")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--input_strategy", type=str, default="both_frames",
                        choices=["both_frames", "frame1_only", "temporal_diff", "per_stream"])
    parser.add_argument("--steps_per_stage", type=int, default=10000)
    parser.add_argument("--feedback_steps", type=int, default=5000)
    parser.add_argument("--no_feedback", action="store_true")
    parser.add_argument("--no_cross_stream", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results/exp7")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input channels
    in_chans = 6 if args.input_strategy in ("both_frames", "temporal_diff", "per_stream") else 3

    # Build graph
    include_feedback = not args.no_feedback
    include_cross_stream = not args.no_cross_stream
    graph = build_dual_stream_graph(
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
        in_chans=in_chans,
        include_feedback=include_feedback,
        include_cross_stream=include_cross_stream,
    )
    logger.info(f"Graph: {graph}")

    # Input builder
    input_builder = DualStreamInputBuilder(
        node_ids=graph.node_ids,
        strategy=args.input_strategy,
        ventral_node_ids=VENTRAL_NODE_IDS,
        dorsal_node_ids=DORSAL_NODE_IDS,
    )

    # Pixel decoder for V1 stage
    v1_node = graph.get_node(0)
    encoder_dim = v1_node.encoder.num_features
    pixel_decoder = PixelDecoder(encoder_dim=encoder_dim)

    # Build cascade stages
    stages = build_dual_stream_cascade_stages(
        steps_per_stage=args.steps_per_stage,
        feedback_steps=args.feedback_steps,
        include_feedback=include_feedback,
    )
    logger.info(f"Stages: {[s.name for s in stages]}")

    # Data
    train_dataset = TemporalPairDataset(
        root=args.data_root, dataset_name=args.dataset, train=True,
    )
    val_dataset = TemporalPairDataset(
        root=args.data_root, dataset_name=args.dataset, train=False,
    )

    # Collate for cascade trainer
    cascade_collate = temporal_to_cascade_collate(input_builder, len(graph.node_ids))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=cascade_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=cascade_collate, pin_memory=True,
    )

    # Train the cascade
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
    training_metrics = trainer.train()

    # === Evaluation ===
    # For probing, we need temporal collate (not cascade collate)
    probe_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=temporal_collate,
    )
    probe_val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=temporal_collate,
    )

    num_identity_classes = train_dataset.num_identity_classes

    logger.info("=" * 60)
    logger.info("STREAM SPECIALIZATION EVALUATION")
    logger.info("=" * 60)

    stream_results = evaluate_stream_specialization(
        graph=graph,
        train_loader=probe_train_loader,
        val_loader=probe_val_loader,
        input_builder=input_builder,
        num_identity_classes=num_identity_classes,
        num_motion_classes=NUM_MOTION_CLASSES,
        device=args.device,
    )

    logger.info("")
    logger.info("=== STREAM SUMMARY ===")
    for key in sorted(stream_results.keys()):
        if key.startswith("stream/"):
            val = stream_results[key]
            if isinstance(val, bool):
                logger.info(f"  {key}: {'YES' if val else 'NO'}")
            else:
                logger.info(f"  {key}: {val:.4f}")

    dd = stream_results.get("stream/double_dissociation", False)
    logger.info(f"\n  ** DOUBLE DISSOCIATION: {'CONFIRMED' if dd else 'NOT FOUND'} **")

    # CKA specialization analysis
    spec_results = measure_specialization(
        graph=graph, loader=val_loader, device=args.device,
    )

    # Save everything
    results = {
        "tag": f"{args.input_strategy}_{'no_cross' if args.no_cross_stream else 'cross'}",
        "args": vars(args),
        "stages": [s.name for s in stages],
        "training_metrics": training_metrics,
        "stream_probing": stream_results,
        "specialization_cka": spec_results,
    }
    tag = results["tag"]
    results_path = output_dir / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
