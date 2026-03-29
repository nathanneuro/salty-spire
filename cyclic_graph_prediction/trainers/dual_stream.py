"""
Dual-stream cascade trainer (Experiment 7: Ventral/Dorsal).

Trains a branching cortical hierarchy:

    V1(0) ─→ V2(1) ─→ V4(2) ─→ IT(3)           [ventral: "what"]
      │                 ↕
      └──→ MT(4) ─→ MST(5) ─→ Parietal(6)       [dorsal: "where/how"]

V1 is shared. The two streams branch after V1, with a cross-stream
bridge between V4 and MT (well-documented anatomically).

Cascade order: V1 (pixels) → V2 + MT (parallel, both predict V1) →
V4 + MST (predict predecessors) → IT + Parietal (predict predecessors).

The central question: does the branching topology cause the ventral
stream to specialize for identity and the dorsal stream for motion,
without any explicit functional objective?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.node import GraphNode
from ..models.graph import PredictionGraph
from ..models.pixel_decoder import PixelDecoder, PixelReconstructionLoss
from ..models.predictor import PrecisionWeightedLoss
from .cortical_cascade import CascadeStage, CorticalCascadeTrainer

logger = logging.getLogger(__name__)


# --- Dual-stream region definitions ---

DUAL_STREAM_REGIONS = {
    # Shared
    "V1":       {"node_id": 0, "stream": "shared"},
    # Ventral ("what") pathway
    "V2":       {"node_id": 1, "stream": "ventral"},
    "V4":       {"node_id": 2, "stream": "ventral"},
    "IT":       {"node_id": 3, "stream": "ventral"},
    # Dorsal ("where/how") pathway
    "MT":       {"node_id": 4, "stream": "dorsal"},
    "MST":      {"node_id": 5, "stream": "dorsal"},
    "parietal": {"node_id": 6, "stream": "dorsal"},
}

VENTRAL_NODE_IDS = [1, 2, 3]
DORSAL_NODE_IDS = [4, 5, 6]

# Feedforward edges
DUAL_STREAM_FF_EDGES = [
    # Ventral pathway
    (0, 1),  # V1 -> V2
    (1, 2),  # V2 -> V4
    (2, 3),  # V4 -> IT
    # Dorsal pathway
    (0, 4),  # V1 -> MT
    (4, 5),  # MT -> MST
    (5, 6),  # MST -> Parietal
    # Cross-stream bridge (anatomically: V4 ↔ MT)
    (2, 4),  # V4 -> MT
    (4, 2),  # MT -> V4
]

# Feedback edges
DUAL_STREAM_FB_EDGES = [
    # Ventral feedback
    (1, 0),  # V2 -> V1
    (2, 1),  # V4 -> V2
    (3, 2),  # IT -> V4
    # Dorsal feedback
    (4, 0),  # MT -> V1
    (5, 4),  # MST -> MT
    (6, 5),  # Parietal -> MST
    # Cross-stream feedback
    (3, 6),  # IT -> Parietal (convergence at top)
    (6, 3),  # Parietal -> IT
]


def build_dual_stream_graph(
    encoder_name: str = "vit_small_patch16_224",
    latent_dim: int = 384,
    in_chans: int = 3,
    include_feedback: bool = False,
    include_cross_stream: bool = True,
    predictor_hidden_dim: int = 512,
    predictor_layers: int = 2,
) -> PredictionGraph:
    """Build the dual-stream visual cortex graph.

    Args:
        encoder_name: timm model for all nodes
        latent_dim: shared latent dimension
        in_chans: input channels (3 for single frame, 6 for frame pairs)
        include_feedback: add top-down feedback edges
        include_cross_stream: add V4 ↔ MT bridge

    Returns:
        PredictionGraph with 7 nodes (V1, V2, V4, IT, MT, MST, Parietal)
    """
    nodes = []
    for region_name, info in DUAL_STREAM_REGIONS.items():
        node = GraphNode(
            node_id=info["node_id"],
            encoder_name=encoder_name,
            latent_dim=latent_dim,
        )
        # If using 6-channel input (frame pairs), adjust the first conv
        if in_chans != 3:
            _patch_input_channels(node, in_chans)
        nodes.append(node)

    edges = []
    for src, tgt in DUAL_STREAM_FF_EDGES:
        if not include_cross_stream and (src, tgt) in [(2, 4), (4, 2)]:
            continue
        edges.append((src, tgt))

    if include_feedback:
        edges += DUAL_STREAM_FB_EDGES

    return PredictionGraph(
        nodes=nodes,
        edges=edges,
        predictor_hidden_dim=predictor_hidden_dim,
        predictor_layers=predictor_layers,
    )


def _patch_input_channels(node: GraphNode, in_chans: int):
    """Modify first conv layer to accept in_chans channels instead of 3.

    For ViT: patch_embed.proj is a Conv2d(3, embed_dim, kernel_size=patch_size).
    We replace it with one that accepts in_chans, initializing the extra
    channels from the mean of the original weights.
    """
    if hasattr(node.encoder, "patch_embed"):
        old_proj = node.encoder.patch_embed.proj
        new_proj = nn.Conv2d(
            in_chans,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        with torch.no_grad():
            # Copy weights for first 3 channels
            new_proj.weight[:, :3] = old_proj.weight
            # Initialize extra channels as mean of RGB weights
            if in_chans > 3:
                mean_weight = old_proj.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_chans):
                    new_proj.weight[:, c : c + 1] = mean_weight
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        node.encoder.patch_embed.proj = new_proj


def build_dual_stream_cascade_stages(
    steps_per_stage: int = 10000,
    feedback_steps: int = 5000,
    include_feedback: bool = False,
) -> list[CascadeStage]:
    """Build cascade stages for dual-stream training.

    Order mirrors cortical development:
    1. V1: pixel reconstruction (sensory grounding)
    2. V2 + MT in parallel: both predict frozen V1
       (this is where the streams diverge)
    3. V4 predicts frozen V2; MST predicts frozen MT
    4. IT predicts frozen V4; Parietal predicts frozen MST
    5. (Optional) Feedback fine-tuning across all nodes
    """
    stages = [
        # Stage 1: V1 pixel reconstruction
        CascadeStage(
            name="stage1_V1_pixel",
            region="V1",
            training_node_id=0,
            target_node_ids=[],
            objective="pixel_reconstruction",
            steps=steps_per_stage,
        ),

        # Stage 2a: V2 (ventral) predicts frozen V1
        CascadeStage(
            name="stage2a_V2_from_V1",
            region="V2",
            training_node_id=1,
            target_node_ids=[0],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        # Stage 2b: MT (dorsal) predicts frozen V1
        # NOTE: In the trainer, 2a and 2b can run sequentially or parallel.
        # Sequential is simpler and still tests stream divergence since
        # V2 and MT don't interact at this stage.
        CascadeStage(
            name="stage2b_MT_from_V1",
            region="MT",
            training_node_id=4,
            target_node_ids=[0],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),

        # Stage 3a: V4 (ventral) predicts frozen V2
        CascadeStage(
            name="stage3a_V4_from_V2",
            region="V4",
            training_node_id=2,
            target_node_ids=[1],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        # Stage 3b: MST (dorsal) predicts frozen MT
        CascadeStage(
            name="stage3b_MST_from_MT",
            region="MST",
            training_node_id=5,
            target_node_ids=[4],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),

        # Stage 4a: IT (ventral terminus) predicts frozen V4
        CascadeStage(
            name="stage4a_IT_from_V4",
            region="IT",
            training_node_id=3,
            target_node_ids=[2],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        # Stage 4b: Parietal (dorsal terminus) predicts frozen MST
        CascadeStage(
            name="stage4b_parietal_from_MST",
            region="parietal",
            training_node_id=6,
            target_node_ids=[5],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
    ]

    if include_feedback:
        stages.append(
            CascadeStage(
                name="stage5_feedback_finetune",
                region="all",
                training_node_id=-1,
                target_node_ids=[],
                objective="bidirectional_finetune",
                steps=feedback_steps,
            )
        )

    return stages
