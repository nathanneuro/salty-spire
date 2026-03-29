"""
Cortical cascade schedule and trainer (Experiment 6).

Mimics visual cortical development: V1 trains on pixels, freezes;
V2 trains on frozen V1 latents, freezes; V4 on frozen V2; parietal on frozen V4.
Then optionally: unfreeze feedback connections for bidirectional fine-tuning.

This is SALT's two-stage principle chained through a hierarchy, testing
whether the frozen-teacher advantage compounds across multiple stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.node import GraphNode
from ..models.graph import PredictionGraph
from ..models.pixel_decoder import PixelDecoder, PixelReconstructionLoss
from ..models.predictor import LatentPredictor, PrecisionWeightedLoss

logger = logging.getLogger(__name__)


# --- Cortical region definitions ---

# Receptive field configs: each cortical area sees patches at different scales
# V1 = fine local, V2 = medium, V4 = larger, Parietal = global
CORTICAL_REGIONS = {
    "V1": {
        "node_id": 0,
        "description": "Primary visual cortex — edge/texture features",
        "encoder_name": "vit_small_patch16_224",
        "receptive_field": "local",      # sees all patches (fine-grained)
        "mask_ratio": 0.75,              # high masking for pixel recon
    },
    "V2": {
        "node_id": 1,
        "description": "Secondary visual cortex — contour/surface features",
        "encoder_name": "vit_small_patch16_224",
        "receptive_field": "local",
    },
    "V4": {
        "node_id": 2,
        "description": "Area V4 — shape/color features",
        "encoder_name": "vit_small_patch16_224",
        "receptive_field": "medium",
    },
    "parietal": {
        "node_id": 3,
        "description": "Parietal cortex — spatial/relational features",
        "encoder_name": "vit_small_patch16_224",
        "receptive_field": "global",
    },
}

# Feedforward hierarchy
FEEDFORWARD_EDGES = [
    (0, 1),  # V1 -> V2
    (1, 2),  # V2 -> V4
    (2, 3),  # V4 -> Parietal
]

# Feedback connections (added in optional final phase)
FEEDBACK_EDGES = [
    (1, 0),  # V2 -> V1
    (2, 1),  # V4 -> V2
    (3, 2),  # Parietal -> V4
]

# Skip connections (lateral / bypass, biologically present)
SKIP_EDGES = [
    (0, 2),  # V1 -> V4  (bypass V2)
    (3, 0),  # Parietal -> V1  (top-down global to local)
]


@dataclass
class CascadeStage:
    """One stage of the cortical cascade training."""
    name: str
    region: str
    training_node_id: int
    target_node_ids: list[int]  # frozen predecessors to predict
    objective: str  # "pixel_reconstruction" or "latent_prediction"
    steps: int


def build_cascade_stages(
    steps_per_stage: int = 10000,
    feedback_steps: int = 5000,
    include_feedback: bool = True,
) -> list[CascadeStage]:
    """Define the sequential training stages mimicking cortical development.

    The cascade:
    1. V1: pixel reconstruction (retinal input grounding)
    2. V2: predict frozen V1 latents
    3. V4: predict frozen V2 latents (and optionally V1)
    4. Parietal: predict frozen V4 latents
    5. (Optional) Feedback: unfreeze feedback edges, fine-tune all
    """
    stages = [
        CascadeStage(
            name="stage1_V1_pixel",
            region="V1",
            training_node_id=0,
            target_node_ids=[],  # no latent targets — pixel recon
            objective="pixel_reconstruction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage2_V2_from_V1",
            region="V2",
            training_node_id=1,
            target_node_ids=[0],  # predict frozen V1
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage3_V4_from_V2",
            region="V4",
            training_node_id=2,
            target_node_ids=[1],  # predict frozen V2
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        CascadeStage(
            name="stage4_parietal_from_V4",
            region="parietal",
            training_node_id=3,
            target_node_ids=[2],  # predict frozen V4
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
    ]

    if include_feedback:
        stages.append(
            CascadeStage(
                name="stage5_feedback_finetune",
                region="all",
                training_node_id=-1,  # all nodes
                target_node_ids=[],    # bidirectional
                objective="bidirectional_finetune",
                steps=feedback_steps,
            )
        )

    return stages


class CorticalCascadeTrainer:
    """Trainer for the cortical cascade experiment.

    Unlike GraphTrainer which uses a schedule to toggle freeze/unfreeze,
    this trainer runs sequential stages where each new cortical area
    trains on the frozen outputs of its predecessors.
    """

    def __init__(
        self,
        graph: PredictionGraph,
        pixel_decoder: PixelDecoder,
        stages: list[CascadeStage],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        log_every: int = 100,
        eval_every: int = 2000,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
    ):
        self.graph = graph.to(device)
        self.pixel_decoder = pixel_decoder.to(device)
        self.stages = stages
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.eval_every = eval_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.pixel_loss_fn = PixelReconstructionLoss(norm_pix_loss=True)
        self.latent_loss_fn = PrecisionWeightedLoss()

        self.global_step = 0
        self.metrics: list[dict] = []

    def _get_data_iter(self):
        return iter(self.train_loader)

    def _next_batch(self, data_iter):
        try:
            return next(data_iter), data_iter
        except StopIteration:
            data_iter = self._get_data_iter()
            return next(data_iter), data_iter

    def _freeze_all(self):
        for nid in self.graph.node_ids:
            self.graph.get_node(nid).freeze()
        self.pixel_decoder.requires_grad_(False)

    def _run_pixel_reconstruction_stage(
        self, stage: CascadeStage, data_iter
    ):
        """Stage 1: Train V1 with pixel reconstruction (SALT Stage 1 analog).

        V1 sees masked input. Its encoder + pixel decoder reconstruct the
        masked patches. This grounds the hierarchy in sensory input.
        """
        logger.info(f"=== {stage.name}: V1 pixel reconstruction ===")

        node = self.graph.get_node(stage.training_node_id)
        node.unfreeze()
        self.pixel_decoder.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            list(node.parameters()) + list(self.pixel_decoder.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        for step in range(stage.steps):
            batch, data_iter = self._next_batch(data_iter)
            image, masked_views, labels = batch

            # V1 gets masked input
            v1_input = masked_views[stage.training_node_id].to(self.device)
            full_image = image.to(self.device)

            # Forward through V1 encoder (patch features)
            patch_features = node.forward_features(v1_input)

            # Decode to pixel space
            predicted_pixels = self.pixel_decoder(patch_features)

            # Build reconstruction target from full image
            target_pixels = self._patchify(full_image)

            # Build mask (which patches were masked out)
            # Patches where v1_input is zero are masked
            v1_patch = self._patchify(v1_input)
            mask = (v1_patch.abs().sum(dim=-1) < 1e-6)  # [B, num_patches]

            loss = self.pixel_loss_fn(predicted_pixels, target_pixels, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(node.parameters()) + list(self.pixel_decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            if step % self.log_every == 0:
                logger.info(
                    f"[{stage.name}] step {step}/{stage.steps}, "
                    f"pixel_loss={loss.item():.4f}"
                )
                self.metrics.append({
                    "global_step": self.global_step,
                    "stage": stage.name,
                    "local_step": step,
                    "pixel_loss": loss.item(),
                })

            self.global_step += 1

        # Freeze V1 after training
        node.freeze()
        logger.info(f"V1 frozen after {stage.steps} steps of pixel reconstruction")
        self._save_checkpoint(f"after_{stage.name}")

        return data_iter

    def _run_latent_prediction_stage(
        self, stage: CascadeStage, data_iter
    ):
        """Stages 2-4: Train one area to predict frozen predecessor's latents."""
        logger.info(
            f"=== {stage.name}: node {stage.training_node_id} predicts "
            f"frozen nodes {stage.target_node_ids} ==="
        )

        # Ensure predecessors are frozen
        for tgt_id in stage.target_node_ids:
            self.graph.get_node(tgt_id).freeze()

        # Unfreeze the training node
        node = self.graph.get_node(stage.training_node_id)
        node.unfreeze()

        # Collect trainable params: the node + its outgoing predictor heads
        trainable_params = list(node.parameters())
        for tgt_id in stage.target_node_ids:
            # Edge from training_node -> target (predicting frozen target's latent)
            # Actually: training node predicts target's latent, so edge is
            # training_node -> target_node
            key = f"{stage.training_node_id}_to_{tgt_id}"
            if key in self.graph.predictors:
                trainable_params.extend(self.graph.predictors[key].parameters())

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay,
        )

        for step in range(stage.steps):
            batch, data_iter = self._next_batch(data_iter)
            image, masked_views, labels = batch

            inputs = {
                nid: view.to(self.device) for nid, view in masked_views.items()
            }

            # Encode training node
            src_latent = self.graph.get_node(stage.training_node_id)(
                inputs[stage.training_node_id]
            )

            # Predict each frozen target's latent
            losses = []
            for tgt_id in stage.target_node_ids:
                with torch.no_grad():
                    tgt_latent = self.graph.get_node(tgt_id)(inputs[tgt_id])

                predictor = self.graph.get_predictor(stage.training_node_id, tgt_id)
                predicted, precision = predictor(src_latent)
                loss = self.latent_loss_fn(predicted, tgt_latent.detach(), precision)
                losses.append(loss)

            total_loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            if step % self.log_every == 0:
                logger.info(
                    f"[{stage.name}] step {step}/{stage.steps}, "
                    f"latent_loss={total_loss.item():.4f}"
                )
                self.metrics.append({
                    "global_step": self.global_step,
                    "stage": stage.name,
                    "local_step": step,
                    "latent_loss": total_loss.item(),
                })

            self.global_step += 1

        # Freeze this node — it becomes a target for the next stage
        node.freeze()
        logger.info(
            f"Node {stage.training_node_id} frozen after {stage.steps} steps"
        )
        self._save_checkpoint(f"after_{stage.name}")

        return data_iter

    def _run_feedback_finetune_stage(
        self, stage: CascadeStage, data_iter
    ):
        """Stage 5 (optional): Unfreeze feedback edges and fine-tune bidirectionally.

        After the feedforward cascade, add feedback connections and let all
        nodes co-adapt briefly with round-robin freezing. Tests whether
        top-down prediction (parietal->V4->V2->V1) further improves
        representations once the hierarchy is established.
        """
        logger.info(f"=== {stage.name}: feedback fine-tuning ===")

        # Unfreeze all nodes
        for nid in self.graph.node_ids:
            self.graph.get_node(nid).unfreeze()

        optimizer = torch.optim.AdamW(
            self.graph.parameters(), lr=self.lr * 0.1,  # lower LR for fine-tuning
            weight_decay=self.weight_decay,
        )

        n_nodes = len(self.graph.node_ids)

        for step in range(stage.steps):
            batch, data_iter = self._next_batch(data_iter)
            image, masked_views, labels = batch
            inputs = {
                nid: view.to(self.device) for nid, view in masked_views.items()
            }

            # Round-robin: one node trains, rest frozen
            active_idx = (step // 500) % n_nodes
            active_id = self.graph.node_ids[active_idx]
            for nid in self.graph.node_ids:
                if nid == active_id:
                    self.graph.get_node(nid).unfreeze()
                else:
                    self.graph.get_node(nid).freeze()

            latents = self.graph.encode_all(inputs)
            edge_losses = self.graph.compute_edge_losses(latents, detach_targets=True)

            # Only edges where source is the active node
            active_losses = []
            for (src_id, _), key in zip(self.graph.edges, edge_losses.keys()):
                if src_id == active_id:
                    active_losses.append(edge_losses[key])

            if active_losses:
                total_loss = torch.stack(active_losses).mean()
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.graph.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                total_loss = torch.tensor(0.0)

            if step % self.log_every == 0:
                logger.info(
                    f"[{stage.name}] step {step}/{stage.steps}, "
                    f"active=node_{active_id}, loss={total_loss.item():.4f}"
                )
                self.metrics.append({
                    "global_step": self.global_step,
                    "stage": stage.name,
                    "local_step": step,
                    "feedback_loss": total_loss.item(),
                    "active_node": active_id,
                })

            self.global_step += 1

        self._save_checkpoint(f"after_{stage.name}")
        return data_iter

    def train(self) -> list[dict]:
        """Run the full cortical cascade."""
        self.graph.train()
        self._freeze_all()
        data_iter = self._get_data_iter()

        for stage in self.stages:
            if stage.objective == "pixel_reconstruction":
                data_iter = self._run_pixel_reconstruction_stage(stage, data_iter)
            elif stage.objective == "latent_prediction":
                data_iter = self._run_latent_prediction_stage(stage, data_iter)
            elif stage.objective == "bidirectional_finetune":
                data_iter = self._run_feedback_finetune_stage(stage, data_iter)
            else:
                raise ValueError(f"Unknown objective: {stage.objective}")

        self._save_checkpoint("final")
        return self.metrics

    def _patchify(self, images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """Convert images to patch sequences.

        Args:
            images: [B, C, H, W]

        Returns:
            patches: [B, num_patches, patch_size*patch_size*C]
        """
        B, C, H, W = images.shape
        p = patch_size
        nh, nw = H // p, W // p
        # [B, C, nh, p, nw, p] -> [B, nh, nw, C, p, p] -> [B, num_patches, C*p*p]
        patches = images.reshape(B, C, nh, p, nw, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, nh * nw, C * p * p)
        return patches

    def _save_checkpoint(self, tag: str):
        path = self.checkpoint_dir / f"cascade_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "graph_state_dict": self.graph.state_dict(),
            "pixel_decoder_state_dict": self.pixel_decoder.state_dict(),
            "metrics": self.metrics,
        }, path)
        logger.info(f"Saved checkpoint to {path}")


def build_cortical_hierarchy(
    encoder_name: str = "vit_small_patch16_224",
    latent_dim: int = 384,
    include_feedback: bool = True,
    include_skip: bool = False,
    predictor_hidden_dim: int = 512,
    predictor_layers: int = 2,
) -> PredictionGraph:
    """Build the visual cortical hierarchy graph.

    Nodes: V1(0), V2(1), V4(2), Parietal(3)
    Feedforward: V1->V2->V4->Parietal
    Feedback (optional): Parietal->V4->V2->V1
    Skip (optional): V1->V4, Parietal->V1

    Returns:
        PredictionGraph with cortical connectivity
    """
    nodes = [
        GraphNode(region["node_id"], encoder_name=encoder_name, latent_dim=latent_dim)
        for region in CORTICAL_REGIONS.values()
    ]

    edges = list(FEEDFORWARD_EDGES)
    if include_feedback:
        edges += FEEDBACK_EDGES
    if include_skip:
        edges += SKIP_EDGES

    return PredictionGraph(
        nodes=nodes,
        edges=edges,
        predictor_hidden_dim=predictor_hidden_dim,
        predictor_layers=predictor_layers,
    )
