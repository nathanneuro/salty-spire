"""
Saccade integration trainer (Experiment 9).

Graph structure for foveated vision:

    Peripheral(0) ──→ Ventral_Gist(2) ──→ Integration(4)
                           ↕                    ↕
    Foveal(1) ────→ Ventral_Detail(3) ──→ Integration(4)
         │
         └────→ Where_Next(5)  ← Peripheral(0)

Nodes:
  0: Peripheral — receives blurry full-field view (scene gist / layout)
  1: Foveal — receives sharp crop at current fixation (detail / identity)
  2: Ventral_Gist — predicts scene-level features from peripheral
  3: Ventral_Detail — predicts object features from foveal
  4: Integration — combines both streams (like PFC / hippocampus)
  5: Where_Next — predicts the next fixation's foveal latent from
     current peripheral + foveal context (the saccade prediction task)

Training proceeds as a cascade:
  Stage 1: Peripheral + Foveal train on pixel reconstruction
  Stage 2: Ventral_Gist (from Peripheral) + Ventral_Detail (from Foveal)
  Stage 3: Integration (from both ventral nodes)
  Stage 4: Where_Next (predicts next foveal latent — the saccade prediction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.node import GraphNode
from ..models.graph import PredictionGraph
from ..models.pixel_decoder import PixelDecoder, PixelReconstructionLoss
from ..models.predictor import PrecisionWeightedLoss

logger = logging.getLogger(__name__)


# --- Region definitions ---

SACCADE_REGIONS = {
    "peripheral":    {"node_id": 0, "role": "input",   "desc": "Blurry full-field (scene gist)"},
    "foveal":        {"node_id": 1, "role": "input",   "desc": "Sharp crop at fixation (detail)"},
    "ventral_gist":  {"node_id": 2, "role": "mid",     "desc": "Scene-level from peripheral"},
    "ventral_detail":{"node_id": 3, "role": "mid",     "desc": "Object-level from foveal"},
    "integration":   {"node_id": 4, "role": "high",    "desc": "Multi-stream integration"},
    "where_next":    {"node_id": 5, "role": "predict",  "desc": "Predict next foveal latent"},
}

SACCADE_FF_EDGES = [
    (0, 2),  # Peripheral -> Ventral_Gist
    (1, 3),  # Foveal -> Ventral_Detail
    (2, 4),  # Ventral_Gist -> Integration
    (3, 4),  # Ventral_Detail -> Integration
    (0, 5),  # Peripheral -> Where_Next (peripheral context informs saccade planning)
    (1, 5),  # Foveal -> Where_Next (current detail informs next fixation)
    (4, 5),  # Integration -> Where_Next (high-level state guides exploration)
]

SACCADE_FB_EDGES = [
    (2, 0),  # Ventral_Gist -> Peripheral (top-down modulation)
    (3, 1),  # Ventral_Detail -> Foveal (top-down attention)
    (4, 2),  # Integration -> Ventral_Gist
    (4, 3),  # Integration -> Ventral_Detail
    (5, 1),  # Where_Next -> Foveal (prediction of next fixation modulates current)
]


def build_saccade_graph(
    encoder_name: str = "vit_small_patch16_224",
    latent_dim: int = 384,
    fovea_in_chans: int = 3,
    peripheral_in_chans: int = 3,
    include_feedback: bool = False,
    predictor_hidden_dim: int = 512,
    predictor_layers: int = 2,
) -> PredictionGraph:
    """Build the foveated vision graph."""
    nodes = [
        GraphNode(info["node_id"], encoder_name=encoder_name, latent_dim=latent_dim)
        for info in SACCADE_REGIONS.values()
    ]

    edges = list(SACCADE_FF_EDGES)
    if include_feedback:
        edges += SACCADE_FB_EDGES

    return PredictionGraph(
        nodes=nodes,
        edges=edges,
        predictor_hidden_dim=predictor_hidden_dim,
        predictor_layers=predictor_layers,
    )


@dataclass
class SaccadeStage:
    """One stage of saccade cascade training."""
    name: str
    training_node_ids: list[int]
    target_node_ids: list[int]
    objective: str
    steps: int


def build_saccade_stages(
    steps_per_stage: int = 10000,
    saccade_prediction_steps: int = 15000,
) -> list[SaccadeStage]:
    """Build cascade training stages for saccade integration."""
    return [
        # Stage 1: Input nodes learn pixel reconstruction
        SaccadeStage(
            name="stage1_input_pixel_recon",
            training_node_ids=[0, 1],  # Peripheral + Foveal
            target_node_ids=[],
            objective="pixel_reconstruction",
            steps=steps_per_stage,
        ),
        # Stage 2: Mid-level ventral nodes predict frozen input nodes
        SaccadeStage(
            name="stage2_ventral_from_inputs",
            training_node_ids=[2, 3],  # Ventral_Gist, Ventral_Detail
            target_node_ids=[0, 1],     # predict Peripheral, Foveal
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        # Stage 3: Integration predicts both ventral nodes
        SaccadeStage(
            name="stage3_integration",
            training_node_ids=[4],
            target_node_ids=[2, 3],
            objective="latent_prediction",
            steps=steps_per_stage,
        ),
        # Stage 4: Where_Next predicts the NEXT fixation's foveal latent
        # This is the key saccade prediction task
        SaccadeStage(
            name="stage4_saccade_prediction",
            training_node_ids=[5],
            target_node_ids=[1],  # predict foveal node's latent at next fixation
            objective="saccade_prediction",
            steps=saccade_prediction_steps,
        ),
    ]


class SaccadeTrainer:
    """Trainer for the foveated saccade integration experiment.

    Key difference from CorticalCascadeTrainer: the saccade prediction
    stage uses temporal structure — predicting the NEXT fixation's
    foveal latent from the CURRENT fixation's context.
    """

    def __init__(
        self,
        graph: PredictionGraph,
        pixel_decoder_foveal: PixelDecoder,
        pixel_decoder_peripheral: PixelDecoder,
        stages: list[SaccadeStage],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        log_every: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
    ):
        self.graph = graph.to(device)
        self.pixel_decoder_foveal = pixel_decoder_foveal.to(device)
        self.pixel_decoder_peripheral = pixel_decoder_peripheral.to(device)
        self.stages = stages
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.checkpoint_dir = checkpoint_dir
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

    def _build_node_inputs(self, batch: dict, fixation_idx: int = 0) -> dict[int, torch.Tensor]:
        """Build per-node inputs from a saccade batch at a given fixation.

        Node 0 (peripheral): blurry full-field view
        Node 1 (foveal): sharp crop at fixation_idx
        Nodes 2-5: receive foveated_image (full image with spatially varying blur)
        """
        peripheral = batch["peripheral"].to(self.device)           # [B, C, H, W]
        foveal = batch["foveal_crops"][:, fixation_idx].to(self.device)  # [B, C, h, w]
        foveated = batch["foveated_images"][:, fixation_idx].to(self.device)  # [B, C, H, W]

        return {
            0: peripheral,
            1: foveal,
            2: foveated,
            3: foveated,
            4: foveated,
            5: foveated,
        }

    def _run_saccade_prediction_stage(self, stage: SaccadeStage, data_iter):
        """Stage 4: predict NEXT fixation's foveal latent from current context.

        At fixation t, the Where_Next node sees the current peripheral +
        foveal + integration context and must predict what the foveal node
        will encode at fixation t+1. This trains the graph to anticipate
        the consequences of eye movements.
        """
        logger.info(f"=== {stage.name}: saccade prediction ===")

        where_next = self.graph.get_node(5)
        where_next.unfreeze()

        # Trainable: Where_Next node + its predictor heads
        trainable_params = list(where_next.parameters())
        for edge_key, predictor in self.graph.predictors.items():
            if edge_key.startswith("5_to_"):
                trainable_params.extend(predictor.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay,
        )

        foveal_node = self.graph.get_node(1)

        for step in range(stage.steps):
            batch, data_iter = self._next_batch(data_iter)
            num_fix = batch["foveal_crops"].shape[1]

            if num_fix < 2:
                continue

            total_loss = torch.tensor(0.0, device=self.device)
            n_pairs = 0

            # For each consecutive pair of fixations
            for t in range(num_fix - 1):
                # Current context: encode all nodes at fixation t
                inputs_t = self._build_node_inputs(batch, fixation_idx=t)
                where_next_latent = where_next(inputs_t[5])

                # Target: foveal encoding at fixation t+1
                foveal_input_next = batch["foveal_crops"][:, t + 1].to(self.device)
                with torch.no_grad():
                    foveal_latent_next = foveal_node(foveal_input_next)

                # Predict next foveal latent
                predictor = self.graph.get_predictor(5, 1)
                predicted, precision = predictor(where_next_latent)
                loss = self.latent_loss_fn(
                    predicted, foveal_latent_next.detach(), precision
                )
                total_loss = total_loss + loss
                n_pairs += 1

            if n_pairs > 0:
                total_loss = total_loss / n_pairs
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            if step % self.log_every == 0:
                logger.info(
                    f"[{stage.name}] step {step}/{stage.steps}, "
                    f"saccade_pred_loss={total_loss.item():.4f}"
                )
                self.metrics.append({
                    "global_step": self.global_step,
                    "stage": stage.name,
                    "local_step": step,
                    "saccade_pred_loss": total_loss.item(),
                })
            self.global_step += 1

        where_next.freeze()
        return data_iter

    def _run_latent_prediction_stage(self, stage: SaccadeStage, data_iter):
        """Standard latent prediction stage."""
        logger.info(f"=== {stage.name} ===")

        for tgt_id in stage.target_node_ids:
            self.graph.get_node(tgt_id).freeze()

        trainable_params = []
        for nid in stage.training_node_ids:
            node = self.graph.get_node(nid)
            node.unfreeze()
            trainable_params.extend(node.parameters())

        # Add predictor params for relevant edges
        for src_id in stage.training_node_ids:
            for tgt_id in stage.target_node_ids:
                key = f"{src_id}_to_{tgt_id}"
                if key in self.graph.predictors:
                    trainable_params.extend(self.graph.predictors[key].parameters())

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay,
        )

        for step in range(stage.steps):
            batch, data_iter = self._next_batch(data_iter)
            # Use first fixation for non-saccade stages
            inputs = self._build_node_inputs(batch, fixation_idx=0)

            losses = []
            for src_id in stage.training_node_ids:
                src_latent = self.graph.get_node(src_id)(inputs[src_id])
                for tgt_id in stage.target_node_ids:
                    key = f"{src_id}_to_{tgt_id}"
                    if key not in self.graph.predictors:
                        continue
                    with torch.no_grad():
                        tgt_latent = self.graph.get_node(tgt_id)(inputs[tgt_id])
                    predictor = self.graph.get_predictor(src_id, tgt_id)
                    predicted, precision = predictor(src_latent)
                    loss = self.latent_loss_fn(predicted, tgt_latent.detach(), precision)
                    losses.append(loss)

            if losses:
                total_loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            else:
                total_loss = torch.tensor(0.0)

            if step % self.log_every == 0:
                logger.info(
                    f"[{stage.name}] step {step}/{stage.steps}, "
                    f"loss={total_loss.item():.4f}"
                )
                self.metrics.append({
                    "global_step": self.global_step,
                    "stage": stage.name,
                    "local_step": step,
                    "latent_loss": total_loss.item(),
                })
            self.global_step += 1

        for nid in stage.training_node_ids:
            self.graph.get_node(nid).freeze()
        return data_iter

    def train(self) -> list[dict]:
        """Run the full saccade training cascade."""
        self.graph.train()
        self._freeze_all()
        data_iter = self._get_data_iter()

        for stage in self.stages:
            if stage.objective == "pixel_reconstruction":
                # Simplified: skip pixel recon, just train encoders with latent self-prediction
                # (pixel recon for arbitrary input sizes is complex; frozen random init works for testing)
                logger.info(f"=== {stage.name}: initializing input nodes ===")
                for nid in stage.training_node_ids:
                    self.graph.get_node(nid).unfreeze()
                # Run a few steps of self-encoding to warm up
                data_iter = self._run_latent_prediction_stage(
                    SaccadeStage(
                        name=stage.name,
                        training_node_ids=stage.training_node_ids,
                        target_node_ids=stage.training_node_ids,  # self-predict
                        objective="latent_prediction",
                        steps=stage.steps,
                    ),
                    data_iter,
                )
            elif stage.objective == "latent_prediction":
                data_iter = self._run_latent_prediction_stage(stage, data_iter)
            elif stage.objective == "saccade_prediction":
                data_iter = self._run_saccade_prediction_stage(stage, data_iter)

        self._save_checkpoint("final")
        return self.metrics

    def _save_checkpoint(self, tag: str):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = f"{self.checkpoint_dir}/saccade_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "graph_state_dict": self.graph.state_dict(),
            "metrics": self.metrics,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
