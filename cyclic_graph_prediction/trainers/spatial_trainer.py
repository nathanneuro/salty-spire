"""
Trainer for spatial prediction graph (Experiment 8).

Like GraphTrainer but operates on patch-level features via
SpatialPredictionGraph. Supports all the same schedules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.spatial_graph import SpatialPredictionGraph
from .schedules import UpdateSchedule, ScheduleState

logger = logging.getLogger(__name__)


class SpatialGraphTrainer:
    """Training loop for spatial (patch-level) prediction graphs."""

    def __init__(
        self,
        graph: SpatialPredictionGraph,
        schedule: UpdateSchedule,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        max_steps: int = 50000,
        log_every: int = 100,
        eval_every: int = 1000,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
    ):
        self.graph = graph.to(device)
        self.schedule = schedule
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            graph.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self.step = 0
        self.metrics: list[dict] = []

    def _apply_schedule(self, state: ScheduleState):
        for nid in state.frozen_node_ids:
            self.graph.get_node(nid).freeze()
        for nid in state.trainable_node_ids:
            self.graph.get_node(nid).unfreeze()

    def _train_step(self, batch: tuple) -> dict[str, float]:
        image, masked_views, labels = batch
        inputs = {nid: view.to(self.device) for nid, view in masked_views.items()}

        state = self.schedule.get_state(self.step, self.max_steps)
        self._apply_schedule(state)

        # Encode to spatial (patch-level) features
        spatial_latents = self.graph.encode_all_spatial(inputs)

        # Compute per-edge patch-level losses
        edge_losses = self.graph.compute_edge_losses(spatial_latents, detach_targets=True)

        trainable_set = set(state.trainable_node_ids)
        active_losses = []
        for (src_id, _), key in zip(self.graph.edges, edge_losses.keys()):
            if src_id in trainable_set:
                active_losses.append(edge_losses[key])

        if active_losses:
            total_loss = torch.stack(active_losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.graph.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            total_loss = torch.tensor(0.0)

        metrics = {
            "step": self.step,
            "total_loss": total_loss.item(),
            "phase": state.phase,
            "num_trainable": len(state.trainable_node_ids),
        }
        for key, loss in edge_losses.items():
            metrics[f"loss/{key}"] = loss.item()
        return metrics

    def _compute_rankme(self, latents: torch.Tensor) -> float:
        latents = latents - latents.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(latents)
        p = s / s.sum()
        p = p[p > 1e-10]
        entropy = -(p * p.log()).sum()
        return entropy.exp().item()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.graph.eval()
        all_latents = {nid: [] for nid in self.graph.node_ids}

        for batch in self.val_loader:
            image, masked_views, labels = batch
            inputs = {nid: view.to(self.device) for nid, view in masked_views.items()}
            # Pool spatial features for evaluation
            pooled = self.graph.encode_all(inputs)
            for nid, lat in pooled.items():
                all_latents[nid].append(lat.cpu())

        metrics = {}
        concat_parts = []
        for nid in self.graph.node_ids:
            node_latents = torch.cat(all_latents[nid], dim=0)
            concat_parts.append(node_latents)
            metrics[f"rankme/node_{nid}"] = self._compute_rankme(node_latents[:2048])

        concat = torch.cat(concat_parts, dim=-1)
        metrics["rankme/concat"] = self._compute_rankme(concat[:2048])

        self.graph.train()
        return metrics

    def train(self) -> list[dict]:
        self.graph.train()
        data_iter = iter(self.train_loader)

        for self.step in range(self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            step_metrics = self._train_step(batch)

            if self.step % self.log_every == 0:
                logger.info(
                    f"step={self.step}, loss={step_metrics['total_loss']:.4f}"
                )
                self.metrics.append(step_metrics)

            if self.val_loader and self.step % self.eval_every == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                logger.info(f"Eval @ step {self.step}: {eval_metrics}")
                self.metrics.append({**step_metrics, **eval_metrics})

        self.save_checkpoint("final")
        return self.metrics

    def save_checkpoint(self, tag: str):
        path = self.checkpoint_dir / f"spatial_graph_{tag}.pt"
        torch.save({
            "step": self.step,
            "graph_state_dict": self.graph.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
