"""
Main training loop for the cyclic graph prediction experiments.

Orchestrates the update schedule, loss computation, logging, and checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.graph import PredictionGraph
from .schedules import UpdateSchedule, ScheduleState

logger = logging.getLogger(__name__)


class GraphTrainer:
    """Training loop for cyclic graph prediction."""

    def __init__(
        self,
        graph: PredictionGraph,
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

        # Single optimizer over all trainable parameters
        # We'll toggle requires_grad per the schedule
        self.optimizer = torch.optim.AdamW(
            graph.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.step = 0
        self.metrics: list[dict] = []

    def _apply_schedule(self, state: ScheduleState):
        """Freeze/unfreeze nodes according to schedule."""
        for nid in state.frozen_node_ids:
            self.graph.get_node(nid).freeze()
        for nid in state.trainable_node_ids:
            self.graph.get_node(nid).unfreeze()

    def _train_step(self, batch: tuple) -> dict[str, float]:
        """Execute one training step."""
        image, masked_views, labels = batch

        # Move masked views to device
        inputs = {
            nid: view.to(self.device)
            for nid, view in masked_views.items()
        }

        # Get schedule state and apply freeze pattern
        state = self.schedule.get_state(self.step, self.max_steps)
        self._apply_schedule(state)

        # Forward: encode all nodes
        latents = self.graph.encode_all(inputs)

        # Compute per-edge losses (targets are detached for frozen nodes)
        edge_losses = self.graph.compute_edge_losses(latents, detach_targets=True)

        # Only backprop through edges whose source is trainable
        trainable_set = set(state.trainable_node_ids)
        active_losses = []
        for (src_id, tgt_id), key in zip(
            self.graph.edges,
            edge_losses.keys(),
        ):
            if src_id in trainable_set:
                active_losses.append(edge_losses[key])

        if active_losses:
            total_loss = torch.stack(active_losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping
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
        """Compute RankMe (effective rank) to detect representation collapse.

        RankMe = exp(entropy of normalized singular values).
        Higher is better; low values indicate collapse.
        """
        # Center the representations
        latents = latents - latents.mean(dim=0, keepdim=True)
        _, s, _ = torch.svd(latents)
        # Normalize singular values to form a distribution
        p = s / s.sum()
        p = p[p > 1e-10]  # avoid log(0)
        entropy = -(p * p.log()).sum()
        return entropy.exp().item()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate representation quality: per-node RankMe and concatenated RankMe."""
        self.graph.eval()
        all_latents = {nid: [] for nid in self.graph.node_ids}

        for batch in self.val_loader:
            image, masked_views, labels = batch
            inputs = {
                nid: view.to(self.device) for nid, view in masked_views.items()
            }
            latents = self.graph.encode_all(inputs)
            for nid, lat in latents.items():
                all_latents[nid].append(lat.cpu())

        metrics = {}
        concat_parts = []
        for nid in self.graph.node_ids:
            node_latents = torch.cat(all_latents[nid], dim=0)
            concat_parts.append(node_latents)
            rankme = self._compute_rankme(node_latents[:2048])  # subsample for speed
            metrics[f"rankme/node_{nid}"] = rankme

        # Concatenated representation across all nodes
        concat = torch.cat(concat_parts, dim=-1)
        metrics["rankme/concat"] = self._compute_rankme(concat[:2048])

        self.graph.train()
        return metrics

    def train(self):
        """Main training loop."""
        self.graph.train()
        data_iter = iter(self.train_loader)

        for self.step in range(self.max_steps):
            # Get next batch, cycling through data
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            step_metrics = self._train_step(batch)

            if self.step % self.log_every == 0:
                loss_str = f"step={self.step}, loss={step_metrics['total_loss']:.4f}"
                logger.info(loss_str)
                self.metrics.append(step_metrics)

            if self.val_loader and self.step % self.eval_every == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                logger.info(f"Eval @ step {self.step}: {eval_metrics}")
                self.metrics.append({**step_metrics, **eval_metrics})

        # Final checkpoint
        self.save_checkpoint("final")
        return self.metrics

    def save_checkpoint(self, tag: str):
        path = self.checkpoint_dir / f"graph_{tag}.pt"
        torch.save({
            "step": self.step,
            "graph_state_dict": self.graph.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.graph.load_state_dict(ckpt["graph_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt["step"]
        self.metrics = ckpt.get("metrics", [])
        logger.info(f"Loaded checkpoint from {path} at step {self.step}")
