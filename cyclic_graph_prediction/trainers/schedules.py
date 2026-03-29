"""
Update schedules for cyclic graph training (Experiment 1).

Each schedule determines which nodes are trainable vs frozen at each step,
implementing different learning dynamics on the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ScheduleState:
    """Which nodes to train and which to freeze at this step."""
    trainable_node_ids: list[int]
    frozen_node_ids: list[int]
    phase: int  # current phase/epoch index for logging


class UpdateSchedule(ABC):
    """Base class for graph update schedules."""

    def __init__(self, node_ids: list[int]):
        self.node_ids = node_ids

    @abstractmethod
    def get_state(self, step: int, total_steps: int) -> ScheduleState:
        """Return the schedule state for a given training step."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class SimultaneousSchedule(UpdateSchedule):
    """All nodes train concurrently (naive co-adaptation baseline).

    This is expected to be unstable and risk collapse, serving as
    the negative baseline that SALT's findings predict will underperform.
    """

    def get_state(self, step: int, total_steps: int) -> ScheduleState:
        return ScheduleState(
            trainable_node_ids=list(self.node_ids),
            frozen_node_ids=[],
            phase=0,
        )

    def name(self) -> str:
        return "simultaneous"


class RoundRobinSchedule(UpdateSchedule):
    """Cycle through nodes: train one, freeze the rest.

    Generalizes SALT's two-stage freeze to N nodes on a graph.
    At each phase, one node is unfrozen and trained to predict its
    neighbors' (frozen) latents.
    """

    def __init__(self, node_ids: list[int], steps_per_phase: int = 1000):
        super().__init__(node_ids)
        self.steps_per_phase = steps_per_phase

    def get_state(self, step: int, total_steps: int) -> ScheduleState:
        n = len(self.node_ids)
        phase = (step // self.steps_per_phase) % n
        active_id = self.node_ids[phase]
        frozen_ids = [nid for nid in self.node_ids if nid != active_id]
        return ScheduleState(
            trainable_node_ids=[active_id],
            frozen_node_ids=frozen_ids,
            phase=phase,
        )

    def name(self) -> str:
        return "round_robin"


class AsyncGibbsSchedule(UpdateSchedule):
    """Asynchronous updates: each node trains with some probability at each step.

    Biologically plausible — different regions don't synchronize their
    plasticity. Connects to Gibbs sampling on graphical models.
    """

    def __init__(
        self,
        node_ids: list[int],
        update_prob: float = 0.5,
        seed: int = 42,
    ):
        super().__init__(node_ids)
        self.update_prob = update_prob
        self.rng = np.random.RandomState(seed)

    def get_state(self, step: int, total_steps: int) -> ScheduleState:
        trainable = []
        frozen = []
        for nid in self.node_ids:
            if self.rng.random() < self.update_prob:
                trainable.append(nid)
            else:
                frozen.append(nid)
        # Ensure at least one node is trainable and one is frozen
        if not trainable:
            pick = self.rng.choice(self.node_ids)
            trainable.append(pick)
            frozen.remove(pick)
        if not frozen:
            pick = self.rng.choice(trainable)
            frozen.append(pick)
            trainable.remove(pick)
        return ScheduleState(
            trainable_node_ids=trainable,
            frozen_node_ids=frozen,
            phase=step,
        )

    def name(self) -> str:
        return "async_gibbs"


class WavePropagationSchedule(UpdateSchedule):
    """Updates propagate from a seed node outward.

    Mimics developmental critical periods: each node only begins training
    after its predecessors have partially stabilized. Nodes are ordered
    by graph distance from the seed.
    """

    def __init__(
        self,
        node_ids: list[int],
        edges: list[tuple[int, int]],
        seed_node: int = 0,
        warmup_steps_per_hop: int = 2000,
    ):
        super().__init__(node_ids)
        self.warmup_steps_per_hop = warmup_steps_per_hop

        # BFS from seed to determine hop distance
        self.distances = self._bfs_distances(node_ids, edges, seed_node)

    @staticmethod
    def _bfs_distances(
        node_ids: list[int],
        edges: list[tuple[int, int]],
        seed: int,
    ) -> dict[int, int]:
        adj = {n: [] for n in node_ids}
        for s, t in edges:
            adj[s].append(t)
            adj[t].append(s)  # treat as undirected for distance

        dist = {seed: 0}
        queue = [seed]
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def get_state(self, step: int, total_steps: int) -> ScheduleState:
        trainable = []
        frozen = []
        for nid in self.node_ids:
            hop = self.distances.get(nid, len(self.node_ids))
            activation_step = hop * self.warmup_steps_per_hop
            if step >= activation_step:
                trainable.append(nid)
            else:
                frozen.append(nid)
        # Seed node is always trainable
        if not trainable:
            trainable.append(self.node_ids[0])
        return ScheduleState(
            trainable_node_ids=trainable,
            frozen_node_ids=frozen,
            phase=step // self.warmup_steps_per_hop,
        )

    def name(self) -> str:
        return "wave_propagation"
