"""Canonical decision-signal contract for pipeline steps.

Every step script writes a ``decision_signal`` block into its ``results.json``
that the orchestrator reads to decide whether to proceed, branch, or halt.
This is the stable machine-readable contract between individual step scripts
and the pipeline orchestrator.

The rest of ``results.json`` is free-form and step-specific; only the
``decision_signal`` key is contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Canonical outcome codes.
#
# Most steps produce ``pass`` or ``fail``. Step 3a is special: exact
# conversion has three qualitative regimes — ``dream_case`` (most circuits
# are small), ``tractable`` (workable but large), ``blown_up`` (exact
# conversion failed for the majority of half-experts). The orchestrator's
# branching rules switch on these codes directly.
OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_DREAM_CASE = "dream_case"
OUTCOME_TRACTABLE = "tractable"
OUTCOME_BLOWN_UP = "blown_up"

ALL_OUTCOMES = frozenset({
    OUTCOME_PASS,
    OUTCOME_FAIL,
    OUTCOME_DREAM_CASE,
    OUTCOME_TRACTABLE,
    OUTCOME_BLOWN_UP,
})


@dataclass
class DecisionSignal:
    """The per-step machine-readable outcome the orchestrator consumes.

    Attributes:
        step: Step identifier, e.g. "0", "1", "2", "3a", "3b", "3c", "3d".
        outcome: One of the OUTCOME_* constants above.
        metrics: Dict of named numeric metrics that pipeline gates can
            threshold against. Keys are stable; values are numbers.
        reason: Human-readable explanation of the outcome, surfaced in
            logs and in the pipeline trace.
    """

    step: str
    outcome: str
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self):
        if self.outcome not in ALL_OUTCOMES:
            raise ValueError(
                f"DecisionSignal.outcome must be one of {sorted(ALL_OUTCOMES)}, "
                f"got {self.outcome!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionSignal":
        return cls(
            step=str(data["step"]),
            outcome=str(data["outcome"]),
            metrics=dict(data.get("metrics", {})),
            reason=str(data.get("reason", "")),
        )


def write_signal_into_results(
    results: dict[str, Any],
    signal: DecisionSignal,
) -> dict[str, Any]:
    """Attach a decision signal to an in-memory results dict.

    This is the canonical way step scripts expose their outcome: they
    call this function on their results dict just before dumping to
    ``results.json``.
    """
    results["decision_signal"] = signal.to_dict()
    return results


def load_signal_from_results(results_path: Path) -> DecisionSignal:
    """Load a decision signal from a step's ``results.json``.

    Raises:
        FileNotFoundError: If the results file does not exist.
        KeyError: If the results file does not contain a ``decision_signal``.
    """
    results_path = Path(results_path)
    with open(results_path) as f:
        data = json.load(f)
    if "decision_signal" not in data:
        raise KeyError(
            f"{results_path} does not contain a 'decision_signal' block. "
            f"The step script must call "
            f"monet_logic_circuit.pipeline.signals.write_signal_into_results."
        )
    return DecisionSignal.from_dict(data["decision_signal"])
