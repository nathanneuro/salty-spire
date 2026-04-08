"""Threshold-gate evaluation for pipeline decision signals.

A ``Gate`` is a single numeric check applied to a named metric on a
:class:`DecisionSignal`. A step passes a gate spec if none of its gates
fail. Gate specs are declared in the pipeline YAML; the orchestrator
loads them via :func:`parse_gates` and evaluates them with
:func:`evaluate_gates`.

Example YAML fragment::

    gates:
      perplexity_delta_nats: { max: 0.1 }
      downstream_loss_pct:   { max: 2.0 }
      mean_output_cardinality: { min: 4 }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from monet_logic_circuit.pipeline.signals import DecisionSignal


@dataclass(frozen=True)
class Gate:
    """A single threshold check on one named metric.

    At least one of ``max_value`` or ``min_value`` must be set. Both may
    be set simultaneously to enforce a range.
    """

    metric: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None

    def __post_init__(self):
        if self.max_value is None and self.min_value is None:
            raise ValueError(
                f"Gate on metric {self.metric!r} must set max_value, "
                f"min_value, or both."
            )

    def check(self, value: float) -> tuple[bool, str]:
        """Return (passed, reason_if_failed)."""
        if self.max_value is not None and value > self.max_value:
            return False, (
                f"{self.metric}={value:.4g} exceeds max {self.max_value:.4g}"
            )
        if self.min_value is not None and value < self.min_value:
            return False, (
                f"{self.metric}={value:.4g} below min {self.min_value:.4g}"
            )
        return True, ""


@dataclass
class GateEvaluation:
    """Result of evaluating a set of gates against a decision signal."""

    passed: bool
    failures: list[str]
    missing_metrics: list[str]

    @property
    def reason(self) -> str:
        parts = []
        if self.failures:
            parts.append("; ".join(self.failures))
        if self.missing_metrics:
            parts.append(
                "missing metrics: " + ", ".join(sorted(self.missing_metrics))
            )
        return "; ".join(parts) if parts else "all gates passed"


def parse_gates(raw: dict[str, Any] | None) -> list[Gate]:
    """Parse a YAML ``gates`` block into a list of :class:`Gate`.

    Accepts either::

        gates:
          metric_name: { max: X, min: Y }

    or the legacy key-style::

        gates:
          metric_name:
            max_value: X
            min_value: Y
    """
    if not raw:
        return []
    gates: list[Gate] = []
    for metric, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Gate {metric!r} spec must be a dict, got {type(spec).__name__}"
            )
        max_value = spec.get("max", spec.get("max_value"))
        min_value = spec.get("min", spec.get("min_value"))
        gates.append(
            Gate(
                metric=metric,
                max_value=float(max_value) if max_value is not None else None,
                min_value=float(min_value) if min_value is not None else None,
            )
        )
    return gates


def evaluate_gates(
    signal: DecisionSignal,
    gates: list[Gate],
    *,
    treat_missing_as_fail: bool = True,
) -> GateEvaluation:
    """Evaluate a list of gates against a signal's metrics.

    Args:
        signal: The step's decision signal.
        gates: List of gates to check.
        treat_missing_as_fail: If True, a gate whose metric is missing
            from the signal counts as a failure. If False, it's recorded
            in ``missing_metrics`` but not treated as a hard failure.

    Returns:
        A :class:`GateEvaluation` summarising pass/fail plus per-gate detail.
    """
    failures: list[str] = []
    missing: list[str] = []
    for gate in gates:
        if gate.metric not in signal.metrics:
            missing.append(gate.metric)
            continue
        ok, reason = gate.check(float(signal.metrics[gate.metric]))
        if not ok:
            failures.append(reason)

    passed = not failures and (not missing or not treat_missing_as_fail)
    return GateEvaluation(
        passed=passed,
        failures=failures,
        missing_metrics=missing,
    )
