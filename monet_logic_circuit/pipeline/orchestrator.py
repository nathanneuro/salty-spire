"""Pipeline orchestrator: runs steps, evaluates gates, branches, halts.

The orchestrator owns the whole state machine. Each step is invoked as
a subprocess so that standalone step scripts continue to work unchanged.
After a step finishes, the orchestrator reads the step's decision signal
from ``results.json``, evaluates the gates from ``pipeline.yaml``, and
decides whether to proceed to the next step, take a branch, or halt.

Single-scale mode runs the pipeline once on a specific checkpoint.
Scale-automation mode runs the whole pipeline across a checkpoint
progression (850M -> 1.4B -> 4.1B by default), advancing only after
the previous scale's final verdict passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from monet_logic_circuit.pipeline.gates import (
    Gate,
    GateEvaluation,
    evaluate_gates,
    parse_gates,
)
from monet_logic_circuit.pipeline.signals import (
    DecisionSignal,
    OUTCOME_BLOWN_UP,
    OUTCOME_DREAM_CASE,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_TRACTABLE,
    load_signal_from_results,
)


# Action codes returned by the orchestrator's decision logic.
ACTION_PROCEED = "proceed"
ACTION_HALT = "halt"
ACTION_WARN = "warn"
ACTION_BRANCH = "branch"


@dataclass
class StepSpec:
    """Declarative spec for one pipeline step, parsed from pipeline.yaml."""

    name: str                                  # "0", "1", "2", "3a", ...
    script: str                                # module path, e.g. "step0_baseline"
    config: str                                # path to the step's YAML config
    always_run: bool = False                   # bypass gates entirely
    gates: list[Gate] = field(default_factory=list)
    on_fail: str = "halt"                      # "halt" | "warn"
    # Branching: map from outcome code -> next step name. If a key is
    # present it overrides the default ``next`` sequencing. Value may be
    # None to indicate "end of pipeline".
    branches: dict[str, Optional[str]] = field(default_factory=dict)
    # Default next step if no branch matches. None means fall through
    # to the next step in the pipeline list.
    next: Optional[str] = None
    # Optional notes surfaced in the trace.
    note: str = ""


@dataclass
class PipelineSpec:
    """Top-level parsed pipeline.yaml."""

    steps: list[StepSpec]
    final_verdict_gates: list[Gate] = field(default_factory=list)
    scale_progression: list[str] = field(default_factory=list)
    scale_automatically: bool = False
    results_root: str = "outputs"

    def step(self, name: str) -> StepSpec:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"Unknown step {name!r}. Known: {[s.name for s in self.steps]}")

    def default_next(self, name: str) -> Optional[str]:
        """Return the step listed immediately after `name`, or None."""
        for i, s in enumerate(self.steps):
            if s.name == name:
                if i + 1 < len(self.steps):
                    return self.steps[i + 1].name
                return None
        return None


@dataclass
class StepRun:
    """One entry in the pipeline trace."""

    step: str
    action: str                        # ACTION_*
    signal: Optional[dict] = None      # DecisionSignal.to_dict()
    gate_eval: Optional[dict] = None   # GateEvaluation dict
    next_step: Optional[str] = None
    error: str = ""
    duration_sec: float = 0.0
    subprocess_returncode: Optional[int] = None


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load and parse pipeline.yaml into a :class:`PipelineSpec`."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    pipeline_block = raw.get("pipeline", raw)

    steps_raw = pipeline_block.get("steps", [])
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError(f"{path}: pipeline.steps must be a non-empty list")

    steps: list[StepSpec] = []
    for s in steps_raw:
        steps.append(
            StepSpec(
                name=str(s["step"]),
                script=str(s["script"]),
                config=str(s["config"]),
                always_run=bool(s.get("always_run", False)),
                gates=parse_gates(s.get("gates")),
                on_fail=str(s.get("on_fail", "halt")),
                branches={
                    str(k): (None if v in (None, "end") else str(v))
                    for k, v in (s.get("branches") or {}).items()
                },
                next=(
                    None
                    if s.get("next") in (None, "end")
                    else str(s["next"])
                ) if "next" in s else None,
                note=str(s.get("note", "")),
            )
        )

    final_gates = parse_gates(pipeline_block.get("final_verdict"))
    scale_progression = [str(c) for c in pipeline_block.get("scale_progression", [])]
    scale_auto = bool(pipeline_block.get("scale_automatically", False))
    results_root = str(pipeline_block.get("results_root", "outputs"))

    return PipelineSpec(
        steps=steps,
        final_verdict_gates=final_gates,
        scale_progression=scale_progression,
        scale_automatically=scale_auto,
        results_root=results_root,
    )


def _results_path(results_root: str | Path, step_name: str) -> Path:
    """Canonical path for a step's results.json."""
    return Path(results_root) / f"step{step_name}" / "results.json"


def _run_step_subprocess(
    step: StepSpec,
    env_overrides: Optional[dict[str, str]] = None,
    dry_run: bool = False,
) -> tuple[int, float]:
    """Run a step as a subprocess. Returns (returncode, duration_sec)."""
    cmd = [
        sys.executable,
        "-m",
        f"monet_logic_circuit.scripts.{step.script}",
        "--config",
        step.config,
    ]
    if dry_run:
        print(f"  [dry-run] would execute: {' '.join(cmd)}")
        return 0, 0.0

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    return result.returncode, time.time() - t0


def decide_next_step(
    step: StepSpec,
    signal: DecisionSignal,
    gate_eval: GateEvaluation,
    default_next: Optional[str],
) -> tuple[str, Optional[str]]:
    """Given a step's outcome + gate evaluation, decide the next action.

    Returns ``(action, next_step_name)`` where ``action`` is one of
    ``ACTION_PROCEED``, ``ACTION_BRANCH``, ``ACTION_WARN``, ``ACTION_HALT``.
    ``next_step_name`` may be None to indicate end-of-pipeline.
    """
    # 1. Outcome-based branching takes precedence. These are the 3a-style
    #    qualitative regimes — if the outcome matches a branch key, follow
    #    it regardless of gate evaluation.
    if signal.outcome in step.branches:
        return ACTION_BRANCH, step.branches[signal.outcome]

    # 2. Gate-based decisions. If the gates pass, proceed along the
    #    default or explicit ``next`` edge.
    if gate_eval.passed:
        next_step = step.next if step.next is not None else default_next
        return ACTION_PROCEED, next_step

    # 3. Gates failed. Honor the step's on_fail policy.
    if step.on_fail == "warn":
        next_step = step.next if step.next is not None else default_next
        return ACTION_WARN, next_step
    return ACTION_HALT, None


class Pipeline:
    """Run a parsed :class:`PipelineSpec`, optionally across scales."""

    def __init__(
        self,
        spec: PipelineSpec,
        *,
        dry_run: bool = False,
        from_step: Optional[str] = None,
    ):
        self.spec = spec
        self.dry_run = dry_run
        self.from_step = from_step
        self.trace: list[StepRun] = []

    def run(self) -> bool:
        """Run the pipeline once at the current checkpoint.

        Returns True if the pipeline completed and the final verdict
        (if any) passed; False if it halted or the verdict failed.
        """
        current: Optional[str] = self.from_step or self.spec.steps[0].name
        visited: set[str] = set()

        while current is not None:
            if current in visited:
                self._record_error(
                    current,
                    f"Step {current!r} already ran in this pipeline invocation; "
                    f"refusing to loop.",
                )
                return False
            visited.add(current)

            try:
                step = self.spec.step(current)
            except KeyError as e:
                self._record_error(current, str(e))
                return False

            print(f"\n=== Step {step.name}: {step.script} ===")
            if step.note:
                print(f"  note: {step.note}")

            rc, duration = _run_step_subprocess(step, dry_run=self.dry_run)
            if rc != 0:
                self._record_error(
                    step.name,
                    f"subprocess exited with code {rc}",
                    returncode=rc,
                    duration=duration,
                )
                return False

            # In dry-run mode, synthesize a passing signal and walk the
            # default next edge so we can preview the flow.
            if self.dry_run:
                synthetic = DecisionSignal(
                    step=step.name,
                    outcome=OUTCOME_PASS,
                    metrics={},
                    reason="dry run",
                )
                gate_eval = GateEvaluation(
                    passed=True, failures=[], missing_metrics=[]
                )
                action, next_step = (
                    ACTION_PROCEED,
                    step.next
                    if step.next is not None
                    else self.spec.default_next(step.name),
                )
                self._record_run(
                    step.name, action, synthetic, gate_eval, next_step, duration
                )
                current = next_step
                continue

            # Real run: read the step's decision signal.
            if step.always_run:
                # Still try to read the signal for tracing, but don't gate
                # on it.
                signal = self._try_load_signal(step.name)
                gate_eval = GateEvaluation(
                    passed=True, failures=[], missing_metrics=[]
                )
                next_step = step.next or self.spec.default_next(step.name)
                self._record_run(
                    step.name, ACTION_PROCEED, signal, gate_eval, next_step, duration
                )
                current = next_step
                continue

            try:
                signal = load_signal_from_results(
                    _results_path(self.spec.results_root, step.name)
                )
            except (FileNotFoundError, KeyError) as e:
                self._record_error(
                    step.name,
                    f"could not load decision signal: {e}",
                    duration=duration,
                )
                return False

            gate_eval = evaluate_gates(signal, step.gates)
            action, next_step = decide_next_step(
                step,
                signal,
                gate_eval,
                default_next=self.spec.default_next(step.name),
            )
            self._record_run(
                step.name, action, signal, gate_eval, next_step, duration
            )

            self._print_step_outcome(step, signal, gate_eval, action, next_step)

            if action == ACTION_HALT:
                return False

            current = next_step

        # Final verdict: evaluate against the last step's signal. Skipped
        # in dry-run mode because synthetic signals have no metrics.
        if (
            self.spec.final_verdict_gates
            and self.trace
            and not self.dry_run
        ):
            last_real = next(
                (r for r in reversed(self.trace) if r.signal), None
            )
            if last_real:
                last_signal = DecisionSignal.from_dict(last_real.signal)
                verdict = evaluate_gates(last_signal, self.spec.final_verdict_gates)
                print("\n=== Final verdict ===")
                if verdict.passed:
                    print("  PASS:", verdict.reason)
                    return True
                print("  FAIL:", verdict.reason)
                return False

        return True

    def run_with_scale_progression(self) -> bool:
        """Run the pipeline at each checkpoint in the scale progression.

        Advances to the next scale only if the previous scale's final
        verdict passes. Halts at the first failure.
        """
        if not self.spec.scale_progression:
            return self.run()

        for checkpoint in self.spec.scale_progression:
            print(f"\n##### Scale: {checkpoint} #####")
            if self.dry_run:
                print(f"  [dry-run] would set checkpoint to {checkpoint}")
            else:
                _apply_checkpoint_override(
                    self.spec, checkpoint, self.spec.results_root
                )
            ok = self.run()
            if not ok:
                print(f"  HALT: pipeline failed at scale {checkpoint}")
                return False
            # Reset per-scale trace state; write it out before moving on.
            self._dump_trace(Path(self.spec.results_root) / f"trace_{_slug(checkpoint)}.json")
            self.trace = []
        return True

    # --- Trace helpers ---

    def _record_run(
        self,
        step_name: str,
        action: str,
        signal: Optional[DecisionSignal],
        gate_eval: GateEvaluation,
        next_step: Optional[str],
        duration: float,
    ):
        self.trace.append(
            StepRun(
                step=step_name,
                action=action,
                signal=signal.to_dict() if signal else None,
                gate_eval={
                    "passed": gate_eval.passed,
                    "failures": gate_eval.failures,
                    "missing_metrics": gate_eval.missing_metrics,
                    "reason": gate_eval.reason,
                },
                next_step=next_step,
                duration_sec=duration,
            )
        )

    def _record_error(
        self,
        step_name: str,
        message: str,
        *,
        returncode: Optional[int] = None,
        duration: float = 0.0,
    ):
        print(f"  ERROR at step {step_name}: {message}")
        self.trace.append(
            StepRun(
                step=step_name,
                action=ACTION_HALT,
                error=message,
                duration_sec=duration,
                subprocess_returncode=returncode,
            )
        )

    def _try_load_signal(self, step_name: str) -> Optional[DecisionSignal]:
        try:
            return load_signal_from_results(
                _results_path(self.spec.results_root, step_name)
            )
        except (FileNotFoundError, KeyError):
            return None

    def _print_step_outcome(
        self,
        step: StepSpec,
        signal: DecisionSignal,
        gate_eval: GateEvaluation,
        action: str,
        next_step: Optional[str],
    ):
        print(f"  outcome: {signal.outcome}")
        if signal.metrics:
            metric_str = ", ".join(
                f"{k}={v:.4g}" for k, v in sorted(signal.metrics.items())
            )
            print(f"  metrics: {metric_str}")
        if signal.reason:
            print(f"  reason: {signal.reason}")
        if gate_eval.failures or gate_eval.missing_metrics:
            print(f"  gates: {gate_eval.reason}")
        else:
            print("  gates: all passed")
        print(f"  action: {action} -> {next_step or 'end'}")

    def _dump_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [
                    {
                        "step": r.step,
                        "action": r.action,
                        "signal": r.signal,
                        "gate_eval": r.gate_eval,
                        "next_step": r.next_step,
                        "error": r.error,
                        "duration_sec": r.duration_sec,
                        "subprocess_returncode": r.subprocess_returncode,
                    }
                    for r in self.trace
                ],
                f,
                indent=2,
            )

    def dump_trace(self, path: str | Path):
        """Write the trace to ``path`` as JSON."""
        self._dump_trace(Path(path))


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def _apply_checkpoint_override(
    spec: PipelineSpec, checkpoint: str, results_root: str
):
    """Patch each step config on disk to point at ``checkpoint``.

    Used by scale-automation mode. Writes the patched config to a
    temporary file under ``{results_root}/_patched/`` and updates each
    StepSpec's ``config`` field to point at it. The original configs on
    disk are never modified.
    """
    patched_dir = Path(results_root) / "_patched" / _slug(checkpoint)
    patched_dir.mkdir(parents=True, exist_ok=True)

    for step in spec.steps:
        src = Path(step.config)
        with open(src) as f:
            cfg = yaml.safe_load(f)

        model_block = cfg.get("model", {})
        # Only override the primary checkpoint keys. Derived checkpoints
        # (ternary_checkpoint, circuits_dir, hybrid_model_dir) are left
        # alone — they live under outputs/ and don't change with scale.
        if "checkpoint" in model_block:
            model_block["checkpoint"] = checkpoint
        if "base_checkpoint" in model_block:
            model_block["base_checkpoint"] = checkpoint
        if "source_checkpoint" in model_block:
            model_block["source_checkpoint"] = checkpoint
        cfg["model"] = model_block

        dst = patched_dir / src.name
        with open(dst, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        step.config = str(dst)
