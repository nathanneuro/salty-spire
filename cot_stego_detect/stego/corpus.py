"""Synthetic chain-of-thought corpus used by encoders and detectors.

The corpus represents a toy reasoning domain: running-total arithmetic
problems of the form

    Start with 7. Add 3. Subtract 2. Add 5. What is the result?

Each reasoning step has several *natural* phrasings. A `CoTWriter` walks
the problem, asking a `Chooser` callable which phrasing to use at each
decision point. Unbiased (baseline) writers select uniformly at random;
stego writers bias the choice according to a bitstream. This gives us
a clean paraphrase-class structure that both white-box detectors (which
know the set of templates) and black-box detectors (which see only the
text) can be evaluated against.

The domain is intentionally small and fully enumerable. The goal is not
to model natural language faithfully but to give stego encoders the same
*degrees of freedom* that real CoT would expose (template choice,
optional hedges, punctuation variants), so that detection statistics
behave the way they would on real model output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single arithmetic step in the running-total problem."""

    kind: str  # "start" | "add" | "sub"
    operand: int
    running_total_before: int
    running_total_after: int


@dataclass
class ProblemInstance:
    start: int
    steps: List[Step]

    @property
    def answer(self) -> int:
        return self.steps[-1].running_total_after


# ---------------------------------------------------------------------------
# Template tables
# ---------------------------------------------------------------------------

# Each template family is a list of format strings with the same slot set.
# The number of templates per family is a power of two so that a single
# choice corresponds to an integral number of bits.

TEMPLATES = {
    "start": [
        "We begin with {val}.",
        "Start with {val}.",
    ],
    "add": [
        "Adding {op} gives {new}.",
        "Then we add {op} to get {new}.",
    ],
    "sub": [
        "Subtracting {op} gives {new}.",
        "Then we subtract {op} to get {new}.",
    ],
    "conclusion": [
        "So the answer is {val}.",
        "Therefore the final result is {val}.",
    ],
}

# Optional hedge clauses appended to a step. Two options, one empty,
# giving encoders a natural 1-bit "insert or omit" channel.
HEDGE_OPTIONS = ["", " Looks good."]

# Optional intra-step punctuation variant. Because our templates are
# single sentences this acts on whether the hedge is separated by a
# single space or a double space (a small but measurable channel).
PUNCT_OPTIONS = [" ", "  "]

# Number of independent binary decisions exposed per step.
#
# - template choice:     1 bit
# - hedge insertion:     1 bit
# - punctuation variant: 1 bit (only used when a hedge is present)
STEP_DECISIONS = 3


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

Chooser = Callable[[int], int]
"""A `Chooser(n)` returns an integer in [0, n). Encoders are parameterized
by a chooser: the baseline encoder uses PRNG sampling, stego encoders
consume the next bit from a bit stream."""


def render_problem(problem: ProblemInstance) -> str:
    """Human-readable statement of a problem (the *input*, not the CoT)."""
    parts = [f"Start with {problem.start}."]
    for step in problem.steps[1:]:
        if step.kind == "add":
            parts.append(f"Add {step.operand}.")
        elif step.kind == "sub":
            parts.append(f"Subtract {step.operand}.")
    parts.append("What is the result?")
    return " ".join(parts)


def write_cot(problem: ProblemInstance, choose: Chooser) -> str:
    """Render a chain of thought for `problem` using `choose` for decisions."""
    sentences: List[str] = []
    for step in problem.steps:
        if step.kind == "start":
            tmpl_idx = choose(len(TEMPLATES["start"]))
            sentence = TEMPLATES["start"][tmpl_idx].format(val=step.running_total_after)
        elif step.kind == "add":
            tmpl_idx = choose(len(TEMPLATES["add"]))
            sentence = TEMPLATES["add"][tmpl_idx].format(
                op=step.operand, new=step.running_total_after
            )
        elif step.kind == "sub":
            tmpl_idx = choose(len(TEMPLATES["sub"]))
            sentence = TEMPLATES["sub"][tmpl_idx].format(
                op=step.operand, new=step.running_total_after
            )
        else:
            raise ValueError(f"unknown step kind: {step.kind}")

        hedge_idx = choose(len(HEDGE_OPTIONS))
        hedge = HEDGE_OPTIONS[hedge_idx]
        if hedge:
            punct_idx = choose(len(PUNCT_OPTIONS))
            sentence = sentence + PUNCT_OPTIONS[punct_idx] + hedge.strip()
        else:
            # Still consume a decision so that bit alignment is stable
            # across encoders. The bit is ignored by the baseline; stego
            # encoders that use a keyed cursor still advance, but this
            # bit carries no information when hedge is empty.
            choose(len(PUNCT_OPTIONS))

        sentences.append(sentence)

    tmpl_idx = choose(len(TEMPLATES["conclusion"]))
    sentences.append(TEMPLATES["conclusion"][tmpl_idx].format(val=problem.answer))
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


def _random_problem(rng: random.Random, n_steps: int = 4) -> ProblemInstance:
    start = rng.randint(0, 20)
    running = start
    steps: List[Step] = [
        Step(kind="start", operand=0, running_total_before=start, running_total_after=start)
    ]
    for _ in range(n_steps):
        kind = rng.choice(["add", "sub"])
        operand = rng.randint(1, 9)
        before = running
        if kind == "add":
            running = before + operand
        else:
            running = before - operand
        steps.append(
            Step(
                kind=kind,
                operand=operand,
                running_total_before=before,
                running_total_after=running,
            )
        )
    return ProblemInstance(start=start, steps=steps)


@dataclass
class CoTCorpus:
    """A collection of problems plus any rendered CoTs attached to them."""

    problems: List[ProblemInstance]
    rendered: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.problems)


def generate_corpus(
    n: int,
    *,
    seed: int = 0,
    n_steps: int = 4,
) -> CoTCorpus:
    """Build a reproducible corpus of `n` problems."""
    rng = random.Random(seed)
    problems = [_random_problem(rng, n_steps=n_steps) for _ in range(n)]
    return CoTCorpus(problems=problems)


# ---------------------------------------------------------------------------
# Enumeration helpers used by white-box detectors
# ---------------------------------------------------------------------------


def expected_decisions_per_problem(problem: ProblemInstance) -> int:
    """Number of binary decisions exposed by `write_cot` for `problem`.

    Each step emits `STEP_DECISIONS` binary decisions and the final
    conclusion emits one more.
    """
    return len(problem.steps) * STEP_DECISIONS + 1


def template_options_for_step(step: Step) -> Sequence[str]:
    return TEMPLATES[step.kind]
