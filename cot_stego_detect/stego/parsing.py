"""Parsing CoTs back into their decision trace.

Given a `ProblemInstance` and a rendered CoT, we know the closed set
of templates that could have produced it. A white-box detector can
recover the decision each step was made with by trying every template
as a prefix of the remaining text. This parser is used by both the
chi-square template detector and the batch-level unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .corpus import HEDGE_OPTIONS, PUNCT_OPTIONS, TEMPLATES, ProblemInstance


@dataclass
class ParsedCoT:
    template_choices: List[int] = field(default_factory=list)
    hedge_choices: List[int] = field(default_factory=list)
    punct_choices: List[Optional[int]] = field(default_factory=list)
    conclusion_choice: Optional[int] = None
    parse_ok: bool = False


def _format_templates(kind: str, step) -> List[str]:
    family = TEMPLATES[kind]
    if kind == "start":
        return [t.format(val=step.running_total_after) for t in family]
    if kind in ("add", "sub"):
        return [
            t.format(op=step.operand, new=step.running_total_after) for t in family
        ]
    raise ValueError(kind)


def parse_cot(problem: ProblemInstance, text: str) -> ParsedCoT:
    parsed = ParsedCoT()
    remaining = text
    hedge_body = HEDGE_OPTIONS[1].strip()  # "Looks good."
    for step in problem.steps:
        options = _format_templates(step.kind, step)
        matched = None
        for idx, candidate in enumerate(options):
            if remaining.startswith(candidate):
                matched = idx
                remaining = remaining[len(candidate):]
                break
        if matched is None:
            return parsed  # parse_ok stays False
        parsed.template_choices.append(matched)

        # Optional hedge, trying each punct variant.
        hedge_idx = 0
        punct_idx: Optional[int] = None
        for pv, punct in enumerate(PUNCT_OPTIONS):
            probe = punct + hedge_body
            if remaining.startswith(probe):
                hedge_idx = 1
                punct_idx = pv
                remaining = remaining[len(probe):]
                break
        parsed.hedge_choices.append(hedge_idx)
        parsed.punct_choices.append(punct_idx)

        # Consume inter-sentence space.
        if remaining.startswith(" "):
            remaining = remaining[1:]

    # Conclusion template.
    conc_family = TEMPLATES["conclusion"]
    conc_options = [t.format(val=problem.answer) for t in conc_family]
    for idx, candidate in enumerate(conc_options):
        if remaining == candidate or remaining.startswith(candidate):
            parsed.conclusion_choice = idx
            break
    parsed.parse_ok = parsed.conclusion_choice is not None
    return parsed
