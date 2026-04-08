"""End-to-end encoder × detector benchmark.

Runs every encoder against every detector, measuring how well each
detector separates clean batches from stego batches via AUROC. Writes
results to `results/benchmark.md` and prints a human-readable table.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Allow running as a script without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from cot_stego_detect.stego.bitstream import prng_bits
from cot_stego_detect.stego.corpus import ProblemInstance, generate_corpus, write_cot
from cot_stego_detect.stego.detectors import (
    BigramSurprisalDetector,
    ChiSquareTemplateDetector,
    Detector,
    EnsembleDetector,
    LengthParityDetector,
    PunctuationRateDetector,
)
from cot_stego_detect.stego.encoders import (
    AcrosticEncoder,
    BaselineEncoder,
    LengthParityEncoder,
    PunctuationEncoder,
    StegoEncoder,
    SynonymEncoder,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney U based AUROC. pos = positive class (stego)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    return wins / total


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def render_batch(
    problems: Sequence[ProblemInstance], encoder: StegoEncoder
) -> List[str]:
    cots = []
    for p in problems:
        encoder.reset(p)
        cots.append(write_cot(p, encoder))
    return cots


def make_payload(seed: int, length: int = 16) -> List[int]:
    return prng_bits(seed, length)


def build_encoder(name: str, seed: int) -> StegoEncoder:
    payload = make_payload(seed, length=16)
    if name == "baseline":
        return BaselineEncoder(seed=seed)
    if name == "synonym":
        return SynonymEncoder(seed=seed, payload=payload)
    if name == "punctuation":
        return PunctuationEncoder(seed=seed, payload=payload)
    if name == "length_parity":
        return LengthParityEncoder(seed=seed, payload=payload)
    if name == "acrostic":
        return AcrosticEncoder(seed=seed, payload=payload)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    n_reference: int = 4000
    n_eval_batches: int = 40
    batch_size: int = 100
    n_steps: int = 4
    master_seed: int = 20260408


def run(config: BenchmarkConfig) -> Dict[str, Dict[str, float]]:
    rng = random.Random(config.master_seed)

    # 1. Build a big reference corpus and render it with the baseline
    #    encoder. This trains the bigram LM and calibrates per-detector
    #    null statistics.
    ref_corpus = generate_corpus(
        config.n_reference, seed=rng.randrange(2**31), n_steps=config.n_steps
    )
    ref_enc = BaselineEncoder(seed=rng.randrange(2**31))
    ref_cots = render_batch(ref_corpus.problems, ref_enc)

    # 2. Initialize detectors.
    chi = ChiSquareTemplateDetector()
    punct = PunctuationRateDetector()
    lenp = LengthParityDetector()
    bigram = BigramSurprisalDetector()
    bigram.fit(ref_corpus.problems, ref_cots)
    ensemble = EnsembleDetector([chi, punct, lenp, bigram])

    detectors: List[Detector] = [chi, punct, lenp, bigram, ensemble]

    # 3. Calibrate the ensemble on clean batches drawn from the reference
    #    corpus.
    cal_batches: List[Tuple[List[ProblemInstance], List[str]]] = []
    cal_size = min(config.batch_size, len(ref_corpus.problems))
    for _ in range(max(10, config.n_eval_batches // 2)):
        idxs = rng.sample(range(len(ref_corpus.problems)), cal_size)
        probs = [ref_corpus.problems[i] for i in idxs]
        cots = [ref_cots[i] for i in idxs]
        cal_batches.append((probs, cots))
    ensemble.fit_calibration(cal_batches)

    # 4. Build eval batches: fresh problems, rendered once per encoder.
    encoder_names = [
        "baseline",
        "synonym",
        "acrostic",
        "punctuation",
        "length_parity",
    ]
    eval_problems: List[List[ProblemInstance]] = []
    for _ in range(config.n_eval_batches):
        sub = generate_corpus(
            config.batch_size, seed=rng.randrange(2**31), n_steps=config.n_steps
        )
        eval_problems.append(sub.problems)

    scores: Dict[str, Dict[str, List[float]]] = {
        det.name: {enc: [] for enc in encoder_names} for det in detectors
    }

    for bi, problems in enumerate(eval_problems):
        for enc_name in encoder_names:
            encoder = build_encoder(enc_name, seed=rng.randrange(2**31))
            cots = render_batch(problems, encoder)
            for det in detectors:
                scores[det.name][enc_name].append(det.score_batch(problems, cots))

    # 5. Compute AUROC for each detector against each encoder,
    #    using baseline scores as the negative class.
    aurocs: Dict[str, Dict[str, float]] = {}
    for det in detectors:
        aurocs[det.name] = {}
        neg = scores[det.name]["baseline"]
        for enc_name in encoder_names:
            if enc_name == "baseline":
                continue
            pos = scores[det.name][enc_name]
            aurocs[det.name][enc_name] = auroc(pos, neg)
    return aurocs


def format_table(aurocs: Dict[str, Dict[str, float]]) -> str:
    encoders = sorted({enc for d in aurocs.values() for enc in d})
    detectors = list(aurocs.keys())
    col_w = max(16, max(len(e) for e in encoders) + 2)
    header = "detector".ljust(24) + "".join(e.ljust(col_w) for e in encoders)
    lines = [header, "-" * len(header)]
    for d in detectors:
        row = d.ljust(24)
        for e in encoders:
            row += f"{aurocs[d].get(e, float('nan')):<{col_w}.3f}"
        lines.append(row)
    return "\n".join(lines)


def format_markdown(aurocs: Dict[str, Dict[str, float]]) -> str:
    encoders = sorted({enc for d in aurocs.values() for enc in d})
    lines = ["# CoT stego detection benchmark", ""]
    lines.append("AUROC for each detector vs. each encoder.")
    lines.append("Negative class = baseline batches; positive class = stego batches.")
    lines.append("")
    lines.append("| detector | " + " | ".join(encoders) + " |")
    lines.append("|" + "---|" * (len(encoders) + 1))
    for det_name, row in aurocs.items():
        vals = [f"{row.get(e, float('nan')):.3f}" for e in encoders]
        lines.append(f"| {det_name} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reference", type=int, default=4000)
    parser.add_argument("--n-eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_reference=args.n_reference,
        n_eval_batches=args.n_eval_batches,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        master_seed=args.seed,
    )
    aurocs = run(config)
    print(format_table(aurocs))

    out_path = args.out or os.path.join(
        os.path.dirname(_HERE), "results", "benchmark.md"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(format_markdown(aurocs))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
