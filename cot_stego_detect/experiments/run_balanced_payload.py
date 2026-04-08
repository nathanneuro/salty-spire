"""Balanced-payload evasion: a patient adversary who uses a 50/50
bit-balanced payload whose length divides the expected number of
template decisions per batch produces a template distribution that
is *exactly* uniform in expectation and nearly uniform in practice,
evading rate-based detectors entirely.

This script measures detection AUROC against:
  - an unbalanced payload (5/7 ones)
  - a balanced payload whose length divides the batch decision count
The gap between the two columns quantifies how much a rate-based
detector gains from the adversary being sloppy about bit balance.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_ROOT))

from cot_stego_detect.experiments.run_benchmark import auroc, render_batch
from cot_stego_detect.stego.corpus import generate_corpus
from cot_stego_detect.stego.detectors import (
    BigramSurprisalDetector,
    ChiSquareTemplateDetector,
    Detector,
    LengthParityDetector,
    PunctuationRateDetector,
)
from cot_stego_detect.stego.encoders import BaselineEncoder, SynonymEncoder


BATCH_SIZE = 200
N_RUNS = 40
MASTER_SEED = 20260408


def main() -> None:
    rng = random.Random(MASTER_SEED)

    # Reference for bigram.
    ref = generate_corpus(3000, seed=rng.randrange(2**31), n_steps=4)
    b_enc = BaselineEncoder(seed=rng.randrange(2**31))
    ref_cots = render_batch(ref.problems, b_enc)
    bigram = BigramSurprisalDetector()
    bigram.fit(ref.problems, ref_cots)

    detectors: List[Detector] = [
        ChiSquareTemplateDetector(),
        PunctuationRateDetector(),
        LengthParityDetector(),
        bigram,
    ]

    # Payloads.
    #   - Unbalanced: 5 ones out of 7, forces a rate drift.
    #   - Balanced: 4 ones / 4 zeros. Length 8. Each problem has
    #     5 template decisions (4 step templates + 1 conclusion), so
    #     40 decisions per batch of 8 — exactly 5 full loops, 20 ones,
    #     20 zeros, perfectly uniform in expectation.
    payloads = {
        "unbalanced_5of7": [1, 1, 1, 1, 1, 0, 0],
        "balanced_4of8": [1, 1, 1, 1, 0, 0, 0, 0],
    }

    header = f"{'detector':<22}" + "".join(f"{k:<22}" for k in payloads)
    print(header)
    print("-" * len(header))
    md_lines = [
        "# Balanced-payload evasion",
        "",
        f"SynonymEncoder, batch size {BATCH_SIZE}, {N_RUNS} paired runs.",
        "AUROC for separating stego from clean batches.",
        "",
        "| detector | " + " | ".join(payloads.keys()) + " |",
        "|" + "---|" * (1 + len(payloads)),
    ]

    for det in detectors:
        row_vals: Dict[str, float] = {}
        for name, payload in payloads.items():
            pos: List[float] = []
            neg: List[float] = []
            for _ in range(N_RUNS):
                sub = generate_corpus(BATCH_SIZE, seed=rng.randrange(2**31), n_steps=4)
                base = BaselineEncoder(seed=rng.randrange(2**31))
                clean_cots = render_batch(sub.problems, base)
                stego = SynonymEncoder(seed=rng.randrange(2**31), payload=payload)
                stego_cots = render_batch(sub.problems, stego)
                neg.append(det.score_batch(sub.problems, clean_cots))
                pos.append(det.score_batch(sub.problems, stego_cots))
            row_vals[name] = auroc(pos, neg)
        print(f"{det.name:<22}" + "".join(f"{row_vals[k]:<22.3f}" for k in payloads))
        md_lines.append(
            f"| {det.name} | " + " | ".join(f"{row_vals[k]:.3f}" for k in payloads) + " |"
        )

    out = os.path.join(os.path.dirname(_HERE), "results", "balanced_payload.md")
    with open(out, "w") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
