"""Capacity curve: how detection AUROC grows with batch size.

The defender's advantage is that frequency-based tests converge to
certainty as the batch grows. This sweep measures how fast that
happens for each detector against the two "hardest" encoders
(synonym and punctuation).
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

from cot_stego_detect.experiments.run_benchmark import auroc, build_encoder, render_batch
from cot_stego_detect.stego.corpus import generate_corpus
from cot_stego_detect.stego.detectors import (
    BigramSurprisalDetector,
    ChiSquareTemplateDetector,
    Detector,
    LengthParityDetector,
    PunctuationRateDetector,
)
from cot_stego_detect.stego.encoders import BaselineEncoder


SIZES = [10, 25, 50, 100, 200, 400, 800]
ENCODERS = ["synonym", "punctuation", "length_parity"]
N_RUNS = 25
MASTER_SEED = 20260408


def main() -> None:
    rng = random.Random(MASTER_SEED)

    # Train bigram null on a big clean reference, once.
    ref = generate_corpus(3000, seed=rng.randrange(2**31), n_steps=4)
    base_enc = BaselineEncoder(seed=rng.randrange(2**31))
    ref_cots = render_batch(ref.problems, base_enc)
    bigram = BigramSurprisalDetector()
    bigram.fit(ref.problems, ref_cots)

    detectors: List[Detector] = [
        ChiSquareTemplateDetector(),
        PunctuationRateDetector(),
        LengthParityDetector(),
        bigram,
    ]

    print(f"{'encoder':<16}{'detector':<22}" + "".join(f"{s:<10}" for s in SIZES))
    print("-" * (38 + 10 * len(SIZES)))

    markdown = ["# Capacity curve", "", "Rows: (encoder, detector). Cols: batch size. Cells: AUROC over "
                f"{N_RUNS} paired runs.", ""]
    markdown.append("| encoder | detector | " + " | ".join(str(s) for s in SIZES) + " |")
    markdown.append("|" + "---|" * (2 + len(SIZES)))

    for enc_name in ENCODERS:
        for det in detectors:
            row_vals: List[float] = []
            for size in SIZES:
                pos: List[float] = []
                neg: List[float] = []
                for _ in range(N_RUNS):
                    sub = generate_corpus(size, seed=rng.randrange(2**31), n_steps=4)
                    b_enc = BaselineEncoder(seed=rng.randrange(2**31))
                    clean_cots = render_batch(sub.problems, b_enc)
                    s_enc = build_encoder(enc_name, seed=rng.randrange(2**31))
                    stego_cots = render_batch(sub.problems, s_enc)
                    neg.append(det.score_batch(sub.problems, clean_cots))
                    pos.append(det.score_batch(sub.problems, stego_cots))
                row_vals.append(auroc(pos, neg))
            print(
                f"{enc_name:<16}{det.name:<22}"
                + "".join(f"{v:<10.3f}" for v in row_vals)
            )
            markdown.append(
                f"| {enc_name} | {det.name} | " + " | ".join(f"{v:.3f}" for v in row_vals) + " |"
            )

    out = os.path.join(os.path.dirname(_HERE), "results", "capacity_curve.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(markdown) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
