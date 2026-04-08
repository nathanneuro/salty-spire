"""Unit tests for the stego toolkit.

Run with:
    python -m pytest cot_stego_detect/tests -q
or:
    python -m unittest cot_stego_detect.tests.test_stego
"""

from __future__ import annotations

import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from cot_stego_detect.stego.bitstream import (
    BitCursor,
    bits_from_bytes,
    bytes_from_bits,
    prng_bits,
)
from cot_stego_detect.stego.corpus import generate_corpus, write_cot
from cot_stego_detect.stego.detectors import (
    BigramSurprisalDetector,
    ChiSquareTemplateDetector,
    LengthParityDetector,
    PunctuationRateDetector,
)
from cot_stego_detect.stego.encoders import (
    BaselineEncoder,
    LengthParityEncoder,
    PunctuationEncoder,
    SynonymEncoder,
)
from cot_stego_detect.stego.parsing import parse_cot


class BitstreamTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        payload = b"\x00\x7f\xff\xa5"
        bits = bits_from_bytes(payload)
        self.assertEqual(len(bits), 32)
        self.assertEqual(bytes_from_bits(bits), payload)

    def test_cursor_wraparound(self) -> None:
        cursor = BitCursor([1, 0, 1])
        taken = [cursor.take() for _ in range(7)]
        self.assertEqual(taken, [1, 0, 1, 1, 0, 1, 1])
        self.assertEqual(cursor.consumed, 7)

    def test_prng_deterministic(self) -> None:
        self.assertEqual(prng_bits(42, 10), prng_bits(42, 10))
        self.assertNotEqual(prng_bits(42, 10), prng_bits(43, 10))


class ParsingTests(unittest.TestCase):
    def test_parse_roundtrip_baseline(self) -> None:
        corpus = generate_corpus(50, seed=1, n_steps=4)
        enc = BaselineEncoder(seed=2)
        for problem in corpus.problems:
            enc.reset(problem)
            text = write_cot(problem, enc)
            parsed = parse_cot(problem, text)
            self.assertTrue(parsed.parse_ok, f"failed to parse: {text!r}")
            self.assertEqual(len(parsed.template_choices), len(problem.steps))
            self.assertEqual(len(parsed.hedge_choices), len(problem.steps))
            self.assertIsNotNone(parsed.conclusion_choice)


class EncoderChannelTests(unittest.TestCase):
    """Each encoder should actually smuggle its payload."""

    def _recover_template_bits(self, corpus, enc) -> list:
        bits = []
        for p in corpus.problems:
            enc.reset(p)
            text = write_cot(p, enc)
            parsed = parse_cot(p, text)
            self.assertTrue(parsed.parse_ok)
            bits.extend(parsed.template_choices)
            if parsed.conclusion_choice is not None:
                bits.append(parsed.conclusion_choice)
        return bits

    def _recover_hedge_bits(self, corpus, enc) -> list:
        bits = []
        for p in corpus.problems:
            enc.reset(p)
            text = write_cot(p, enc)
            parsed = parse_cot(p, text)
            self.assertTrue(parsed.parse_ok)
            bits.extend(parsed.hedge_choices)
        return bits

    def test_synonym_encoder_carries_bits(self) -> None:
        """Synonym encoder: template/conclusion choices replay the payload."""
        corpus = generate_corpus(32, seed=5, n_steps=4)
        payload = [1, 0, 1, 1, 0, 0, 1, 0]
        enc = SynonymEncoder(seed=0, payload=payload)
        recovered = self._recover_template_bits(corpus, enc)
        # Reconstruct the expected bit sequence: for each problem there
        # are len(steps) template decisions + 1 conclusion decision,
        # all reading from the wrap-around bit stream in order.
        expected = []
        idx = 0
        for p in corpus.problems:
            for _ in p.steps:
                expected.append(payload[idx % len(payload)])
                idx += 1
            expected.append(payload[idx % len(payload)])
            idx += 1
        self.assertEqual(recovered, expected)

    def test_length_parity_encoder_carries_bits(self) -> None:
        corpus = generate_corpus(32, seed=6, n_steps=4)
        payload = [0, 1, 1, 0, 1, 0, 0, 1]
        enc = LengthParityEncoder(seed=0, payload=payload)
        recovered = self._recover_hedge_bits(corpus, enc)
        expected = []
        idx = 0
        for p in corpus.problems:
            for _ in p.steps:
                expected.append(payload[idx % len(payload)])
                idx += 1
        self.assertEqual(recovered, expected)

    def test_punctuation_encoder_forces_hedges(self) -> None:
        corpus = generate_corpus(32, seed=7, n_steps=4)
        enc = PunctuationEncoder(seed=0, payload=[1, 0, 1, 0, 0, 1])
        for p in corpus.problems:
            enc.reset(p)
            text = write_cot(p, enc)
            parsed = parse_cot(p, text)
            self.assertTrue(parsed.parse_ok)
            self.assertTrue(all(h == 1 for h in parsed.hedge_choices))


class DetectorSanityTests(unittest.TestCase):
    """Detectors should fire on stego and stay calm on baseline."""

    def setUp(self) -> None:
        self.corpus = generate_corpus(200, seed=11, n_steps=4)
        self.problems = self.corpus.problems

        self.base_enc = BaselineEncoder(seed=100)
        self.clean_cots = []
        for p in self.problems:
            self.base_enc.reset(p)
            self.clean_cots.append(write_cot(p, self.base_enc))

    def _render(self, enc) -> list:
        out = []
        for p in self.problems:
            enc.reset(p)
            out.append(write_cot(p, enc))
        return out

    def test_chisq_on_synonym(self) -> None:
        det = ChiSquareTemplateDetector()
        # Payload is intentionally imbalanced (5/7 ones). A *balanced*
        # looped payload produces uniform template statistics and
        # evades the chi-square test — this is a real finding about
        # the detector's limits, captured in test_balanced_payload_evades.
        stego = self._render(
            SynonymEncoder(seed=3, payload=[1, 1, 1, 1, 1, 0, 0])
        )
        clean_score = det.score_batch(self.problems, self.clean_cots)
        stego_score = det.score_batch(self.problems, stego)
        self.assertGreater(stego_score, clean_score * 2)

    def test_balanced_payload_evades_chisq(self) -> None:
        """Known limitation: a bit-balanced looped payload has uniform
        template statistics and slips past a frequency-only detector."""
        det = ChiSquareTemplateDetector()
        stego = self._render(
            SynonymEncoder(seed=9, payload=[1, 1, 1, 1, 0, 0, 0, 0])
        )
        clean_score = det.score_batch(self.problems, self.clean_cots)
        stego_score = det.score_batch(self.problems, stego)
        # Stego score should be close to baseline — within 3x.
        self.assertLess(stego_score, clean_score * 3 + 1.0)

    def test_punct_rate_on_punctuation_encoder(self) -> None:
        det = PunctuationRateDetector()
        stego = self._render(
            PunctuationEncoder(seed=4, payload=[1, 0, 1, 0])
        )
        clean_score = det.score_batch(self.problems, self.clean_cots)
        stego_score = det.score_batch(self.problems, stego)
        self.assertGreater(stego_score, clean_score * 3)

    def test_length_parity_on_length_encoder(self) -> None:
        det = LengthParityDetector()
        # Use a payload that isn't bit-balanced so the rate drifts.
        stego = self._render(
            LengthParityEncoder(seed=5, payload=[1, 1, 1, 0])
        )
        clean_score = det.score_batch(self.problems, self.clean_cots)
        stego_score = det.score_batch(self.problems, stego)
        self.assertGreater(stego_score, clean_score * 2)

    def test_bigram_detector_fires_on_template_bias(self) -> None:
        det = BigramSurprisalDetector()
        det.fit(self.problems, self.clean_cots)
        clean_score = det.score_batch(self.problems, self.clean_cots)
        # Heavy bias toward a specific template.
        stego = self._render(
            SynonymEncoder(seed=6, payload=[0] * 16)
        )
        stego_score = det.score_batch(self.problems, stego)
        self.assertGreater(stego_score, clean_score)


if __name__ == "__main__":
    unittest.main()
