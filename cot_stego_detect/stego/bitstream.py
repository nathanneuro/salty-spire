"""Utilities for converting secret messages to and from bit streams."""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List


def bits_from_bytes(data: bytes) -> List[int]:
    """Return a list of bits (MSB first) for the given bytes."""
    out: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            out.append((byte >> shift) & 1)
    return out


def bytes_from_bits(bits: Iterable[int]) -> bytes:
    """Inverse of `bits_from_bytes`. Extra trailing bits are zero-padded."""
    bit_list = list(bits)
    # Pad to a byte boundary.
    pad = (-len(bit_list)) % 8
    bit_list.extend([0] * pad)
    out = bytearray()
    for i in range(0, len(bit_list), 8):
        byte = 0
        for b in bit_list[i : i + 8]:
            byte = (byte << 1) | (b & 1)
        out.append(byte)
    return bytes(out)


def prng_bits(seed: int, n: int) -> List[int]:
    """Deterministic pseudo-random bit stream of length `n`.

    Used for benchmarks where we want to measure capacity without
    choosing a specific plaintext payload.
    """
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(n)]


class BitCursor:
    """A rewindable cursor over a bit stream, with wrap-around.

    Wrap-around lets short payloads be repeated over arbitrarily long
    cover text. That matches a realistic "I have 8 bits of information
    to smuggle out, please encode them redundantly" threat model and
    keeps the encoder from running out of bits mid-CoT.
    """

    def __init__(self, bits: List[int]):
        if not bits:
            raise ValueError("bitstream must be non-empty")
        self._bits = list(bits)
        self._idx = 0

    def take(self) -> int:
        bit = self._bits[self._idx % len(self._bits)]
        self._idx += 1
        return bit

    @property
    def consumed(self) -> int:
        return self._idx

    def __iter__(self) -> Iterator[int]:  # pragma: no cover - convenience
        while True:
            yield self.take()
