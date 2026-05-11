from __future__ import annotations
from typing import Callable

MASK32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    # JS Math.imul: 32-bit signed multiplication.
    # Perform unsigned multiplication mod 2^32, then reinterpret as signed.
    r = (a * b) & MASK32
    if r >= 0x80000000:
        r -= 0x100000000
    return r


def mulberry32(seed: int) -> Callable[[], float]:
    s = seed & MASK32

    def gen() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & MASK32
        t = s
        # Math.imul(t ^ (t >>> 15), t | 1)
        t = _imul(t ^ (t >> 15), t | 1) & MASK32
        # t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        add = (t + _imul(t ^ (t >> 7), t | 61)) & MASK32
        t = (t ^ add) & MASK32
        # ((t ^ (t >>> 14)) >>> 0) / 4294967296
        out = (t ^ (t >> 14)) & MASK32
        return out / 4294967296.0

    return gen
