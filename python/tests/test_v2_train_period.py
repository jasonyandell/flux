"""Tests for v2 trainer cadence helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v2 import _starts_ai_accum_window


def test_accumulator_window_starts_for_single_tick_ai():
    assert [_starts_ai_accum_window(t, 1) for t in range(1, 6)] == [
        True, True, True, True, True,
    ]


def test_accumulator_window_starts_for_multi_tick_ai():
    assert [_starts_ai_accum_window(t, 5) for t in range(1, 12)] == [
        True, False, False, False, False,
        True, False, False, False, False,
        True,
    ]
