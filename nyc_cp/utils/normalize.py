"""Normalisation helpers for 1-D series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZScoreParams:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def invert(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def invert_std(self, x: np.ndarray) -> np.ndarray:
        return x * self.std


@dataclass(frozen=True)
class MinMaxParams:
    lo: float
    hi: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def invert(self, x: np.ndarray) -> np.ndarray:
        return x * (self.hi - self.lo) + self.lo


def zscore(x: np.ndarray) -> tuple[np.ndarray, ZScoreParams]:
    p = ZScoreParams(mean=float(np.mean(x)), std=float(np.std(x)) + 1e-8)
    return p.transform(x), p


def minmax(x: np.ndarray) -> tuple[np.ndarray, MinMaxParams]:
    p = MinMaxParams(lo=float(np.min(x)), hi=float(np.max(x)) + 1e-8)
    return p.transform(x), p
