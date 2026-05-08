"""Deterministic seeding across numpy / torch / random."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
