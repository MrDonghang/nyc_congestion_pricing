"""PCN training loop with hierarchical free-energy regulariser.

Splits a single z-scored series into rolling input/target windows, then trains
with four loss terms:
  * Gaussian NLL on (mu, sigma)
  * ``alpha`` weight on  ``(mu[0] - x[-1])^2``      — boundary continuity
  * ``beta``  weight on  ``SmoothL1(diff(mu), diff(y))`` — first-derivative match
  * ``gamma`` weight on  ``mean(layer_errors^2)``    — hierarchical free energy

Mirrors ``pcn_model_new.py`` from the original research repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    alpha: float = 10.0    # boundary continuity weight
    beta: float = 0.5      # first-derivative match weight
    gamma: float = 0.1     # hierarchical free-energy weight
    test_split: float = 0.2
    val_split: float = 0.1
    patience: int = 10


class _EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.stop = float("inf"), 0, False

    def step(self, val: float) -> None:
        if val < self.best - self.min_delta:
            self.best, self.counter = val, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def make_windows(series: np.ndarray, batch_length: int, pred_length: int):
    n = len(series) - batch_length - pred_length
    if n <= 0:
        raise ValueError(
            f"Series too short ({len(series)}) for batch_length={batch_length}, pred_length={pred_length}"
        )
    xs, ys = [], []
    for t in range(n):
        xs.append(series[t : t + batch_length])
        ys.append(series[t + batch_length : t + batch_length + pred_length])
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


def train_pcn(
    model: nn.Module,
    series: np.ndarray,
    batch_length: int,
    pred_length: int,
    cfg: TrainConfig,
    device: str | torch.device = "cpu",
) -> float:
    """Fit ``model`` on a single z-scored ``series``. Returns final epoch's avg loss."""
    x, y = make_windows(series, batch_length, pred_length)

    # Mirror the original: split off test, then val, train on what's left.
    train_x, _, train_y, _ = train_test_split(x, y, test_size=cfg.test_split, shuffle=False)
    train_x, _, train_y, _ = train_test_split(train_x, train_y, test_size=cfg.val_split, shuffle=False)

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    smooth_l1 = nn.SmoothL1Loss()
    stopper = _EarlyStopping(patience=cfg.patience)

    last_avg = float("nan")
    for _epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)                       # (B, L)
            yb = yb.to(device)                       # (B, pred_length)

            xm = xb.mean(dim=1, keepdim=True)        # (B, 1)
            xs = xb.std(dim=1, keepdim=True) + 1e-6
            xn = (xb - xm) / xs

            mu_n, sigma_n, layer_errors = model(xn.float())

            xs_e = xs.expand(xb.shape[0], pred_length)
            xm_e = xm.expand(xb.shape[0], pred_length)
            mu = mu_n * xs_e + xm_e
            sigma = sigma_n * xs_e

            nll = (torch.log(sigma) + 0.5 * ((yb - mu) / sigma) ** 2).mean()
            continuity = ((mu[:, 0] - xb[:, -1]) ** 2).mean()
            slope = smooth_l1(mu[:, 1:] - mu[:, :-1], yb[:, 1:] - yb[:, :-1])
            hier = torch.stack([(e ** 2).mean() for e in layer_errors]).mean()

            loss = nll + cfg.alpha * continuity + cfg.beta * slope + cfg.gamma * hier

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()

        last_avg = total / max(len(loader), 1)
        stopper.step(total)
        if stopper.stop:
            break

    return last_avg
