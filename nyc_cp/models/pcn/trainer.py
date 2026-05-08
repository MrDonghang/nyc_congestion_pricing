"""PCN training loop.

Splits a single z-scored series into rolling input/target windows, then trains
with Gaussian NLL plus two regularisers:
  * ``alpha`` weight on  ``(mu[0] - x[-1])^2``    — continuity at the boundary
  * ``beta``  weight on  ``MSE(diff(mu), diff(y))`` — first-derivative smoothness
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
    alpha: float = 10.0
    beta: float = 0.5
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

    train_x, _, train_y, _ = train_test_split(x, y, test_size=cfg.test_split, shuffle=False)
    train_x, _, train_y, _ = train_test_split(train_x, train_y, test_size=cfg.val_split, shuffle=False)

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    derivative_loss = nn.SmoothL1Loss()
    stopper = _EarlyStopping(patience=cfg.patience)

    last_avg = float("nan")
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.unsqueeze(0).to(device)  # (1, B, L)
            yb = yb.unsqueeze(0).to(device)

            xm = xb.mean(dim=2, keepdim=True)
            xs = xb.std(dim=2, keepdim=True) + 1e-6
            xn = (xb - xm) / xs

            mu_n, sigma_n = model(xn.float())
            xs_e = xs.expand(1, xb.shape[1], pred_length)
            xm_e = xm.expand(1, xb.shape[1], pred_length)
            mu = mu_n * xs_e + xm_e
            sigma = sigma_n * xs_e

            nll = (torch.log(sigma) + 0.5 * ((yb - mu) / sigma) ** 2).mean()
            constraint = ((mu[:, :, 0] - xb[:, :, -1]) ** 2).mean()
            d_pred = mu[:, 1:] - mu[:, :-1]
            d_true = yb[:, 1:] - yb[:, :-1]
            deriv = derivative_loss(d_pred, d_true)

            loss = nll + cfg.alpha * constraint + cfg.beta * deriv
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()

        last_avg = total / max(len(loader), 1)
        stopper.step(total)
        if stopper.stop:
            break
    return last_avg
