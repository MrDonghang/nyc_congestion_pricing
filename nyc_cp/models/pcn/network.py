"""Predictive Coding Network architectures.

Two variants:
  * :class:`MultiLayerPCN`   — unidirectional, iterative top-down refinement
  * :class:`MultiLayerPCNBi` — bidirectional with iterative inference
Both produce per-horizon Gaussian (mu, sigma) heads.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class _PCNLayer(nn.Module):
    """Single PCN layer: top-down predict + bottom-up encode."""

    def __init__(self, input_size: int, hidden_size: int, activation: Callable | None = torch.relu):
        super().__init__()
        self.predict = nn.Linear(hidden_size, input_size)
        self.represent = nn.Linear(input_size, hidden_size)
        self.activation = activation

    def forward(self, x_below: torch.Tensor, h_above: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.predict(h_above)
        error = x_below - predicted
        new_repr = self.represent(x_below)
        if self.activation is not None:
            new_repr = self.activation(new_repr)
        return error, new_repr


class MultiLayerPCN(nn.Module):
    """Unidirectional multi-layer PCN with iterative refinement."""

    def __init__(
        self,
        layer_sizes: list[int],
        iterations: int,
        activation: Callable | None = torch.relu,
        pred_length: int = 24,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.iterations = iterations
        self.pred_length = pred_length

        self.layers = nn.ModuleList(
            [_PCNLayer(layer_sizes[i], layer_sizes[i + 1], activation) for i in range(len(layer_sizes) - 1)]
        )
        self.mu_layer = nn.Linear(layer_sizes[-1], pred_length)
        self.sigma_layer = nn.Linear(layer_sizes[-1], pred_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        device = x.device
        reps = [
            torch.zeros(b, self.layer_sizes[i + 1], device=device)
            for i in range(len(self.layer_sizes) - 1)
        ]
        for _ in range(self.iterations):
            inp = x
            new_reps = []
            for i, layer in enumerate(self.layers):
                _err, h = layer(inp, reps[i])
                new_reps.append(h)
                inp = h
            reps = new_reps

        mu = self.mu_layer(inp)
        sigma = nn.functional.softplus(self.sigma_layer(inp)) + 1e-6
        return mu, sigma


class _PCNLayerBi(nn.Module):
    """Bidirectional PCN layer with separate top-down and bottom-up mappings."""

    def __init__(self, input_size: int, hidden_size: int, activation: Callable | None = torch.tanh):
        super().__init__()
        self.predict_down = nn.Linear(hidden_size, input_size)
        self.encode_up = nn.Linear(input_size, hidden_size)
        self.activation = activation

    def bottom_up(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_up(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def top_down(self, h: torch.Tensor) -> torch.Tensor:
        return self.predict_down(h)


class MultiLayerPCNBi(nn.Module):
    """Bidirectional PCN with per-iteration error correction.

    Inference iteratively refines representations with both top-down predictions
    and bottom-up updates, scaled by ``inference_lr``.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        iterations: int = 3,
        activation: Callable | None = torch.relu,
        pred_length: int = 24,
        inference_lr: float = 0.05,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [_PCNLayerBi(layer_sizes[i], layer_sizes[i + 1], activation) for i in range(len(layer_sizes) - 1)]
        )
        self.iterations = iterations
        self.inference_lr = inference_lr
        self.mu_layer = nn.Linear(layer_sizes[-1], pred_length)
        self.sigma_layer = nn.Linear(layer_sizes[-1], pred_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialise representations bottom-up.
        reps: list[torch.Tensor] = []
        inp = x
        for layer in self.layers:
            inp = layer.bottom_up(inp)
            reps.append(inp)

        # Iterative refinement.
        for _ in range(self.iterations):
            new_reps = []
            for i, layer in enumerate(self.layers):
                below = x if i == 0 else reps[i - 1]
                pred = layer.top_down(reps[i])
                err_down = below - pred
                err_up = layer.bottom_up(below) - reps[i]
                new_reps.append(reps[i] + self.inference_lr * (err_up - layer.encode_up(err_down)))
            reps = new_reps

        top = reps[-1]
        mu = self.mu_layer(top)
        sigma = nn.functional.softplus(self.sigma_layer(top)) + 1e-6
        return mu, sigma
