"""Predictive Coding Network for time-series forecasting.

Bidirectional multi-layer PCN with iterative inference. The forward pass
returns ``(mu, sigma, final_layer_errors)`` — the layer errors are used by
the training loop to add a hierarchical free-energy regulariser
(``gamma * mean(error**2)``) on top of the standard NLL.

This implementation matches the production architecture from
``pcn_model_new.py`` in the original research repo.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNLayer(nn.Module):
    """Single PCN layer with separate top-down and bottom-up mappings."""

    def __init__(self, input_size: int, hidden_size: int, activation: Callable | None = torch.tanh):
        super().__init__()
        self.predict_down = nn.Linear(hidden_size, input_size)   # top → bottom
        self.encode_up = nn.Linear(input_size, hidden_size)      # bottom → top
        self.activation = activation

    def bottom_up_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_up(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def top_down_predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.predict_down(h)


class MultiLayerPCN(nn.Module):
    """Multi-layer bidirectional PCN for seq-to-seq forecasting.

    Forward returns ``(mu, sigma, final_layer_errors)``. Each iteration
    refines representations using *bottom-up* error feedback (gradient of
    free energy w.r.t. each rep, approximated via decoder weight transpose)
    and *top-down* consistency from the layer above.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        iterations: int = 10,
        activation: Callable | None = torch.tanh,
        pred_length: int = 24,
        inference_lr: float = 0.05,
    ):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.iterations = iterations
        self.inference_lr = inference_lr
        self.layers = nn.ModuleList(
            [PCNLayer(layer_sizes[i], layer_sizes[i + 1], activation) for i in range(len(layer_sizes) - 1)]
        )
        self.mu_layer = nn.Linear(layer_sizes[-1], pred_length)
        self.sigma_layer = nn.Linear(layer_sizes[-1], pred_length)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """``x`` is ``(batch, input_dim)``; returns ``(mu, sigma, layer_errors)``."""
        # Bottom-up initialisation.
        inputs_per_layer: list[torch.Tensor] = [x]
        reps: list[torch.Tensor] = []
        inp = x
        for layer in self.layers:
            h = layer.bottom_up_encode(inp)
            reps.append(h)
            inp = h
            inputs_per_layer.append(inp)

        n_layers = len(self.layers)

        # Iterative bidirectional refinement.
        for _ in range(self.iterations):
            errors = [
                inputs_per_layer[i] - layer.top_down_predict(reps[i])
                for i, layer in enumerate(self.layers)
            ]

            for i, layer in enumerate(self.layers):
                # Bottom-up: error @ decoder.weight  ↔  W^T err  (free-energy gradient).
                bottom_up = errors[i] @ layer.predict_down.weight

                # Top-down consistency from the layer above.
                if i < n_layers - 1:
                    top_down_pred = self.layers[i + 1].predict_down(reps[i + 1])
                    top_down = reps[i] - top_down_pred
                else:
                    top_down = torch.zeros_like(reps[i])

                reps[i] = reps[i] + self.inference_lr * (bottom_up - top_down)

                if i > 0:
                    inputs_per_layer[i] = reps[i - 1]

        # Forecast heads.
        top = reps[-1]
        mu = self.mu_layer(top)
        sigma = F.softplus(self.sigma_layer(top)) + 1e-6

        # Final layer errors after the last refinement (fed to gamma loss).
        final_layer_errors = [
            inputs_per_layer[i] - layer.top_down_predict(reps[i])
            for i, layer in enumerate(self.layers)
        ]

        return mu, sigma, final_layer_errors
