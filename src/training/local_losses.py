from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborPredictionLoss(nn.Module):
    """Local objective: each token state predicts adjacent hidden states."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.left = nn.Linear(d_model, d_model)
        self.right = nn.Linear(d_model, d_model)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.size(1) < 2:
            return mu.new_tensor(0.0)
        left_loss = F.mse_loss(self.left(mu[:, 1:]), mu[:, :-1].detach())
        right_loss = F.mse_loss(self.right(mu[:, :-1]), mu[:, 1:].detach())
        return 0.5 * (left_loss + right_loss)


def moment_smoothness_loss(log_var: torch.Tensor | None, weight: float = 1e-4) -> torch.Tensor:
    if log_var is None or log_var.size(1) < 2 or weight <= 0:
        device = log_var.device if log_var is not None else "cpu"
        return torch.tensor(0.0, device=device)
    return weight * (log_var[:, 1:] - log_var[:, :-1]).pow(2).mean()

