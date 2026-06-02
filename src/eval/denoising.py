from __future__ import annotations

import torch


@torch.no_grad()
def reconstruction_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float()
    if loss_mask is not None:
        weights = loss_mask.float()
        return (correct * weights).sum() / weights.sum().clamp_min(1.0)
    return correct.mean()

