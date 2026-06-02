from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    vocab = logits.size(-1)
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    if loss_mask is None:
        return per_token.mean()
    weights = loss_mask.float()
    denom = weights.sum().clamp_min(1.0)
    return (per_token * weights).sum() / denom


def variance_noncollapse_loss(
    log_var: torch.Tensor | None,
    target_std: float = 0.5,
    weight: float = 1e-3,
) -> torch.Tensor:
    if log_var is None or weight <= 0:
        device = log_var.device if log_var is not None else "cpu"
        return torch.tensor(0.0, device=device)
    std = log_var.float().std()
    return weight * (std - target_std).pow(2)


def basis_entropy_loss(
    basis: torch.Tensor | None,
    min_entropy: float = 0.2,
    weight: float = 1e-4,
) -> torch.Tensor:
    if basis is None or weight <= 0:
        device = basis.device if basis is not None else "cpu"
        return torch.tensor(0.0, device=device)
    probs = basis.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return weight * torch.relu(torch.tensor(min_entropy, device=basis.device) - entropy)

