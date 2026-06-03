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


def hcr_denominator_stability_loss(
    denominator: torch.Tensor | None,
    min_abs: float = 0.05,
    negative_weight: float = 1.0,
    weight: float = 0.0,
) -> torch.Tensor:
    """Discourage invalid or nearly singular HCR conditional normalizers."""

    if denominator is None or weight <= 0:
        device = denominator.device if denominator is not None else "cpu"
        return torch.tensor(0.0, device=device)
    denom = denominator.float()
    small = torch.relu(torch.tensor(min_abs, device=denom.device) - denom.abs()).pow(2).mean()
    negative = torch.relu(-denom).pow(2).mean()
    return weight * (small + negative_weight * negative)


def hcr_conditional_coefficient_loss(
    coefficients: torch.Tensor | None,
    max_rms: float = 4.0,
    weight: float = 0.0,
) -> torch.Tensor:
    """Limit non-normalizer conditional coefficients to avoid density blow-ups."""

    if coefficients is None or weight <= 0:
        device = coefficients.device if coefficients is not None else "cpu"
        return torch.tensor(0.0, device=device)
    nontrivial = coefficients.float()[..., 1:]
    if nontrivial.numel() == 0:
        return torch.tensor(0.0, device=coefficients.device)
    rms = nontrivial.pow(2).mean().sqrt()
    return weight * torch.relu(rms - torch.tensor(max_rms, device=coefficients.device)).pow(2)


def hcr_variance_bound_loss(
    variance: torch.Tensor | None,
    max_variance: float = 0.25,
    weight: float = 0.0,
) -> torch.Tensor:
    """Keep normalized conditional variances inside the [0, 1] variable range."""

    if variance is None or weight <= 0:
        device = variance.device if variance is not None else "cpu"
        return torch.tensor(0.0, device=device)
    upper = torch.relu(variance.float() - torch.tensor(max_variance, device=variance.device))
    return weight * upper.pow(2).mean()
