from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    n_bins: int = 15,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    if loss_mask is not None:
        keep = loss_mask.bool()
        conf = conf[keep]
        correct = correct[keep]
    else:
        conf = conf.reshape(-1)
        correct = correct.reshape(-1)
    if conf.numel() == 0:
        return logits.new_tensor(0.0)
    ece = logits.new_tensor(0.0)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece = ece + in_bin.float().mean() * (conf[in_bin].mean() - correct[in_bin].mean()).abs()
    return ece


@torch.no_grad()
def brier_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    true_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    score = (1.0 - true_probs).pow(2)
    if loss_mask is not None:
        weights = loss_mask.float()
        return (score * weights).sum() / weights.sum().clamp_min(1.0)
    return score.mean()

