from __future__ import annotations

from typing import Any

import torch

from ..model.hcr_state import basis_entropy


@torch.no_grad()
def summarize_state(state: dict[str, torch.Tensor | None] | None) -> dict[str, float]:
    if not state:
        return {}
    out: dict[str, float] = {}
    mu = state.get("mu")
    if mu is not None:
        out["mu_mean"] = float(mu.mean().item())
        out["mu_std"] = float(mu.std().item())
    log_var = state.get("log_var")
    if log_var is not None:
        var = log_var.exp()
        out["log_var_mean"] = float(log_var.mean().item())
        out["log_var_std"] = float(log_var.std().item())
        out["variance_mean"] = float(var.mean().item())
    corr = state.get("corr")
    if corr is not None:
        out["corr_mean"] = float(corr.mean().item())
        out["corr_std"] = float(corr.std().item())
    basis = state.get("basis")
    entropy = basis_entropy(basis)
    if entropy is not None:
        out["basis_entropy"] = float(entropy.mean().item())
    for key, value in state.items():
        if key.startswith("hcr_") and value is not None:
            if value.numel() == 1:
                out[key] = float(value.float().item())
                continue
            out[f"{key}_mean"] = float(value.float().mean().item())
            if value.numel() > 1:
                out[f"{key}_std"] = float(value.float().std().item())
    return out


def prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}
