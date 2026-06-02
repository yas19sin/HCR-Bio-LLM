from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HCRState:
    mu: torch.Tensor
    log_var: torch.Tensor | None = None
    corr: torch.Tensor | None = None
    basis: torch.Tensor | None = None

    def with_mu(self, mu: torch.Tensor) -> "HCRState":
        return HCRState(mu=mu, log_var=self.log_var, corr=self.corr, basis=self.basis)


def stabilize_logvar(
    log_var: torch.Tensor,
    min_value: float = -8.0,
    max_value: float = 4.0,
) -> torch.Tensor:
    return torch.clamp(log_var, min=min_value, max=max_value)


def basis_entropy(basis: torch.Tensor | None) -> torch.Tensor | None:
    if basis is None:
        return None
    probs = basis.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)

