from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCRKANNeuron(nn.Module):
    """KAN-inspired expected-value block using radial basis functions.

    This is deliberately not a faithful KAN layer from arXiv:2404.19756. KANs
    put learnable spline functions on edges and avoid ordinary linear weights as
    the main transformation. This block is the cheaper approximation requested
    in the project brief for a first Transformer-compatible comparison.
    """

    def __init__(self, d_model: int, d_hidden: int, n_basis: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_hidden)
        self.centers = nn.Parameter(torch.linspace(-2.0, 2.0, n_basis))
        self.widths = nn.Parameter(torch.ones(n_basis))
        self.coeffs = nn.Parameter(torch.randn(d_hidden, n_basis) * 0.02)
        self.out_proj = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        widths = F.softplus(self.widths).view(1, 1, 1, -1) + 1e-4
        return torch.exp(-((z - centers) ** 2) / (2.0 * widths.pow(2)))

    def forward(self, x: torch.Tensor, return_basis: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.in_proj(x)
        basis_values = self.basis(hidden)
        mixed = torch.einsum("bthk,hk->bth", basis_values, self.coeffs)
        out = self.dropout(self.out_proj(mixed))
        if return_basis:
            return out, basis_values
        return out


class HCRKANFFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, n_basis: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.neuron = HCRKANNeuron(d_model, mlp_ratio * d_model, n_basis, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(x)
