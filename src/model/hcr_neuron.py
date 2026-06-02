from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hcr_state import HCRState, stabilize_logvar


class HCRMomentNeuron(nn.Module):
    """Distributional-state approximation carrying mean and log variance."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.init_logvar_bias = nn.Parameter(torch.full((d_model,), -3.0))
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu_update = nn.Linear(hidden, d_model)
        self.var_update = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def ensure_logvar(self, state: HCRState) -> torch.Tensor:
        if state.log_var is not None:
            return state.log_var
        return self.init_logvar_bias.view(1, 1, -1).expand_as(state.mu)

    def forward(self, state: HCRState) -> HCRState:
        log_var = self.ensure_logvar(state)
        features = torch.cat([state.mu, log_var], dim=-1)
        hidden = self.trunk(features)
        delta_mu = self.dropout(self.mu_update(hidden))
        delta_logvar = 0.25 * torch.tanh(self.var_update(hidden))
        return HCRState(
            mu=state.mu + delta_mu,
            log_var=stabilize_logvar(log_var + delta_logvar),
            corr=state.corr,
            basis=state.basis,
        )


class HCRDensityNeuron(nn.Module):
    """Latent density-channel approximation.

    The HCR paper's density coefficients are mixed moments in a product basis.
    This module instead keeps a compact learned per-token coefficient vector, so
    it should be treated as a trainable bridge toward HCR rather than a literal
    local joint-density neuron.
    """

    def __init__(self, d_model: int, hidden: int, n_basis: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.moment = HCRMomentNeuron(d_model, hidden, dropout)
        self.init_basis = nn.Linear(2 * d_model, n_basis)
        self.basis_update = nn.Sequential(
            nn.Linear(2 * d_model + n_basis, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_basis),
        )

    def ensure_basis(self, mu: torch.Tensor, log_var: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
        if basis is not None:
            return basis
        return F.softmax(self.init_basis(torch.cat([mu, log_var], dim=-1)), dim=-1)

    def forward(self, state: HCRState) -> HCRState:
        state = self.moment(state)
        assert state.log_var is not None
        basis = self.ensure_basis(state.mu, state.log_var, state.basis)
        features = torch.cat([state.mu, state.log_var, basis], dim=-1)
        logits = torch.log(basis.clamp_min(1e-8)) + self.basis_update(features)
        return HCRState(
            mu=state.mu,
            log_var=state.log_var,
            corr=state.corr,
            basis=F.softmax(logits, dim=-1),
        )


class HCRJointNeuron(nn.Module):
    """Compressed pairwise-channel approximation, not a dense HCR tensor."""

    def __init__(self, d_model: int, hidden: int, pairwise_rank: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pairwise_rank = pairwise_rank
        self.moment = HCRMomentNeuron(d_model, hidden, dropout)
        self.init_corr = nn.Linear(2 * d_model, pairwise_rank)
        self.corr_update = nn.Sequential(
            nn.Linear(2 * d_model + pairwise_rank, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pairwise_rank),
        )
        self.corr_norm = nn.LayerNorm(pairwise_rank)

    def ensure_corr(self, mu: torch.Tensor, log_var: torch.Tensor, corr: torch.Tensor | None) -> torch.Tensor:
        if corr is not None:
            return corr
        return torch.tanh(self.init_corr(torch.cat([mu, log_var], dim=-1)))

    def forward(self, state: HCRState) -> HCRState:
        state = self.moment(state)
        assert state.log_var is not None
        corr = self.ensure_corr(state.mu, state.log_var, state.corr)
        features = torch.cat([state.mu, state.log_var, corr], dim=-1)
        corr = self.corr_norm(corr + 0.25 * torch.tanh(self.corr_update(features)))
        return HCRState(mu=state.mu, log_var=state.log_var, corr=corr, basis=state.basis)
