from __future__ import annotations

import torch
import torch.nn as nn

from .hcr_state import HCRState


class DistributionalLMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        pairwise_rank: int = 0,
        n_basis: int = 0,
        use_distributional_features: bool = True,
    ) -> None:
        super().__init__()
        self.use_distributional_features = use_distributional_features
        in_dim = d_model
        if use_distributional_features:
            in_dim += d_model
            in_dim += pairwise_rank
            in_dim += n_basis
        self.proj = nn.Linear(in_dim, vocab_size, bias=False)

    def forward(self, state: HCRState) -> torch.Tensor:
        features = [state.mu]
        if self.use_distributional_features:
            features.append(state.log_var if state.log_var is not None else torch.zeros_like(state.mu))
            if state.corr is not None:
                features.append(state.corr)
            if state.basis is not None:
                features.append(state.basis)
        return self.proj(torch.cat(features, dim=-1))

