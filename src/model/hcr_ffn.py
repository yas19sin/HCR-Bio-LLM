from __future__ import annotations

import torch.nn as nn

from .baseline_transformer import RMSNorm
from .hcr_attention import HCRSelfAttention
from .hcr_neuron import HCRDensityNeuron, HCRJointNeuron, HCRMomentNeuron
from .hcr_state import HCRState


class HCRBlock(nn.Module):
    def __init__(
        self,
        config,
        kind: str = "moment",
        n_basis: int = 16,
        pairwise_rank: int = 8,
        causal: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(config.mlp_ratio * config.d_model)
        self.attn = HCRSelfAttention(config, causal=causal)
        self.norm = RMSNorm(config.d_model)
        if kind == "moment":
            self.neuron = HCRMomentNeuron(config.d_model, hidden, config.dropout)
        elif kind == "density":
            self.neuron = HCRDensityNeuron(config.d_model, hidden, n_basis, config.dropout)
        elif kind == "joint":
            self.neuron = HCRJointNeuron(config.d_model, hidden, pairwise_rank, config.dropout)
        else:
            raise ValueError(f"unknown HCR block kind: {kind}")

    def forward(self, state: HCRState) -> HCRState:
        state = self.attn(state)
        normed = HCRState(
            mu=self.norm(state.mu),
            log_var=state.log_var,
            corr=state.corr,
            basis=state.basis,
        )
        updated = self.neuron(normed)
        return HCRState(
            mu=state.mu + (updated.mu - normed.mu),
            log_var=updated.log_var,
            corr=updated.corr,
            basis=updated.basis,
        )

