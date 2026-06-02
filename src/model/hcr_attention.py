from __future__ import annotations

import torch
import torch.nn as nn

from .baseline_transformer import RMSNorm, SelfAttention, TransformerConfig
from .hcr_state import HCRState, stabilize_logvar


class HCRSelfAttention(nn.Module):
    """Routes mean and uncertainty as separate messages."""

    def __init__(self, config: TransformerConfig, causal: bool = True) -> None:
        super().__init__()
        self.mu_norm = RMSNorm(config.d_model)
        self.mu_attn = SelfAttention(config, causal=causal)
        self.var_norm = RMSNorm(config.d_model)
        self.var_attn = SelfAttention(config, causal=causal)

    def forward(self, state: HCRState) -> HCRState:
        mu = state.mu + self.mu_attn(self.mu_norm(state.mu))
        log_var = state.log_var
        if log_var is not None:
            delta_var = 0.25 * torch.tanh(self.var_attn(self.var_norm(log_var)))
            log_var = stabilize_logvar(log_var + delta_var)
        return HCRState(mu=mu, log_var=log_var, corr=state.corr, basis=state.basis)

