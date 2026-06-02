from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..training.losses import cross_entropy_loss
from .baseline_transformer import RMSNorm, TransformerBlock, TransformerConfig
from .hcr_basis import HCRKANFFN
from .hcr_ffn import HCRBlock
from .hcr_state import HCRState
from .lm_head import DistributionalLMHead


class HCRKANMeanLM(nn.Module):
    def __init__(self, config: TransformerConfig, n_basis: int = 16) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    ffn=HCRKANFFN(config.d_model, config.mlp_ratio, n_basis, config.dropout),
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_state: bool = False,
        return_steps: bool = False,
    ) -> dict:
        _, steps = input_ids.shape
        pos = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = cross_entropy_loss(logits, targets, loss_mask)
        if return_state:
            out["state"] = {"mu": x}
        return out


class HCRDistributionalLM(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        kind: str = "moment",
        n_basis: int = 16,
        pairwise_rank: int = 8,
        use_distributional_head: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.kind = kind
        self.n_basis = n_basis if kind == "density" else 0
        self.pairwise_rank = pairwise_rank if kind == "joint" else 0
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.init_logvar = nn.Parameter(torch.full((config.d_model,), -3.0))
        self.blocks = nn.ModuleList(
            [
                HCRBlock(
                    config,
                    kind=kind,
                    n_basis=n_basis,
                    pairwise_rank=pairwise_rank,
                    causal=config.causal,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.head = DistributionalLMHead(
            config.d_model,
            config.vocab_size,
            pairwise_rank=self.pairwise_rank,
            n_basis=self.n_basis,
            use_distributional_features=use_distributional_head,
        )

    def init_state(self, input_ids: torch.Tensor) -> HCRState:
        _, steps = input_ids.shape
        pos = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        mu = self.token_embedding(input_ids) + self.position_embedding(pos)
        mu = self.drop(mu)
        log_var = self.init_logvar.view(1, 1, -1).expand_as(mu)
        return HCRState(mu=mu, log_var=log_var)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_state: bool = False,
        return_steps: bool = False,
    ) -> dict[str, Any]:
        state = self.init_state(input_ids)
        for block in self.blocks:
            state = block(state)
        state = HCRState(
            mu=self.norm(state.mu),
            log_var=state.log_var,
            corr=state.corr,
            basis=state.basis,
        )
        logits = self.head(state)
        out: dict[str, Any] = {"logits": logits}
        if targets is not None:
            out["loss"] = cross_entropy_loss(logits, targets, loss_mask)
        if return_state:
            out["state"] = {
                "mu": state.mu,
                "log_var": state.log_var,
                "corr": state.corr,
                "basis": state.basis,
            }
        return out
