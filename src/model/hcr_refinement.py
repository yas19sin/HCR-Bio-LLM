from __future__ import annotations

import torch
import torch.nn as nn

from ..training.losses import cross_entropy_loss
from .baseline_transformer import RMSNorm, TransformerConfig
from .hcr_ffn import HCRBlock
from .hcr_state import HCRState
from .lm_head import DistributionalLMHead


class HCRRefinementLM(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        n_basis: int = 16,
        refinement_steps: int = 4,
        mask_token_id: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.refinement_steps = refinement_steps
        self.mask_token_id = mask_token_id
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.mask_logvar = nn.Parameter(torch.full((config.d_model,), 1.0))
        self.obs_logvar = nn.Parameter(torch.full((config.d_model,), -3.0))
        self.blocks = nn.ModuleList(
            [
                HCRBlock(config, kind="density", n_basis=n_basis, causal=False)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.head = DistributionalLMHead(
            config.d_model,
            config.vocab_size,
            n_basis=n_basis,
            use_distributional_features=True,
        )

    def init_state(self, input_ids: torch.Tensor) -> HCRState:
        batch, steps = input_ids.shape
        pos = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        mu = self.token_embedding(input_ids) + self.position_embedding(pos)
        is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
        log_var = torch.where(
            is_mask,
            self.mask_logvar.view(1, 1, -1),
            self.obs_logvar.view(1, 1, -1),
        )
        return HCRState(mu=mu, log_var=log_var)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_state: bool = False,
        return_steps: bool = False,
    ) -> dict:
        state = self.init_state(input_ids)
        step_logits = []
        for _ in range(self.refinement_steps):
            for block in self.blocks:
                state = block(state)
            state = HCRState(
                mu=self.norm(state.mu),
                log_var=state.log_var,
                corr=state.corr,
                basis=state.basis,
            )
            if return_steps:
                step_logits.append(self.head(state))
        logits = step_logits[-1] if step_logits else self.head(state)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = cross_entropy_loss(logits, targets, loss_mask)
            if return_steps and len(step_logits) > 1:
                out["step_losses"] = torch.stack(
                    [cross_entropy_loss(step, targets, loss_mask).detach() for step in step_logits]
                )
        if return_state:
            out["state"] = {
                "mu": state.mu,
                "log_var": state.log_var,
                "basis": state.basis,
            }
        return out

