from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baseline_transformer import TinyTransformerLM, TransformerConfig
from .hcr_joint_block import HCRBlockwiseJointLM
from .hcr_lm import HCRDistributionalLM, HCRKANMeanLM
from .hcr_refinement import HCRRefinementLM


def build_model(config: dict[str, Any], vocab_size: int, mask_token_id: int = 1) -> nn.Module:
    model_type = config.get("model_type", "transformer_baseline")
    causal = model_type != "hcr_bidirectional_refinement" and config.get("task", "causal") == "causal"
    tconfig = TransformerConfig.from_dict({**config, "causal": causal}, vocab_size)
    n_basis = int(config.get("n_basis", 16))
    pairwise_rank = int(config.get("pairwise_rank", 8))

    if model_type == "transformer_baseline":
        return TinyTransformerLM(tconfig)
    if model_type == "hcr_kan_mean":
        return HCRKANMeanLM(tconfig, n_basis=n_basis)
    if model_type == "hcr_moment":
        return HCRDistributionalLM(tconfig, kind="moment")
    if model_type == "hcr_density":
        return HCRDistributionalLM(tconfig, kind="density", n_basis=n_basis)
    if model_type == "hcr_joint_pairwise":
        return HCRDistributionalLM(tconfig, kind="joint", pairwise_rank=pairwise_rank)
    if model_type == "hcr_blockwise_joint":
        return HCRBlockwiseJointLM(
            tconfig,
            block_size=int(config.get("hcr_block_size", 2)),
            degree=int(config.get("hcr_degree", 2)),
            output_scale=float(config.get("hcr_output_scale", 0.1)),
            density_state_mix=float(config.get("hcr_density_state_mix", 0.5)),
        )
    if model_type == "hcr_bidirectional_refinement":
        return HCRRefinementLM(
            tconfig,
            n_basis=n_basis,
            refinement_steps=int(config.get("refinement_steps", 4)),
            mask_token_id=mask_token_id,
        )
    raise ValueError(f"unknown model_type: {model_type}")
