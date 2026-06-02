from __future__ import annotations

import torch


def corrupt_tokens(
    input_ids: torch.Tensor,
    vocab_size: int,
    probability: float = 0.1,
    mask_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(input_ids.float()) < probability
    corrupted = input_ids.clone()
    if mask_token_id is None:
        replacement = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device)
        corrupted[mask] = replacement[mask]
    else:
        corrupted[mask] = mask_token_id
    return corrupted, mask

