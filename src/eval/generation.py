from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits[:, -1, :] / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_ids(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int | None = 50,
) -> torch.Tensor:
    model.eval()
    ids = input_ids
    for _ in range(max_new_tokens):
        context = model.crop_context(ids) if hasattr(model, "crop_context") else ids[:, -model.config.context_length :]
        logits = model(context)["logits"]
        next_id = sample_next_token(logits, temperature=temperature, top_k=top_k)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device | str,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int | None = 50,
) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = generate_ids(model, input_ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(out[0].tolist())

