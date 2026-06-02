from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.losses import cross_entropy_loss


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    mlp_ratio: int = 4
    causal: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any], vocab_size: int) -> "TransformerConfig":
        return cls(
            vocab_size=vocab_size,
            context_length=int(config.get("context_length", 128)),
            d_model=int(config.get("d_model", 128)),
            n_layers=int(config.get("n_layers", 4)),
            n_heads=int(config.get("n_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
            mlp_ratio=int(config.get("mlp_ratio", 4)),
            causal=bool(config.get("causal", True)),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, causal: bool | None = None) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.causal = config.causal if causal is None else causal
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            scores = scores.masked_fill(self.causal_mask[:, :, :steps, :steps] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, ffn: nn.Module | None = None) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = ffn if ffn is not None else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_state: bool = False,
        return_steps: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, steps = input_ids.shape
        if steps > self.config.context_length:
            raise ValueError("sequence length exceeds context_length")
        pos = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"logits": logits}
        if targets is not None:
            out["loss"] = cross_entropy_loss(logits, targets, loss_mask)
        if return_state:
            out["state"] = {"mu": x}
        return out

    @torch.no_grad()
    def crop_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids[:, -self.config.context_length :]
