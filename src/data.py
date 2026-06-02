from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .tokenizer import CharTokenizer


FALLBACK_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us.
""".strip()


@dataclass
class Batch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor | None = None

    def to_dict(self) -> dict[str, torch.Tensor]:
        out = {"input_ids": self.input_ids, "targets": self.targets}
        if self.loss_mask is not None:
            out["loss_mask"] = self.loss_mask
        return out


class CharacterDataset:
    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        context_length: int,
        split: str,
        val_fraction: float = 0.1,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if len(text) < context_length + 2:
            repeat = (context_length + 2) // max(1, len(text)) + 1
            text = (text + "\n") * repeat

        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_val = max(context_length + 2, int(len(ids) * val_fraction))
        n_val = min(n_val, max(context_length + 2, len(ids) // 2))
        if split == "train":
            data = ids[:-n_val]
        else:
            data = ids[-n_val:]
        if len(data) < context_length + 2:
            data = ids

        self.data = data.contiguous()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.split = split

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length - 1)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        task: str = "causal",
        mask_probability: float = 0.15,
    ) -> Batch:
        max_start = len(self.data) - self.context_length - 1
        if max_start <= 0:
            raise ValueError("dataset is too short for the configured context_length")
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[i : i + self.context_length] for i in starts])

        if task in {"denoising", "masked", "fim"}:
            targets = x.clone()
            mask = torch.rand_like(x.float()) < mask_probability
            # Keep at least one target per sequence for stable masked losses.
            empty_rows = ~mask.any(dim=1)
            if empty_rows.any():
                cols = torch.randint(0, self.context_length, (int(empty_rows.sum()),))
                mask[empty_rows, cols] = True
            corrupted = x.clone()
            corrupted[mask] = self.tokenizer.mask_id
            return Batch(
                input_ids=corrupted.to(device),
                targets=targets.to(device),
                loss_mask=mask.to(device),
            )

        targets = torch.stack(
            [self.data[i + 1 : i + self.context_length + 1] for i in starts]
        )
        return Batch(input_ids=x.to(device), targets=targets.to(device))


@dataclass
class DatasetInfo:
    source: str
    chars: int
    fallback: bool = False


def load_text_dataset(config: dict[str, Any]) -> tuple[str, DatasetInfo]:
    data_path = config.get("data_path")
    if data_path:
        path = Path(data_path)
        text = path.read_text(encoding="utf-8")
        return text, DatasetInfo(source=str(path), chars=len(text), fallback=False)

    dataset = config.get("dataset", "tiny_shakespeare")
    if dataset == "inline":
        text = str(config["text"])
        return text, DatasetInfo(source="inline", chars=len(text), fallback=False)

    if dataset == "tiny_shakespeare":
        local_path = Path("data") / "tiny_shakespeare.txt"
        if local_path.exists():
            text = local_path.read_text(encoding="utf-8")
            return text, DatasetInfo(source=str(local_path), chars=len(text), fallback=False)
        if not bool(config.get("allow_fallback_dataset", True)):
            raise FileNotFoundError(
                "data/tiny_shakespeare.txt is required for this config. "
                "The built-in fallback excerpt is too small for benchmark runs and will "
                "produce memorization like near-zero train loss with exploding validation loss. "
                "Use configs/smoke.yaml for an offline smoke test, provide data_path=..., "
                "or set allow_fallback_dataset=true only for non-benchmark debugging."
            )
        return FALLBACK_SHAKESPEARE, DatasetInfo(
            source="built-in fallback Shakespeare excerpt",
            chars=len(FALLBACK_SHAKESPEARE),
            fallback=True,
        )

    candidate = Path(str(dataset))
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
        return text, DatasetInfo(source=str(candidate), chars=len(text), fallback=False)

    raise ValueError(
        f"unknown dataset {dataset!r}; use data_path, inline text, or data/tiny_shakespeare.txt"
    )


def build_datasets(
    config: dict[str, Any],
) -> tuple[CharacterDataset, CharacterDataset, CharTokenizer, DatasetInfo]:
    text, info = load_text_dataset(config)
    tokenizer = CharTokenizer.from_text(text)
    context_length = int(config.get("context_length", 128))
    val_fraction = float(config.get("val_fraction", 0.1))
    train_data = CharacterDataset(text, tokenizer, context_length, "train", val_fraction)
    val_data = CharacterDataset(text, tokenizer, context_length, "val", val_fraction)
    return train_data, val_data, tokenizer, info
