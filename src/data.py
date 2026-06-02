from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import torch

from .tokenizer import CharTokenizer


HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
HF_TEXT_COLUMN_CANDIDATES = (
    "text",
    "Text",
    "content",
    "Content",
    "sentence",
    "Sentence",
    "prompt",
    "Prompt",
)


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
    rows: int | None = None
    cache_path: str | None = None


def load_text_dataset(config: dict[str, Any]) -> tuple[str, DatasetInfo]:
    data_path = config.get("data_path")
    if data_path:
        path = Path(data_path)
        text = path.read_text(encoding="utf-8")
        return text, DatasetInfo(source=str(path), chars=len(text), fallback=False)

    dataset = str(config.get("dataset", "tiny_shakespeare"))
    if dataset in {"hf", "huggingface"} or dataset.startswith("hf:"):
        repo_id = str(config.get("hf_dataset") or dataset.removeprefix("hf:"))
        return load_huggingface_text_dataset(config, repo_id)

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
        f"unknown dataset {dataset!r}; use dataset=huggingface with hf_dataset=..., "
        "data_path, inline text, or data/tiny_shakespeare.txt"
    )


def load_huggingface_text_dataset(
    config: dict[str, Any],
    repo_id: str,
) -> tuple[str, DatasetInfo]:
    if not repo_id:
        raise ValueError("hf_dataset is required when dataset is 'huggingface'")
    hf_config = str(config.get("hf_config", "default"))
    split = str(config.get("hf_split", "train"))
    text_column = config.get("hf_text_column")
    text_column = None if text_column in {None, ""} else str(text_column)
    max_rows = config.get("hf_max_rows")
    max_rows = None if max_rows in {None, ""} else int(max_rows)
    cache_dir = Path(str(config.get("hf_cache_dir", Path("data") / "hf_cache")))
    cache_path = _hf_cache_path(cache_dir, repo_id, hf_config, split, text_column, max_rows)

    if cache_path.exists() and not bool(config.get("hf_refresh_cache", False)):
        text = cache_path.read_text(encoding="utf-8")
        return text, DatasetInfo(
            source=_hf_source(repo_id, hf_config, split),
            chars=len(text),
            fallback=False,
            rows=None,
            cache_path=str(cache_path),
        )

    rows = _fetch_hf_rows(
        repo_id=repo_id,
        hf_config=hf_config,
        split=split,
        max_rows=max_rows,
        timeout=float(config.get("hf_timeout_seconds", 30.0)),
        token=os.environ.get(str(config.get("hf_token_env", "HF_TOKEN"))),
    )
    if not rows:
        raise RuntimeError(f"Hugging Face dataset {repo_id!r} returned no rows for split {split!r}")

    first_row = _row_payload(rows[0])
    resolved_column = text_column or _detect_text_column(first_row)
    parts = []
    for item in rows:
        row = _row_payload(item)
        if resolved_column not in row:
            raise KeyError(
                f"text column {resolved_column!r} not present in Hugging Face row; "
                f"available columns: {sorted(row)}"
            )
        value = row[resolved_column]
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(json.dumps(value, ensure_ascii=False))
    text = "\n\n".join(parts)
    if not text:
        raise RuntimeError(f"Hugging Face column {resolved_column!r} produced empty text")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text, DatasetInfo(
        source=f"{_hf_source(repo_id, hf_config, split)}#{resolved_column}",
        chars=len(text),
        fallback=False,
        rows=len(rows),
        cache_path=str(cache_path),
    )


def _fetch_hf_rows(
    repo_id: str,
    hf_config: str,
    split: str,
    max_rows: int | None,
    timeout: float,
    token: str | None,
) -> list[dict[str, Any]]:
    page_size = 100
    offset = 0
    rows: list[dict[str, Any]] = []
    while True:
        length = page_size if max_rows is None else min(page_size, max_rows - len(rows))
        if length <= 0:
            break
        payload = _fetch_hf_json(
            "rows",
            {
                "dataset": repo_id,
                "config": hf_config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=timeout,
            token=token,
        )
        page_rows = payload.get("rows", [])
        if not isinstance(page_rows, list):
            raise RuntimeError(f"unexpected Hugging Face rows payload: {payload}")
        rows.extend(page_rows)
        total = payload.get("num_rows_total")
        offset += len(page_rows)
        if not page_rows or (isinstance(total, int) and offset >= total):
            break
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _fetch_hf_json(
    endpoint: str,
    params: dict[str, Any],
    timeout: float,
    token: str | None,
) -> dict[str, Any]:
    url = f"{HF_DATASET_SERVER}/{endpoint}?{urlencode(params)}"
    headers = {"User-Agent": "HCR-Bio-LLM/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected Hugging Face response from {url}")
    if "error" in payload:
        raise RuntimeError(f"Hugging Face dataset error for {url}: {payload['error']}")
    return payload


def _row_payload(item: dict[str, Any]) -> dict[str, Any]:
    row = item.get("row", item)
    if not isinstance(row, dict):
        raise RuntimeError(f"unexpected Hugging Face row payload: {item}")
    return row


def _detect_text_column(row: dict[str, Any]) -> str:
    for column in HF_TEXT_COLUMN_CANDIDATES:
        if isinstance(row.get(column), str):
            return column
    string_columns = [key for key, value in row.items() if isinstance(value, str)]
    if string_columns:
        return max(string_columns, key=lambda key: len(row[key]))
    raise ValueError(f"could not auto-detect a string text column from columns: {sorted(row)}")


def _hf_cache_path(
    cache_dir: Path,
    repo_id: str,
    hf_config: str,
    split: str,
    text_column: str | None,
    max_rows: int | None,
) -> Path:
    label = "__".join(
        [
            repo_id,
            hf_config,
            split,
            text_column or "auto",
            "all" if max_rows is None else str(max_rows),
        ]
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return cache_dir / f"{safe}.txt"


def _hf_source(repo_id: str, hf_config: str, split: str) -> str:
    return f"hf://datasets/{repo_id}/{hf_config}/{split}"


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
