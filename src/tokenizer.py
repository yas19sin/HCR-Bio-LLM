from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<mask>"]


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        itos = SPECIAL_TOKENS + [ch for ch in chars if ch not in SPECIAL_TOKENS]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def mask_id(self) -> int:
        return self.stoi[self.mask_token]

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"text contains characters outside tokenizer vocab: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {
            "itos": self.itos,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        itos = list(payload["itos"])
        return cls(
            stoi={ch: i for i, ch in enumerate(itos)},
            itos=itos,
            pad_token=payload.get("pad_token", "<pad>"),
            mask_token=payload.get("mask_token", "<mask>"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

