from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        clean = {}
        for key, value in payload.items():
            if hasattr(value, "item"):
                value = value.item()
            clean[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, sort_keys=True) + "\n")


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)

