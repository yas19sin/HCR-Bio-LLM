from __future__ import annotations

import argparse
from typing import Any

from src.training.trainer import load_config, train


def parse_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    config = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        config[key] = parse_value(value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an HCR-LLM experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    parser.add_argument("--set", action="append", default=[], help="Override config key=value.")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    train(config)


if __name__ == "__main__":
    main()

