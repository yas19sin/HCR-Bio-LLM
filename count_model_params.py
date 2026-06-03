from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.model import build_model
from src.training.trainer import count_parameters, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count trainable parameters for model configs.")
    parser.add_argument("configs", nargs="+", help="Config file(s) or directories containing *.yaml files.")
    parser.add_argument("--vocab-size", type=int, default=67, help="Character vocab size used for the count.")
    parser.add_argument("--mask-token-id", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a Markdown table.")
    return parser.parse_args()


def expand_configs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.yaml")))
            paths.extend(sorted(path.glob("*.yml")))
        else:
            paths.append(path)
    return paths


def count_config(path: Path, vocab_size: int, mask_token_id: int) -> dict[str, Any]:
    config = load_config(path)
    model = build_model(config, vocab_size, mask_token_id)
    params = count_parameters(model)
    target = config.get("parameter_target")
    delta = None
    if target not in {None, ""}:
        delta = params - int(target)
    return {
        "config": str(path),
        "model_type": config.get("model_type"),
        "task": config.get("task", "causal"),
        "d_model": config.get("d_model"),
        "n_layers": config.get("n_layers"),
        "n_heads": config.get("n_heads"),
        "mlp_ratio": config.get("mlp_ratio", ""),
        "params": params,
        "parameter_target": target,
        "target_delta": delta,
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| config | model | task | d_model | layers | heads | mlp | params | target_delta |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        delta = "" if row["target_delta"] is None else str(row["target_delta"])
        print(
            "| {config} | {model_type} | {task} | {d_model} | {n_layers} | {n_heads} | "
            "{mlp_ratio} | {params} | {delta} |".format(**row, delta=delta)
        )


def main() -> None:
    args = parse_args()
    rows = [count_config(path, args.vocab_size, args.mask_token_id) for path in expand_configs(args.configs)]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
