from __future__ import annotations

import argparse
import json

import torch

from src.data import build_datasets
from src.eval.moment_analysis import summarize_state
from src.training.trainer import get_task, load_checkpoint, resolve_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hidden distribution channels.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device({"device": args.device})
    model, _, config, meta = load_checkpoint(args.checkpoint, device)
    _, val_data, _ = build_datasets(config)
    task = get_task(config)
    summaries: list[dict[str, float]] = []
    for _ in range(args.batches):
        batch = val_data.get_batch(
            int(config.get("batch_size", 32)),
            device,
            task=task,
            mask_probability=float(config.get("mask_probability", 0.15)),
        ).to_dict()
        out = model(
            batch["input_ids"],
            targets=batch["targets"],
            loss_mask=batch.get("loss_mask"),
            return_state=True,
            return_steps=bool(config.get("return_steps", False)),
        )
        summaries.append(summarize_state(out.get("state")))

    merged: dict[str, float] = {}
    keys = sorted({key for item in summaries for key in item})
    for key in keys:
        values = [item[key] for item in summaries if key in item]
        merged[key] = sum(values) / len(values)
    print(json.dumps({"checkpoint_step": meta["step"], **merged}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

