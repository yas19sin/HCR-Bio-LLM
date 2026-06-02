from __future__ import annotations

import argparse
import json

import torch

from src.data import build_datasets
from src.eval.corruption import corrupt_tokens
from src.training.trainer import estimate_metrics, load_checkpoint, resolve_device


@torch.no_grad()
def estimate_corrupted_loss(model, dataset, config, device, batches: int, probability: float, mask_token_id: int):
    model.eval()
    losses = []
    task = str(config.get("task", "causal"))
    if task != "causal":
        return None
    for _ in range(batches):
        batch = dataset.get_batch(
            int(config.get("batch_size", 32)),
            device,
            task="causal",
            mask_probability=float(config.get("mask_probability", 0.15)),
        ).to_dict()
        corrupted, _ = corrupt_tokens(
            batch["input_ids"],
            int(config["_vocab_size"]),
            probability=probability,
            mask_token_id=mask_token_id,
        )
        out = model(corrupted, targets=batch["targets"])
        losses.append(float(out["loss"].item()))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on validation data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--corruption-prob", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device({"device": args.device})
    model, tokenizer, config, meta = load_checkpoint(args.checkpoint, device)
    config["_vocab_size"] = tokenizer.vocab_size
    _, val_data, _, _ = build_datasets(config)
    metrics = estimate_metrics(model, val_data, config, device, args.eval_batches)
    corrupted_loss = estimate_corrupted_loss(
        model,
        val_data,
        config,
        device,
        args.eval_batches,
        args.corruption_prob,
        tokenizer.mask_id,
    )
    if corrupted_loss is not None:
        metrics["corrupted_loss"] = corrupted_loss
        metrics["corruption_degradation"] = corrupted_loss / max(metrics["loss"], 1e-8)
    print(json.dumps({"checkpoint_step": meta["step"], **metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
