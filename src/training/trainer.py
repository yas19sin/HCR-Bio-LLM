from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ..data import CharacterDataset, build_datasets
from ..eval.calibration import brier_score, expected_calibration_error
from ..eval.denoising import reconstruction_accuracy
from ..eval.generation import generate_text
from ..eval.moment_analysis import summarize_state
from ..model import build_model
from ..tokenizer import CharTokenizer
from .local_losses import NeighborPredictionLoss, moment_smoothness_loss
from .logging import JSONLLogger, format_metrics
from .losses import basis_entropy_loss, variance_noncollapse_loss


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    return payload


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_task(config: dict[str, Any]) -> str:
    if "task" in config:
        return str(config["task"])
    if config.get("model_type") == "hcr_bidirectional_refinement":
        return "denoising"
    return "causal"


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    tokenizer: CharTokenizer,
    step: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    tokenizer: CharTokenizer,
    step: int,
    metrics: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, optimizer, config, tokenizer, step, metrics), path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, CharTokenizer, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = dict(payload["config"])
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = build_model(config, tokenizer.vocab_size, tokenizer.mask_id)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    meta = {"step": payload.get("step", 0), "metrics": payload.get("metrics", {})}
    return model, tokenizer, config, meta


@torch.no_grad()
def estimate_metrics(
    model: torch.nn.Module,
    dataset: CharacterDataset,
    config: dict[str, Any],
    device: torch.device,
    eval_batches: int,
) -> dict[str, float]:
    model.eval()
    task = get_task(config)
    losses = []
    accs = []
    eces = []
    briers = []
    state_summaries: list[dict[str, float]] = []
    for _ in range(eval_batches):
        batch = dataset.get_batch(
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
        logits = out["logits"]
        losses.append(float(out["loss"].item()))
        accs.append(float(reconstruction_accuracy(logits, batch["targets"], batch.get("loss_mask")).item()))
        eces.append(float(expected_calibration_error(logits, batch["targets"], batch.get("loss_mask")).item()))
        briers.append(float(brier_score(logits, batch["targets"], batch.get("loss_mask")).item()))
        state_summaries.append(summarize_state(out.get("state")))
    metrics: dict[str, float] = {
        "loss": sum(losses) / len(losses),
        "accuracy": sum(accs) / len(accs),
        "ece": sum(eces) / len(eces),
        "brier": sum(briers) / len(briers),
        "perplexity": math.exp(min(20.0, sum(losses) / len(losses))),
    }
    keys = sorted({key for summary in state_summaries for key in summary})
    for key in keys:
        values = [summary[key] for summary in state_summaries if key in summary]
        metrics[key] = sum(values) / len(values)
    return metrics


def build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
    local_loss_module: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    params = list(model.parameters())
    if local_loss_module is not None:
        params += list(local_loss_module.parameters())
    return torch.optim.AdamW(
        params,
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.1)),
        betas=tuple(config.get("betas", [0.9, 0.95])),
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 1337)))
    output_dir = Path(config.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    logger = JSONLLogger(output_dir / "metrics.jsonl")

    device = resolve_device(config)
    train_data, val_data, tokenizer, dataset_info = build_datasets(config)
    model = build_model(config, tokenizer.vocab_size, tokenizer.mask_id).to(device)

    local_weight = float(config.get("local_neighbor_weight", 0.0))
    local_loss_module = None
    if local_weight > 0:
        local_loss_module = NeighborPredictionLoss(int(config.get("d_model", 128))).to(device)

    optimizer = build_optimizer(model, config, local_loss_module)
    max_steps = int(config.get("max_steps", 5000))
    eval_interval = int(config.get("eval_interval", 250))
    eval_batches = int(config.get("eval_batches", 10))
    batch_size = int(config.get("batch_size", 32))
    context_length = int(config.get("context_length", 128))
    task = get_task(config)
    best_val = float("inf")
    t0 = time.perf_counter()

    start_metrics = {
        "event": "start",
        "params": count_parameters(model),
        "device": str(device),
        "model_type": config.get("model_type", "transformer_baseline"),
        "task": task,
        "dataset_source": dataset_info.source,
        "dataset_chars": dataset_info.chars,
        "dataset_fallback": dataset_info.fallback,
        "dataset_rows": dataset_info.rows,
        "dataset_cache_path": dataset_info.cache_path,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
    }
    print(format_metrics(start_metrics))
    logger.write(start_metrics)

    for step in range(1, max_steps + 1):
        model.train()
        batch = train_data.get_batch(
            batch_size,
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
        loss = out["loss"]
        state = out.get("state", {})
        if isinstance(state, dict):
            if state.get("log_var") is not None:
                loss = loss + variance_noncollapse_loss(
                    state.get("log_var"),
                    weight=float(config.get("variance_noncollapse_weight", 0.0)),
                )
                loss = loss + moment_smoothness_loss(
                    state.get("log_var"),
                    weight=float(config.get("moment_smoothness_weight", 0.0)),
                )
            if state.get("basis") is not None:
                loss = loss + basis_entropy_loss(
                    state.get("basis"),
                    weight=float(config.get("basis_entropy_weight", 0.0)),
                )
            if local_loss_module is not None and state.get("mu") is not None:
                loss = loss + local_weight * local_loss_module(state["mu"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip = float(config.get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if local_loss_module is not None:
                torch.nn.utils.clip_grad_norm_(local_loss_module.parameters(), grad_clip)
        optimizer.step()

        if step == 1 or step % int(config.get("log_interval", 25)) == 0:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            metrics = {
                "event": "train",
                "step": step,
                "loss": float(loss.item()),
                "tokens_per_sec": step * batch_size * context_length / elapsed,
            }
            metrics.update({f"state_{k}": v for k, v in summarize_state(state).items()})
            print(format_metrics(metrics))
            logger.write(metrics)

        if step == 1 or step % eval_interval == 0 or step == max_steps:
            val_metrics = estimate_metrics(model, val_data, config, device, eval_batches)
            prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            record = {"event": "eval", "step": step, **prefixed}
            print(format_metrics(record))
            logger.write(record)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    config,
                    tokenizer,
                    step,
                    val_metrics,
                )
            if step % int(config.get("checkpoint_interval", eval_interval)) == 0 or step == max_steps:
                save_checkpoint(
                    output_dir / "last.pt",
                    model,
                    optimizer,
                    config,
                    tokenizer,
                    step,
                    val_metrics,
                )

    final = {"best_val_loss": best_val, "output_dir": str(output_dir), "params": count_parameters(model)}
    if task == "causal" and bool(config.get("sample_at_end", True)):
        prompt = str(config.get("sample_prompt", "First "))
        try:
            final["sample"] = generate_text(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=int(config.get("sample_tokens", 200)),
                temperature=float(config.get("temperature", 0.9)),
                top_k=int(config.get("top_k", 50)),
            )
        except Exception as exc:  # pragma: no cover
            final["sample_error"] = str(exc)
    logger.write({"event": "final", **final})
    print(format_metrics({"event": "final", **{k: v for k, v in final.items() if k != "sample"}}))
    return final
