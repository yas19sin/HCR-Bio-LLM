from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from src.data import build_datasets
from src.model.hcm_ntp_lm import (
    HCMFeatureNextTokenLanguageModel,
    HCMNextTokenLanguageModel,
    token_windows,
)
from src.tokenizer import SPECIAL_TOKENS


class BackoffNGramLM:
    def __init__(
        self,
        counts_by_order: list[dict[tuple[int, ...], torch.Tensor]],
        vocab_size: int,
        allowed_mask: torch.Tensor,
        smoothing: float,
    ) -> None:
        self.counts_by_order = counts_by_order
        self.vocab_size = vocab_size
        self.allowed_mask = allowed_mask
        self.smoothing = smoothing

    @classmethod
    def fit(
        cls,
        token_ids: torch.Tensor,
        context_length: int,
        vocab_size: int,
        disallowed_token_ids: tuple[int, ...],
        max_train_windows: int | None,
        seed: int,
        smoothing: float = 1e-3,
    ) -> "BackoffNGramLM":
        windows = token_windows(token_ids, context_length).to(dtype=torch.long)
        if max_train_windows is not None and windows.size(0) > max_train_windows:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(windows.size(0), generator=generator)[
                :max_train_windows]
            windows = windows[indices].contiguous()
        allowed_mask = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in disallowed_token_ids:
            if 0 <= token_id < vocab_size:
                allowed_mask[token_id] = False
        counts_by_order: list[dict[tuple[int, ...], torch.Tensor]] = [
            defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.float32))
            for _ in range(context_length + 1)
        ]
        for row in windows:
            target = int(row[context_length].item())
            for order in range(context_length + 1):
                key = tuple(
                    int(value) for value in row[context_length - order: context_length].tolist())
                counts_by_order[order][key][target] += 1.0
        return cls([dict(counts) for counts in counts_by_order], vocab_size, allowed_mask, smoothing)

    def next_token_probs(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim == 1:
            context_ids = context_ids.unsqueeze(0)
        rows = []
        for row in context_ids.tolist():
            counts = None
            for order in range(min(len(row), len(self.counts_by_order) - 1), -1, -1):
                key = tuple(int(value)
                            for value in row[-order:]) if order > 0 else ()
                counts = self.counts_by_order[order].get(key)
                if counts is not None and counts.sum() > 0:
                    break
            if counts is None:
                counts = torch.zeros(self.vocab_size, dtype=torch.float32)
            scores = counts.clone() + self.smoothing
            scores = scores * self.allowed_mask.to(dtype=scores.dtype)
            rows.append(scores / scores.sum().clamp_min(1e-12))
        return torch.stack(rows, dim=0)


class LogLinearHybridLM:
    def __init__(self, base_model, hcm_model: HCMNextTokenLanguageModel, hcm_weight: float) -> None:
        if not 0.0 <= hcm_weight <= 1.0:
            raise ValueError("hcm_weight must be in [0, 1]")
        self.base_model = base_model
        self.hcm_model = hcm_model
        self.hcm_weight = hcm_weight

    def next_token_probs(self, context_ids: torch.Tensor) -> torch.Tensor:
        base_probs = self.base_model.next_token_probs(
            context_ids).clamp_min(1e-12)
        hcm_probs = self.hcm_model.next_token_probs(
            context_ids).clamp_min(1e-12)
        log_probs = (1.0 - self.hcm_weight) * base_probs.log() + \
            self.hcm_weight * hcm_probs.log()
        probs = torch.softmax(log_probs, dim=-1)
        return probs


def build_char_token_features(itos: list[str]) -> torch.Tensor:
    """Build deterministic normalized character features for feature-HCM."""

    codepoints = []
    for token in itos:
        if token in SPECIAL_TOKENS or len(token) != 1:
            continue
        codepoints.append(ord(token))
    code_min = min(codepoints) if codepoints else 0
    code_max = max(codepoints) if codepoints else 1
    denom = max(code_max - code_min, 1)
    features = []
    for token in itos:
        if token in SPECIAL_TOKENS or len(token) != 1:
            features.append([0.0, 0.0])
            continue
        char = token
        code_feature = (ord(char) - code_min) / denom
        if char == "\n":
            class_feature = 0.10
        elif char.isspace():
            class_feature = 0.20
        elif char.isdigit():
            class_feature = 0.35
        elif char.islower():
            class_feature = 0.55
        elif char.isupper():
            class_feature = 0.65
        elif char in ".,;:!?-_'\"`":
            class_feature = 0.80
        else:
            class_feature = 0.95
        features.append([float(code_feature), float(class_feature)])
    return torch.tensor(features, dtype=torch.float32)


def evaluate_distribution_model(model, token_ids: torch.Tensor, context_length: int, max_windows: int | None, batch_size: int) -> dict[str, float | int]:
    windows = token_windows(token_ids, context_length).to(dtype=torch.long)
    if max_windows is not None:
        windows = windows[:max_windows]
    context = windows[:, :context_length]
    targets = windows[:, context_length]
    nll = 0.0
    correct = 0.0
    brier = 0.0
    for start in range(0, context.size(0), batch_size):
        end = min(start + batch_size, context.size(0))
        probs = model.next_token_probs(context[start:end])
        target = targets[start:end]
        true_probs = probs.gather(-1, target.unsqueeze(-1)
                                  ).squeeze(-1).clamp_min(1e-12)
        nll += float((-true_probs.log()).sum().item())
        correct += float((probs.argmax(dim=-1) == target).sum().item())
        brier += float((1.0 - true_probs).pow(2).sum().item())
    total = int(context.size(0))
    loss = nll / max(total, 1)
    return {
        "loss": loss,
        "accuracy": correct / max(total, 1),
        "perplexity": math.exp(min(loss, 50.0)),
        "brier": brier / max(total, 1),
        "evaluated_tokens": total,
    }


def generate_from_distribution_model(
    model,
    prompt_ids: list[int],
    context_length: int,
    fallback_context_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        if len(ids) >= context_length:
            context = ids[-context_length:]
        else:
            context = [fallback_context_id] * (context_length - len(ids)) + ids
        probs = model.next_token_probs(torch.tensor(
            context, dtype=torch.long)).squeeze(0)
        if temperature != 1.0:
            logits = probs.clamp_min(1e-12).log() / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(probs, min(top_k, probs.numel()))
            filtered = torch.zeros_like(probs)
            filtered[indices] = values
            probs = filtered / filtered.sum().clamp_min(1e-12)
        ids.append(int(torch.multinomial(
            probs, num_samples=1, generator=generator).item()))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct faithful HCM/HCR next-token language model.")
    parser.add_argument("--dataset", default="tiny_shakespeare")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--hf-dataset", default="Trelis/tiny-shakespeare")
    parser.add_argument("--hf-config", default="default")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-max-rows", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--max-total-degree", type=int, default=4)
    parser.add_argument("--max-train-windows", type=int, default=50000)
    parser.add_argument("--eval-windows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prior-smoothing", type=float, default=1e-4)
    parser.add_argument(
        "--calibration", choices=["floor", "softplus", "exp"], default="softplus")
    parser.add_argument("--calibration-floor", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--feature-hcm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-degree", type=int, default=3)
    parser.add_argument("--feature-max-total-degree", type=int, default=3)
    parser.add_argument("--prompt", default="First ")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--hybrid-weights",
        default="0.1,0.25,0.5",
        help="Comma-separated HCM log-linear weights for n-gram+HCM rescoring.",
    )
    parser.add_argument(
        "--sample-model",
        choices=["hcm", "feature_hcm", "ngram", "hybrid", "feature_hybrid"],
        default="feature_hybrid",
    )
    parser.add_argument("--sample-hybrid-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = {
        "dataset": "huggingface" if args.hf_dataset else args.dataset,
        "data_path": args.data_path,
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_max_rows": args.hf_max_rows,
        "context_length": args.context_length,
        "val_fraction": args.val_fraction,
        "allow_fallback_dataset": True,
    }
    train_data, val_data, tokenizer, info = build_datasets(config)
    disallowed = (tokenizer.pad_id, tokenizer.mask_id)

    hcm = HCMNextTokenLanguageModel.fit(
        train_data.data,
        context_length=args.context_length,
        vocab_size=tokenizer.vocab_size,
        degree=args.degree,
        max_total_degree=args.max_total_degree,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
        disallowed_token_ids=disallowed,
        prior_smoothing=args.prior_smoothing,
        calibration=args.calibration,
        calibration_floor=args.calibration_floor,
        beta=args.beta,
    )
    hcm_metrics = hcm.evaluate(
        val_data.data, max_windows=args.eval_windows, batch_size=args.batch_size)

    ngram = BackoffNGramLM.fit(
        train_data.data,
        context_length=args.context_length,
        vocab_size=tokenizer.vocab_size,
        disallowed_token_ids=disallowed,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
    )
    ngram_metrics = evaluate_distribution_model(
        ngram,
        val_data.data,
        context_length=args.context_length,
        max_windows=args.eval_windows,
        batch_size=args.batch_size,
    )
    feature_hcm = None
    feature_hcm_metrics = None
    if args.feature_hcm:
        token_features = build_char_token_features(tokenizer.itos)
        feature_hcm = HCMFeatureNextTokenLanguageModel.fit(
            train_data.data,
            token_features=token_features,
            context_length=args.context_length,
            degree=args.feature_degree,
            max_total_degree=args.feature_max_total_degree,
            max_train_windows=args.max_train_windows,
            seed=args.seed,
            disallowed_token_ids=disallowed,
            prior_smoothing=args.prior_smoothing,
            calibration=args.calibration,
            calibration_floor=args.calibration_floor,
            beta=args.beta,
        )
        feature_hcm_metrics = feature_hcm.evaluate(
            val_data.data,
            max_windows=args.eval_windows,
            batch_size=args.batch_size,
        )
    hybrid_results = {}
    feature_hybrid_results = {}
    for weight in _parse_weights(args.hybrid_weights):
        hybrid = LogLinearHybridLM(ngram, hcm, hcm_weight=weight)
        hybrid_results[f"{weight:.3g}"] = evaluate_distribution_model(
            hybrid,
            val_data.data,
            context_length=args.context_length,
            max_windows=args.eval_windows,
            batch_size=args.batch_size,
        )
        if feature_hcm is not None:
            feature_hybrid = LogLinearHybridLM(
                ngram, feature_hcm, hcm_weight=weight)
            feature_hybrid_results[f"{weight:.3g}"] = evaluate_distribution_model(
                feature_hybrid,
                val_data.data,
                context_length=args.context_length,
                max_windows=args.eval_windows,
                batch_size=args.batch_size,
            )
    best_hybrid_weight = None
    best_hybrid_metrics = None
    if hybrid_results:
        best_key, best_hybrid_metrics = min(
            hybrid_results.items(),
            key=lambda item: float(item[1]["loss"]),
        )
        best_hybrid_weight = float(best_key)
    best_feature_hybrid_weight = None
    best_feature_hybrid_metrics = None
    if feature_hybrid_results:
        best_key, best_feature_hybrid_metrics = min(
            feature_hybrid_results.items(),
            key=lambda item: float(item[1]["loss"]),
        )
        best_feature_hybrid_weight = float(best_key)

    prompt_ids = tokenizer.encode(args.prompt)
    if args.sample_model == "hcm":
        sample_backend = hcm
    elif args.sample_model == "feature_hcm":
        if feature_hcm is None:
            raise SystemExit(
                "--sample-model feature_hcm requires --feature-hcm")
        sample_backend = feature_hcm
    elif args.sample_model == "ngram":
        sample_backend = ngram
    elif args.sample_model == "hybrid":
        sample_backend = LogLinearHybridLM(
            ngram, hcm, hcm_weight=args.sample_hybrid_weight)
    else:
        if feature_hcm is None:
            raise SystemExit(
                "--sample-model feature_hybrid requires --feature-hcm")
        sample_backend = LogLinearHybridLM(
            ngram, feature_hcm, hcm_weight=args.sample_hybrid_weight)
    generated_ids = generate_from_distribution_model(
        sample_backend,
        prompt_ids,
        context_length=args.context_length,
        fallback_context_id=hcm.fallback_context_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    sample = tokenizer.decode(generated_ids)

    result = {
        "status": "ok",
        "model": "hcm_ntp_lm",
        "interpretation": (
            "Direct HCM/HCR local next-token language model. HCM conditional "
            "density directly scores next-token candidates; this is not a Transformer."
        ),
        "dataset": {
            "source": info.source,
            "chars": info.chars,
            "fallback": info.fallback,
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_data.data.numel()),
            "val_tokens": int(val_data.data.numel()),
        },
        "config": {
            "context_length": args.context_length,
            "degree": args.degree,
            "max_total_degree": args.max_total_degree,
            "max_train_windows": args.max_train_windows,
            "eval_windows": args.eval_windows,
            "calibration": args.calibration,
            "prior_smoothing": args.prior_smoothing,
            "feature_hcm": args.feature_hcm,
            "feature_degree": args.feature_degree,
            "feature_max_total_degree": args.feature_max_total_degree,
        },
        "hcm": {
            "coefficient_shape": list(hcm.density.coefficients.shape),
            "coefficient_count": hcm.coefficient_count,
            "nonzero_coefficients": hcm.nonzero_coefficients,
            "loss": hcm_metrics.loss,
            "accuracy": hcm_metrics.accuracy,
            "perplexity": hcm_metrics.perplexity,
            "ece": hcm_metrics.ece,
            "brier": hcm_metrics.brier,
            "evaluated_tokens": hcm_metrics.evaluated_tokens,
        },
        "feature_hcm": _feature_hcm_result(feature_hcm, feature_hcm_metrics),
        "backoff_ngram": ngram_metrics,
        "hybrid_ngram_hcm": hybrid_results,
        "hybrid_ngram_feature_hcm": feature_hybrid_results,
        "best_hybrid_by_loss": {
            "hcm_weight": best_hybrid_weight,
            "metrics": best_hybrid_metrics,
        },
        "best_feature_hybrid_by_loss": {
            "hcm_weight": best_feature_hybrid_weight,
            "metrics": best_feature_hybrid_metrics,
        },
        "sample_model": {
            "type": args.sample_model,
            "hybrid_weight": args.sample_hybrid_weight
            if args.sample_model in {"hybrid", "feature_hybrid"}
            else None,
        },
        "sample": sample,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(result,
                                                            indent=2, sort_keys=True), encoding="utf-8")
        (output_dir / "summary.md").write_text(format_summary_markdown(result), encoding="utf-8")
        (output_dir / "sample.txt").write_text(sample, encoding="utf-8")


def _parse_weights(raw: str) -> list[float]:
    weights = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        weight = float(item)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("hybrid weights must be in [0, 1]")
        weights.append(weight)
    return weights


def _feature_hcm_result(model, metrics):
    if model is None or metrics is None:
        return None
    return {
        "coefficient_shape": list(model.density.coefficients.shape),
        "coefficient_count": model.coefficient_count,
        "feature_count": model.feature_count,
        "nonzero_coefficients": model.nonzero_coefficients,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "perplexity": metrics.perplexity,
        "ece": metrics.ece,
        "brier": metrics.brier,
        "evaluated_tokens": metrics.evaluated_tokens,
    }


def format_summary_markdown(result: dict) -> str:
    lines = [
        "# HCM NTP LM Summary",
        "",
        f"- dataset: `{result['dataset']['source']}`",
        f"- chars: `{result['dataset']['chars']}`",
        f"- vocab_size: `{result['dataset']['vocab_size']}`",
        f"- context_length: `{result['config']['context_length']}`",
        f"- degree: `{result['config']['degree']}`",
        f"- max_total_degree: `{result['config']['max_total_degree']}`",
        f"- feature_hcm: `{result['config']['feature_hcm']}`",
        "",
        "## Metrics",
        "",
        "| model | loss | ppl | acc | brier | ece |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    hcm = result["hcm"]
    lines.append(_metric_row("pure_hcm_id", hcm, include_ece=True))
    feature_hcm = result.get("feature_hcm")
    if feature_hcm is not None:
        lines.append(_metric_row("pure_hcm_feature",
                     feature_hcm, include_ece=True))
    lines.append(_metric_row("backoff_ngram",
                 result["backoff_ngram"], include_ece=False))
    for weight, metrics in result.get("hybrid_ngram_hcm", {}).items():
        lines.append(_metric_row(
            f"ngram+hcm_id w={weight}", metrics, include_ece=False))
    for weight, metrics in result.get("hybrid_ngram_feature_hcm", {}).items():
        lines.append(_metric_row(
            f"ngram+hcm_feature w={weight}", metrics, include_ece=False))
    best = result.get("best_hybrid_by_loss", {})
    feature_best = result.get("best_feature_hybrid_by_loss", {})
    lines.extend(
        [
            "",
            "## Best By Loss",
            "",
            f"- id-HCM hybrid: `w={best.get('hcm_weight')}` loss `{_metric(best.get('metrics'), 'loss')}`",
            f"- feature-HCM hybrid: `w={feature_best.get('hcm_weight')}` loss `{_metric(feature_best.get('metrics'), 'loss')}`",
            "",
            "## Sample",
            "",
            "```text",
            str(result.get("sample", "")),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(label: str, metrics: dict, include_ece: bool) -> str:
    ece = metrics.get("ece") if include_ece else None
    return (
        f"| {label} | {_metric(metrics, 'loss')} | {_metric(metrics, 'perplexity')} | "
        f"{_metric(metrics, 'accuracy')} | {_metric(metrics, 'brier')} | "
        f"{'' if ece is None else f'{float(ece):.4f}'} |"
    )


def _metric(metrics: dict | None, key: str) -> str:
    if metrics is None or key not in metrics:
        return ""
    value = metrics[key]
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
