from __future__ import annotations

import argparse
import json
import math

import torch

from src.model.hcr_sequence import (
    ColumnEmpiricalCDFNormalizer,
    HCRSequenceDensityModel,
    make_sequence_windows,
)


def build_nonlinear_sequence(length: int, noise: float, seed: int) -> torch.Tensor:
    """Generate a bounded nonlinear autoregressive process for HCR diagnostics."""

    if length < 4:
        raise ValueError("length must be at least 4")
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.empty(length, dtype=torch.float32)
    sequence[:2] = torch.rand(2, generator=generator)
    for index in range(2, length):
        driver = torch.rand((), generator=generator) - 0.5
        eps = noise * torch.randn((), generator=generator)
        prevalue = (
            -0.65
            + 1.35 * sequence[index - 1]
            + 0.85 * torch.sin(2.0 * math.pi * sequence[index - 2])
            + 0.55 * driver
            + eps
        )
        sequence[index] = torch.sigmoid(prevalue)
    return sequence


def build_nonlinear_transition_windows(
    n_windows: int,
    context_length: int,
    noise: float,
    seed: int,
) -> torch.Tensor:
    """Generate independent nonlinear causal transition windows."""

    if context_length < 1:
        raise ValueError("context_length must be positive")
    generator = torch.Generator().manual_seed(seed)
    context = torch.rand(n_windows, context_length, generator=generator)
    signal = 0.5 + 0.34 * torch.sin(2.0 * math.pi * context[:, 0])
    if context_length > 1:
        signal = signal + 0.18 * torch.cos(2.0 * math.pi * context[:, 1])
        signal = signal + 0.10 * (context[:, 0] - 0.5) * (context[:, -1] - 0.5)
    if context_length > 2:
        signal = signal + 0.08 * torch.sin(2.0 * math.pi * context[:, 2:].mean(dim=-1))
    target = (signal + noise * torch.randn(n_windows, generator=generator)).clamp(0.0, 1.0)
    return torch.cat([context, target.unsqueeze(-1)], dim=-1)


def fit_linear(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    design = torch.cat([torch.ones(context.size(0), 1), context], dim=-1)
    return torch.linalg.lstsq(design, target.unsqueeze(-1)).solution.squeeze(-1)


def predict_linear(context: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    design = torch.cat([torch.ones(context.size(0), 1), context], dim=-1)
    return (design @ weights).clamp(0.0, 1.0)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).pow(2).mean().item())


def summarize_predictions(
    raw_context: torch.Tensor,
    raw_target: torch.Tensor,
    raw_hcr_mean: torch.Tensor,
    raw_hcr_mode: torch.Tensor,
    raw_linear: torch.Tensor,
    raw_persistence: torch.Tensor,
    limit: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(min(limit, raw_target.size(0))):
        rows.append(
            {
                "context": [round(float(value), 4) for value in raw_context[index].tolist()],
                "target": round(float(raw_target[index].item()), 4),
                "hcr_mean": round(float(raw_hcr_mean[index].item()), 4),
                "hcr_mode": round(float(raw_hcr_mode[index].item()), 4),
                "linear": round(float(raw_linear[index].item()), 4),
                "persistence": round(float(raw_persistence[index].item()), 4),
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithful small HCM/HCR sequence-density prototype."
    )
    parser.add_argument("--source", choices=["transition", "autoregressive"], default="transition")
    parser.add_argument("--length", type=int, default=6000)
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--max-total-degree", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--normalization", choices=["empirical", "identity"], default="empirical")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--calibration", choices=["floor", "softplus", "exp"], default="softplus")
    parser.add_argument("--calibration-floor", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must be in (0, 1)")

    if args.source == "transition":
        raw_windows = build_nonlinear_transition_windows(
            args.length,
            context_length=args.context_length,
            noise=args.noise,
            seed=args.seed,
        )
    else:
        raw_sequence = build_nonlinear_sequence(args.length, args.noise, args.seed)
        raw_windows = make_sequence_windows(raw_sequence, args.context_length)
    split = int(raw_windows.size(0) * args.train_fraction)
    if split < 8 or raw_windows.size(0) - split < 8:
        raise ValueError("train/test split is too small")

    if args.normalization == "empirical":
        normalizer = ColumnEmpiricalCDFNormalizer.fit(raw_windows[:split])
        windows = normalizer.transform(raw_windows)
    else:
        normalizer = None
        windows = raw_windows
    train_windows = windows[:split]
    test_windows = windows[split:]
    test_raw_windows = raw_windows[split:]

    model = HCRSequenceDensityModel.from_windows(
        train_windows,
        context_length=args.context_length,
        degree=args.degree,
        max_total_degree=args.max_total_degree,
    )

    train_context = train_windows[:, : args.context_length]
    train_target = train_windows[:, args.context_length]
    test_context = test_windows[:, : args.context_length]
    test_target = test_windows[:, args.context_length]

    hcr_mean = model.predict_mean(test_context)
    hcr_mode = model.predict_mode(
        test_context,
        grid_size=args.grid_size,
        calibration_floor=args.calibration_floor,
        calibration=args.calibration,
        beta=args.beta,
    )
    hcr_var = model.predict_variance(test_context)
    log_prob = model.conditional_log_prob(
        test_context,
        test_target,
        calibration_floor=args.calibration_floor,
        calibration=args.calibration,
        beta=args.beta,
    )

    train_mean = torch.full_like(test_target, train_target.mean())
    persistence = test_context[:, -1]
    linear_weights = fit_linear(train_context, train_target)
    linear = predict_linear(test_context, linear_weights)

    reverse_target_index = max(args.context_length - 1, 0)
    hcr_reverse = model.conditional_mean_known(test_windows, target_index=reverse_target_index)
    reverse_known_train = torch.cat(
        [
            train_windows[:, :reverse_target_index],
            train_windows[:, reverse_target_index + 1 :],
        ],
        dim=-1,
    )
    reverse_known_test = torch.cat(
        [
            test_windows[:, :reverse_target_index],
            test_windows[:, reverse_target_index + 1 :],
        ],
        dim=-1,
    )
    reverse_target_train = train_windows[:, reverse_target_index]
    reverse_target_test = test_windows[:, reverse_target_index]
    reverse_linear_weights = fit_linear(reverse_known_train, reverse_target_train)
    reverse_linear = predict_linear(reverse_known_test, reverse_linear_weights)

    if normalizer is not None:
        raw_hcr_mean = normalizer.inverse_column(hcr_mean, args.context_length)
        raw_hcr_mode = normalizer.inverse_column(hcr_mode, args.context_length)
        raw_linear = normalizer.inverse_column(linear, args.context_length)
        raw_train_mean = normalizer.inverse_column(train_mean, args.context_length)
        raw_hcr_reverse = normalizer.inverse_column(hcr_reverse, reverse_target_index)
        raw_reverse_linear = normalizer.inverse_column(reverse_linear, reverse_target_index)
    else:
        raw_hcr_mean = hcr_mean
        raw_hcr_mode = hcr_mode
        raw_linear = linear
        raw_train_mean = train_mean
        raw_hcr_reverse = hcr_reverse
        raw_reverse_linear = reverse_linear
    raw_persistence = test_raw_windows[:, args.context_length - 1]

    raw_target = test_raw_windows[:, args.context_length]
    raw_reverse_target = test_raw_windows[:, reverse_target_index]

    result = {
        "status": "ok",
        "interpretation": (
            "Standalone faithful local HCM/HCR sequence-density prototype; "
            "not a Transformer and not a language-model success claim."
        ),
        "length": args.length,
        "source": args.source,
        "context_length": args.context_length,
        "degree": args.degree,
        "max_total_degree": args.max_total_degree,
        "noise": args.noise,
        "seed": args.seed,
        "normalization": args.normalization,
        "train_windows": int(train_windows.size(0)),
        "test_windows": int(test_windows.size(0)),
        "coefficient_shape": list(model.density.coefficients.shape),
        "coefficient_count": int(model.density.coefficients.numel()),
        "nonzero_coefficients": int((model.density.coefficients.abs() > 1e-12).sum().item()),
        "forward_norm_mse": {
            "hcr_mean": mse(hcr_mean, test_target),
            "hcr_mode": mse(hcr_mode, test_target),
            "linear": mse(linear, test_target),
            "persistence": mse(persistence, test_target),
            "train_mean": mse(train_mean, test_target),
        },
        "forward_raw_mse": {
            "hcr_mean": mse(raw_hcr_mean, raw_target),
            "hcr_mode": mse(raw_hcr_mode, raw_target),
            "linear": mse(raw_linear, raw_target),
            "persistence": mse(raw_persistence, raw_target),
            "train_mean": mse(raw_train_mean, raw_target),
        },
        "reverse_norm_mse": {
            "hcr_same_density": mse(hcr_reverse, reverse_target_test),
            "linear": mse(reverse_linear, reverse_target_test),
        },
        "reverse_raw_mse": {
            "hcr_same_density": mse(raw_hcr_reverse, raw_reverse_target),
            "linear": mse(raw_reverse_linear, raw_reverse_target),
        },
        "conditional_variance_mean": float(hcr_var.mean().item()),
        "conditional_log_prob_mean": float(log_prob.mean().item()),
        "conditional_nll_mean": float((-log_prob).mean().item()),
        "sample_next_norm_mean": float(_sample_next_mean(model, test_context[:16], args.seed)),
        "examples": summarize_predictions(
            test_raw_windows[:, : args.context_length],
            raw_target,
            raw_hcr_mean,
            raw_hcr_mode,
            raw_linear,
            raw_persistence,
        ),
    }
    if normalizer is not None:
        result["normalizer_train_min_by_column"] = [
            float(column.sorted_values.min().item()) for column in normalizer.normalizers
        ]
        result["normalizer_train_max_by_column"] = [
            float(column.sorted_values.max().item()) for column in normalizer.normalizers
        ]

    print(json.dumps(result, indent=2, sort_keys=True))


def _sample_next_mean(
    model: HCRSequenceDensityModel,
    context: torch.Tensor,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    return model.sample_next(context, n_samples=16).mean().item()


if __name__ == "__main__":
    main()
