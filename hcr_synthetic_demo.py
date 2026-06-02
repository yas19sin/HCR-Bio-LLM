from __future__ import annotations

import argparse
import json

import torch

from src.model.hcr_moments import (
    HCRLocalJointDensity,
    hcr_mean_from_coefficients,
    hcr_pairwise_mutual_information,
    hcr_variance_from_coefficients,
    shifted_legendre_basis,
)


def build_correlated_samples(n_samples: int, noise: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 1, generator=generator)
    y = (0.15 + 0.75 * x + noise * torch.randn(n_samples, 1, generator=generator)).clamp(0.0, 1.0)
    return torch.cat([x, y], dim=-1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-direct HCR conditional-density demo.")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--max-total-degree", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    samples = build_correlated_samples(args.samples, args.noise, args.seed)
    model = HCRLocalJointDensity.from_samples(
        samples,
        degree=args.degree,
        max_total_degree=args.max_total_degree,
    )

    grid_x = torch.linspace(0.0, 1.0, 101).unsqueeze(-1)
    known_for_y = torch.cat([grid_x, torch.zeros_like(grid_x)], dim=-1)
    pred_y = model.conditional_mean(known_for_y, target_index=1)
    var_y = model.conditional_variance(known_for_y, target_index=1)
    mode_y = model.conditional_mode(known_for_y, target_index=1)
    true_y = (0.15 + 0.75 * grid_x.squeeze(-1)).clamp(0.0, 1.0)
    forward_mse = (pred_y - true_y).pow(2).mean()
    mode_mse = (mode_y - true_y).pow(2).mean()

    point_x_coeffs = shifted_legendre_basis(grid_x.squeeze(-1), args.degree)
    propagated_y_coeffs = model.propagate_density_coefficients(point_x_coeffs, target_index=1)
    direct_y_coeffs = model.conditional_coefficients(known_for_y, target_index=1)
    propagation_coeff_max_abs_diff = (propagated_y_coeffs - direct_y_coeffs).abs().max()
    propagation_mean_mse = (hcr_mean_from_coefficients(propagated_y_coeffs) - pred_y).pow(2).mean()
    propagation_variance_mse = (
        hcr_variance_from_coefficients(propagated_y_coeffs) - var_y
    ).pow(2).mean()

    grid_y = torch.linspace(0.15, 0.9, 101).unsqueeze(-1)
    known_for_x = torch.cat([torch.zeros_like(grid_y), grid_y], dim=-1)
    pred_x = model.conditional_mean(known_for_x, target_index=0)
    true_x = ((grid_y.squeeze(-1) - 0.15) / 0.75).clamp(0.0, 1.0)
    reverse_mse = (pred_x - true_x).pow(2).mean()
    samples_y = model.sample_conditional(known_for_y[::20], target_index=1, n_samples=32)
    marginal_x = model.marginal_coefficients(0)

    result = {
        "samples": args.samples,
        "degree": args.degree,
        "max_total_degree": args.max_total_degree,
        "noise": args.noise,
        "forward_y_given_x_mse": float(forward_mse.item()),
        "reverse_x_given_y_mse": float(reverse_mse.item()),
        "conditional_y_variance_mean": float(var_y.mean().item()),
        "conditional_y_mode_mse": float(mode_mse.item()),
        "propagation_coeff_max_abs_diff": float(propagation_coeff_max_abs_diff.item()),
        "propagation_mean_mse": float(propagation_mean_mse.item()),
        "propagation_variance_mse": float(propagation_variance_mse.item()),
        "pairwise_mi_approx": float(hcr_pairwise_mutual_information(model.coefficients).item()),
        "conditional_sample_mean": float(samples_y.mean().item()),
        "conditional_sample_std": float(samples_y.std().item()),
        "coeff_shape": list(model.coefficients.shape),
        "nonzero_coefficients": int((model.coefficients.abs() > 1e-12).sum().item()),
        "marginal_x_coeff_shape": list(marginal_x.shape),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
