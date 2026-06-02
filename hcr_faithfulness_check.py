from __future__ import annotations

import json

import torch

from src.model import build_model
from src.model.hcr_joint_block import HCRBlockwiseJointNeuron
from src.model.hcr_moments import (
    HCRLocalJointDensity,
    hcr_mean_from_coefficients,
    hcr_variance_from_coefficients,
    shifted_legendre_basis,
)


def build_correlated_samples(n_samples: int, noise: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 1, generator=generator)
    y = (0.15 + 0.75 * x + noise * torch.randn(n_samples, 1, generator=generator)).clamp(0.0, 1.0)
    return torch.cat([x, y], dim=-1)


@torch.no_grad()
def check_basis_orthonormality(degree: int = 4, n_grid: int = 8192) -> dict[str, float]:
    grid = (torch.arange(n_grid, dtype=torch.float32) + 0.5) / n_grid
    basis = shifted_legendre_basis(grid, degree)
    gram = basis.transpose(0, 1) @ basis / n_grid
    identity = torch.eye(degree + 1)
    return {"basis_gram_max_abs_error": float((gram - identity).abs().max().item())}


@torch.no_grad()
def check_local_hcr_density() -> dict[str, float | int | list[int]]:
    degree = 4
    samples = build_correlated_samples(n_samples=512, noise=0.03, seed=1337)
    density = HCRLocalJointDensity.from_samples(samples, degree=degree, max_total_degree=4)

    grid_x = torch.linspace(0.0, 1.0, 101).unsqueeze(-1)
    known_for_y = torch.cat([grid_x, torch.zeros_like(grid_x)], dim=-1)
    direct_y_coeffs = density.conditional_coefficients(known_for_y, target_index=1)
    point_x_coeffs = shifted_legendre_basis(grid_x.squeeze(-1), degree)
    propagated_y_coeffs = density.propagate_density_coefficients(point_x_coeffs, target_index=1)

    pred_y = density.conditional_mean(known_for_y, target_index=1)
    propagated_y = hcr_mean_from_coefficients(propagated_y_coeffs)
    var_y = density.conditional_variance(known_for_y, target_index=1)
    propagated_var_y = hcr_variance_from_coefficients(propagated_y_coeffs)
    mode_y = density.conditional_mode(known_for_y, target_index=1)
    true_y = (0.15 + 0.75 * grid_x.squeeze(-1)).clamp(0.0, 1.0)

    grid_y = torch.linspace(0.15, 0.9, 101).unsqueeze(-1)
    known_for_x = torch.cat([torch.zeros_like(grid_y), grid_y], dim=-1)
    pred_x = density.conditional_mean(known_for_x, target_index=0)
    true_x = ((grid_y.squeeze(-1) - 0.15) / 0.75).clamp(0.0, 1.0)

    return {
        "local_coeff_shape": list(density.coefficients.shape),
        "local_forward_y_given_x_mse": float((pred_y - true_y).pow(2).mean().item()),
        "local_reverse_x_given_y_mse": float((pred_x - true_x).pow(2).mean().item()),
        "local_conditional_y_mode_mse": float((mode_y - true_y).pow(2).mean().item()),
        "local_conditional_y_variance_mean": float(var_y.mean().item()),
        "local_propagation_coeff_max_abs_diff": float(
            (propagated_y_coeffs - direct_y_coeffs).abs().max().item()
        ),
        "local_propagation_mean_mse": float((propagated_y - pred_y).pow(2).mean().item()),
        "local_propagation_variance_mse": float((propagated_var_y - var_y).pow(2).mean().item()),
        "local_nonzero_coefficients": int((density.coefficients.abs() > 1e-12).sum().item()),
        "local_marginal_x_coeff_shape": list(density.marginal_coefficients(0).shape),
    }


@torch.no_grad()
def check_blockwise_neuron() -> dict[str, float | list[int]]:
    torch.manual_seed(2026)
    neuron = HCRBlockwiseJointNeuron(d_model=8, block_size=2, degree=2)
    values = torch.randn(2, 3, 8)
    means, variances = neuron.conditional_moments(values)
    direct = neuron.conditional_coefficients(values)
    normalized = torch.sigmoid(values).view(2, 3, neuron.n_blocks, neuron.block_size)
    point_coeffs = neuron._product_basis(normalized)
    propagated = neuron.propagate_density_coefficients(point_coeffs)
    carried_out, carried_coeffs = neuron.forward_with_density(values, input_coefficients=propagated)

    return {
        "blockwise_mean_shape": list(means.shape),
        "blockwise_variance_shape": list(variances.shape),
        "blockwise_conditional_coeff_shape": list(direct.shape),
        "blockwise_carried_output_shape": list(carried_out.shape),
        "blockwise_carried_coeff_shape": list(carried_coeffs.shape),
        "blockwise_propagation_coeff_max_abs_diff": float((propagated - direct).abs().max().item()),
        "blockwise_variance_min": float(variances.min().item()),
    }


@torch.no_grad()
def check_blockwise_lm_state() -> dict[str, bool | list[str] | list[int]]:
    torch.manual_seed(2026)
    config = {
        "model_type": "hcr_blockwise_joint",
        "context_length": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "hcr_block_size": 2,
        "hcr_degree": 2,
        "hcr_density_state_mix": 0.5,
    }
    vocab_size = 32
    model = build_model(config, vocab_size=vocab_size)
    input_ids = torch.randint(0, vocab_size, (2, 8))
    targets = torch.randint(0, vocab_size, (2, 8))
    out = model(input_ids, targets=targets, return_state=True)
    state = out["state"]
    expected_keys = {
        "hcr_conditional_mean",
        "hcr_conditional_variance",
        "hcr_conditional_coefficients",
        "hcr_density_coefficients",
        "hcr_density_state_layers",
        "hcr_effective_density_coefficients",
        "hcr_input_density_coefficients",
        "hcr_denominator",
    }
    return {
        "lm_logits_shape": list(out["logits"].shape),
        "lm_loss_finite": bool(torch.isfinite(out["loss"]).item()),
        "lm_state_keys": sorted(state.keys()),
        "lm_has_hcr_state": expected_keys.issubset(state.keys()),
        "lm_density_coeff_shape": list(state["hcr_density_coefficients"].shape),
        "lm_density_state_layers_shape": list(state["hcr_density_state_layers"].shape),
    }


def assert_thresholds(results: dict[str, object]) -> None:
    thresholds = {
        "basis_gram_max_abs_error": 1e-3,
        "local_forward_y_given_x_mse": 5e-3,
        "local_reverse_x_given_y_mse": 1e-2,
        "local_conditional_y_mode_mse": 1e-2,
        "local_propagation_coeff_max_abs_diff": 1e-5,
        "local_propagation_mean_mse": 1e-10,
        "local_propagation_variance_mse": 1e-10,
        "blockwise_propagation_coeff_max_abs_diff": 1e-7,
    }
    failures: list[str] = []
    for key, max_value in thresholds.items():
        value = float(results[key])
        if value > max_value:
            failures.append(f"{key}={value:.6g} > {max_value:.6g}")
    if float(results["blockwise_variance_min"]) < -1e-7:
        failures.append(f"blockwise_variance_min={results['blockwise_variance_min']}")
    if not bool(results["lm_loss_finite"]):
        failures.append("lm_loss_finite=False")
    if not bool(results["lm_has_hcr_state"]):
        failures.append("lm_has_hcr_state=False")
    if results["blockwise_carried_coeff_shape"] != [2, 3, 4, 9]:
        failures.append(f"blockwise_carried_coeff_shape={results['blockwise_carried_coeff_shape']}")
    if results["lm_density_coeff_shape"] != [2, 8, 8, 9]:
        failures.append(f"lm_density_coeff_shape={results['lm_density_coeff_shape']}")
    if results["lm_density_state_layers_shape"] != [2, 2, 8, 8, 9]:
        failures.append(
            f"lm_density_state_layers_shape={results['lm_density_state_layers_shape']}"
        )
    if failures:
        results["failures"] = failures
        print(json.dumps(results, indent=2, sort_keys=True))
        raise SystemExit(1)


def main() -> None:
    results: dict[str, object] = {}
    results.update(check_basis_orthonormality())
    results.update(check_local_hcr_density())
    results.update(check_blockwise_neuron())
    results.update(check_blockwise_lm_state())
    assert_thresholds(results)
    results["status"] = "pass"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
