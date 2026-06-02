from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch


def shifted_legendre_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Orthonormal shifted Legendre basis on [0, 1].

    The HCR paper uses this family as a convenient example where coefficients
    can be interpreted as moments of normalized variables.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    x = x.clamp(0.0, 1.0)
    t = 2.0 * x - 1.0
    polys = [torch.ones_like(t)]
    if degree >= 1:
        polys.append(t)
    for n in range(1, degree):
        next_poly = ((2 * n + 1) * t * polys[n] - n * polys[n - 1]) / (n + 1)
        polys.append(next_poly)
    basis = [torch.sqrt(x.new_tensor(2 * n + 1)) * p for n, p in enumerate(polys)]
    return torch.stack(basis, dim=-1)


def estimate_hcr_coefficients(
    samples: torch.Tensor,
    degree: int,
    max_total_degree: int | None = None,
) -> torch.Tensor:
    """Estimate dense HCR mixed-moment coefficients from normalized samples.

    Args:
        samples: Tensor of shape [N, D], expected to be normalized to [0, 1].
        degree: Maximum basis degree per variable.
        max_total_degree: Optional sparse HCR restriction. If set, coefficients
            whose product-basis multi-index has sum larger than this value are
            left at zero.

    Returns:
        Coefficient tensor with shape [degree + 1] * D. Entry (0, ..., 0) is
        one for non-empty samples because f_0 is constant one.
    """

    if samples.ndim != 2:
        raise ValueError("samples must have shape [N, D]")
    n_samples, n_variables = samples.shape
    if n_samples == 0:
        raise ValueError("cannot estimate HCR coefficients from an empty sample")
    basis = shifted_legendre_basis(samples, degree)
    n_basis = degree + 1
    indices = _basis_indices(
        n_variables,
        n_basis,
        samples.device,
        max_total_degree=max_total_degree,
    )
    products = samples.new_ones(n_samples, indices.size(0))
    for variable in range(n_variables):
        products = products * basis[:, variable, indices[:, variable]]
    coefficients = samples.new_zeros([n_basis] * n_variables)
    coefficients[tuple(indices[:, variable] for variable in range(n_variables))] = products.mean(dim=0)
    return coefficients


def hcr_pairwise_mutual_information(coefficients: torch.Tensor) -> torch.Tensor:
    """Small-coefficient HCR approximation of I(X;Y) for a 2D coefficient table."""

    if coefficients.ndim != 2:
        raise ValueError("pairwise mutual information expects a 2D coefficient table")
    return coefficients[1:, 1:].pow(2).sum()


def hcr_mean_from_coefficients(coefficients: torch.Tensor) -> torch.Tensor:
    """Expected value of a normalized 1D HCR density from basis coefficients."""

    mean = torch.full_like(coefficients[..., 0], 0.5)
    if coefficients.size(-1) > 1:
        mean = mean + coefficients[..., 1] / torch.sqrt(coefficients.new_tensor(12.0))
    return mean


def hcr_variance_from_coefficients(coefficients: torch.Tensor) -> torch.Tensor:
    """Variance of a normalized 1D HCR density from shifted-Legendre moments."""

    second_moment = torch.full_like(coefficients[..., 0], 1.0 / 3.0)
    if coefficients.size(-1) > 1:
        second_moment = second_moment + coefficients[..., 1] / (2.0 * torch.sqrt(coefficients.new_tensor(3.0)))
    if coefficients.size(-1) > 2:
        second_moment = second_moment + coefficients[..., 2] / (6.0 * torch.sqrt(coefficients.new_tensor(5.0)))
    mean = hcr_mean_from_coefficients(coefficients)
    return (second_moment - mean.pow(2)).clamp_min(0.0)


@dataclass
class HCRLocalJointDensity:
    """Small dense local HCR joint-density model.

    This class is intentionally for small blocks or synthetic tests. Dense HCR
    tensors grow as (degree + 1) ** n_variables, so language-scale models should
    use blockwise, sparse, or low-rank variants.
    """

    coefficients: torch.Tensor

    @classmethod
    def from_samples(
        cls,
        samples: torch.Tensor,
        degree: int,
        max_total_degree: int | None = None,
    ) -> "HCRLocalJointDensity":
        return cls(estimate_hcr_coefficients(samples, degree, max_total_degree=max_total_degree))

    @property
    def n_variables(self) -> int:
        return self.coefficients.ndim

    @property
    def n_basis(self) -> int:
        return self.coefficients.size(0)

    @property
    def degree(self) -> int:
        return self.n_basis - 1

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate rho(x) for x with shape [..., D]."""

        if x.size(-1) != self.n_variables:
            raise ValueError(f"expected last dimension {self.n_variables}, got {x.size(-1)}")
        basis = shifted_legendre_basis(x, self.degree)
        indices = _basis_indices(self.n_variables, self.n_basis, x.device)
        products = x.new_ones(*x.shape[:-1], indices.size(0))
        for variable in range(self.n_variables):
            products = products * basis[..., variable, indices[:, variable]]
        coeffs = self.coefficients.to(device=x.device, dtype=x.dtype).reshape(-1)
        return products @ coeffs

    def calibrated_density(
        self,
        x: torch.Tensor,
        calibration: str = "floor",
        floor: float = 1e-4,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Evaluate a positive calibrated density proxy at x."""

        return _calibrate_raw_density(self.density(x), calibration=calibration, floor=floor, beta=beta)

    def log_density(
        self,
        x: torch.Tensor,
        calibration: str = "floor",
        floor: float = 1e-4,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Evaluate log of a positive calibrated density proxy at x."""

        return self.calibrated_density(x, calibration=calibration, floor=floor, beta=beta).log()

    def conditional_coefficients(
        self,
        known_values: torch.Tensor,
        target_index: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Return normalized coefficients for rho(x_target | known others).

        `known_values` has shape [..., D]. The value at `target_index` is ignored.
        """

        if not 0 <= target_index < self.n_variables:
            raise ValueError("target_index out of range")
        if known_values.size(-1) != self.n_variables:
            raise ValueError(
                f"expected last dimension {self.n_variables}, got {known_values.size(-1)}"
            )
        basis = shifted_legendre_basis(known_values, self.degree)
        indices = _basis_indices(self.n_variables, self.n_basis, known_values.device)
        coeffs = self.coefficients.to(device=known_values.device, dtype=known_values.dtype).reshape(-1)
        out = []
        for target_basis in range(self.n_basis):
            keep = indices[:, target_index] == target_basis
            kept_indices = indices[keep]
            terms = coeffs[keep]
            products = known_values.new_ones(*known_values.shape[:-1], kept_indices.size(0))
            for variable in range(self.n_variables):
                if variable == target_index:
                    continue
                products = products * basis[..., variable, kept_indices[:, variable]]
            out.append(products @ terms)
        cond = torch.stack(out, dim=-1)
        denom = cond[..., :1]
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        denom = torch.where(denom.abs() < eps, sign * eps, denom)
        return cond / denom

    def conditional_mean(self, known_values: torch.Tensor, target_index: int) -> torch.Tensor:
        """Expected normalized value E[x_target | known others].

        For the shifted Legendre basis, only coefficients 0 and 1 contribute to
        the first raw moment of x on [0, 1].
        """

        cond = self.conditional_coefficients(known_values, target_index)
        mean = torch.full_like(cond[..., 0], 0.5)
        if self.n_basis > 1:
            mean = mean + cond[..., 1] / torch.sqrt(cond.new_tensor(12.0))
        return mean.clamp(0.0, 1.0)

    def conditional_variance(self, known_values: torch.Tensor, target_index: int) -> torch.Tensor:
        """Variance of x_target under rho(x_target | known others)."""

        cond = self.conditional_coefficients(known_values, target_index)
        return hcr_variance_from_coefficients(cond)

    def conditional_density_grid(
        self,
        known_values: torch.Tensor,
        target_index: int,
        grid_size: int = 256,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a positive calibrated 1D conditional density grid."""

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        cond = self.conditional_coefficients(known_values, target_index)
        grid = torch.linspace(
            0.0,
            1.0,
            grid_size,
            dtype=known_values.dtype,
            device=known_values.device,
        )
        basis = shifted_legendre_basis(grid, self.degree)
        density = cond @ basis.transpose(0, 1)
        density = _calibrate_raw_density(
            density,
            calibration=calibration,
            floor=calibration_floor,
            beta=beta,
        )
        density = density / density.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return grid, density

    def conditional_mode(
        self,
        known_values: torch.Tensor,
        target_index: int,
        grid_size: int = 256,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Grid-mode estimate of argmax_x rho(x_target | known others)."""

        grid, density = self.conditional_density_grid(
            known_values,
            target_index,
            grid_size=grid_size,
            calibration_floor=calibration_floor,
            calibration=calibration,
            beta=beta,
        )
        return grid[density.argmax(dim=-1)]

    def marginal_coefficients(self, keep_indices: int | tuple[int, ...]) -> torch.Tensor:
        """Return HCR coefficients for a marginal over selected variables."""

        if isinstance(keep_indices, int):
            keep = (keep_indices,)
        else:
            keep = tuple(keep_indices)
        if len(set(keep)) != len(keep):
            raise ValueError("keep_indices must not contain duplicates")
        if any(index < 0 or index >= self.n_variables for index in keep):
            raise ValueError("keep_indices contains an out-of-range variable")

        sorted_keep = tuple(sorted(keep))
        selector = [slice(None) if variable in sorted_keep else 0 for variable in range(self.n_variables)]
        marginal = self.coefficients[tuple(selector)]
        if keep != sorted_keep and len(keep) > 1:
            order = [sorted_keep.index(index) for index in keep]
            marginal = marginal.permute(order)
        return marginal

    def propagate_density_coefficients(
        self,
        known_coefficients: torch.Tensor | None,
        target_index: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Propagate a known density/moment vector through rho(x_target | known).

        This is the density-vector analogue of value conditioning: concrete
        basis values f_j(x_known) are replaced by coefficients b_j describing a
        distribution over the known variables.
        """

        if not 0 <= target_index < self.n_variables:
            raise ValueError("target_index out of range")

        known_variables = tuple(index for index in range(self.n_variables) if index != target_index)
        n_known = len(known_variables)
        coeffs = self.coefficients
        if n_known == 0:
            raw = coeffs.to(dtype=coeffs.dtype)
        else:
            if known_coefficients is None:
                raise ValueError("known_coefficients is required when conditioning on variables")
            expected_shape = (self.n_basis,) * n_known
            if tuple(known_coefficients.shape[-n_known:]) != expected_shape:
                raise ValueError(
                    f"expected trailing known coefficient shape {expected_shape}, "
                    f"got {tuple(known_coefficients.shape[-n_known:])}"
                )
            known = known_coefficients.to(device=coeffs.device, dtype=coeffs.dtype)
            sum_dims = tuple(range(known.ndim - n_known, known.ndim))
            raw_terms = []
            for target_basis in range(self.n_basis):
                selector = [slice(None)] * self.n_variables
                selector[target_index] = target_basis
                coeff_slice = coeffs[tuple(selector)]
                raw_terms.append((known * coeff_slice).sum(dim=sum_dims))
            raw = torch.stack(raw_terms, dim=-1)

        denom = raw[..., :1]
        return raw / _safe_denominator(denom, eps=eps)

    def sample_conditional(
        self,
        known_values: torch.Tensor,
        target_index: int,
        n_samples: int = 1,
        grid_size: int = 256,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Sample x_target from a calibrated grid approximation of rho(x_target | known)."""

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        grid, probs = self.conditional_density_grid(
            known_values,
            target_index,
            grid_size=grid_size,
            calibration_floor=calibration_floor,
            calibration=calibration,
            beta=beta,
        )
        flat_probs = probs.reshape(-1, grid_size)
        draw = torch.multinomial(flat_probs, num_samples=n_samples, replacement=True)
        samples = grid[draw]
        return samples.reshape(*known_values.shape[:-1], n_samples)


def _basis_indices(
    n_variables: int,
    n_basis: int,
    device: torch.device,
    max_total_degree: int | None = None,
) -> torch.Tensor:
    if max_total_degree is not None and max_total_degree < 0:
        raise ValueError("max_total_degree must be non-negative")
    raw_indices = list(product(range(n_basis), repeat=n_variables))
    if max_total_degree is not None:
        raw_indices = [index for index in raw_indices if sum(index) <= max_total_degree]
    indices = torch.tensor(
        raw_indices,
        dtype=torch.long,
        device=device,
    )
    return indices


def _safe_denominator(denominator: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(denominator >= 0, torch.ones_like(denominator), -torch.ones_like(denominator))
    return torch.where(denominator.abs() < eps, sign * eps, denominator)


def _calibrate_raw_density(
    density: torch.Tensor,
    calibration: str,
    floor: float,
    beta: float,
) -> torch.Tensor:
    if calibration == "floor":
        return density.clamp_min(floor)
    if calibration == "softplus":
        return torch.nn.functional.softplus(beta * density) / max(beta, 1e-12) + floor
    if calibration == "exp":
        return torch.exp(beta * density).clamp_min(floor)
    raise ValueError(f"unknown calibration: {calibration}")
