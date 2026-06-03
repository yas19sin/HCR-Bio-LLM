from __future__ import annotations

from dataclasses import dataclass

import torch

from .hcr_moments import HCRLocalJointDensity


def make_sequence_windows(sequence: torch.Tensor, context_length: int) -> torch.Tensor:
    """Build causal windows [x_t, ..., x_{t+k-1}, x_{t+k}] from a 1D sequence."""

    if context_length < 1:
        raise ValueError("context_length must be positive")
    if sequence.ndim != 1:
        raise ValueError("sequence must have shape [T]")
    if sequence.size(0) <= context_length:
        raise ValueError("sequence is too short for the requested context_length")
    return sequence.unfold(0, context_length + 1, 1).contiguous()


@dataclass
class EmpiricalCDFNormalizer:
    """Empirical CDF normalizer for HCR variables.

    HCR conditioning expects variables normalized to approximately uniform
    values in [0, 1]. This keeps that step explicit instead of hiding it behind
    a sigmoid proxy.
    """

    sorted_values: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "EmpiricalCDFNormalizer":
        flat = values.detach().flatten().to(dtype=torch.float32)
        if flat.numel() == 0:
            raise ValueError("cannot fit an empirical CDF on empty values")
        return cls(torch.sort(flat).values)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        sorted_values = self.sorted_values.to(device=values.device, dtype=values.dtype)
        ranks = torch.searchsorted(sorted_values, values.contiguous(), right=False)
        denom = values.new_tensor(max(sorted_values.numel(), 1))
        return ((ranks.to(dtype=values.dtype) + 0.5) / denom).clamp(0.0, 1.0)

    def inverse(self, normalized: torch.Tensor) -> torch.Tensor:
        sorted_values = self.sorted_values.to(device=normalized.device, dtype=normalized.dtype)
        if sorted_values.numel() == 1:
            return torch.full_like(normalized, sorted_values[0])
        positions = normalized.clamp(0.0, 1.0) * (sorted_values.numel() - 1)
        lower = positions.floor().to(dtype=torch.long)
        upper = positions.ceil().to(dtype=torch.long)
        weight = positions - lower.to(dtype=positions.dtype)
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@dataclass
class ColumnEmpiricalCDFNormalizer:
    """Column-wise empirical CDF normalizer for local HCR windows."""

    normalizers: tuple[EmpiricalCDFNormalizer, ...]

    @classmethod
    def fit(cls, windows: torch.Tensor) -> "ColumnEmpiricalCDFNormalizer":
        if windows.ndim != 2:
            raise ValueError("windows must have shape [N, D]")
        return cls(tuple(EmpiricalCDFNormalizer.fit(windows[:, index]) for index in range(windows.size(1))))

    def transform(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.size(-1) != len(self.normalizers):
            raise ValueError(f"expected last dimension {len(self.normalizers)}")
        columns = [
            normalizer.transform(windows[..., index])
            for index, normalizer in enumerate(self.normalizers)
        ]
        return torch.stack(columns, dim=-1)

    def inverse_column(self, values: torch.Tensor, column_index: int) -> torch.Tensor:
        if not 0 <= column_index < len(self.normalizers):
            raise ValueError("column_index out of range")
        return self.normalizers[column_index].inverse(values)


@dataclass
class HCRSequenceDensityModel:
    """Faithful small HCR sequence-density model.

    This is intentionally not a Transformer. It estimates one local joint
    density over normalized causal windows and uses HCR conditioning to infer
    the next value or any missing value in the same window.
    """

    density: HCRLocalJointDensity
    context_length: int

    @classmethod
    def from_sequence(
        cls,
        sequence: torch.Tensor,
        context_length: int,
        degree: int,
        max_total_degree: int | None = None,
    ) -> "HCRSequenceDensityModel":
        windows = make_sequence_windows(sequence, context_length)
        return cls.from_windows(
            windows,
            context_length=context_length,
            degree=degree,
            max_total_degree=max_total_degree,
        )

    @classmethod
    def from_windows(
        cls,
        windows: torch.Tensor,
        context_length: int,
        degree: int,
        max_total_degree: int | None = None,
    ) -> "HCRSequenceDensityModel":
        expected_width = context_length + 1
        if windows.ndim != 2 or windows.size(-1) != expected_width:
            raise ValueError(f"windows must have shape [N, {expected_width}]")
        if windows.numel() == 0:
            raise ValueError("cannot fit HCR sequence model on empty windows")
        if float(windows.min().item()) < 0.0 or float(windows.max().item()) > 1.0:
            raise ValueError("HCR sequence windows must be normalized to [0, 1]")
        density = HCRLocalJointDensity.from_samples(
            windows,
            degree=degree,
            max_total_degree=max_total_degree,
        )
        return cls(density=density, context_length=context_length)

    @property
    def target_index(self) -> int:
        return self.context_length

    def predict_mean(self, context: torch.Tensor) -> torch.Tensor:
        known = self._known_values(context)
        return self.density.conditional_mean(known, target_index=self.target_index)

    def predict_variance(self, context: torch.Tensor) -> torch.Tensor:
        known = self._known_values(context)
        return self.density.conditional_variance(known, target_index=self.target_index)

    def predict_mode(
        self,
        context: torch.Tensor,
        grid_size: int = 256,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> torch.Tensor:
        known = self._known_values(context)
        return self.density.conditional_mode(
            known,
            target_index=self.target_index,
            grid_size=grid_size,
            calibration_floor=calibration_floor,
            calibration=calibration,
            beta=beta,
        )

    def conditional_log_prob(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> torch.Tensor:
        known = self._known_values(context)
        return self.density.conditional_log_density(
            known,
            target,
            target_index=self.target_index,
            calibration_floor=calibration_floor,
            calibration=calibration,
            beta=beta,
        )

    def sample_next(
        self,
        context: torch.Tensor,
        n_samples: int = 1,
        grid_size: int = 256,
        calibration_floor: float = 1e-4,
        calibration: str = "floor",
        beta: float = 1.0,
    ) -> torch.Tensor:
        known = self._known_values(context)
        return self.density.sample_conditional(
            known,
            target_index=self.target_index,
            n_samples=n_samples,
            grid_size=grid_size,
            calibration_floor=calibration_floor,
            calibration=calibration,
            beta=beta,
        )

    def conditional_mean_known(self, known_values: torch.Tensor, target_index: int) -> torch.Tensor:
        self._validate_known_values(known_values)
        return self.density.conditional_mean(known_values, target_index=target_index)

    def conditional_variance_known(self, known_values: torch.Tensor, target_index: int) -> torch.Tensor:
        self._validate_known_values(known_values)
        return self.density.conditional_variance(known_values, target_index=target_index)

    def _known_values(self, context: torch.Tensor) -> torch.Tensor:
        if context.size(-1) != self.context_length:
            raise ValueError(
                f"expected context last dimension {self.context_length}, got {context.size(-1)}"
            )
        known = context.new_zeros(*context.shape[:-1], self.context_length + 1)
        known[..., : self.context_length] = context
        return known

    def _validate_known_values(self, known_values: torch.Tensor) -> None:
        expected = self.context_length + 1
        if known_values.size(-1) != expected:
            raise ValueError(f"expected known_values last dimension {expected}")
