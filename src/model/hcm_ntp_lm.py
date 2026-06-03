from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hcr_moments import HCRLocalJointDensity, shifted_legendre_basis
from .hcr_sequence import ColumnEmpiricalCDFNormalizer, make_sequence_windows


def token_windows(token_ids: torch.Tensor, context_length: int) -> torch.Tensor:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [T]")
    return make_sequence_windows(token_ids.to(dtype=torch.float32), context_length)


def token_feature_windows(
    token_ids: torch.Tensor,
    token_features: torch.Tensor,
    context_length: int,
) -> torch.Tensor:
    """Build HCM windows from per-token feature vectors."""

    windows = make_sequence_windows(token_ids.to(dtype=torch.long), context_length)
    feature_windows = token_features[windows]
    return feature_windows.reshape(windows.size(0), -1).contiguous()


@dataclass
class HCMLanguageMetrics:
    loss: float
    accuracy: float
    perplexity: float
    ece: float
    brier: float
    evaluated_tokens: int


@dataclass
class HCMNextTokenLanguageModel:
    """Direct HCM/HCR next-token language model.

    This model estimates one local joint density over token windows:

        [token_{t-k}, ..., token_{t-1}, token_t]

    Each window column is normalized with its own empirical CDF. HCR conditional
    density over the target column is converted back to a discrete token
    distribution by multiplying candidate densities by their empirical target
    CDF bin mass, then normalizing over the vocabulary.
    """

    density: HCRLocalJointDensity
    normalizer: ColumnEmpiricalCDFNormalizer
    context_length: int
    vocab_size: int
    candidate_ids: torch.Tensor
    target_bin_mass: torch.Tensor
    allowed_mask: torch.Tensor
    calibration: str = "softplus"
    calibration_floor: float = 1e-6
    beta: float = 1.0
    fallback_context_id: int = 0

    @classmethod
    def fit(
        cls,
        token_ids: torch.Tensor,
        context_length: int,
        vocab_size: int,
        degree: int,
        max_total_degree: int | None = None,
        max_train_windows: int | None = 50000,
        seed: int = 1337,
        disallowed_token_ids: tuple[int, ...] = (),
        prior_smoothing: float = 1e-4,
        calibration: str = "softplus",
        calibration_floor: float = 1e-6,
        beta: float = 1.0,
    ) -> "HCMNextTokenLanguageModel":
        windows = token_windows(token_ids, context_length)
        if max_train_windows is not None and windows.size(0) > max_train_windows:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(windows.size(0), generator=generator)[:max_train_windows]
            windows = windows[indices].contiguous()

        normalizer = ColumnEmpiricalCDFNormalizer.fit(windows)
        normalized_windows = normalizer.transform(windows)
        density = HCRLocalJointDensity.from_samples(
            normalized_windows,
            degree=degree,
            max_total_degree=max_total_degree,
        )

        allowed_mask = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in disallowed_token_ids:
            if 0 <= token_id < vocab_size:
                allowed_mask[token_id] = False

        targets = windows[:, context_length].to(dtype=torch.long)
        counts = torch.bincount(targets.clamp(0, vocab_size - 1), minlength=vocab_size).to(dtype=torch.float32)
        mass = counts + prior_smoothing
        mass = mass * allowed_mask.to(dtype=mass.dtype)
        if mass.sum() <= 0:
            raise ValueError("all candidate tokens are disallowed")
        mass = mass / mass.sum()

        context_counts = torch.bincount(
            windows[:, :context_length].reshape(-1).to(dtype=torch.long).clamp(0, vocab_size - 1),
            minlength=vocab_size,
        )
        fallback_context_id = int(context_counts.argmax().item())
        if not bool(allowed_mask[fallback_context_id]):
            fallback_context_id = int(torch.nonzero(allowed_mask, as_tuple=False)[0].item())

        return cls(
            density=density,
            normalizer=normalizer,
            context_length=context_length,
            vocab_size=vocab_size,
            candidate_ids=torch.arange(vocab_size, dtype=torch.long),
            target_bin_mass=mass,
            allowed_mask=allowed_mask,
            calibration=calibration,
            calibration_floor=calibration_floor,
            beta=beta,
            fallback_context_id=fallback_context_id,
        )

    @property
    def coefficient_count(self) -> int:
        return int(self.density.coefficients.numel())

    @property
    def nonzero_coefficients(self) -> int:
        return int((self.density.coefficients.abs() > 1e-12).sum().item())

    def next_token_probs(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim == 1:
            context_ids = context_ids.unsqueeze(0)
        if context_ids.size(-1) != self.context_length:
            raise ValueError(
                f"expected context last dimension {self.context_length}, got {context_ids.size(-1)}"
            )
        device = context_ids.device
        dtype = torch.float32
        known = torch.zeros(context_ids.size(0), self.context_length + 1, device=device, dtype=dtype)
        for index in range(self.context_length):
            values = context_ids[:, index].to(dtype=dtype)
            known[:, index] = self.normalizer.normalizers[index].transform(values).to(device=device)

        cond = self.density.conditional_coefficients(known, target_index=self.context_length)
        candidate_values = self.candidate_ids.to(device=device, dtype=dtype)
        candidate_values = self.normalizer.normalizers[self.context_length].transform(candidate_values)
        basis = shifted_legendre_basis(candidate_values, self.density.degree)
        raw_density = cond @ basis.transpose(0, 1)
        density = self._positive_density(raw_density)

        mass = self.target_bin_mass.to(device=device, dtype=dtype)
        allowed = self.allowed_mask.to(device=device)
        scores = density * mass.unsqueeze(0)
        scores = scores.masked_fill(~allowed.unsqueeze(0), 0.0)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return scores / denom

    def next_token_logits(self, context_ids: torch.Tensor) -> torch.Tensor:
        return self.next_token_probs(context_ids).clamp_min(1e-12).log()

    def evaluate(
        self,
        token_ids: torch.Tensor,
        max_windows: int | None = None,
        batch_size: int = 512,
    ) -> HCMLanguageMetrics:
        windows = token_windows(token_ids, self.context_length)
        if max_windows is not None:
            windows = windows[:max_windows]
        if windows.numel() == 0:
            raise ValueError("no evaluation windows")
        context = windows[:, : self.context_length].to(dtype=torch.long)
        targets = windows[:, self.context_length].to(dtype=torch.long)

        nll_sum = 0.0
        correct_sum = 0.0
        brier_sum = 0.0
        conf_parts: list[torch.Tensor] = []
        correct_parts: list[torch.Tensor] = []
        for start in range(0, context.size(0), batch_size):
            end = min(start + batch_size, context.size(0))
            probs = self.next_token_probs(context[start:end])
            target = targets[start:end]
            true_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            nll_sum += float((-true_probs.log()).sum().item())
            conf, pred = probs.max(dim=-1)
            correct = (pred == target).to(dtype=torch.float32)
            correct_sum += float(correct.sum().item())
            brier_sum += float((1.0 - true_probs).pow(2).sum().item())
            conf_parts.append(conf.detach().cpu())
            correct_parts.append(correct.detach().cpu())

        total = int(context.size(0))
        loss = nll_sum / max(total, 1)
        return HCMLanguageMetrics(
            loss=loss,
            accuracy=correct_sum / max(total, 1),
            perplexity=math.exp(min(loss, 50.0)),
            ece=_ece(torch.cat(conf_parts), torch.cat(correct_parts)),
            brier=brier_sum / max(total, 1),
            evaluated_tokens=total,
        )

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 1337,
    ) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = self._context_from_prefix(ids)
            probs = self.next_token_probs(torch.tensor(context, dtype=torch.long)).squeeze(0)
            if temperature != 1.0:
                logits = probs.clamp_min(1e-12).log() / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
            if top_k is not None and top_k > 0:
                keep = min(top_k, probs.numel())
                values, indices = torch.topk(probs, keep)
                filtered = torch.zeros_like(probs)
                filtered[indices] = values
                probs = filtered / filtered.sum().clamp_min(1e-12)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            ids.append(next_id)
        return ids

    def _context_from_prefix(self, ids: list[int]) -> list[int]:
        if len(ids) >= self.context_length:
            return ids[-self.context_length :]
        prefix = [self.fallback_context_id] * (self.context_length - len(ids))
        return prefix + ids

    def _positive_density(self, density: torch.Tensor) -> torch.Tensor:
        if self.calibration == "floor":
            return density.clamp_min(self.calibration_floor)
        if self.calibration == "softplus":
            return F.softplus(self.beta * density) / max(self.beta, 1e-12) + self.calibration_floor
        if self.calibration == "exp":
            return torch.exp(self.beta * density).clamp_min(self.calibration_floor)
        raise ValueError(f"unknown calibration: {self.calibration}")


@dataclass
class HCMFeatureNextTokenLanguageModel:
    """Direct HCM/HCR NTP model over explicit token feature variables.

    The one-variable token-rank model is faithful but crude for characters. This
    model keeps the HCM mechanics and changes only the observed variables: each
    context/target token is represented by a small vector of deterministic
    normalized features, then HCR scores candidate target feature vectors from
    the local joint density.
    """

    density: HCRLocalJointDensity
    normalizer: ColumnEmpiricalCDFNormalizer
    active_indices: torch.Tensor
    active_coefficients: torch.Tensor
    context_length: int
    vocab_size: int
    token_features: torch.Tensor
    feature_count: int
    target_bin_mass: torch.Tensor
    allowed_mask: torch.Tensor
    calibration: str = "softplus"
    calibration_floor: float = 1e-6
    beta: float = 1.0
    fallback_context_id: int = 0

    @classmethod
    def fit(
        cls,
        token_ids: torch.Tensor,
        token_features: torch.Tensor,
        context_length: int,
        degree: int,
        max_total_degree: int | None = None,
        max_train_windows: int | None = 50000,
        seed: int = 1337,
        disallowed_token_ids: tuple[int, ...] = (),
        prior_smoothing: float = 1e-4,
        calibration: str = "softplus",
        calibration_floor: float = 1e-6,
        beta: float = 1.0,
    ) -> "HCMFeatureNextTokenLanguageModel":
        if token_features.ndim != 2:
            raise ValueError("token_features must have shape [vocab, features]")
        if float(token_features.min().item()) < 0.0 or float(token_features.max().item()) > 1.0:
            raise ValueError("token_features must be normalized to [0, 1]")
        vocab_size = int(token_features.size(0))
        feature_count = int(token_features.size(1))
        windows = token_windows(token_ids, context_length).to(dtype=torch.long)
        if max_train_windows is not None and windows.size(0) > max_train_windows:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(windows.size(0), generator=generator)[:max_train_windows]
            windows = windows[indices].contiguous()
        feature_windows = token_features[windows].reshape(windows.size(0), -1).contiguous()
        normalizer = ColumnEmpiricalCDFNormalizer.fit(feature_windows)
        normalized_windows = normalizer.transform(feature_windows)
        density = HCRLocalJointDensity.from_samples(
            normalized_windows,
            degree=degree,
            max_total_degree=max_total_degree,
        )

        allowed_mask = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in disallowed_token_ids:
            if 0 <= token_id < vocab_size:
                allowed_mask[token_id] = False

        targets = windows[:, context_length].to(dtype=torch.long)
        counts = torch.bincount(targets.clamp(0, vocab_size - 1), minlength=vocab_size).to(dtype=torch.float32)
        mass = (counts + prior_smoothing) * allowed_mask.to(dtype=torch.float32)
        if mass.sum() <= 0:
            raise ValueError("all candidate tokens are disallowed")
        mass = mass / mass.sum()

        context_counts = torch.bincount(
            windows[:, :context_length].reshape(-1).clamp(0, vocab_size - 1),
            minlength=vocab_size,
        )
        fallback_context_id = int(context_counts.argmax().item())
        if not bool(allowed_mask[fallback_context_id]):
            fallback_context_id = int(torch.nonzero(allowed_mask, as_tuple=False)[0].item())

        active_indices = torch.nonzero(density.coefficients.abs() > 1e-12, as_tuple=False)
        active_coefficients = density.coefficients[
            tuple(active_indices[:, dim] for dim in range(active_indices.size(1)))
        ]

        return cls(
            density=density,
            normalizer=normalizer,
            active_indices=active_indices.to(dtype=torch.long),
            active_coefficients=active_coefficients.to(dtype=torch.float32),
            context_length=context_length,
            vocab_size=vocab_size,
            token_features=token_features.to(dtype=torch.float32),
            feature_count=feature_count,
            target_bin_mass=mass,
            allowed_mask=allowed_mask,
            calibration=calibration,
            calibration_floor=calibration_floor,
            beta=beta,
            fallback_context_id=fallback_context_id,
        )

    @property
    def coefficient_count(self) -> int:
        return int(self.density.coefficients.numel())

    @property
    def nonzero_coefficients(self) -> int:
        return int(self.active_coefficients.numel())

    def next_token_probs(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim == 1:
            context_ids = context_ids.unsqueeze(0)
        if context_ids.size(-1) != self.context_length:
            raise ValueError(
                f"expected context last dimension {self.context_length}, got {context_ids.size(-1)}"
            )
        device = context_ids.device
        token_features = self.token_features.to(device=device, dtype=torch.float32)
        context_features = token_features[context_ids.to(dtype=torch.long)].reshape(
            context_ids.size(0),
            self.context_length * self.feature_count,
        )
        candidate_features = token_features.unsqueeze(0).expand(context_ids.size(0), -1, -1)
        full = torch.cat(
            [
                context_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
                candidate_features,
            ],
            dim=-1,
        )
        flat = full.reshape(-1, full.size(-1))
        normalized = self.normalizer.transform(flat)
        raw_density = self._density_from_active_coefficients(normalized).reshape(
            context_ids.size(0),
            self.vocab_size,
        )
        density = self._positive_density(raw_density)
        mass = self.target_bin_mass.to(device=device, dtype=torch.float32)
        allowed = self.allowed_mask.to(device=device)
        scores = density * mass.unsqueeze(0)
        scores = scores.masked_fill(~allowed.unsqueeze(0), 0.0)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def evaluate(
        self,
        token_ids: torch.Tensor,
        max_windows: int | None = None,
        batch_size: int = 512,
    ) -> HCMLanguageMetrics:
        windows = token_windows(token_ids, self.context_length).to(dtype=torch.long)
        if max_windows is not None:
            windows = windows[:max_windows]
        context = windows[:, : self.context_length]
        targets = windows[:, self.context_length]
        nll_sum = 0.0
        correct_sum = 0.0
        brier_sum = 0.0
        conf_parts: list[torch.Tensor] = []
        correct_parts: list[torch.Tensor] = []
        for start in range(0, context.size(0), batch_size):
            end = min(start + batch_size, context.size(0))
            probs = self.next_token_probs(context[start:end])
            target = targets[start:end]
            true_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            nll_sum += float((-true_probs.log()).sum().item())
            conf, pred = probs.max(dim=-1)
            correct = (pred == target).to(dtype=torch.float32)
            correct_sum += float(correct.sum().item())
            brier_sum += float((1.0 - true_probs).pow(2).sum().item())
            conf_parts.append(conf.detach().cpu())
            correct_parts.append(correct.detach().cpu())
        total = int(context.size(0))
        loss = nll_sum / max(total, 1)
        return HCMLanguageMetrics(
            loss=loss,
            accuracy=correct_sum / max(total, 1),
            perplexity=math.exp(min(loss, 50.0)),
            ece=_ece(torch.cat(conf_parts), torch.cat(correct_parts)),
            brier=brier_sum / max(total, 1),
            evaluated_tokens=total,
        )

    def _positive_density(self, density: torch.Tensor) -> torch.Tensor:
        if self.calibration == "floor":
            return density.clamp_min(self.calibration_floor)
        if self.calibration == "softplus":
            return F.softplus(self.beta * density) / max(self.beta, 1e-12) + self.calibration_floor
        if self.calibration == "exp":
            return torch.exp(self.beta * density).clamp_min(self.calibration_floor)
        raise ValueError(f"unknown calibration: {self.calibration}")

    def _density_from_active_coefficients(self, values: torch.Tensor) -> torch.Tensor:
        if self.active_coefficients.numel() == 0:
            return values.new_zeros(values.shape[:-1])
        indices = self.active_indices.to(device=values.device)
        coeffs = self.active_coefficients.to(device=values.device, dtype=values.dtype)
        basis = shifted_legendre_basis(values, self.density.degree)
        products = values.new_ones(values.shape[0], indices.size(0))
        for variable in range(values.size(-1)):
            products = products * basis[:, variable, indices[:, variable]]
        return products @ coeffs


def _ece(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    if confidence.numel() == 0:
        return 0.0
    ece = confidence.new_tensor(0.0)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece = ece + in_bin.to(dtype=torch.float32).mean() * (
                confidence[in_bin].mean() - correct[in_bin].mean()
            ).abs()
    return float(ece.item())
