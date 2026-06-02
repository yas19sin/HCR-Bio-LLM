from __future__ import annotations

from itertools import product

import torch
import torch.nn as nn

from .baseline_transformer import RMSNorm, SelfAttention, TransformerConfig
from .hcr_moments import shifted_legendre_basis


class HCRBlockwiseJointNeuron(nn.Module):
    """Explicit blockwise HCR joint-density neuron.

    Each block models a local joint density over input variables x and output
    variables y:

        rho(x, y) = sum_{i,j} a_{i,j} prod_k f_{i_k}(x_k) prod_l f_{j_l}(y_l)

    Forward propagation substitutes x, normalizes the conditional coefficients
    for rho(y | x), and emits E[y | x]. Reverse propagation uses the same
    coefficients transposed to compute E[x | y].
    """

    def __init__(
        self,
        d_model: int,
        block_size: int = 2,
        degree: int = 2,
        eps: float = 1e-5,
        output_scale: float = 0.1,
        density_state_mix: float = 0.5,
    ) -> None:
        super().__init__()
        if d_model % block_size != 0:
            raise ValueError("d_model must be divisible by block_size")
        if degree < 1:
            raise ValueError("degree must be at least 1 for expected-value propagation")
        if not 0.0 <= density_state_mix <= 1.0:
            raise ValueError("density_state_mix must be in [0, 1]")
        self.d_model = d_model
        self.block_size = block_size
        self.degree = degree
        self.n_basis = degree + 1
        self.n_blocks = d_model // block_size
        self.eps = eps
        self.density_state_mix = density_state_mix

        term_indices = torch.tensor(
            list(product(range(self.n_basis), repeat=block_size)),
            dtype=torch.long,
        )
        self.register_buffer("term_indices", term_indices)
        self.n_terms = int(term_indices.size(0))
        self.zero_term_index = 0
        first_terms = []
        second_terms = []
        for variable in range(block_size):
            pattern = [0] * block_size
            pattern[variable] = 1
            matches = (term_indices == torch.tensor(pattern, dtype=torch.long)).all(dim=1)
            first_terms.append(int(matches.nonzero(as_tuple=False)[0].item()))
            second_pattern = [0] * block_size
            second_pattern[variable] = 2
            second_matches = (term_indices == torch.tensor(second_pattern, dtype=torch.long)).all(dim=1)
            if degree >= 2 and second_matches.any():
                second_terms.append(int(second_matches.nonzero(as_tuple=False)[0].item()))
            else:
                second_terms.append(-1)
        self.register_buffer("first_moment_terms", torch.tensor(first_terms, dtype=torch.long))
        self.register_buffer("second_moment_terms", torch.tensor(second_terms, dtype=torch.long))

        coeffs = torch.zeros(self.n_blocks, self.n_terms, self.n_terms)
        coeffs[:, self.zero_term_index, self.zero_term_index] = 1.0
        coeffs += 0.01 * torch.randn_like(coeffs)
        coeffs[:, self.zero_term_index, self.zero_term_index] = 1.0
        self.coefficients = nn.Parameter(coeffs)
        self.output_scale = nn.Parameter(torch.full((d_model,), output_scale))
        self.output_bias = nn.Parameter(torch.zeros(d_model))
        self.last_conditional_means: torch.Tensor | None = None
        self.last_conditional_variance: torch.Tensor | None = None
        self.last_conditional_coefficients: torch.Tensor | None = None
        self.last_point_density_coefficients: torch.Tensor | None = None
        self.last_input_density_coefficients: torch.Tensor | None = None
        self.last_effective_density_coefficients: torch.Tensor | None = None
        self.last_denominator: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_density(x)
        return y

    def forward_with_density(
        self,
        values: torch.Tensor,
        input_coefficients: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond, _ = self.conditional_coefficients_with_density(
            values,
            input_coefficients=input_coefficients,
            reverse=False,
            return_denominator=True,
        )
        means = self._means_from_conditional_coefficients(cond)
        variance = self._variances_from_conditional_coefficients(cond, means)
        self.last_conditional_means = means.detach()
        self.last_conditional_variance = variance.detach()
        output = (2.0 * means - 1.0) * self.output_scale.view(1, 1, -1) + self.output_bias.view(1, 1, -1)
        return output, cond

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        x = self.conditional_expected_value(y, reverse=True)
        return (2.0 * x - 1.0) * self.output_scale.view(1, 1, -1) + self.output_bias.view(1, 1, -1)

    def conditional_expected_value(self, values: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        means, _ = self.conditional_moments(values, reverse=reverse)
        return means

    def conditional_moments(
        self,
        values: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.conditional_coefficients(values, reverse=reverse)
        means = self._means_from_conditional_coefficients(cond)
        variances = self._variances_from_conditional_coefficients(cond, means)
        return means, variances

    def conditional_coefficients(self, values: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if values.size(-1) != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {values.size(-1)}")
        batch, steps, _ = values.shape
        basis_products = self.point_density_coefficients(values)
        cond, denominator = self.propagate_density_coefficients(
            basis_products,
            reverse=reverse,
            return_denominator=True,
        )
        self.last_point_density_coefficients = basis_products.detach()
        self.last_input_density_coefficients = None
        self.last_effective_density_coefficients = basis_products.detach()
        self.last_conditional_coefficients = cond.detach()
        self.last_denominator = denominator.detach()
        return cond

    def conditional_coefficients_with_density(
        self,
        values: torch.Tensor,
        input_coefficients: torch.Tensor | None = None,
        reverse: bool = False,
        return_denominator: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Condition on a carried density vector blended with the current point value."""

        point_coefficients = self.point_density_coefficients(values)
        effective_coefficients = self.combine_density_coefficients(
            point_coefficients,
            input_coefficients,
        )
        cond, denominator = self.propagate_density_coefficients(
            effective_coefficients,
            reverse=reverse,
            return_denominator=True,
        )
        self.last_point_density_coefficients = point_coefficients.detach()
        self.last_input_density_coefficients = (
            None if input_coefficients is None else input_coefficients.detach()
        )
        self.last_effective_density_coefficients = effective_coefficients.detach()
        self.last_conditional_coefficients = cond.detach()
        self.last_denominator = denominator.detach()
        if return_denominator:
            return cond, denominator
        return cond

    def point_density_coefficients(self, values: torch.Tensor) -> torch.Tensor:
        if values.size(-1) != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {values.size(-1)}")
        batch, steps, _ = values.shape
        normalized = torch.sigmoid(values).view(batch, steps, self.n_blocks, self.block_size)
        return self._product_basis(normalized)

    def combine_density_coefficients(
        self,
        point_coefficients: torch.Tensor,
        input_coefficients: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_coefficients is None:
            return point_coefficients
        if input_coefficients.size(-2) != self.n_blocks or input_coefficients.size(-1) != self.n_terms:
            raise ValueError(
                "expected input coefficients with trailing shape "
                f"({self.n_blocks}, {self.n_terms}), got {tuple(input_coefficients.shape[-2:])}"
            )
        carried = input_coefficients.to(device=point_coefficients.device, dtype=point_coefficients.dtype)
        while carried.ndim < point_coefficients.ndim:
            carried = carried.unsqueeze(0)
        mix = self.density_state_mix
        combined = (1.0 - mix) * point_coefficients + mix * carried
        zero = combined[..., self.zero_term_index : self.zero_term_index + 1]
        return combined / self._safe_denominator(zero)

    def propagate_density_coefficients(
        self,
        input_coefficients: torch.Tensor,
        reverse: bool = False,
        return_denominator: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Propagate an input density/moment vector through the HCR tensor."""

        if input_coefficients.size(-2) != self.n_blocks or input_coefficients.size(-1) != self.n_terms:
            raise ValueError(
                "expected input coefficients with trailing shape "
                f"({self.n_blocks}, {self.n_terms}), got {tuple(input_coefficients.shape[-2:])}"
            )
        coeffs = self.coefficients.transpose(1, 2) if reverse else self.coefficients
        raw_cond = torch.einsum("...ni,nio->...no", input_coefficients, coeffs)
        denominator = raw_cond[..., self.zero_term_index : self.zero_term_index + 1]
        safe_denominator = self._safe_denominator(denominator)
        cond = raw_cond / safe_denominator
        if return_denominator:
            return cond, denominator
        return cond

    def _means_from_conditional_coefficients(self, cond: torch.Tensor) -> torch.Tensor:
        first_coeffs = cond.index_select(dim=-1, index=self.first_moment_terms.to(cond.device))
        means = 0.5 + first_coeffs / torch.sqrt(cond.new_tensor(12.0))
        return means.clamp(0.0, 1.0).reshape(*cond.shape[:-2], self.d_model)

    def _variances_from_conditional_coefficients(
        self,
        cond: torch.Tensor,
        means: torch.Tensor,
    ) -> torch.Tensor:
        first_coeffs = cond.index_select(dim=-1, index=self.first_moment_terms.to(cond.device))
        second_moment = 1.0 / 3.0 + first_coeffs / (2.0 * torch.sqrt(cond.new_tensor(3.0)))
        if self.degree >= 2:
            second_terms = self.second_moment_terms.to(cond.device)
            second_coeffs = cond.new_zeros(*cond.shape[:-1], self.block_size)
            valid = second_terms >= 0
            if valid.any():
                second_coeffs[..., valid] = cond.index_select(dim=-1, index=second_terms[valid])
            second_moment = second_moment + second_coeffs / (6.0 * torch.sqrt(cond.new_tensor(5.0)))
        mean_blocks = means.reshape(*cond.shape[:-2], self.n_blocks, self.block_size)
        variance = (second_moment - mean_blocks.pow(2)).clamp_min(0.0)
        return variance.reshape(*cond.shape[:-2], self.d_model)

    def coefficient_stats(self) -> dict[str, torch.Tensor]:
        nontrivial = self.coefficients[:, 1:, 1:]
        return {
            "hcr_coeff_mean": self.coefficients.mean(),
            "hcr_coeff_std": self.coefficients.std(),
            "hcr_pairwise_energy": nontrivial.pow(2).mean(),
        }

    def _product_basis(self, values: torch.Tensor) -> torch.Tensor:
        basis = shifted_legendre_basis(values, self.degree)
        products_out = values.new_ones(*values.shape[:-1], self.n_terms)
        term_indices = self.term_indices.to(values.device)
        for variable in range(self.block_size):
            products_out = products_out * basis[..., variable, term_indices[:, variable]]
        return products_out

    def _safe_denominator(self, denominator: torch.Tensor) -> torch.Tensor:
        sign = torch.where(
            denominator >= 0.0,
            torch.ones_like(denominator),
            -torch.ones_like(denominator),
        )
        return torch.where(denominator.abs() < self.eps, sign * self.eps, denominator)


class HCRBlockwiseJointFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        block_size: int = 2,
        degree: int = 2,
        dropout: float = 0.1,
        output_scale: float = 0.1,
        density_state_mix: float = 0.5,
    ) -> None:
        super().__init__()
        self.neuron = HCRBlockwiseJointNeuron(
            d_model=d_model,
            block_size=block_size,
            degree=degree,
            output_scale=output_scale,
            density_state_mix=density_state_mix,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def last_conditional_means(self) -> torch.Tensor | None:
        return self.neuron.last_conditional_means

    @property
    def last_conditional_variance(self) -> torch.Tensor | None:
        return self.neuron.last_conditional_variance

    @property
    def last_conditional_coefficients(self) -> torch.Tensor | None:
        return self.neuron.last_conditional_coefficients

    @property
    def last_point_density_coefficients(self) -> torch.Tensor | None:
        return self.neuron.last_point_density_coefficients

    @property
    def last_input_density_coefficients(self) -> torch.Tensor | None:
        return self.neuron.last_input_density_coefficients

    @property
    def last_effective_density_coefficients(self) -> torch.Tensor | None:
        return self.neuron.last_effective_density_coefficients

    @property
    def last_denominator(self) -> torch.Tensor | None:
        return self.neuron.last_denominator

    def coefficient_stats(self) -> dict[str, torch.Tensor]:
        return self.neuron.coefficient_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.neuron(x))

    def forward_with_density(
        self,
        x: torch.Tensor,
        input_coefficients: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, output_coefficients = self.neuron.forward_with_density(
            x,
            input_coefficients=input_coefficients,
        )
        return self.dropout(y), output_coefficients


class HCRBlockwiseJointBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        block_size: int,
        degree: int,
        output_scale: float,
        density_state_mix: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = HCRBlockwiseJointFFN(
            config.d_model,
            block_size=block_size,
            degree=degree,
            dropout=config.dropout,
            output_scale=output_scale,
            density_state_mix=density_state_mix,
        )

    def forward(
        self,
        x: torch.Tensor,
        density_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        delta, next_density_state = self.ffn.forward_with_density(self.norm2(x), density_state)
        x = x + delta
        return x, next_density_state


class HCRBlockwiseJointLM(nn.Module):
    """Transformer shell with paper-direct HCR blockwise joint neurons in FFNs."""

    def __init__(
        self,
        config: TransformerConfig,
        block_size: int = 2,
        degree: int = 2,
        output_scale: float = 0.1,
        density_state_mix: float = 0.5,
    ) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                HCRBlockwiseJointBlock(
                    config,
                    block_size=block_size,
                    degree=degree,
                    output_scale=output_scale,
                    density_state_mix=density_state_mix,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_state: bool = False,
        return_steps: bool = False,
    ) -> dict:
        from ..training.losses import cross_entropy_loss

        _, steps = input_ids.shape
        pos = torch.arange(steps, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.drop(x)
        last_hcr_means = None
        last_hcr_variance = None
        last_hcr_coefficients = None
        last_point_density_coefficients = None
        last_input_density_coefficients = None
        last_effective_density_coefficients = None
        last_hcr_denominator = None
        density_state = None
        density_state_layers = []
        stats: dict[str, torch.Tensor] = {}
        for block in self.blocks:
            x, density_state = block(x, density_state)
            density_state_layers.append(density_state.detach())
            if hasattr(block.ffn, "last_conditional_means"):
                last_hcr_means = block.ffn.last_conditional_means
            if hasattr(block.ffn, "last_conditional_variance"):
                last_hcr_variance = block.ffn.last_conditional_variance
            if hasattr(block.ffn, "last_conditional_coefficients"):
                last_hcr_coefficients = block.ffn.last_conditional_coefficients
            if hasattr(block.ffn, "last_point_density_coefficients"):
                last_point_density_coefficients = block.ffn.last_point_density_coefficients
            if hasattr(block.ffn, "last_input_density_coefficients"):
                last_input_density_coefficients = block.ffn.last_input_density_coefficients
            if hasattr(block.ffn, "last_effective_density_coefficients"):
                last_effective_density_coefficients = block.ffn.last_effective_density_coefficients
            if hasattr(block.ffn, "last_denominator"):
                last_hcr_denominator = block.ffn.last_denominator
            if hasattr(block.ffn, "coefficient_stats"):
                stats = block.ffn.coefficient_stats()
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = cross_entropy_loss(logits, targets, loss_mask)
        if return_state:
            state = {"mu": x}
            if last_hcr_means is not None:
                state["hcr_conditional_mean"] = last_hcr_means
            if last_hcr_variance is not None:
                state["hcr_conditional_variance"] = last_hcr_variance
            if last_hcr_coefficients is not None:
                state["hcr_conditional_coefficients"] = last_hcr_coefficients
                state["hcr_density_coefficients"] = last_hcr_coefficients
            if density_state_layers:
                state["hcr_density_state_layers"] = torch.stack(density_state_layers, dim=0)
            if last_point_density_coefficients is not None:
                state["hcr_point_density_coefficients"] = last_point_density_coefficients
            if last_input_density_coefficients is not None:
                state["hcr_input_density_coefficients"] = last_input_density_coefficients
            if last_effective_density_coefficients is not None:
                state["hcr_effective_density_coefficients"] = last_effective_density_coefficients
            if last_hcr_denominator is not None:
                state["hcr_denominator"] = last_hcr_denominator
            state.update(stats)
            out["state"] = state
        return out
