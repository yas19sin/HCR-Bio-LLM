# Implementation Gap Audit

This audit compares the source-grounded HCR/KAN requirements with the current HCR-Bio-LLM code.

## Current Repo Artifacts

Relevant implementation files:

- `src/model/hcr_moments.py`
- `src/model/hcr_joint_block.py`
- `src/model/hcr_basis.py`
- `src/model/hcr_neuron.py`
- `src/model/hcr_lm.py`
- `src/model/hcr_refinement.py`
- `docs/paper_alignment.md`
- `docs/upstream_grounding.md`
- `results/final_report.md`

## What Is Faithful Enough

### `src/model/hcr_moments.py`

Status: close for small static HCR local densities.

Implemented:

- Shifted Legendre basis on `[0, 1]`.
- Dense product-basis coefficient estimation from samples, with optional
  total-degree restriction.
- Joint density evaluation.
- Positive calibrated density and log-density proxies.
- Conditional coefficient extraction.
- Conditional expected value from first shifted-Legendre moment.
- Conditional variance from shifted-Legendre moments.
- Grid-based conditional mode.
- Marginal coefficient extraction.
- Density-vector propagation by replacing concrete basis values with moment
  vectors.
- Grid conditional density with positivity floor.
- Conditional sampling by grid probabilities.
- Pairwise mutual-information approximation via nontrivial coefficient energy.

Gaps:

- No calibrated log-likelihood training.
- One executable faithfulness check covers basis orthonormality, conditional
  propagation identities, reverse synthetic inference, and blockwise state
  exposure; broader density-recovery tests against analytic distributions are
  still missing.

### `src/model/hcr_joint_block.py`

Status: close for a trainable blockwise HCR expected-value neuron.

Implemented:

- Explicit coefficient tensor per hidden block.
- Product-basis terms for input and output block variables.
- Conditional coefficient extraction by substituting sigmoid-normalized hidden values as a practical normalization proxy.
- Conditional expected-value propagation.
- Conditional variance diagnostics.
- Density-vector propagation method over block coefficient vectors.
- Density coefficient state carried between HCR FFNs with configurable blending
  against the current point hidden state.
- Denominator normalization.
- Reverse method using coefficient tensor transpose.
- Coefficient statistics.

Gaps:

- Uses `sigmoid(hidden)` as normalization proxy, not learned/empirical CDF normalization.
- Carries conditional coefficient vectors between HCR FFNs, but attention and
  projections still operate on point hidden values rather than density states.
- Reverse method is available but not integrated into training/evaluation.
- No conditional entropy, mode, sampling, or full density-state propagation
  through attention/projection operations.
- No sparsity or total-degree restriction.
- No coefficient positivity/calibration path.
- No local HCR loss beyond downstream language modeling.

## Bridge Models

### `hcr_moment`

Status: HCR-inspired bridge.

Implemented:

- Carries `mu` and `log_var`.
- Exposes variance statistics.

Gap:

- Mean/variance are not product-basis mixed moments. This model tests distributional state, not HCR density mechanics.

### `hcr_density`

Status: HCR-inspired bridge.

Implemented:

- Adds a normalized latent basis vector.
- Basis entropy can be tracked.

Gap:

- The basis vector is not HCR `a_j` product-basis coefficients.
- No conditioning formula.

### `hcr_joint_pairwise`

Status: HCR-inspired bridge.

Implemented:

- Adds compressed pairwise/correlation channels.

Gap:

- Not a dense or sparse HCR coefficient tensor.
- No conditional density semantics.

### `hcr_bidirectional_refinement`

Status: useful denoising/refinement experiment.

Implemented:

- Non-causal masked denoising.
- Iterative refinement and per-step loss support.

Gap:

- Bidirectional refinement is not the same as HCR reverse conditioning unless tied to the same joint-density coefficients.

## KAN Path

### `hcr_kan_mean`

Status: FastKAN/RBF-like baseline, not faithful KAN.

Implemented:

- Radial basis functions over hidden activations.
- Learned mixture into expected-value-like FFN output.

Gaps:

- No B-spline edge functions.
- No per-edge KANLayer behavior.
- No adaptive grid, pruning, symbolification, or edge-curve diagnostics.
- Missing FastKAN details such as LayerNorm and base update.

## Fidelity Checklist

| Requirement | Current support | Verdict |
| --- | --- | --- |
| Product-basis mixed moments | `hcr_moments`, `hcr_blockwise_joint` | partial but real |
| CDF/EDF normalization | none; sigmoid proxy in blockwise path | missing |
| Conditional expected value | `hcr_moments`, `hcr_blockwise_joint` | present |
| Conditional density | `hcr_moments` grid only | partial |
| Conditional variance/mode | variance in `hcr_moments` and `hcr_blockwise_joint`; mode in `hcr_moments` grid path | partial |
| Conditional sampling | `hcr_moments` grid only | partial |
| Reverse conditioning | block neuron method exists | not evaluated |
| Density-vector propagation | `hcr_moments`; `hcr_blockwise_joint` methods plus carried HCR FFN density state; attention remains point-value only | partial |
| Calibration/positivity | grid floor only | weak |
| Total-degree / sparse basis | optional total-degree restriction in `hcr_moments`; none in blockwise LM | partial |
| Coefficient pruning | none | missing |
| HCR local losses / IB | none | missing |
| Faithful KAN baseline | none | missing |
| FastKAN-like baseline | `hcr_kan_mean` | partial |
| KAT-style grouped rational KAN | none | future |
| IBNN baseline | none | future |

## Highest-Risk Overclaims

1. Calling all HCR variants faithful. Only `hcr_moments` and `hcr_blockwise_joint` are close to paper mechanics.
2. Treating `hcr_kan_mean` as KAN. It is RBF/FastKAN-like.
3. Treating denoising refinement as HCR reverse inference. It is not unless tied to the same joint-density model.
4. Claiming layer-to-layer density propagation in the LM before moment vectors are carried as state.
5. Claiming uncertainty propagation from `log_var` alone without conditional variance from HCR coefficients.
