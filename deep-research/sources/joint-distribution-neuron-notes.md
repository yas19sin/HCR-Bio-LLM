# Source Notes: vfd-org/joint-distribution-neuron

Repo: https://github.com/vfd-org/joint-distribution-neuron  
Local path: `deep-research/source-repos/joint-distribution-neuron`  
Commit: `9298614a59d1256e92e496271daf863abba90cc4`

## README Positioning

The repo presents a prototype of joint-distribution neurons and multi-layer HCRNNs. The README explicitly emphasizes:

- Forward inference `X -> Y`.
- Reverse inference `Y -> X`.
- Conditional sampling.
- Uncertainty propagation.
- Reversible regression.
- Multi-layer density transformations.
- Test coverage for basis, conditional inference, joint density, network behavior, pipelines, topology, and torch integration.

## Key Files

- `joint-distribution-neuron/hcrnn/basis.py`
- `joint-distribution-neuron/hcrnn/joint_density.py`
- `joint-distribution-neuron/hcrnn/conditionals.py`
- `joint-distribution-neuron/hcrnn/network.py`
- `joint-distribution-neuron/hcrnn/torch_integration.py`
- `joint-distribution-neuron/tests/*.py`

## Basis Implementation

The basis module implements shifted Legendre-style orthonormal bases and product bases. It also includes total-degree basis support and orthonormality checks.

Implication for this repo:

- `src/model/hcr_moments.py` has full dense product basis support but not total-degree masking.
- Adding total-degree basis support is a near-term fidelity improvement because it directly addresses coefficient explosion while preserving the HCR basis structure.

## Joint Density Implementation

The joint density module provides:

- Basis expansion coefficients.
- Moment matching from samples.
- Density and log-density evaluation.
- Sampling.
- Marginalization.
- Conditioning.
- Conditional expected value and variance.

Implication for this repo:

- Our local utility has density, conditioning, mean, grid conditional density, and sampling.
- Missing relative to upstream: conditional variance, conditional mode, marginalization API, richer log-density/calibration, and tested sampling quality.

## Conditional Helpers

The conditionals module includes grid-based conditional density helpers, expectation, variance, mode, marginal density, and conditional sampling.

Implication for this repo:

- Our `conditional_density_grid` and `sample_conditional` are a start.
- We should add variance and mode because they are cheap from the grid path and directly support uncertainty evaluation.

## Network Implementation

The network module builds multi-layer HCR networks with layer specs. It supports:

- Joint densities over input/output spaces.
- Forward layer inference through conditional expectation.
- Reverse layer inference by conditioning in the opposite direction.
- Uncertainty through conditional variance.
- Resonance or consistency penalties.
- Alternating / coordinate / CMA-like training options.
- Information-bottleneck-style losses.
- Coefficient pruning.

Implication for this repo:

- Our `hcr_blockwise_joint` implements the core conditional-expected-value idea inside a Transformer FFN.
- We do not yet have reverse reconstruction evaluation, density-vector propagation, resonance loss, information-bottleneck local loss, or pruning.

## Highest-Value Borrowing Targets

1. Total-degree basis masks.
2. Conditional variance/mode/sampling in `HCRLocalJointDensity`.
3. Reverse reconstruction tests for `HCRBlockwiseJointNeuron.reverse`.
4. Coefficient pruning and sparsity metrics.
5. Multi-layer consistency/resonance loss for bidirectional paths.
6. Density-vector propagation where inputs and outputs are moment vectors.

