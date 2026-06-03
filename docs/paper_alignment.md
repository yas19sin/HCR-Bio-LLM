# Paper Alignment Notes

These notes are based on the local PDFs in `papers/`, not on memory or secondary
summaries. For the additional Wolfram/GitHub sources, see
[upstream_grounding.md](upstream_grounding.md).

## Local Sources

| File | Paper | What matters for this repo |
|---|---|---|
| `papers/2405.05097v8.pdf` | Jarek Duda, *Biology-inspired joint distribution neurons based on Hierarchical Correlation Reconstruction for multidirectional propagation of values and densities* | HCR neurons model a local joint density with coefficients that are mixed moments in a product basis. Conditioning substitutes known variables and normalizes. Density propagation replaces concrete basis values with moment vectors. |
| `papers/2404.19756v5.pdf` | Liu et al., *KAN: Kolmogorov-Arnold Networks* | KAN proper uses learnable univariate functions on edges, parametrized as splines, instead of fixed node activations and linear weights. |
| `papers/2605.30370v2.pdf` | Mohedano et al., *Updating the standard neuron model in artificial neural networks* | The proposed neuron is an implicit-bias neural network unit: the activation prevalue depends on a dendritic nonlinear bias term coupled to same-layer activity. This is related motivation, not the same mechanism as HCR joint-density neurons. |

## HCR Formula Grounding

The HCR paper normalizes variables to approximately uniform values in `[0, 1]`
and represents a local joint density using an orthonormal product basis:

```text
rho(x) = sum_j a_j prod_i f_{j_i}(x_i)
```

with `f_0 = 1`. Coefficients with more nonzero indexes represent higher-order
dependencies:

- `a_000... = 1`: normalization
- one nonzero index: marginal moment
- two nonzero indexes: pairwise mixed moment
- three or more nonzero indexes: higher-order dependency

The repo now includes [src/model/hcr_moments.py](../src/model/hcr_moments.py),
which implements:

- shifted Legendre basis functions on `[0, 1]`
- static coefficient estimation from samples
- local joint-density evaluation
- one-variable conditional coefficient extraction
- conditional mean from the first conditional moment
- conditional variance from shifted-Legendre moments
- grid-based conditional mode
- marginal coefficients
- density-vector propagation by replacing concrete basis values with moment vectors
- positivity-floor grid-based conditional sampling
- the HCR small-coefficient mutual-information approximation for 2D coefficients

This module is the most literal implementation of the HCR paper primitive in the
repo.

The repo also includes
[src/model/hcr_sequence.py](../src/model/hcr_sequence.py), a standalone small
sequence-density wrapper around the same primitive. It keeps per-variable
empirical-CDF normalization explicit, estimates product-basis mixed moments over
causal context/target windows, and exposes forward conditioning, reverse
conditioning, conditional variance, conditional log density, mode, and sampling.
This is faithful local HCR/HCM mechanics, not a Transformer language model.

For next-token prediction, the repo now includes
[src/model/hcm_ntp_lm.py](../src/model/hcm_ntp_lm.py). It is a direct local HCM
language model: character-token windows are converted to per-column empirical
CDF variables, one HCR joint density is estimated over context plus target, and
the HCR conditional density directly scores next-token candidates. The CLI in
`hcm_ntp_lm.py` also supports a feature-HCM variant over deterministic
character code/class variables and log-linear HCM rescoring over a discrete
backoff n-gram LM, which is currently the more useful language-model form.

The root-level `hcr_faithfulness_check.py` script is the executable check for
these claims. It verifies shifted-Legendre orthonormality, local HCR
conditioning/reverse inference, density-vector propagation identity, and the
standalone sequence-density nonlinear transition task, a direct HCM NTP token
guardrail, plus the
`hcr_blockwise_joint` carried density-state and conditional coefficient/variance
state surface.

## KAN Accuracy

The `hcr_kan_mean` model is not a faithful KAN implementation. It is a
KAN-inspired radial-basis FFN approximation closer to FastKAN/RBF-KAN, matching
the project brief's
practical first bridge:

```text
basis(x) -> learned combination -> output mean
```

Faithful KAN would put learnable spline functions on edges and avoid ordinary
linear weights as the main transformation. The current model keeps linear
projections for a cheap Transformer-compatible baseline.

## Current LM Approximation Levels

| Component | Paper-faithful? | Current status |
|---|---|---|
| `src/model/hcr_moments.py` | Close for small dense local HCR distributions | Implements product-basis mixed-moment mechanics directly. |
| `src/model/hcr_sequence.py` | Close for small local HCR sequence densities | Uses explicit CDF normalization and local window coefficients for forward/reverse conditioning. |
| `src/model/hcm_ntp_lm.py` | Close for a small local HCR NTP language model | Uses HCR conditional density to score next-token candidates from token CDF variables or deterministic character feature variables; hybrid mode uses HCR as a direct log-linear rescoring factor over backoff n-gram probabilities. |
| `hcr_blockwise_joint` | Close for a trainable expected-value HCR neuron | Uses explicit blockwise product-basis coefficient tensors, substitutes sigmoid-normalized hidden inputs as a practical proxy, normalizes `rho(y | x)`, carries conditional density coefficients between HCR FFNs, exposes conditional coefficient vectors and conditional variance, and feeds expected values on the residual hidden path. Reverse conditioning uses the same tensor transposed, but reverse reconstruction is not yet an LM evaluation path. |
| `hcr_kan_mean` | Partial | KAN-inspired radial-basis mean propagation, not spline edge KAN. |
| `hcr_moment` | Partial | Carries `mu` and `log_var`, matching the value/variance motivation but not full mixed-moment density coefficients. |
| `hcr_density` | Partial | Adds normalized latent density coefficients, but these are not yet HCR `a_j` product-basis coefficients. |
| `hcr_joint_pairwise` | Partial | Adds compressed correlation features, but not explicit blockwise mixed-moment tensors. |
| `hcr_bidirectional_refinement` | Partial | Tests multidirectional refinement in a denoising task; it does not yet condition an explicit HCR joint-density tensor. |
| cortical-cell paper support | Motivation only | No IBNN implicit dendritic-bias layer is implemented yet. |

## Next Corrections for Stronger Paper Fidelity

1. Replace or supplement `hcr_kan_mean` with a true spline-edge KAN FFN, or
   rename it as a FastKAN/RBF-KAN baseline and add LayerNorm/base-update parity.
2. Propagate density states through attention/projection operations, not only
   through the HCR FFN chain.
3. Add reverse reconstruction evaluation for `hcr_blockwise_joint`.
4. Add learned/empirical CDF normalization or another measured normalization
   path instead of relying only on `sigmoid(hidden)`.
5. Add an IBNN-style implicit-bias layer as a separate cortical-neuron baseline,
   not as an HCR component.
