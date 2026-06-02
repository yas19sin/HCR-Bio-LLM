# Paper Notes: HCRNN / HCR Joint-Distribution Neurons

Local source: `papers/2405.05097v8.pdf`  
Extracted text: `deep-research/papers/2405.05097v8.txt`

## Core Claim

The HCRNN paper proposes neurons that carry a local model of a joint probability density. The model is written as a linear combination of product-basis functions, and its coefficients are interpretable as mixed moments. This gives one local object that can support forward inference, reverse inference, conditional densities, expected values, uncertainty, and density propagation.

The paper is not mainly proposing a KAN implementation. It argues that a restricted HCR expected-value path can become KAN-like, while HCR itself is broader because it models joint and conditional distributions.

## Normalization

Variables are expected to be transformed into approximately uniform coordinates on `[0, 1]`, using CDF, empirical distribution function, or a parameterized approximation. This is not cosmetic. The shifted Legendre example and coefficient interpretation depend on a stable normalized domain.

Implementation consequence:

- A faithful LM experiment should not just apply `sigmoid(hidden)` and call it normalized. That is a practical proxy, but the paper's normalization is distributional and should be measured or learned.
- For embeddings, we need a design for per-feature or per-block normalization and inverse normalization if values are to be interpreted.

## Density Representation

For normalized variables `x in [0,1]^d`, HCR writes a density as:

```text
rho(x) = sum_j a_j prod_i f_{j_i}(x_i)
```

where `f_0 = 1`, the basis is preferably orthonormal, and `a_0...0 = 1` represents normalization.

Interpretation:

- A single nonzero index gives a marginal moment.
- Two nonzero indexes give pairwise mixed moments.
- Three or more nonzero indexes give higher-order dependencies.
- This hierarchy is why the method is named hierarchical correlation reconstruction.

Implementation consequence:

- `src/model/hcr_moments.py` is aligned with this: it uses shifted Legendre product bases and estimates dense mixed-moment coefficients from normalized samples.
- `hcr_blockwise_joint` is aligned in spirit: it uses blockwise coefficient tensors over input/output product bases.
- `hcr_density` and `hcr_joint_pairwise` are bridge models, not literal HCR coefficient models.

## Conditional Inference

The paper obtains a conditional distribution by substituting known variables into the joint-density expansion and normalizing by the zero-index coefficient for the predicted variable.

For one target variable, the conditional coefficient vector is effectively:

```text
conditional_coeff[target_basis] =
    sum_over_known_basis a[target_basis, known_basis] prod f_known(value)
    / same expression with target_basis = 0
```

For shifted Legendre basis, the expected normalized value only needs the constant and first basis coefficient:

```text
E[x | known] = 1/2 + coeff_1 / sqrt(12)
```

Implementation consequence:

- Our `HCRLocalJointDensity.conditional_coefficients` and `.conditional_mean` follow this mechanism.
- `HCRBlockwiseJointNeuron.conditional_expected_value` also follows the same denominator and first-moment extraction structure.
- The current trainable block path uses a sigmoid proxy for normalization and returns only conditional expected values, not the full conditional density.

## Density Propagation

The paper explicitly generalizes value propagation to density propagation. If the input is already a distribution represented by basis coefficients `b_j`, then concrete basis values `f_j(y)` can be replaced by `b_j`, giving an approximate propagated output density.

Implementation consequence:

- This is a major missing piece in the LM path. We currently propagate hidden values and a few bridge state channels, but `hcr_blockwise_joint` does not propagate density vectors through layers.
- A faithful next step is to let each block consume and emit moment vectors, not just normalized scalar expected values.

## Tensor Decomposition and Sparse Basis

The paper acknowledges the exponential coefficient growth: `(degree + 1)^d` for dense product bases. It suggests restrictions and reductions:

- Pairwise-only dependencies.
- Total-degree restrictions such as `sum_i j_i <= m`.
- Basis optimization with SVD / CCA-like procedures.
- Sparse selection, possibly with L1 penalties.
- Tensor decomposition for intermediate-layer transformations.

Implementation consequence:

- Blockwise tensors are a practical first step but not enough for scale.
- We should add total-degree basis support and sparse masks before increasing block size or degree.
- Low-rank or tensor-factorized coefficient tensors are the route for non-toy dimensions.

## Information Bottleneck / HSIC Direction

The paper connects HCR mixed moments to information measures. For finite orthonormal bases, dependencies can be estimated using sums of squared nontrivial coefficients, avoiding full kernel matrices in HSIC-style objectives.

Implementation consequence:

- Local auxiliary losses can target dependency structure, not just next-token cross entropy.
- Useful metrics include pairwise energy, conditional entropy estimates, coefficient sparsity, and changes under ablation of variables.
- This aligns with adding local losses to keep HCR coefficients from collapsing or becoming arbitrary.

## Calibration and Positivity

Linear density expansions can go negative. The paper discusses calibration functions `phi: R -> R+`, including floors, softplus-like functions, and exponential calibration. It also notes the normalization difficulty in high dimensions.

Implementation consequence:

- Our grid conditional sampling uses a simple clamp/floor and renormalization. That is acceptable as a smoke path, but it is not the calibrated log-likelihood training described in the paper.
- If we train density likelihoods, calibration must be explicit and measured.
- If we only train expected values, we should not claim density-likelihood fidelity.

## Transformers and Density Embeddings

The paper speculates that transformer embeddings could be interpreted as vectors of moments of real-world properties, and that unembedding/softmax can be viewed as a calibrated density overlap when features are mixed moments.

Implementation consequence:

- This is a research hypothesis, not an established implementation.
- A grounded LM architecture should test whether groups of embedding dimensions behave like moments, rather than assuming they do.
- Candidate tests: moment decay by order, density-overlap unembedding, reconstruction of known token attributes, and uncertainty reduction under added context.

## Faithful Takeaways for HCR-Bio-LLM

Required for a strong HCR claim:

1. Explicit product-basis mixed-moment coefficients, dense or sparse.
2. Real normalization strategy for variables/features.
3. Conditional inference with denominator normalization.
4. Reverse inference using the same coefficients.
5. Conditional density or moment-vector propagation, not only point means.
6. Positivity/calibration story if evaluating densities or sampling.
7. Tests that target conditional means, conditional density quality, reverse reconstruction, uncertainty, and coefficient interpretability.

