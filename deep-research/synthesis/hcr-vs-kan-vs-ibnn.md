# HCR vs KAN vs IBNN

## One-Line Distinction

- HCR: learns or estimates local joint densities and uses conditioning to propagate values, densities, uncertainty, and reverse inference.
- KAN: replaces scalar weights with learnable univariate edge functions, usually splines.
- IBNN: replaces standard neuron computation with implicit same-layer coupling inspired by cortical-cell nonlinearities.

## HCR

Primary source: `papers/2405.05097v8.pdf`

Core object:

```text
rho(x_1, ..., x_d) = sum_j a_j prod_i f_{j_i}(x_i)
```

The coefficients `a_j` are mixed moments when the basis is orthonormal and variables are normalized to `[0, 1]`.

What makes HCR distinctive:

- Local joint-density model, not just a nonlinear activation.
- Conditional inference by substituting known variables.
- Same coefficient tensor can support forward and reverse direction.
- Expected value, conditional density, variance, sampling, and density-vector propagation all come from one density representation.
- Coefficients can be interpreted hierarchically: marginal, pairwise, triplewise, etc.

Failure mode for our repo:

- A hidden state with mean/variance channels is HCR-inspired but not HCR unless tied to product-basis mixed moments and conditioning.

## KAN

Primary source: `papers/2404.19756v5.pdf`

Core object:

```text
x_{l+1,j} = sum_i phi_{j,i}(x_{l,i})
```

Each edge function `phi_{j,i}` is learned, typically as a spline plus a base residual function.

What makes KAN distinctive:

- Learnable functions on edges.
- Nodes mostly sum incoming transformed values.
- Interpretability through plotting, sparsification, pruning, and symbolic snapping.
- Grid extension and basis adaptation are central to the paper's accuracy story.

Failure mode for our repo:

- An RBF basis inside a standard FFN is not faithful KAN unless explicitly framed as FastKAN-like.
- KAN does not imply joint-density modeling.

## IBNN

Primary source: `papers/2605.30370v2.pdf`

Core object:

- A layer output is defined implicitly through coupled neuron states.
- The layer requires fixed-point or optimization-based computation.

What makes IBNN distinctive:

- Same-layer nonlinear coupling.
- Biological motivation from cortical cells and dendritic nonlinearities.
- Robustness, sample efficiency, expressivity, and memorization claims.

Failure mode for our repo:

- IBNN is not a density model and not an edge-function model.
- It should not be used as support for HCR or KAN claims except at the high-level "richer neuron models can matter" motivation layer.

## Where They Touch

HCR and KAN:

- HCR conditional expected-value propagation restricted to pairwise dependencies can look like a KAN-style sum of univariate functions.
- KAN's spline basis could be used as an HCR basis, but that alone does not add joint-density conditioning.

KAN and IBNN:

- Both replace parts of standard neuron computation.
- KAN changes the edge transformation; IBNN changes the neuron/layer equation.

HCR and IBNN:

- Both are biologically or statistically motivated alternatives to standard point neurons.
- HCR is explicit probabilistic density modeling; IBNN is implicit coupled computation.

## Correct Language for Reports

Use:

- "HCR-inspired bridge" for `hcr_moment`, `hcr_density`, and `hcr_joint_pairwise`.
- "paper-direct HCR small-block utility" for `src/model/hcr_moments.py`.
- "paper-direct trainable blockwise HCR expected-value path" for `hcr_blockwise_joint`.
- "FastKAN/RBF-like baseline" for `hcr_kan_mean`.
- "efficient spline KAN baseline" only after adding an efficient-kan-style layer.

Avoid:

- "HCR KAN" as a single mechanism.
- "faithful KAN" for RBF-only layers.
- "density propagation" unless moment vectors or density coefficients are actually propagated.
- "bidirectional HCR" unless reverse inference is exposed and evaluated.

