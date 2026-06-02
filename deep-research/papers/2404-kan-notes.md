# Paper Notes: Kolmogorov-Arnold Networks

Local source: `papers/2404.19756v5.pdf`  
Extracted text: `deep-research/papers/2404.19756v5.txt`

## Core Claim

KANs replace fixed node activations and scalar linear weights with learnable univariate functions on edges. In the paper's main implementation, these edge functions are parameterized as splines.

This is distinct from HCR:

- KAN: learnable edge functions, nodes sum incoming transformed scalar signals.
- HCR: local joint density coefficients, conditioning, density/moment propagation.

HCR can reduce to KAN-like expected-value sums under restricted pairwise assumptions, but KAN does not by itself represent a local joint distribution.

## Architecture Mechanics

A KAN layer has edge functions `phi_{out,in}(x_in)`. Each output node sums the corresponding edge outputs:

```text
x_{l+1,j} = sum_i phi_{l,j,i}(x_{l,i})
```

The implementation details emphasized in the paper include:

- Spline basis parameterization, usually cubic.
- A base residual activation term, often SiLU-like.
- Learnable scale factors for base and spline contributions.
- Adaptive grid updates based on activation distributions.
- Grid extension: train coarse grids first, then refine grids.

Implementation consequence:

- Our current `hcr_kan_mean` is not faithful KAN. It is closer to a FastKAN/RBF-style basis expansion inside an FFN.
- A serious KAN baseline should use a `KANLinear` or `KANLayer` with edge functions and expose grid, basis, and regularization behavior.

## Interpretability and Sparsification

The paper's interpretability story depends on:

- Edge functions being inspectable 1D curves.
- Sparse activation-function usage.
- Pruning unimportant nodes.
- Symbolification: fitting learned curves to known symbolic functions with affine input/output parameters.

Implementation consequence:

- If we add a faithful KAN baseline, we need diagnostic exports for edge functions, pruning masks, and regularization.
- Language-model perplexity alone is not enough to evaluate the reason KAN was proposed.

## Scaling Caution

The paper positions KANs as promising for small-scale AI + science problems where interpretability and accuracy matter and slow training is tolerable. It does not establish that KANs are a plug-in improvement for LLMs.

Implementation consequence:

- KAN-in-GPT repos should be treated as experimental baselines, not as evidence that KAN scales to language modeling.
- HCR-Bio-LLM should keep a clear baseline table: Transformer, FastKAN/RBF-like, efficient spline KAN, HCR bridge states, faithful HCR blockwise joint.

## What to Borrow

Useful KAN ideas for this repo:

- Edge/local function diagnostics.
- Grid or basis adaptivity.
- Sparsification and pruning.
- Low-dimensional interpretable synthetic tasks before LM claims.

What not to borrow blindly:

- Replacing every projection with a heavy KAN layer without memory accounting.
- Calling RBF basis expansion "KAN" without saying it is a variant.
- Treating KAN performance as HCR performance.

