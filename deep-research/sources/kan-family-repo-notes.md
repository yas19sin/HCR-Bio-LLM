# Source Notes: KAN Repo Family

This file covers the requested KAN-related repos and separates the implementation families.

## pykan

Repo: https://github.com/kindxiaoming/pykan  
Local path: `deep-research/source-repos/pykan`  
Commit: `ecde4ec3274d3bef1ad737479cf126aed38ab530`

Key files:

- `kan/KANLayer.py`
- `kan/spline.py`
- `kan/MultKAN.py`
- `kan/Symbolic_KANLayer.py`

Observed mechanics:

- `KANLayer` stores spline coefficients per edge.
- It includes base residual activation, spline scale, base scale, masks, and grid update.
- Forward paths expose preactivations, postactivations, and postspline values for diagnostics.
- The broader package supports pruning, symbolic layers, plotting, and interpretability workflows.

Implication:

- This is the reference for a faithful KAN baseline.
- It is not the best direct drop-in for LM scale because per-edge spline bookkeeping is heavy.

## efficient-kan

Repo: https://github.com/Blealtan/efficient-kan  
Local path: `deep-research/source-repos/efficient-kan`  
Commit: `7b6ce1c87f18c8bc90c208f6b494042344216b11`

Key file:

- `src/efficient_kan/kan.py`

Observed mechanics:

- `KANLinear` computes B-spline basis values for inputs, flattens the basis dimension, and uses `F.linear` against flattened spline weights.
- It keeps `base_weight`, `spline_weight`, optional `spline_scaler`, grid update, and an efficient reformulation that avoids expanding to `(batch, out_features, in_features)`.
- The README explicitly says the efficiency tradeoff changes the original sample-based sparsification regularizer into a weight-based proxy.

Implication:

- This is the practical starting point for a faithful B-spline KAN baseline in this repo.
- We should document that its regularization is not identical to pykan's original interpretability path.

## fast-kan

Repo: https://github.com/ZiyaoLi/fast-kan  
Local path: `deep-research/source-repos/fast-kan`  
Commit: `17b65401c252334fffb5e63c9852dd8316d29e69`

Key file:

- `fastkan/fastkan.py`

Observed mechanics:

- Replaces B-spline bases with Gaussian radial basis functions.
- Uses `LayerNorm` before RBF expansion.
- Has optional base update through a linear transform of `SiLU(x)`.
- Includes an attention module using FastKAN transforms.
- The README frames this as evidence that KANs are closely related to RBF networks.

Implication:

- Our `hcr_kan_mean` is closer to this family than to pykan.
- To make it a serious FastKAN-style baseline, add LayerNorm, base update, and explicit naming as `fastkan_like` rather than `hcr_kan_mean`.

## kan-gpt

Repo: https://github.com/adityang/kan-gpt  
Local path: `deep-research/source-repos/kan-gpt`  
Commit: `0c6e4c2582d9a0e23c612c3b695846d4caceac1c`

Key files:

- `kan_gpt/model.py`
- `kan_gpt/efficient_kan/model.py`
- `kan_gpt/kan/KAN.py`
- `kan_gpt/kan/KANLayer.py`
- `docs/results.md`

Observed mechanics:

- The GPT model replaces QKV projection, attention output projection, MLP projection layers, and LM head with KAN modules.
- It supports either `EFFICIENT_KAN` or `ORIGINAL_KAN`.
- The efficient KAN implementation expects 2D inputs internally; the model path depends on KAN wrappers being shape-compatible for `[B, T, C]`.
- The repo reports a Tiny Shakespeare comparison where KAN-GPT is slightly better than an equivalent MLP-GPT, while saying more experiments are needed.

Implication:

- This is a useful LM baseline design, but it is KAN-in-GPT, not HCR.
- Replacing attention projections with KAN is much more aggressive than only replacing the FFN.

## kansformers

Repo: https://github.com/akaashdash/kansformers  
Local path: `deep-research/source-repos/kansformers`  
Commit: `e58d5bc28a5fba09a6996454f6a121529bcf38f3`

Key files:

- `model.py`
- `kan.py`
- `README.md`

Observed mechanics:

- minGPT-style model.
- Replaces QKV projection, attention output projection, FFN, and LM head with KAN modules.
- The local `KANLinear` is efficient-kan-like but modified to accept arbitrary leading dimensions by flattening and restoring shape.
- README notes compute constraints, tiny-model training, high-memory settings, and future work to scale toward GPT-2.

Implication:

- Shape handling in `KANLinear.forward` is worth borrowing.
- The repo reinforces that naive KAN replacement can become memory/compute constrained quickly.

## KAT

Repo: https://github.com/Adamdad/kat  
Local path: `deep-research/source-repos/kat`  
Commit: `d254de7c14b6c050bd00cac3689b0a5614659a7f`

Key files:

- `katransformer.py`
- `README.md`

Observed mechanics:

- The code is a Vision Transformer variant.
- Attention remains standard linear QKV/projection.
- The MLP sublayer is replaced by a `KAN` module that uses `KAT_Group` rational activations.
- The README says vanilla ViT+KAN struggles to scale, and KAT addresses this with rational base functions, grouped KAN weight sharing, and activation-magnitude-aware initialization.
- It depends on an external CUDA/Triton rational KAT implementation.

Implication:

- This is the strongest warning against naive "replace all linears with KAN" scaling.
- For large models, grouped/shared activation functions and rational kernels are more realistic than full per-edge splines.
- KAT is a vision-transformer source. It is relevant for scalable KAN design, not for HCR fidelity.

## Summary for HCR-Bio-LLM

Recommended KAN baseline ladder:

1. FastKAN-like FFN only: cheap and close to our existing RBF path.
2. efficient-kan-style B-spline FFN only: faithful enough for KAN mechanics, manageable memory.
3. KAN-GPT / Kansformer-style all-projection replacement: only after memory measurements.
4. KAT-style grouped rational KAN: future scalable direction, especially if we add custom kernels or grouped sharing.

Keep all KAN baselines separate from HCR evidence.

