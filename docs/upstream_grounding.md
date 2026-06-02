# Upstream Grounding Review

Reviewed on 2026-06-02 from the user-provided links and the local PDFs in
`papers/`.

## HCR / Joint-Distribution Sources

| Source | What it contributes | Implication for this repo |
|---|---|---|
| Wolfram Community: `Hierarchical correlation reconstruction: between statistics and machine learning` | Public HCR notebook/post by Jarek Duda. The visible page confirms the HCR statistical/machine-learning framing and notebook attachment. | Treat HCR as a statistical joint-density reconstruction method first, not as an FFN trick. |
| Wolfram Community: `HCRNN: biology-inspired joint distribution artificial neural networks reducing to KAN-like` | Public HCRNN notebook/post by Jarek Duda. The visible page title and image caption describe an HCR neuron/neural network containing a local joint-distribution model. | The faithful path must center local joint densities, bidirectional inference, conditioning, density/value propagation, and uncertainty. |
| `vfd-org/joint-distribution-neuron` | Prototype HCRNN repo. Its README describes joint-distribution neural units, local joint probability densities, `X -> Y` and `Y -> X` inference, conditional sampling, uncertainty propagation, reversible regression, and tests for basis orthonormality / conditional inference / joint-density estimation. | Our `HCRLocalJointDensity` and `hcr_blockwise_joint` align with this direction. `HCRLocalJointDensity` now includes grid-based conditional sampling. Remaining gaps are explicit uncertainty propagation across layers and tested reverse reconstruction. |
| Local `papers/2405.05097v8.pdf` | Primary HCRNN paper. Defines product-basis density coefficients as mixed moments, CDF/EDF normalization to `[0, 1]`, conditional propagation by substituting variables and normalizing, value and density propagation, tensor decomposition, and local/information-bottleneck training directions. | The faithful core should remain explicit coefficient tensors or sparse/low-rank approximations of them. A mean/variance hidden state is only a bridge. |

## KAN Sources

| Source | What it contributes | Implication for this repo |
|---|---|---|
| `kindxiaoming/pykan` | Official KAN repo for the KAN paper. It emphasizes edge activations, interpretability, grid extension, sparsification/pruning, symbolic regression, and cautions that KANs are not yet a simple plug-in for large ML/LLM settings. | A faithful KAN baseline should use edge functions and expose plotting/sparsification hooks. Do not infer that KAN will help LMs without experiments. |
| `Blealtan/efficient-kan` | Memory-efficient PyTorch KAN. It reformulates B-spline basis expansion so computation becomes a matrix multiplication, while changing regularization for efficiency. | If we add a spline KAN baseline, use this style rather than expanding `(batch, out, in)` tensors. |
| `ZiyaoLi/fast-kan` | Replaces KAN's B-spline basis with Gaussian RBFs, uses LayerNorm and optional base update, and frames FastKAN as connecting KANs and RBF networks. | Our `hcr_kan_mean` is closer to FastKAN/RBF-KAN than to pykan. It needs LayerNorm/base-update parity before being called a serious FastKAN baseline. |
| `adityang/kan-gpt` | GPT language model using KANs, with PyPI package and GPT2-style config usage. References minGPT, pykan, webtext, and tinyshakespeare. | Confirms that KAN-in-GPT is a known baseline family. It is separate from HCR; compare against it, do not conflate it with joint-density neurons. |
| `akaashdash/kansformers` | minGPT-style Transformer where KANs are swapped in for linear layers using efficient KAN; notes high memory use and small-model limitations. | For language modeling, KAN replacement can be expensive. Track memory and context length carefully. |
| `Adamdad/kat` | ICLR 2025 Kolmogorov-Arnold Transformer. Uses GR-KANs, rational functions instead of B-splines, grouped edge sharing, initialization for activation scale, and large-scale ImageNet-oriented code. | For scalable Transformer KANs, grouped/rational KANs are more relevant than naive per-edge splines. This is a vision Transformer reference, not HCR. |
| Local `papers/2404.19756v5.pdf` | KAN paper. KANs place learnable univariate functions on edges, commonly spline-parametrized, rather than fixed node activations. | Keep KAN baselines separate from HCR experiments. |

## Cortical-Cell Source

| Source | What it contributes | Implication for this repo |
|---|---|---|
| Local `papers/2605.30370v2.pdf` | Implicit-bias neural network model based on dendritic nonlinearities and same-layer coupling. | This motivates moving beyond point neurons, but it is not HCR and not KAN. Implement as a separate baseline only if needed. |

## Design Decisions From This Review

1. `hcr_blockwise_joint` is the closest trainable HCR path in this repo, but it
   still uses sigmoid hidden normalization as a proxy and only carries density
   coefficient state through the HCR FFN chain, not through attention.
2. `hcr_moment`, `hcr_density`, and `hcr_joint_pairwise` stay labeled as bridges.
3. `hcr_kan_mean` should be described as RBF/FastKAN-like, not faithful KAN.
4. A future faithful KAN baseline should be named separately, e.g.
   `efficient_spline_kan` or `fast_kan_transformer`.
5. HCR success metrics must include reverse inference, conditional sampling,
   uncertainty propagation, and reconstruction, not only next-token loss.

## Concrete Gaps To Close

- Add reverse reconstruction evaluation for `hcr_blockwise_joint`.
- Propagate density-vector state through attention/projection operations, not
  just through the HCR FFN chain.
- Add learned/empirical CDF normalization or another measured normalization
  path for hidden variables.
- Add a real KAN baseline with either efficient B-spline KAN or FastKAN-style
  RBF + LayerNorm + base update.
- Keep all KAN baselines in the comparison table as baselines, not HCR evidence.
