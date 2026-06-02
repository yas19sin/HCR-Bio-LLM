# Architecture Roadmap

## Principle

Keep three lanes separate:

1. Baseline Transformer lane.
2. KAN baseline lane.
3. HCR density lane.

This avoids confusing KAN improvements with HCR evidence and avoids overstating bridge models.

## Phase 1: Make the Current HCR Primitive Honest

Completed in `src/model/hcr_moments.py`:

- Total-degree basis index generation.
- Conditional variance and mode from the grid density path.
- Marginal density extraction.
- Optional calibration function choices: floor, softplus, exponential.
- Density-vector propagation by substituting moment vectors for concrete basis
  values.

Added root-level executable check:

- `hcr_faithfulness_check.py` covers basis orthonormality, local conditional and
  reverse inference, density-vector propagation identity, and trainable
  blockwise HCR state exposure.

Still add broader tests:

- Coefficient recovery on known independent and correlated distributions.
- Reverse conditional consistency for symmetric synthetic tasks.
- Sampling mean/variance checks.

Why first:

- This is low-risk and directly grounded in the HCR paper plus the HCRNN prototype repo.

## Phase 2: Evaluate Blockwise Reverse Inference

Expose reverse inference for `hcr_blockwise_joint` outside the module and add a synthetic task:

- Train forward `x -> y`.
- Evaluate `E[y | x]`.
- Evaluate reverse `E[x | y]` using the same coefficient tensor.
- Compare against an MLP and against a forward-only KAN baseline.

Metrics:

- Forward MSE.
- Reverse MSE.
- Conditional denominator stability.
- Coefficient sparsity / energy.
- Sensitivity to block size and degree.

Why:

- Reverse inference is central to HCR. Without it, `hcr_blockwise_joint` is only a custom FFN.

## Phase 3: Add Density-Vector Propagation

Partially completed for the HCR FFN chain. The trainable block can consume:

- A point value represented by basis products.
- Or a moment vector representing input density.

Then output:

- Conditional expected value for LM compatibility.
- And/or output density coefficients for the next layer.

Minimal implementation:

- For block size 1 or 2, keep a small moment vector per hidden group.
- Replace `f_j(x)` with input coefficient vector `b_j` where appropriate.
- Normalize by the propagated zero coefficient.

Current remaining limitation:

- Attention and projection operations still use point hidden values, so density
  state is carried through HCR FFNs rather than all Transformer operations.

Why:

- This is the paper's actual density propagation story.

## Phase 4: Add Local HCR Objectives

Cross entropy alone may not force coefficients to behave as meaningful density moments.

Candidate losses:

- Denominator stability penalty.
- Coefficient L1 / group sparsity.
- Pairwise dependency energy regularizer.
- Reverse consistency loss.
- Conditional entropy / uncertainty calibration loss.
- Information-bottleneck-style penalty using squared nontrivial coefficients.

Why:

- HCR needs local structure. A Transformer objective may use coefficient tensors as arbitrary nonlinear parameters unless constrained.

## Phase 5: Add Clean KAN Baselines

Add separate model names:

- `fastkan_ffn`: RBF + LayerNorm + base update.
- `efficient_kan_ffn`: B-spline efficient KANLinear replacing FFN only.
- `kan_gpt_projection`: optional all-projection replacement after memory checks.

Metrics:

- Validation loss and accuracy.
- Wall-clock time.
- Peak memory.
- Parameter count.
- Edge/basis diagnostics where available.

Why:

- KAN is the nearest comparison family but not evidence for HCR.

## Phase 6: Token/Embedding Density Hypothesis

Only after Phases 1-5:

- Group embedding dimensions as moment sets.
- Add moment-order penalties.
- Replace or augment softmax unembedding with a density-overlap interpretation.
- Test on token attributes where some ground truth exists.

Metrics:

- Attribute reconstruction.
- Moment decay by order.
- Uncertainty reduction under context.
- Calibration under ambiguous prompts.

Why:

- The transformer-density-embedding idea is the most speculative part of the HCR paper. It needs careful tests.

## Phase 7: Optional IBNN Baseline

Do not add IBNN inside the main LM first.

Start with:

- Small synthetic classification.
- Small text classification.
- Robustness and label-noise tests.

Why:

- IBNN has different mechanics and computational costs. It is a related baseline, not a prerequisite for HCR.
