# Experiment Plan

## Experiment 1: HCR Synthetic Conditional Density

Goal:

- Verify `HCRLocalJointDensity` against known distributions.

Tasks:

- Independent variables: conditional should equal marginal.
- Correlated 2D distribution: conditional mean should match Monte Carlo estimate.
- Nonlinear relation with noise: conditional density should be multimodal or broad where appropriate.

Metrics:

- Conditional mean MSE.
- Conditional variance error.
- Negative density rate before calibration.
- KL or grid cross-entropy when a reference density is available.
- Conditional sampling mean and variance.

Pass condition:

- HCR beats a naive mean baseline and behaves correctly under independence.

## Experiment 2: Reverse Inference

Goal:

- Test whether one coefficient tensor supports both directions.

Tasks:

- Train or estimate on paired data `(x, y)`.
- Evaluate `E[y | x]`.
- Evaluate `E[x | y]`.

Models:

- `HCRLocalJointDensity`.
- `HCRBlockwiseJointNeuron`.
- MLP forward-only baseline.
- KAN/FastKAN forward-only baseline.

Metrics:

- Forward MSE.
- Reverse MSE.
- Denominator min/mean/std.
- Coefficient energy and sparsity.

Pass condition:

- HCR reverse inference works without training a separate reverse model.

## Experiment 3: Density-Vector Propagation

Goal:

- Test the paper's claim that density/moment vectors can propagate through HCR neurons.

Tasks:

- Represent input uncertainty as a moment vector.
- Propagate through a trained conditional model.
- Compare output moment vector to Monte Carlo propagation.

Metrics:

- Output mean error.
- Output variance error.
- Moment-vector cosine similarity.
- Runtime and memory vs sampling.

Pass condition:

- Moment propagation approximates Monte Carlo output with lower compute on small tasks.

## Experiment 4: Language Modeling Smoke Ladder

Goal:

- Keep LM comparisons honest and cheap.

Models:

- `transformer_baseline`.
- `hcr_moment`.
- `hcr_density`.
- `hcr_joint_pairwise`.
- `hcr_blockwise_joint`.
- `fastkan_ffn` once added.
- `efficient_kan_ffn` once added.

Metrics:

- Validation loss.
- Accuracy.
- Expected calibration error.
- Brier score.
- Corruption degradation.
- Parameter count.
- Wall-clock time.
- Peak memory.
- HCR-specific state stats.

Pass condition:

- Any improvement must survive comparison against both Transformer and KAN baselines.

## Experiment 5: HCR-Specific LM Diagnostics

Goal:

- Determine whether HCR variables carry meaningful uncertainty or density information.

Metrics:

- Conditional denominator stability.
- Coefficient entropy and sparsity.
- Pairwise energy per layer.
- Conditional variance vs token error.
- Reverse reconstruction of hidden states.
- Sensitivity to masking/noise.

Pass condition:

- HCR diagnostics correlate with ambiguity/error and do not collapse to constants.

## Experiment 6: Denoising / Refinement

Goal:

- Separate refinement benefits from HCR density benefits.

Models:

- Non-causal Transformer denoising baseline.
- `hcr_bidirectional_refinement`.
- HCR blockwise variant with reverse consistency, once implemented.

Metrics:

- Masked-token loss.
- Per-step loss.
- Improvement from step 1 to final step.
- Calibration on masked positions.
- Variance/error correlation.

Pass condition:

- Refinement improves over a matched non-causal baseline, and HCR-specific state adds measurable benefit.

## Experiment 7: Robustness and Sample Efficiency

Goal:

- Borrow the right evaluation style from IBNN without conflating mechanisms.

Tasks:

- Train with fractions of data.
- Token corruption at eval.
- Label-noise style synthetic classification.
- Adversarial or worst-case embedding perturbation if feasible.

Metrics:

- Data fraction needed for target validation loss.
- Corruption slope.
- Robust accuracy / loss.
- Memorization under corrupted labels.

Pass condition:

- HCR or KAN variants show a clear, reproducible benefit over parameter-matched baselines.

