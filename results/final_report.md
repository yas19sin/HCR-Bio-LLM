# Initial HCR-LLM Prototype Report

This report records the implementation and smoke-test state of the project. It
has been revised against the local PDFs in `papers/`; see
`docs/paper_alignment.md` for the source-to-code mapping. It has also been
grounded against the user-provided Wolfram Community and GitHub sources; see
`docs/upstream_grounding.md`. The numbers below are not benchmark results; they
are short CPU runs on the built-in fallback Shakespeare excerpt to verify that
every model family builds, trains, evaluates, checkpoints, and exposes the
intended state statistics.

## 1. Motivation

The project tests whether HCRNN-style joint-distribution neurons can become a
practical language-model primitive rather than only an FFN replacement. The
implementation keeps a Transformer baseline and adds progressively richer HCR
state channels: mean, variance, density coefficients, pairwise features, and
bidirectional refinement.

## 2. HCRNN/KAN/Cortical-Neuron Inspiration

The implemented path follows three bridges from the brief:

- KAN-inspired expected-value propagation through radial basis functions. This
  is not a faithful KAN layer, because KAN proper uses learnable edge functions
  and the scalable KAN family splits into efficient B-spline KAN, FastKAN/RBF,
  and KAT/grouped rational KAN.
- Moment-carrying token states with `mu` and `log_var`.
- Multidirectional message passing through a non-causal denoising refinement
  model.

The code now also includes `src/model/hcr_moments.py`, a small-block HCR utility
that directly implements product-basis mixed-moment coefficients, conditional
coefficients, conditional means, and the 2D mutual-information approximation
from the HCR paper.

After the initial smoke report, `hcr_blockwise_joint` was added as the closest
trainable paper-direct HCR neuron path. It uses explicit blockwise coefficient
tensors for `rho(x, y)`, substitutes sigmoid-normalized hidden variables as a
practical proxy, normalizes `rho(y | x)`, exposes conditional coefficient
vectors and conditional variance, and feeds conditional expected values to the
Transformer residual path. It now carries density coefficient state between HCR
FFNs with configurable blending against the current point hidden state. The same
coefficients can be transposed for reverse conditioning, but reverse
reconstruction is not yet a language-model evaluation path.

## 3. Implemented Models

- `transformer_baseline`: GPT-style causal Transformer.
- `hcr_kan_mean`: Transformer with a radial-basis KAN-inspired FFN.
- `hcr_moment`: causal LM with `HCRState(mu, log_var)`.
- `hcr_density`: causal LM with latent density coefficients, not yet explicit
  HCR mixed-moment coefficients.
- `hcr_joint_pairwise`: causal LM with compressed pairwise channels, not yet a
  dense HCR coefficient tensor.
- `hcr_blockwise_joint`: causal LM with explicit blockwise HCR coefficient
  tensors and conditional expected-value propagation.
- `hcr_bidirectional_refinement`: non-causal masked denoising model with
  iterative refinement.

## 4. Training Setup

Smoke checks used:

- CPU execution.
- Character-level tokenizer.
- Built-in fallback Shakespeare excerpt.
- One-step architecture checks for all model families.
- Five-step HCR moment run for checkpoint, evaluation, uncertainty analysis,
  and sampling checks.
- Local paper audit against `papers/2405.05097v8.pdf`,
  `papers/2404.19756v5.pdf`, and `papers/2605.30370v2.pdf`.

For real experiments, benchmark configs now fetch `Trelis/tiny-shakespeare`
from Hugging Face through the Dataset Viewer API and cache the resulting text in
`data/hf_cache/`. You can override `hf_dataset`, `hf_split`, and `hf_max_rows`
from the command line, or set `data_path` for a local file.

Important correction from a later full baseline run: the old benchmark config
silently used the built-in fallback excerpt when `data/tiny_shakespeare.txt` was
missing. That run memorized the tiny train fragment immediately (`loss` near
zero) while validation loss climbed above `8`, so it is not a valid baseline.
Benchmark configs now use Hugging Face data by default, and the trainer logs
dataset source, character count, row count, cache path, and train/validation
window counts.

## 5. Smoke Results

| Model | Params | Task | Steps | Val loss | ECE | Extra state signal |
|---|---:|---|---:|---:|---:|---|
| Transformer | 12,032 | causal LM | 1 | 3.8375 | 0.0785 | `mu_std=0.9994` |
| HCR-KAN mean | 13,088 | causal LM | 1 | 3.6969 | 0.0688 | `mu_std=0.9998` |
| HCR moment | 25,888 | causal LM | 5 | 3.4443 | 0.0543 | `log_var_std=0.1015` |
| HCR density | 33,776 | causal LM | 1 | 3.9487 | 0.1131 | `basis_entropy=2.3726` |
| HCR joint | 31,928 | causal LM | 1 | 4.0177 | 0.2158 | `corr_std=1.0005` |
| HCR blockwise joint | 9,232 | causal LM | 3 | 3.7939 | 0.0859 | `hcr_conditional_mean_std=0.0072` |
| HCR refinement | 33,808 | denoising | 1 | 3.7615 | 0.1079 | `basis_entropy=1.7049` |

The five-step HCR moment run also passed checkpoint load/eval, corruption
evaluation, uncertainty analysis, and text sampling.

The paper-direct HCR faithfulness check passed before the density-state carry
extension via `hcr_faithfulness_check.py`:

- shifted-Legendre basis Gram max error: `4.77e-7`
- local forward `E[y | x]` MSE: `0.000268`
- local reverse `E[x | y]` MSE: `0.001477`
- local density-vector propagation coefficient max error: `2.38e-7`
- propagated mean/variance MSE: `< 1e-15`
- blockwise HCR propagation coefficient max error: `0.0`
- blockwise LM exposed conditional coefficients, conditional mean/variance,
  denominator diagnostics, and finite loss in a one-batch check

The check has since been extended to require carried HCR density-state keys and
shapes in a two-layer `hcr_blockwise_joint` LM. That latest runtime check is
pending because the local Python launcher was blocked by the environment usage
limit after the implementation change.

## 6. Denoising / Masked Reconstruction

The refinement model trains in `task: denoising` mode. Its input batch randomly
masks characters, supervises only masked positions, initializes masked-token
variance higher than observed-token variance, and uses non-causal HCR-inspired
density bridge blocks for iterative refinement. This is not yet reverse
conditioning through an explicit HCR joint-density tensor.

## 7. Calibration and Uncertainty

Evaluation reports expected calibration error and Brier score. HCR models also
report available state statistics:

- `log_var_mean`, `log_var_std`, `variance_mean`
- `basis_entropy`
- `corr_mean`, `corr_std`

The initial HCR moment smoke run showed non-constant variance after five steps,
which is only a sign that the channel is active, not evidence of useful
calibration yet.

## 8. Corruption Robustness

`eval.py` supports corrupted causal-context evaluation. On the five-step HCR
moment smoke checkpoint:

- clean validation loss: `3.4443`
- corrupted validation loss: `3.4542`
- degradation ratio: `1.0029`

This is a runtime-path check only; real robustness conclusions require longer
matched runs.

## 9. Refinement Dynamics

`hcr_bidirectional_refinement` supports `return_steps: true`, and the model
internally computes per-refinement-step losses when requested. The trainer and
metrics path exercise this mode.

## 10. Visualizations

The run logs are JSONL files under `runs/<experiment>/metrics.jsonl`. Suggested
plots:

- train and validation loss curves
- variance statistics over steps
- variance/error correlation
- calibration reliability diagrams
- clean vs corrupted loss
- refinement step vs loss
- density-basis entropy over training
- pairwise/correlation feature statistics

## 11. What Worked

- All model families build and complete at least one train/eval/checkpoint
  cycle.
- HCR moment, density, joint, and refinement models expose inspectable
  distributional state statistics.
- The same training interface covers causal LM and denoising refinement.
- Checkpoint loading, evaluation, corruption metrics, uncertainty summaries, and
  sampling were verified.
- The repo now has a paper-direct HCR mixed-moment utility separate from the
  scalable LM approximations.

## 12. What Failed

No architectural runtime failures remain from the smoke tests. The local paper
audit did reveal that the first implementation language overstated the KAN and
HCR density fidelity. That wording has been corrected, and the literal HCR
mixed-moment mechanics now live in `src/model/hcr_moments.py`.

## 13. Most Promising Next Direction

Run matched longer experiments for:

1. `transformer_baseline`
2. `hcr_moment`
3. `hcr_bidirectional_refinement`

Then compare validation loss, ECE, corruption degradation, and variance/error
correlation. This is the shortest path to seeing whether the distributional
channels carry useful signal beyond being trainable.

In parallel, use `HCRLocalJointDensity` and `hcr_synthetic_demo.py` on synthetic
normalized variables to validate paper-faithful conditional propagation, then
compare that behavior with `hcr_blockwise_joint` inside the language model.

The upstream review changes the priority order:

1. Add reverse reconstruction for the explicit HCR path.
2. Propagate density-vector state through attention/projection operations, not
   just the HCR FFN chain.
3. Add learned/empirical CDF normalization or another measured normalization
   path for hidden variables.
4. Add a separate KAN baseline, preferably efficient spline KAN or FastKAN-like
   RBF plus LayerNorm and base update.
5. Keep KAN/GPT/KAT results as baselines, not as proof of HCR behavior.

## 14. Open Questions

- Does hidden variance correlate with token error or ambiguity after real
  training?
- Do density coefficients avoid collapse without stronger auxiliary objectives?
- Do pairwise channels help on synthetic correlation tasks before language data?
- Does refinement improve masked reconstruction over multiple steps?
- Can local losses improve stability or sample efficiency?
