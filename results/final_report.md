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
practical next-token prediction primitive rather than only an FFN replacement.
The implementation keeps a Transformer baseline and adds progressively richer
causal HCR state channels: mean, variance, density coefficients, pairwise
features, and explicit blockwise joint-density propagation. Bidirectional
refinement is recorded as a side experiment, not as evidence for causal LM
quality.

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
- `hcr_bidirectional_refinement`: non-causal masked denoising side branch with
  iterative refinement. It is not a next-token prediction model.

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

## 5.1 Kaggle Native-Config Results From Previous Run

The first Kaggle GPU run used the native benchmark configs, not the later
`fair_4m` parameter-matched configs. These results are useful as a progress
marker, but only causal models should be compared against each other. The
bidirectional refinement result is a separate masked reconstruction side task,
not an NTP result.

For `hcr_bidirectional_refinement` on `Trelis/tiny-shakespeare`, checkpoint
step `10000`:

| Metric | Value |
|---|---:|
| Task | denoising / masked reconstruction |
| Params | 3,284,656 |
| Train-time best val loss | 1.2187 |
| Final eval loss | 1.1949 |
| Accuracy | 0.6294 |
| Perplexity | 3.3034 |
| ECE | 0.0565 |
| Brier | 0.3557 |
| Basis entropy | 0.2444 |
| `log_var_mean` | 0.1711 |
| `log_var_std` | 1.4535 |
| `mu_std` | 0.7782 |
| `variance_mean` | 2.9083 |

The follow-up uncertainty summary for the same checkpoint reported
`basis_entropy=0.2418`, `log_var_mean=0.1545`, `log_var_std=1.4213`,
`mu_std=0.7782`, and `variance_mean=2.7664`.

The generated `sample.py` text from this checkpoint was broken and heavily
uppercase/repetitive. That is expected: `sample.py` performs autoregressive
next-token generation, while `hcr_bidirectional_refinement` was trained as a
non-causal denoising model that reconstructs masked positions using surrounding
context. Its good validation loss means it is good at masked reconstruction,
not at causal text generation. This result should not be compared directly to
causal LM validation losses.

## 5.2 Fair 4M Causal NTP Results

The first parameter-matched causal benchmark used:

- suite: `fair-causal`
- steps: `10000`
- seed: `1337`
- eval batches: `20`
- dataset: `Trelis/tiny-shakespeare`
- output directory: `runs/benchmark_suites/fair_4m_10k`

All rows are causal next-token prediction models near a 4M trainable-parameter
budget. They are not compute-matched, so throughput matters.

| Model | Params | Loss | d_loss vs baseline | Acc | PPL | ECE | Brier | Corrupt x | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer baseline | 4,069,728 | 1.4893 |  | 0.5613 | 4.4340 | 0.0466 | 0.4215 | 1.3203 | 72,117 |
| HCR-KAN mean | 4,031,508 | 1.4851 | -0.0042 | 0.5615 | 4.4155 | 0.0511 | 0.4207 | 1.4978 | 13,699 |
| HCR moment | 3,959,124 | 1.5472 | +0.0579 | 0.5416 | 4.6982 | 0.0393 | 0.4434 | 1.3471 | 65,326 |
| HCR density | 3,914,900 | 1.5337 | +0.0444 | 0.5427 | 4.6354 | 0.0347 | 0.4448 | 1.3310 | 61,544 |
| HCR joint pairwise | 4,090,696 | 1.5536 | +0.0643 | 0.5419 | 4.7286 | 0.0408 | 0.4449 | 1.3498 | 61,070 |
| HCR blockwise joint | 4,033,808 | 1.5605 | +0.0712 | 0.5399 | 4.7614 | 0.0551 | 0.4428 | 1.3131 | 17,486 |

This run does not support a causal HCR NTP advantage. The only model with lower
loss than the baseline is `hcr_kan_mean`, by `0.0042`, but that model is a
KAN/RBF-style bridge rather than HCR joint-density evidence, and it is roughly
5.3x slower than the baseline in this run. The HCR-specific causal variants
lost on first-pass language-modeling quality, although `hcr_moment` and
`hcr_density` had better ECE than the baseline. That calibration signal is
interesting but not enough to claim better generation or better NTP.

## 5.3 Focused Faithful-HCR NTP Run

After the fair 4M suite, the focused paper-direct run trained only
`hcr_blockwise_joint` with `configs/faithful_hcr/ntp_stable.yaml`:

- suite: `faithful-hcr`
- steps: `10000`
- seed: `1337`
- eval batches: `20`
- output directory: `runs/benchmark_suites/faithful_hcr_ntp_10k`

| Model | Note | Params | Loss | Acc | PPL | ECE | Brier | Corrupt x | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HCR blockwise joint | faithful_hcr_ntp_stable | 4,033,808 | 1.5956 | 0.5300 | 4.9312 | 0.0482 | 0.4553 | 1.2561 | 17,400 |

This is a negative result for the current faithful causal implementation. The
stability-regularized faithful config was worse than the fair Transformer
baseline (`1.4893`) and worse than the earlier fair `hcr_blockwise_joint` run
(`1.5605`). The lower corruption ratio is not enough to offset the main NTP
loss/perplexity regression. The right conclusion is that this implementation
still needs fidelity and optimization work before it can test the paper's HCR
claim inside a causal LM.

## 5.4 Standalone Faithful Local HCM/HCR Prototype

The repo now includes a smaller faithful path that does not use a Transformer
shell: `HCRSequenceDensityModel` in `src/model/hcr_sequence.py`, with the
runnable script `faithful_hcm_sequence_demo.py`.

This prototype estimates one local HCR joint density over normalized
context/target windows:

```text
[x_t, x_{t+1}, ..., x_{t+k}] -> product-basis mixed-moment coefficients
```

It uses per-variable empirical-CDF normalization, then performs conditional
mean prediction, reverse conditioning, variance estimation, conditional log
density, grid mode, and sampling from the same coefficient tensor.

Default controlled nonlinear transition run:

```bash
python faithful_hcm_sequence_demo.py
```

Key metrics from the default run:

| Metric | Value |
|---|---:|
| source | transition |
| context length | 2 |
| degree | 5 |
| max total degree | 5 |
| coefficient shape | `[6, 6, 6]` |
| nonzero coefficients | 56 |
| HCR mean raw forward MSE | 0.0069 |
| linear raw forward MSE | 0.0446 |
| persistence raw forward MSE | 0.1590 |
| train-mean raw forward MSE | 0.0775 |
| HCR same-density reverse raw MSE | 0.0800 |
| linear reverse raw MSE | 0.0804 |

This is the most faithful implementation in the repo for local HCR/HCM
conditional mechanics. It is explicitly not a language-model win: it validates
the local normalized joint-density primitive on a controlled problem. The gap is
still translating this into a useful, scalable causal LM architecture without
falling back to a mostly ordinary Transformer.

## 5.5 Direct HCM Next-Token Language Model

The repo now includes a direct HCM/HCR next-token language model:

- source: `src/model/hcm_ntp_lm.py`
- CLI: `hcm_ntp_lm.py`

This is not a Transformer. It is a small local character LM where HCM directly
contributes to NTP:

1. Build token windows `[x_{t-k}, ..., x_{t-1}, x_t]`.
2. Normalize each window column with its own empirical CDF.
3. Estimate HCR product-basis mixed-moment coefficients over the full local
   context/target window.
4. Condition on the context with the HCR formula.
5. Score every next-token candidate from the HCR conditional density.
6. Convert continuous target density back to discrete token probability using
   the target token's empirical CDF bin mass.

The current CLI reports two direct HCM scorers:

- ID-CDF HCM: one empirical-CDF variable per token position.
- Feature HCM: two deterministic character features per token position
  (normalized codepoint and coarse character class), then HCR scoring over the
  full context/target feature window.

The CLI also evaluates a discrete backoff n-gram baseline and a log-linear
hybrid where HCM is a direct rescoring factor:

```text
p(token | context) proportional to p_ngram(token | context)^(1-w)
                                * p_hcm(token | context)^w
```

Local sanity run on the stable project file `hcr_transformer_intern_project.md`:

```bash
python hcm_ntp_lm.py --data-path hcr_transformer_intern_project.md --context-length 4 --degree 4 --max-total-degree 4 --feature-degree 3 --feature-max-total-degree 3 --max-train-windows 30000 --eval-windows 3000 --hybrid-weights 0.05,0.1,0.25,0.5 --sample-model feature_hybrid --sample-hybrid-weight 0.5 --output-dir runs/hcm_ntp_lm/local_feature_3k
```

| Model | Loss | PPL | Acc | Brier |
|---|---:|---:|---:|---:|
| Pure ID-CDF HCM NTP | 3.4831 | 32.56 | 0.1187 | 0.8922 |
| Pure feature-HCM NTP | 3.6448 | 38.28 | 0.1283 | 0.8862 |
| Backoff n-gram | 2.3132 | 10.11 | 0.6050 | 0.3771 |
| N-gram + ID-CDF HCM, `w=0.05` | 2.2395 | 9.39 | 0.6040 | 0.3774 |
| N-gram + ID-CDF HCM, `w=0.10` | 2.1692 | 8.75 | 0.6053 | 0.3783 |
| N-gram + ID-CDF HCM, `w=0.25` | 1.9872 | 7.29 | 0.6063 | 0.3865 |
| N-gram + ID-CDF HCM, `w=0.50` | 1.8733 | 6.51 | 0.5890 | 0.4479 |
| N-gram + feature HCM, `w=0.05` | 2.2426 | 9.42 | 0.6037 | 0.3771 |
| N-gram + feature HCM, `w=0.10` | 2.1763 | 8.81 | 0.6050 | 0.3778 |
| N-gram + feature HCM, `w=0.25` | 2.0164 | 7.51 | 0.6057 | 0.3879 |
| N-gram + feature HCM, `w=0.50` | 1.9536 | 7.05 | 0.5847 | 0.4413 |

Interpretation: pure HCM is a real NTP language model but is weak on raw
categorical characters because a small continuous HCR window is a crude token
representation. The hybrid result is more promising: HCM directly improves
negative log-likelihood when used as a conditional-density rescoring factor,
although Brier score worsens at larger HCM weights. On this file, ID-CDF HCM
beats the feature-HCM variant by loss, while feature-HCM has slightly better
Brier at larger hybrid weights. This is the current closest artifact to
"faithful HCM as a direct contributor to language modeling."

The paper-direct HCR faithfulness check now passes via
`hcr_faithfulness_check.py`:

- shifted-Legendre basis Gram max error: `4.77e-7`
- local forward `E[y | x]` MSE: `0.000268`
- local reverse `E[x | y]` MSE: `0.001477`
- local density-vector propagation coefficient max error: `2.38e-7`
- propagated mean/variance MSE: `< 1e-15`
- standalone sequence-density HCR raw forward MSE: `0.00776`
- standalone sequence-density linear raw forward MSE: `0.04075`
- direct HCM NTP ordinal-token loss: `0.9717`
- direct HCM NTP ordinal-token accuracy: `0.7975`
- blockwise HCR propagation coefficient max error: `0.0`
- blockwise LM exposed conditional coefficients, conditional mean/variance,
  denominator diagnostics, and finite loss in a one-batch check

The check also requires carried HCR density-state keys and shapes in a two-layer
`hcr_blockwise_joint` LM.

## 6. Optional Denoising / Masked Reconstruction Side Branch

The refinement model trains in `task: denoising` mode. Its input batch randomly
masks characters, supervises only masked positions, initializes masked-token
variance higher than observed-token variance, and uses non-causal HCR-inspired
density bridge blocks for iterative refinement. This is not a next-token
prediction path. It should stay out of the main HCR-vs-Transformer causal LM
claim unless a causal refinement variant is implemented.

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

## 9. State Dynamics

The causal HCR models expose moment, density, pairwise, and blockwise
conditional-state statistics through the same metrics path. The denoising side
branch also supports `return_steps: true`, but that only measures masked
reconstruction refinement, not NTP generation.

## 10. Visualizations

The run logs are JSONL files under `runs/<experiment>/metrics.jsonl`. Suggested
plots:

- train and validation loss curves
- variance statistics over steps
- variance/error correlation
- calibration reliability diagrams
- clean vs corrupted loss
- causal state statistics vs loss
- density-basis entropy over training
- pairwise/correlation feature statistics

## 11. What Worked

- All model families build and complete at least one train/eval/checkpoint
  cycle.
- HCR moment, density, joint, and blockwise causal models expose inspectable
  distributional state statistics for next-token prediction.
- The same training interface can run the optional denoising side branch, but
  that branch is excluded from causal LM comparisons.
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

The parameter-matched causal NTP suite has now run, and the HCR-specific causal
variants did not beat the Transformer baseline on validation loss or
perplexity. The focused faithful-HCR run also underperformed. The next step is
no longer more same-shape benchmarking; it is to fix the HCR translation into a
causal Transformer.

The most promising engineering direction is:

1. Stabilize `hcr_blockwise_joint` denominators and coefficients during causal
   NTP.
2. Replace `sigmoid(hidden)` with learned/empirical CDF normalization or another
   measured normalization path for hidden variables.
3. Propagate density-vector state through attention/projection operations, not
   just the HCR FFN chain.
4. Add auxiliary conditional/reverse objectives for the explicit HCR path, so
   the coefficient tensors are trained to be useful distributions rather than
   only indirectly optimized through next-token cross entropy.
5. Keep `hcr_kan_mean` as a separate KAN/RBF-style baseline. Its tiny loss win
   is not HCR evidence, especially given the large throughput cost.

This direction is now represented by `configs/faithful_hcr/ntp_stable.yaml` and
the `faithful-hcr` benchmark suite. That suite trains only
`hcr_blockwise_joint`, the closest paper-direct causal model in this repo. The
config keeps explicit blockwise product-basis coefficients and carried density
state, while adding auxiliary losses that penalize:

- small or negative HCR conditional denominators,
- exploding non-normalizer conditional coefficients,
- normalized conditional variance above the `[0, 1]` variable bound.

The first focused 10k run of that config reached loss `1.5956`, which is worse
than the fair Transformer baseline and the earlier fair blockwise HCR result.
The command remains useful as the focused regression target:

```bash
python run_benchmark_suite.py --suite faithful-hcr --steps 10000 --run-name faithful_hcr_ntp_10k
```

In parallel, use `HCRLocalJointDensity` and `hcr_synthetic_demo.py` on synthetic
normalized variables to validate paper-faithful conditional propagation, then
compare that behavior with `hcr_blockwise_joint` inside the language model.

The upstream review changes the priority order:

1. Stabilize the explicit blockwise HCR denominator/coefficient path for causal
   NTP.
2. Add learned/empirical CDF normalization or another measured normalization
   path for hidden variables.
3. Propagate density-vector state through attention/projection operations, not
   just the HCR FFN chain.
4. Add a separate KAN baseline, preferably efficient spline KAN or FastKAN-like
   RBF plus LayerNorm and base update.
5. Keep KAN/GPT/KAT results as baselines, not as proof of HCR behavior.

## 14. Open Questions

- Does hidden variance correlate with token error or ambiguity after real
  training?
- Do density coefficients avoid collapse without stronger auxiliary objectives?
- Do pairwise channels help on synthetic correlation tasks before language data?
- Can explicit blockwise HCR coefficients remain stable during causal NTP?
- Can local losses improve stability or sample efficiency?
