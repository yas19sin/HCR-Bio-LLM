## Vibelab: Exploratory HCR Research

**⚠️ Experimental & Exploratory Code**  
This is a personal research sandbox for exploring HCR-inspired neurons. It's vibeoded—meant for fun experimentation and learning. No PRs or issues. Just for fun.

Trainable research prototype for testing joint-distribution-inspired neurons in
small language models.

The project follows `hcr_transformer_intern_project.md`: start with an
undeniable tiny Transformer baseline, then compare causal next-token prediction
variants that carry expected values, hidden moments, density coefficients, and
pairwise features. Bidirectional refinement is a side experiment, not part of
the main NTP benchmark.

The implementation is grounded against the local PDFs in `papers/`. See
[docs/paper_alignment.md](docs/paper_alignment.md) for the exact mapping between
the papers and the current code, including what is faithful and what is still an
approximation.

External upstream sources from Wolfram Community and the KAN/HCR GitHub repos
are summarized in [docs/upstream_grounding.md](docs/upstream_grounding.md).

## Current Status

The fair 4M causal next-token benchmark did not validate an HCR advantage yet.
The Transformer baseline reached loss `1.4893`; the HCR-specific causal models
were worse (`hcr_density=1.5337`, `hcr_moment=1.5472`,
`hcr_joint_pairwise=1.5536`, `hcr_blockwise_joint=1.5605`). `hcr_kan_mean`
barely beat the baseline at `1.4851`, but it is a KAN/RBF-style bridge rather
than HCR joint-density evidence and was much slower.

The first focused faithful-HCR run also underperformed: `hcr_blockwise_joint`
with the stable faithful config reached loss `1.5956`, worse than the fair
baseline and worse than the earlier fair blockwise run. That means the current
paper-direct implementation still needs debugging and architectural fidelity
work before it can support an HCR causal-LM claim.

A standalone faithful local HCM/HCR sequence-density prototype now exists in
[src/model/hcr_sequence.py](src/model/hcr_sequence.py), with a runnable demo in
[faithful_hcm_sequence_demo.py](faithful_hcm_sequence_demo.py). This path uses
per-variable empirical-CDF normalization, product-basis mixed-moment
coefficients over causal context/target windows, HCR conditional means, reverse
conditioning with the same density, conditional variance, conditional log
density, grid mode, and sampling. On the default controlled nonlinear
transition task it reaches raw forward MSE `0.0069` versus linear `0.0446`.
This validates the local HCR conditional-density mechanics, not the failed
Transformer-shell language-model claim.

A direct HCM next-token language model now exists in
[src/model/hcm_ntp_lm.py](src/model/hcm_ntp_lm.py), with a runnable CLI in
[hcm_ntp_lm.py](hcm_ntp_lm.py). It estimates HCR product-basis coefficients over
empirical-CDF-normalized token windows, then scores every next-token candidate
from the HCR conditional density. On a local stable project-file character-LM
sanity run (`hcr_transformer_intern_project.md`), pure HCM reached loss
`3.4831` / PPL `32.56`; a count backoff n-gram reached loss `2.3132` / PPL
`10.11`; and an HCM-rescored n-gram with HCM weight `0.5` reached loss `1.8733`
/ PPL `6.51`. This is now a real HCM-driven NTP language model path, but it is
still a small local model and not a Transformer-scale LLM.

## Implemented Models

- `transformer_baseline`: small GPT-style causal Transformer.
- `hcr_kan_mean`: replaces the FFN with a KAN-inspired radial-basis
  expected-value neuron. This is closer to FastKAN/RBF-KAN than to faithful
  spline-edge KAN.
- `hcr_moment`: carries `HCRState(mu, log_var)` through attention and FFN
  updates.
- `hcr_density`: adds normalized latent density coefficients. These are not yet
  the HCR paper's mixed-moment product-basis coefficients.
- `hcr_joint_pairwise`: adds compressed pairwise/correlation channels. This is
  not yet a dense blockwise HCR coefficient tensor.
- `hcr_blockwise_joint`: uses explicit blockwise HCR joint-density coefficient
  tensors in the FFN path. This is the closest trainable paper-direct HCR path:
  sigmoid-normalized hidden variables as a proxy, product-basis mixed moments,
  conditional coefficient vectors, conditional mean/variance propagation,
  density coefficient state carried between blockwise HCR FFNs, and exposed
  reverse conditioning on the same coefficients. Attention still operates on
  point hidden values rather than full density states.
- `hcr_bidirectional_refinement`: non-causal denoising side branch with
  iterative refinement and per-step loss support. It is not a next-token
  prediction model and should not be used for causal generation claims.
- `HCRSequenceDensityModel`: standalone small local HCR/HCM sequence-density
  prototype, not a Transformer. It estimates explicit product-basis
  coefficients over normalized context/target windows and performs causal and
  reverse conditioning from that same local density.
- `HCMNextTokenLanguageModel`: direct small HCM/HCR character NTP language
  model. It turns token windows into per-column empirical-CDF variables, uses
  local HCR conditional density to score next-token candidates, and can also be
  used as a log-linear HCM rescoring factor over a discrete backoff n-gram LM.

For a literal small-block HCR primitive, see
[src/model/hcr_moments.py](src/model/hcr_moments.py). It implements shifted
Legendre basis functions, mixed-moment coefficient estimation, density
evaluation, conditional coefficients, conditional mean/variance/mode,
marginals, density-vector propagation, conditional sampling, and the HCR
small-coefficient mutual-information approximation.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a tiny offline smoke experiment. This is the only training command meant to
use the built-in fallback excerpt:

```bash
python train.py --config configs/smoke.yaml
```

Run the paper-direct HCR smoke path:

```bash
python train.py --config configs/smoke_faithful_hcr.yaml
```

Run the small synthetic HCR conditional-density demo:

```bash
python hcr_synthetic_demo.py
```

Run the standalone faithful local HCM/HCR sequence-density demo:

```bash
python faithful_hcm_sequence_demo.py
```

Run the direct HCM next-token language model on a local text file:

```bash
python hcm_ntp_lm.py --data-path hcr_transformer_intern_project.md --context-length 4 --degree 4 --max-total-degree 4 --max-train-windows 30000 --eval-windows 3000 --hybrid-weights 0.05,0.1,0.25,0.5
```

Run it on the same Tiny Shakespeare source used by the benchmark configs:

```bash
python hcm_ntp_lm.py --hf-dataset Trelis/tiny-shakespeare --hf-max-rows 1000 --context-length 4 --degree 4 --max-total-degree 4 --max-train-windows 50000 --eval-windows 5000
```

Run the source-grounded HCR faithfulness check:

```bash
python hcr_faithfulness_check.py
```

Run the baseline. Benchmark configs automatically fetch and cache
`Trelis/tiny-shakespeare` from Hugging Face:

```bash
python train.py --config configs/transformer_baseline.yaml
```

Run HCR variants on the same real dataset:

```bash
python train.py --config configs/hcr_moment.yaml
python train.py --config configs/hcr_density.yaml
python train.py --config configs/hcr_joint_pairwise.yaml
python train.py --config configs/hcr_blockwise_joint.yaml
```

Run a matched causal benchmark suite from one command. This is the recommended
Kaggle workflow because it captures raw logs to files and prints only a compact
status/result stream. The `fair-causal` suite uses `configs/fair_4m/` and keeps
all causal models near a 4M trainable-parameter budget:

```bash
python run_benchmark_suite.py --suite fair-causal --steps 10000
```

The fair causal suite runs the Transformer baseline, HCR-KAN mean, HCR moment,
HCR density, HCR joint-pairwise, and HCR blockwise-joint with matched dataset,
seed, context length, batch size, optimizer settings, step count, eval interval,
eval batch count, and approximate parameter budget. It is still not compute-
matched because the architectures do different work per token, so the generated
report shows throughput as well as parameter counts.

After the first fair 4M run, the HCR-specific causal variants did not beat the
Transformer baseline. The project now focuses on the most paper-direct path
instead of broad model comparisons. Run only the faithful HCR NTP track with:

```bash
python run_benchmark_suite.py --suite faithful-hcr --steps 10000 --run-name faithful_hcr_ntp_10k
```

This uses `configs/faithful_hcr/ntp_stable.yaml`, which trains
`hcr_blockwise_joint` with explicit blockwise product-basis coefficients,
carried density state, and auxiliary stability losses for conditional
denominators, coefficient magnitude, and normalized conditional variance.
Suite runs mirror compact training progress to the notebook by default while
still writing complete logs to `suite_logs/train.log`. Use `--progress off` for
quiet runs or `--progress all` to also mirror eval/uncertainty/sample output.

The first 10k faithful-HCR Kaggle run reached loss `1.5956`, accuracy `0.5300`,
perplexity `4.9312`, ECE `0.0482`, Brier `0.4553`, corruption ratio `1.2561`,
and throughput around `17.4k tok/s`. Treat this as a negative result for the
current faithful causal implementation, not as evidence against the paper in
general.

The denoising refinement model is a separate, optional reconstruction task. It
is intentionally outside the NTP benchmark:

```bash
python run_benchmark_suite.py --suite fair-denoising --steps 10000
```

The older `causal` and `all` suites use each causal model's native config sizes
and are useful for continuity, but not for parameter-matched claims. The
`research-all` and `fair-research-all` suites are the only suite aliases that
include the denoising side branch.

Suite outputs live under `runs/benchmark_suites/<suite_name>/`:

- `summary.md`: compact comparison table and interpretation checklist.
- `summary.json`: machine-readable result summary.
- `<model>/metrics.jsonl`: full training metrics.
- `<model>/suite_logs/train.log`: captured stdout/stderr from training.
- `<model>/suite_logs/eval.json`: final evaluation output.
- `<model>/suite_logs/uncertainty.json`: distribution-channel summary.
- `<model>/suite_logs/sample.txt`: generated sample for causal models.

For a less noisy manual training cell, use compact stdout:

```bash
python train.py --config configs/hcr_density.yaml --set stdout=compact --set stdout_events=start,eval,final --set log_interval=250
```

Check exact trainable parameter counts for the fair configs without loading
data:

```bash
python count_model_params.py configs/fair_4m
```

If your dataset is elsewhere, pass it explicitly:

```bash
python train.py --config configs/hcr_blockwise_joint.yaml --set hf_dataset=namespace/dataset --set hf_split=train
```

Local files still work when you want them:

```bash
python train.py --config configs/hcr_blockwise_joint.yaml --set data_path=path/to/text.txt
```

Sample from a causal checkpoint:

```bash
python sample.py --checkpoint runs/transformer_baseline/best.pt --prompt "First "
```

`sample.py` is intentionally restricted to causal LM checkpoints. Denoising
checkpoints such as `hcr_bidirectional_refinement` can have strong masked
reconstruction loss while producing nonsense under autoregressive sampling,
because they were not trained for next-token generation.

Evaluate a checkpoint:

```bash
python eval.py --checkpoint runs/hcr_moment/best.pt
```

Summarize distribution channels:

```bash
python analyze_uncertainty.py --checkpoint runs/hcr_moment/best.pt
```

For quick debugging without network, use the smoke configs. To force the tiny
fallback through a benchmark config, override the dataset explicitly and treat
the run as non-comparable:

```bash
python train.py --config configs/transformer_baseline.yaml --set dataset=tiny_shakespeare --set allow_fallback_dataset=true
```

## Data

By default, benchmark configs use Hugging Face:

```yaml
dataset: huggingface
hf_dataset: Trelis/tiny-shakespeare
hf_config: default
hf_split: train
hf_max_rows: 1000
```

The loader uses the Hugging Face Dataset Viewer API, concatenates the text
column, and caches it under:

```text
data/hf_cache/
```

Useful overrides:

```bash
python train.py --config configs/transformer_baseline.yaml --set hf_dataset=roneneldan/TinyStories --set hf_split=train --set hf_max_rows=5000
```

Local files still work:

```yaml
data_path: path/to/text.txt
```

For offline debugging only:

```yaml
allow_fallback_dataset: true
```

Training logs include `dataset_source`, `dataset_chars`, `train_windows`, and
`val_windows`; treat runs with the built-in fallback as smoke tests only.

## Outputs

Each training run writes:

- `runs/<name>/config.yaml`
- `runs/<name>/metrics.jsonl`
- `runs/<name>/best.pt`
- `runs/<name>/last.pt`

Metrics include validation loss, perplexity, accuracy, ECE, Brier score, and
available state statistics such as `log_var_std`, `variance_mean`,
`basis_entropy`, and `corr_std`.

## Suggested Experimental Sequence

1. Train `transformer_baseline` and confirm loss, sampling, tokens/sec, and
   checkpointing.
2. Train `hcr_kan_mean` with approximately matched dimensions and compare NTP
   loss and throughput.
3. Train `hcr_moment`; inspect variance statistics with
   `analyze_uncertainty.py`.
4. Train `hcr_density` and inspect basis entropy.
5. Train `hcr_joint_pairwise` and inspect correlation statistics.
6. Train `hcr_blockwise_joint` as the first explicit blockwise HCR
   expected-value model.

Keep `hcr_bidirectional_refinement` out of this sequence unless you explicitly
want a masked reconstruction side experiment.

## Report Template

Use `results/final_report.md` for the research write-up:

```text
1. Motivation
2. HCRNN/KAN/cortical-neuron inspiration
3. Implemented models
4. Training setup
5. Language modeling results
6. Optional denoising / masked reconstruction side results
7. Calibration and uncertainty results
8. Corruption robustness
9. Causal HCR state dynamics
10. Variance, basis, and correlation visualizations
11. What worked
12. What failed
13. Most promising next direction
14. Open questions
```

## Notes

This is deliberately a compact research skeleton. It is meant to produce real
training curves and inspectable distributional channels quickly, then leave room
for more radical HCR experiments such as local learning and richer joint-density
representations.

Paper-fidelity caveat: most trainable language models here are still
HCR-inspired approximations. The paper-direct paths are
`src/model/hcr_moments.py` for small static local densities and
`hcr_blockwise_joint` for a trainable blockwise conditional-expected-value
neuron. `hcr_blockwise_joint` carries density coefficient state through the HCR
FFN stack, but still uses sigmoid hidden normalization as a proxy and does not
yet propagate full density states through attention. KAN-based Transformer
variants are useful baselines, but they are not evidence for HCR unless they
explicitly model local joint densities.
