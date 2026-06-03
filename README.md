## Vibelab: Exploratory HCR Research

**⚠️ Experimental & Exploratory Code**  
This is a personal research sandbox for exploring HCR-inspired neurons. It's vibeoded—meant for fun experimentation and learning. No PRs or issues. Just for fun.

Trainable research prototype for testing joint-distribution-inspired neurons in
small language models.

The project follows `hcr_transformer_intern_project.md`: start with an
undeniable tiny Transformer baseline, then compare HCR-inspired variants that
carry expected values, hidden moments, density coefficients, pairwise features,
and bidirectional refinement dynamics.

The implementation is grounded against the local PDFs in `papers/`. See
[docs/paper_alignment.md](docs/paper_alignment.md) for the exact mapping between
the papers and the current code, including what is faithful and what is still an
approximation.

External upstream sources from Wolfram Community and the KAN/HCR GitHub repos
are summarized in [docs/upstream_grounding.md](docs/upstream_grounding.md).

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
- `hcr_bidirectional_refinement`: non-causal denoising model with iterative
  refinement and per-step loss support. This is a refinement bridge, not yet
  reverse conditioning through an explicit HCR joint-density tensor.

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
python train.py --config configs/hcr_bidirectional_refinement.yaml
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

The denoising refinement model is a separate task:

```bash
python run_benchmark_suite.py --suite fair-denoising --steps 10000
```

The older `causal`, `denoising`, and `all` suites use each model's native config
sizes and are useful for continuity, but not for parameter-matched claims.

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
2. Train `hcr_kan_mean` with approximately matched dimensions and compare loss
   and throughput.
3. Train `hcr_moment`; inspect variance statistics with
   `analyze_uncertainty.py`.
4. Train `hcr_density` and inspect basis entropy.
5. Train `hcr_joint_pairwise` and inspect correlation statistics.
6. Train `hcr_blockwise_joint` as the first explicit blockwise HCR
   expected-value model.
7. Train `hcr_bidirectional_refinement` on denoising and compare loss per
   refinement step by enabling `return_steps: true`.

## Report Template

Use `results/final_report.md` for the research write-up:

```text
1. Motivation
2. HCRNN/KAN/cortical-neuron inspiration
3. Implemented models
4. Training setup
5. Language modeling results
6. Denoising / masked reconstruction results
7. Calibration and uncertainty results
8. Corruption robustness
9. Refinement dynamics
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
