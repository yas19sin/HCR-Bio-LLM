# Project: Building a Trainable Joint-Distribution Neural Language Model

## Working title

**HCR-LLM: Toward Language Models with Joint-Distribution Neurons and Multidirectional Propagation**

## Goal

Build a working neural language model that explores the core claim behind HCRNN-style neurons:

> A neuron should not merely output a scalar activation.  
> It should represent and transform a local joint probability distribution.

This project is not just “replace an MLP with a fancy block.”

The real goal is to test whether joint-distribution neurons can become a practical foundation for a new kind of language model — one that can propagate values, uncertainties, densities, correlations, constraints, and partial information in multiple directions.

The model should train. It should generate text. But it should also expose whether the HCRNN direction can lead to capabilities that standard Transformers do not naturally have.

---

## Why this matters

Standard Transformers are incredibly powerful, but they mostly operate as deterministic point-vector machines during the forward pass:

```text
token -> vector -> attention -> MLP -> vector -> logits
```

All uncertainty is mostly implicit in hidden activations and explicit only at the final softmax.

The HCRNN paper suggests a different primitive:

```text
neuron = local joint probability model
```

Instead of asking:

```text
What scalar value should this neuron output?
```

we ask:

```text
What distribution over possible values does this neuron represent?
What correlations does it preserve?
What conditional information can be propagated forward, backward, or sideways?
```

This opens research directions that ordinary Transformer blocks are not designed for:

- multidirectional inference
- missing-value reconstruction
- uncertainty-aware hidden states
- conditional density propagation
- local alternatives to backpropagation
- constraint satisfaction inside the network
- KAN-like function learning as a special case
- probabilistic embeddings and probabilistic logits
- iterative refinement rather than single-pass prediction

---

## Core hypothesis

A language model built from joint-distribution neurons may be able to represent ambiguity, uncertainty, and correlation more directly than a standard Transformer.

This could matter for:

- reasoning under uncertainty
- fill-in-the-middle generation
- denoising corrupted context
- resolving ambiguous references
- bidirectional constraints
- self-correction
- calibrated generation
- learning with less data
- alternative training rules

The project should not assume this works.

The project should build the smallest serious system that can test it.

---

## Research stance

Do not reduce this project to “just another Transformer FFN variant.”

The HCRNN idea deserves to be tested on its own terms.

That means we should try at least three increasingly radical versions:

```text
1. HCR-inspired neurons inside a Transformer
2. HCR-style token states carrying distributions and moments
3. HCR-style multidirectional propagation / iterative inference
```

The first version gets us a working model quickly.

The second version tests whether distributional hidden states are useful.

The third version tests the actual breakthrough possibility.

---

## Papers grounding the project

Primary papers:

1. **arXiv:2405.05097**  
   *Biology-inspired joint distribution neurons based on Hierarchical Correlation Reconstruction allowing for multidirectional propagation of values and densities*  
   Jarek Duda

2. **arXiv:2605.30370**  
   *Updating the standard neuron model in artificial neural networks*  
   Cortical-cell / improved point-neuron model

3. **arXiv:2404.19756**  
   *KAN: Kolmogorov-Arnold Networks*  
   Liu et al.

The project should treat these as serious inspiration, not as decoration.

---

# Current implementation faithfulness audit

This project now has a `deep-research/` folder that checks the local papers,
Wolfram pages, and requested GitHub repos against the code. The important
distinction is:

```text
HCR = local joint density + product-basis mixed moments + conditioning
KAN = learnable univariate edge functions, usually splines
IBNN = implicit same-layer cortical-neuron coupling
```

So the current implementation should be described with these fidelity levels:

| Model / module | Current faithfulness | What it really implements |
|---|---|---|
| `src/model/hcr_moments.py` | close for small static HCR densities | shifted Legendre product basis, mixed-moment coefficient estimation, density evaluation, conditional coefficients, conditional mean/variance/mode, marginal coefficients, density-vector propagation, grid conditional density, sampling |
| `hcr_blockwise_joint` | closest trainable HCR path | explicit blockwise coefficient tensors, conditional coefficient vectors, conditional mean/variance, and density coefficient state carried between HCR FFNs; uses `sigmoid(hidden)` as a normalization proxy and still feeds point hidden values through attention |
| `hcr_moment` | HCR-inspired bridge | mean and log-variance state, not product-basis mixed moments |
| `hcr_density` | HCR-inspired bridge | normalized latent basis vector, not HCR `a_j` mixed-moment coefficients |
| `hcr_joint_pairwise` | HCR-inspired bridge | compressed pairwise/correlation features, not a dense or sparse HCR coefficient tensor |
| `hcr_bidirectional_refinement` | refinement bridge | non-causal denoising and iterative refinement, not reverse conditioning through the same HCR joint density |
| `hcr_kan_mean` | FastKAN/RBF-like baseline | radial-basis FFN approximation, not faithful spline-edge KAN |
| IBNN / cortical-cell baseline | not implemented | the cortical-neuron paper is motivation only |

The near-term fidelity gaps are:

```text
1. learned/empirical CDF normalization instead of sigmoid hidden proxies
2. reverse reconstruction evaluation for hcr_blockwise_joint
3. propagate density states through attention/projection operations, not only the HCR FFN chain
4. conditional mode/sampling in the trainable blockwise path
5. total-degree or sparse basis masks in the trainable blockwise path
6. coefficient pruning and HCR local/IB losses
7. a separate faithful KAN baseline using efficient spline KAN or a clearly named FastKAN variant
```

This means the repo is already useful, but the correct language is:

```text
most trainable models are HCR-inspired bridges;
hcr_moments.py is the literal small HCR primitive;
hcr_blockwise_joint is the first trainable paper-direct HCR expected-value path,
now carrying conditional density coefficients across its HCR FFN stack while
exposing conditional coefficients and variance as diagnostics.
```

`hcr_faithfulness_check.py` is the executable guard for this claim. It checks
basis orthonormality, local conditional/reverse inference, density-vector
propagation identities, carried blockwise density state, and the trainable
blockwise HCR state surface.

---

# Big picture architecture

We want to move from:

```text
Transformer:
  token state = vector
  neuron = scalar nonlinearity
  propagation = forward-only activations
```

toward:

```text
HCR-style model:
  token state = distributional object
  neuron = local joint distribution
  propagation = conditional update of values, densities, moments
```

A simplified distributional token state could be:

```text
state_t = {
  mean: μ_t ∈ R^d,
  variance: σ²_t ∈ R^d,
  correlations: c_t ∈ R^k,
  density_basis: b_t ∈ R^m,
  confidence / precision: p_t ∈ R^d
}
```

At minimum:

```text
state_t = (μ_t, log_var_t)
```

At more ambitious stages:

```text
state_t = (μ_t, log_var_t, low_rank_corr_t, basis_coefficients_t)
```

---

# Model family

The intern should implement a family of models, not just one model.

## Model 0: Standard tiny Transformer baseline

This is the control.

```text
embedding -> causal attention -> MLP -> logits
```

Purpose:

- prove the training loop works
- provide a serious comparison
- avoid fooling ourselves

This is not the final research object.

---

## Model 1: HCR-KAN / FastKAN-like mean neuron

This model tests the claim that HCRNN reduces to KAN-like expected-value propagation when using expected values only.

Instead of a standard MLP:

```text
W2 GELU(W1 x)
```

use a learnable edge/basis transformation:

```text
φ_i_j(x_j)
```

or a practical approximation:

```text
basis(x) -> learned combination -> output mean
```

Token state:

```text
state_t = μ_t
```

This gives the KAN-like bridge.

Current implementation note:

```text
`hcr_kan_mean` is not faithful KAN.
It is a radial-basis / FastKAN-like FFN approximation.
Faithful KAN would use learnable spline edge functions, adaptive grids, and KAN-specific diagnostics.
```

Question:

```text
Does a basis-function / expected-value neuron behave differently from a standard MLP in a language model?
```

---

## Model 2: HCR moment-state bridge

This is the first distributional-state bridge. It is useful, but it is not yet a
faithful HCR joint-density neuron because mean and variance are not product-basis
mixed moments.

Token state:

```text
state_t = {
  μ_t,
  log_var_t
}
```

The neuron transforms value and variance-proxy information.

Possible update:

```text
μ_out       = f_μ(μ_in, log_var_in)
log_var_out = f_σ(μ_in, log_var_in)
```

Residual update:

```text
μ       = μ + Δμ
log_var = stabilize(log_var + Δlog_var)
```

The model still produces logits from the mean at first:

```text
logits = LMHead(μ)
```

But it logs and evaluates variance.

Question:

```text
Does hidden uncertainty become meaningful during next-token prediction?
```

---

## Model 3: HCR-inspired joint / pairwise bridge

This is closer to the HCRNN spirit.

Token state:

```text
state_t = {
  μ_t,
  log_var_t,
  corr_t
}
```

where `corr_t` approximates selected pairwise or low-rank correlations.

Avoid full covariance:

```text
full covariance: O(d²) too expensive
low-rank covariance: O(dr)
selected pairwise moments: O(k)
```

Use either:

```text
cov ≈ diag(exp(log_var)) + U Uᵀ
```

or:

```text
corr_t = learned selected pairwise moment features
```

Question:

```text
Do correlation channels improve reconstruction, denoising, ambiguity handling, or language modeling?
```

Current implementation note:

```text
`hcr_joint_pairwise` is a compressed correlation-channel bridge.
It is not yet an HCR coefficient tensor and does not implement conditional density semantics.
```

---

## Model 3b: Blockwise HCR joint-density neuron

This is the current closest trainable implementation to the HCR paper.

Partition hidden dimensions into small blocks:

```text
d_model = num_blocks * block_size
```

Each block stores coefficients for a local joint density over input block
variables `x` and output block variables `y`:

```text
rho(x, y) = sum_{i,j} a_{i,j} prod_k f_{i_k}(x_k) prod_l f_{j_l}(y_l)
```

Forward propagation substitutes sigmoid-normalized hidden input values, normalizes
`rho(y | x)`, and reads the first output moment:

```text
E[y | x] = 1/2 + a_1(y | x) / sqrt(12)
```

The same coefficients can be transposed for reverse conditioning:

```text
E[x | y]
```

Current implementation caveats:

```text
normalization is sigmoid(hidden), not learned/empirical CDF normalization
density coefficient vectors are carried between HCR FFNs, but attention still uses point hidden values
reverse conditioning exists at module level but is not yet an evaluated LM objective
conditional entropy, mode, and sampling are not yet in the trainable block
```

Question:

```text
Does an explicit local joint-density coefficient tensor learn useful conditional structure inside a Transformer FFN?
```

---

## Model 4: Bidirectional refinement bridge

This is the breakthrough-oriented model.

Instead of a single left-to-right forward pass, allow iterative refinement:

```text
initial token distributions
forward causal pass
backward / side update through context
refine token distributions
predict
```

This can be tested first on denoising or masked modeling before causal LM.

Example task:

```text
Input:  The [MASK] was sitting on the mat.
Target: cat
```

But unlike BERT, the hidden state explicitly carries uncertainty and correlations.

Current implementation note:

```text
`hcr_bidirectional_refinement` is currently a non-causal denoising/refinement model using HCR-inspired density bridge blocks.
It is not yet HCR reverse inference unless the refinement step is tied to the same explicit joint-density coefficients.
```

Possible loop:

```python
state = initialize_distribution(tokens)

for step in range(num_refinement_steps):
    messages = hcr_message_passing(state, mask)
    state = hcr_update(state, messages)

logits = readout(state.mean, state.variance, state.corr)
```

Question:

```text
Can multidirectional propagation solve tasks that are awkward for a strictly causal Transformer?
```

---

## Model 5: Local learning / non-backprop experiment

This is optional and risky, but should not be dismissed.

HCRNN motivates alternative training because neurons model local distributions.

Try a local auxiliary objective:

```text
Each HCR block predicts / reconstructs neighboring hidden states.
Each block learns conditional moments locally.
Global LM loss still exists.
```

Possible local losses:

```text
reconstruct masked input features
predict next-layer mean
predict uncertainty from corruption
match empirical moments in mini-batch
contrastive score for correct vs corrupted context
```

Question:

```text
Can local distributional objectives reduce dependence on pure end-to-end backprop?
```

This is a stretch goal, but it belongs in the project because it tests the actual philosophical promise of the papers.

---

# What “working model” means

A working model must:

```text
train end-to-end
show decreasing loss
generate text or reconstruct masked text
save and load checkpoints
log distributional statistics
compare against baselines
produce a short report
```

A working model does not need to beat GPT.

A working model must make the HCR idea experimentally visible.

---

# Repository structure

```text
hcr-llm/
  README.md
  requirements.txt
  train.py
  sample.py
  eval.py
  analyze_uncertainty.py
  configs/
    transformer_baseline.yaml
    hcr_kan_mean.yaml
    hcr_moment.yaml
    hcr_density.yaml
    hcr_joint_pairwise.yaml
    hcr_blockwise_joint.yaml
    hcr_bidirectional_refinement.yaml
    smoke_faithful_hcr.yaml
  src/
    data.py
    tokenizer.py
    model/
      baseline_transformer.py
      hcr_state.py
      hcr_basis.py
      hcr_moments.py
      hcr_neuron.py
      hcr_joint_block.py
      hcr_lm.py
      hcr_ffn.py
      hcr_attention.py
      hcr_refinement.py
      lm_head.py
    training/
      trainer.py
      losses.py
      local_losses.py
      logging.py
    eval/
      calibration.py
      corruption.py
      denoising.py
      generation.py
      moment_analysis.py
  notebooks/
    01_training_curves.ipynb
    02_uncertainty_analysis.ipynb
    03_correlation_analysis.ipynb
    04_refinement_dynamics.ipynb
  results/
    README.md
```

---

# Phase 1: Make the baseline undeniable

Before inventing anything, implement a tiny GPT-style model.

Recommended first dataset:

```text
Tiny Shakespeare
```

Then optionally:

```text
WikiText-2
enwik8 subset
character-level Wikipedia subset
small code corpus
```

The baseline should support:

```text
causal next-token prediction
sampling
validation loss
checkpointing
parameter count
tokens/sec
```

Baseline architecture:

```text
token embedding
positional embedding or RoPE
causal self-attention
MLP
RMSNorm or LayerNorm
LM head
```

Exit criteria:

```text
baseline trains
loss decreases
can overfit a tiny batch
sample generation works
```

---

# Phase 2: HCR-KAN expected-value neuron

Implement a neuron that acts like expected-value propagation through basis functions.

This connects HCRNN to KAN-like behavior.

A practical version:

```python
class HCRKANNeuron(nn.Module):
    def __init__(self, d_model, d_hidden, n_basis):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_hidden)
        self.centers = nn.Parameter(torch.randn(n_basis))
        self.widths = nn.Parameter(torch.ones(n_basis))
        self.coeffs = nn.Parameter(torch.randn(d_hidden, n_basis))
        self.out_proj = nn.Linear(d_hidden, d_model)

    def basis(self, x):
        # x: [B, T, H]
        # returns basis-expanded features
        z = x.unsqueeze(-1)
        c = self.centers.view(1, 1, 1, -1)
        w = F.softplus(self.widths).view(1, 1, 1, -1) + 1e-4
        return torch.exp(-((z - c) ** 2) / (2 * w ** 2))

    def forward(self, x):
        h = self.in_proj(x)
        b = self.basis(h)
        h2 = torch.einsum("bthk,hk->bth", b, self.coeffs)
        return self.out_proj(h2)
```

Compare this against a normal MLP.

Important:

```text
Match parameter count approximately.
Track compute.
Do not overclaim.
```

Exit criteria:

```text
HCR-KAN model trains
loss is comparable to baseline
basis activations are inspectable
```

---

# Phase 3: HCR moment neuron

Now introduce actual distributional state.

Create a class:

```python
@dataclass
class HCRState:
    mu: torch.Tensor
    log_var: torch.Tensor | None = None
    corr: torch.Tensor | None = None
    basis: torch.Tensor | None = None
```

Moment neuron:

```python
class HCRMomentBlock(nn.Module):
    def forward(self, state: HCRState) -> HCRState:
        mu = state.mu
        log_var = state.log_var

        if log_var is None:
            log_var = self.init_logvar(mu)

        features = torch.cat([mu, log_var], dim=-1)

        delta_mu = self.mu_update(features)
        delta_logvar = self.var_update(features)

        new_mu = mu + delta_mu
        new_logvar = stabilize(log_var + delta_logvar)

        return HCRState(mu=new_mu, log_var=new_logvar)
```

Stabilization:

```python
def stabilize_logvar(log_var):
    return torch.clamp(log_var, min=-8.0, max=4.0)
```

But do not let clipping become the whole solution. Also test:

```text
tanh-scaled updates
softplus variance
precision instead of variance
normalization of log_var
variance dropout
```

Exit criteria:

```text
mean+variance model trains
variance does not trivially collapse
variance correlates with prediction difficulty or corruption
```

---

# Phase 4: Density / basis coefficients

The HCRNN paper is about local joint distributions, not just Gaussian variance.

So add a density-basis representation.

For each hidden dimension or hidden block, represent density using basis coefficients:

```text
p(x) ≈ Σ_i a_i φ_i(x)
```

Practical implementation:

```text
basis coefficients per token
small number of basis functions
shared basis across dimensions
learned or fixed centers
```

Token state:

```text
state_t = {
  μ_t,
  log_var_t,
  basis_coeff_t
}
```

The block updates:

```text
basis_coeff_next = normalize_basis_update(f(μ, log_var, basis_coeff))
```

Possible constraints:

```text
softmax over basis coefficients
softplus + normalization
signed coefficients with penalty for invalid density
```

Questions:

```text
Do density coefficients carry useful information beyond mean and variance?
Can they represent multimodal ambiguity?
Can they help masked reconstruction?
```

This is where the project moves closer to the HCRNN paper rather than generic uncertainty modeling.

Current implementation note:

```text
`hcr_density` is only a bridge model.
Its latent basis vector is normalized and inspectable, but it is not yet the HCR paper's product-basis mixed-moment coefficient table.
The literal coefficient mechanics live in `src/model/hcr_moments.py`, and the trainable blockwise version starts in `hcr_blockwise_joint`.
```

---

# Phase 5: Pairwise / low-rank joint distribution features

Full joint distributions are impossible at scale, but HCR is hierarchical. So test compressed joint structure.

Options:

## Option A: Low-rank covariance

```text
cov_t ≈ diag(exp(log_var_t)) + U_t U_tᵀ
```

where:

```text
U_t ∈ R^{d × r}
```

But storing `U_t` per token can be expensive.

Use compressed features:

```text
corr_t ∈ R^r
```

and decode when needed.

## Option B: Blockwise joint neurons

Partition hidden dimensions:

```text
d_model = num_blocks * block_size
```

Each block models a small local joint distribution:

```text
block_size = 4, 8, or 16
```

This is closer to the HCR idea:

```text
small joint models composed hierarchically
```

## Option C: Selected pairwise moments

Learn a small set of pairwise interactions:

```text
m_ij = E[x_i x_j]
```

Use a learned projection to choose interactions:

```text
corr = W_pair features
```

Question:

```text
Which joint approximation gives the most benefit per FLOP?
```

---

# Phase 6: HCR-style attention or message passing

Attention already moves information between tokens.

But HCR-style propagation should move not only values, but also uncertainty, density, and constraints.

Start by modifying attention to propagate distributions:

```text
attention over means gives value messages
attention over precision gives confidence messages
attention over variance gives uncertainty messages
```

Possible update:

```python
attn_weights = softmax(Q_mu K_mu^T / sqrt(d))

mu_msg = attn_weights @ V_mu
var_msg = attn_weights @ V_var
precision_msg = attn_weights @ V_precision

state = hcr_update(state, mu_msg, var_msg, precision_msg)
```

Then test bidirectional or non-causal attention for masked reconstruction.

This gives a concrete HCR-style bridge:

```text
Transformer attention = message routing
HCR neuron = message interpretation as conditional distribution update
```

---

# Phase 7: Multidirectional refinement

This phase tests the most important HCRNN promise.

Instead of a single pass:

```text
input -> hidden -> output
```

use iterative inference:

```text
state_0 = initialize distributions

for k in range(K):
    state_k+1 = HCRRefinementBlock(state_k, observed_tokens, mask)

output = readout(state_K)
```

Tasks:

```text
masked token reconstruction
fill-in-the-middle
corrupted text denoising
causal LM with refinement before decoding
```

The model should support:

```text
left-to-right causal mode
bidirectional denoising mode
fill-in-the-middle mode
```

This may be where HCR-style neurons show advantages first.

---

# Phase 8: Local training objectives

End-to-end backprop is allowed.

But the project should also test whether local distributional objectives help.

Add auxiliary losses:

## Moment reconstruction loss

Corrupt hidden states and ask the HCR block to reconstruct their moments:

```text
state_clean -> corrupt -> reconstruct μ and log_var
```

## Neighbor prediction loss

Each token distribution predicts neighboring token distributions:

```text
state_t predicts state_{t-1}, state_{t+1}
```

## Conditional consistency loss

Forward and backward predictions should agree:

```text
p(x_i | left context) ≈ p(x_i | right context)
```

## Empirical moment matching

Within a batch or sequence window, predicted moments should match empirical statistics:

```text
predicted mean ≈ observed mean
predicted variance ≈ observed variance
```

This tests whether the HCR-style neurons can be trained partly through local statistical consistency.

---

# Evaluation

Do not evaluate only perplexity.

Perplexity matters, but it may not reveal the main advantage.

Evaluate:

## 1. Standard language modeling

```text
train loss
validation loss
bits per character/token
sample quality
```

## 2. Calibration

```text
Expected Calibration Error
Brier score
entropy vs correctness
variance vs error correlation
```

## 3. Corruption robustness

Evaluate on:

```text
random character replacement
random token masking
deleted tokens
noisy prefix
ambiguous context
```

Metrics:

```text
clean loss
corrupted loss
degradation ratio
reconstruction accuracy
```

## 4. Missing-value reconstruction

Because HCRNN is about multidirectional propagation, test:

```text
given left and right context, reconstruct missing middle
given partial word, reconstruct full word
given corrupted sentence, recover original
```

## 5. Ambiguity representation

Create synthetic ambiguous tasks:

```text
"The bank was near the ___"
river/money ambiguity
pronoun resolution
homographs
ambiguous arithmetic symbols
```

Measure whether variance/density channels spike during ambiguity.

## 6. Refinement dynamics

For iterative models, log:

```text
loss after refinement step 0
loss after refinement step 1
loss after refinement step 2
...
```

A real result would be:

```text
prediction improves with refinement steps
uncertainty decreases when context resolves ambiguity
```

## 7. Compute

Always report:

```text
parameter count
tokens/sec
GPU memory
training wall-clock
FLOPs estimate if possible
```

---

# Key plots

The intern should produce:

```text
training loss curves
validation loss curves
variance mean/std over training
variance vs token error
calibration reliability diagram
clean vs corrupted validation loss
refinement step vs loss
basis coefficient heatmaps
correlation feature heatmaps
sample generations
```

---

# Config examples

## Transformer baseline

```yaml
model_type: transformer_baseline
dataset: tiny_shakespeare
context_length: 128
batch_size: 32
d_model: 128
n_layers: 4
n_heads: 4
dropout: 0.1
learning_rate: 3e-4
max_steps: 5000
eval_interval: 250
```

## HCR-KAN mean

```yaml
model_type: hcr_kan_mean
dataset: tiny_shakespeare
context_length: 128
batch_size: 32
d_model: 128
n_layers: 4
n_heads: 4
n_basis: 16
dropout: 0.1
learning_rate: 3e-4
max_steps: 5000
eval_interval: 250
```

## HCR moment

```yaml
model_type: hcr_moment
dataset: tiny_shakespeare
context_length: 128
batch_size: 32
d_model: 128
n_layers: 4
n_heads: 4
state_channels:
  mean: true
  log_var: true
variance_update: residual
variance_stabilization: tanh_clamp
dropout: 0.1
learning_rate: 3e-4
max_steps: 5000
eval_interval: 250
```

## HCR joint pairwise

```yaml
model_type: hcr_joint_pairwise
dataset: tiny_shakespeare
context_length: 128
batch_size: 32
d_model: 128
n_layers: 4
n_heads: 4
state_channels:
  mean: true
  log_var: true
  pairwise: true
pairwise_rank: 8
block_joint_size: 8
learning_rate: 3e-4
max_steps: 10000
eval_interval: 250
```

## HCR blockwise joint

```yaml
model_type: hcr_blockwise_joint
dataset: tiny_shakespeare
context_length: 128
batch_size: 32
d_model: 128
n_layers: 4
n_heads: 4
hcr_block_size: 2
hcr_degree: 2
learning_rate: 3e-4
max_steps: 10000
eval_interval: 250
```

## HCR bidirectional refinement

```yaml
model_type: hcr_bidirectional_refinement
dataset: tiny_shakespeare
task: denoising
context_length: 128
mask_probability: 0.15
refinement_steps: 4
d_model: 128
n_layers: 4
n_heads: 4
state_channels:
  mean: true
  log_var: true
  basis_coefficients: true
n_basis: 16
learning_rate: 3e-4
max_steps: 10000
eval_interval: 250
```

---

# Implementation milestones

## Milestone 1: baseline model trains

Deliver:

```text
tiny GPT
training loop
sampling
checkpointing
evaluation
```

Success:

```text
loss decreases
tiny batch overfits
samples become structured
```

---

## Milestone 2: HCR-KAN expected-value neuron

Deliver:

```text
basis-function FFN
KAN-like expected-value propagation
comparison to MLP
basis visualization
```

Success:

```text
model trains
results table vs baseline
```

---

## Milestone 3: HCR moment state

Deliver:

```text
HCRState(mu, log_var)
moment block
variance propagation
variance logging
```

Success:

```text
model trains without collapse/explosion
variance carries signal
```

---

## Milestone 4: HCR density basis

Deliver:

```text
basis coefficients
density-inspired state update
coefficient normalization
basis heatmap visualization
```

Success:

```text
density channel trains
coefficients are nontrivial
```

---

## Milestone 5: joint / correlation features

Deliver:

```text
blockwise joint neurons or low-rank correlation channels
correlation logging
pairwise moment analysis
```

Success:

```text
correlation features change with context
at least one task benefits or reveals failure clearly
```

## Milestone 5b: explicit HCR blockwise fidelity

Deliver:

```text
blockwise HCR coefficient tensor
conditional expected-value propagation
reverse reconstruction evaluation
denominator stability diagnostics
conditional variance or density-vector output
```

Success:

```text
forward and reverse conditioning work on synthetic data
the same coefficient tensor supports both directions
HCR-specific metrics are logged, not just LM loss
```

---

## Milestone 6: multidirectional inference

Deliver:

```text
masked reconstruction task
bidirectional HCR refinement block
loss per refinement step
```

Success:

```text
refinement improves predictions
uncertainty decreases when ambiguity resolves
```

---

## Milestone 7: local distributional losses

Deliver:

```text
moment matching loss
neighbor prediction loss
conditional consistency loss
ablation study
```

Success:

```text
local loss improves stability, calibration, robustness, or sample efficiency
```

---

# Recommended first two weeks

## Week 1: build the trainable skeleton

Day 1:

```text
repo setup
data pipeline
tiny GPT baseline
overfit tiny batch
```

Day 2:

```text
training configs
checkpointing
sampling
logging
```

Day 3:

```text
HCR-KAN mean neuron
run against baseline
```

Day 4:

```text
HCRState class
mean+variance block
variance statistics
```

Day 5:

```text
clean evaluation
corruption evaluation
first plots
```

## Week 2: push into actual HCR territory

Day 6:

```text
basis-density coefficients
normalization choices
visualizations
```

Day 7:

```text
blockwise joint neurons or low-rank pairwise moments
```

Day 8:

```text
masked reconstruction task
bidirectional refinement loop
```

Day 9:

```text
calibration metrics
variance/error analysis
refinement dynamics
```

Day 10:

```text
write short report
recommend which path looks most promising
```

---

# Important design principles

## 1. Keep the breakthrough path open

Do not prematurely reduce HCRNN to “variance in an MLP.”

Mean and variance are only the first approximation.

The project should leave room for:

```text
basis densities
joint block distributions
pairwise correlations
multidirectional propagation
local learning
iterative inference
```

## 2. Still make something that trains

Ambition does not mean chaos.

Every experimental model should be runnable from config.

Every model should produce:

```text
loss curve
checkpoint
sample/reconstruction
metrics
```

## 3. Compare fairly, but do not worship the baseline

The Transformer baseline matters.

But the HCR model may show value on metrics other than validation loss:

```text
calibration
robustness
ambiguity handling
refinement
missing data reconstruction
sample efficiency
```

## 4. Look for signs of life

A breakthrough rarely appears first as “beats baseline perplexity.”

Signs of life include:

```text
uncertainty spikes on ambiguous tokens
refinement steps improve predictions
density coefficients separate different contexts
correlation features track long-range dependencies
local losses improve stability
model degrades gracefully under corruption
```

These are important findings even if perplexity is not yet better.

---

# Failure modes to investigate, not dismiss

## Variance collapse

If variance goes to a constant, ask:

```text
Is the task too easy?
Is the loss ignoring uncertainty?
Do we need corruption or masked modeling?
Do we need uncertainty-aware logits?
```

## Density coefficients unused

If basis coefficients become flat, ask:

```text
Are they connected to the loss?
Do they need an auxiliary reconstruction objective?
Do they need a bottleneck?
Are basis functions badly initialized?
```

## Correlation features noisy

If pairwise channels look random, ask:

```text
Is the rank too small?
Is the task too local?
Should pairwise features be blockwise?
Do we need a synthetic correlation task first?
```

## Worse perplexity

If perplexity is worse, ask:

```text
Did the HCR model trade perplexity for calibration or robustness?
Is compute budget equal?
Is optimization harder?
Does the architecture need a different task?
```

Do not treat “worse validation loss on Tiny Shakespeare” as proof the idea is bad.

---

# Synthetic tasks that may reveal HCR advantages

Before or alongside language modeling, build small tests where joint distributions matter.

## Ambiguous XOR

Input contains partial evidence. Correct output depends on interaction between variables.

## Missing variable reconstruction

Given correlated variables with some missing, reconstruct the missing one.

## Noisy sequence denoising

Input is corrupted. Model must infer original sequence.

## Bidirectional arithmetic

Example:

```text
A + B = C
```

Given any two, infer the third.

This directly tests multidirectional propagation.

## Constraint satisfaction strings

Example:

```text
open brackets must close
variables must agree
copy constraints across distance
```

Evaluate whether iterative HCR refinement improves constraint satisfaction.

---

# Possible breakthrough demos

The intern should aim for one of these demos.

## Demo A: uncertainty tracks ambiguity

Show that variance/density channels spike in ambiguous contexts and drop after disambiguating evidence.

Example:

```text
"The bat flew out of the..."
"The bat hit the..."
```

## Demo B: refinement improves reconstruction

Show masked-token predictions improving over refinement steps.

```text
step 0: uncertain
step 1: plausible
step 2: correct
```

## Demo C: multidirectional arithmetic

Train on equations.

At test time:

```text
2 + ? = 5
? + 3 = 5
2 + 3 = ?
```

The same model handles all directions.

## Demo D: corruption robustness

Show the HCR model losing less performance than baseline under noisy context.

## Demo E: local learning helps

Show that a local moment-matching or reconstruction loss improves stability or sample efficiency.

---

# Final report template

```text
1. Motivation
2. Summary of HCRNN/KAN/cortical-neuron inspiration
3. Implemented models
4. Training setup
5. Language modeling results
6. Denoising / masked reconstruction results
7. Calibration and uncertainty results
8. Corruption robustness
9. Refinement dynamics
10. Visualizations of variance, basis coefficients, correlations
11. What worked
12. What failed
13. Most promising next direction
14. Open questions
```

Results table:

| Model | Params | Task | Val loss | ECE | Corrupted loss | Refinement gain | Tokens/sec | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Transformer | | causal LM | | | | | | |
| HCR-KAN mean | | causal LM | | | | | | |
| HCR moment | | causal LM | | | | | | |
| HCR density | | denoising | | | | | | |
| HCR joint | | denoising | | | | | | |
| HCR refinement | | masked / FIM | | | | | | |

---

# Definition of success

## Minimum success

```text
A trainable model exists.
It includes at least one HCR-inspired neuron.
It runs on a real dataset.
It logs uncertainty or moment statistics.
```

## Good success

```text
Mean+variance or basis-density states train stably.
The model shows meaningful uncertainty, calibration, or robustness behavior.
```

## Great success

```text
A multidirectional HCR refinement model works on masked reconstruction, denoising, or equation-style inference.
Refinement improves predictions over steps.
```

## Breakthrough signal

```text
The model demonstrates a capability that a similarly sized causal Transformer does not naturally show:
  - multidirectional inference
  - better uncertainty handling
  - graceful degradation under corruption
  - local distributional learning
  - useful hidden density/correlation structure
```

---

# Final instruction

Do not merely build a conservative Transformer variant.

Try to find out whether the HCRNN papers are pointing toward a genuinely different computational primitive.

The model must train, but the project should remain open to surprise.

If something weird works, investigate it.

If a distribution channel behaves unexpectedly, plot it.

If the model is worse on perplexity but better at uncertainty or reconstruction, that may be the interesting result.

The purpose is not just to get a benchmark score.

The purpose is to test whether joint-distribution neurons can become a real path toward language models beyond standard Transformers.
