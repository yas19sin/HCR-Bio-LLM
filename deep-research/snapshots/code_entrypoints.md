# Code Entrypoints Snapshot

These are the concrete files to revisit when implementing the next changes.

## Current Repo

- `src/model/hcr_moments.py`
  - Add total-degree basis masks.
  - Add conditional variance/mode.
  - Add marginalization.
  - Add calibration choices.

- `src/model/hcr_joint_block.py`
  - Integrate reverse inference into model/eval.
  - Emit conditional coefficient vectors or density moments.
  - Add denominator diagnostics to returned state.
  - Add sparse/total-degree term masks.

- `src/model/hcr_basis.py`
  - Rename or duplicate as `fastkan_like` if keeping RBF semantics.
  - Add LayerNorm and base update to match FastKAN more closely.

- `src/model/hcr_neuron.py`
  - Keep bridge labels clear.
  - Do not describe latent `basis` as HCR mixed moments unless changed.

- `src/eval/*.py`
  - Add reverse reconstruction, conditional-density, and moment diagnostics.

## HCR Prototype Repo

- `deep-research/source-repos/joint-distribution-neuron/joint-distribution-neuron/hcrnn/basis.py`
  - Total-degree basis and orthonormality verification.

- `deep-research/source-repos/joint-distribution-neuron/joint-distribution-neuron/hcrnn/joint_density.py`
  - Joint density, marginalization, conditioning, sampling.

- `deep-research/source-repos/joint-distribution-neuron/joint-distribution-neuron/hcrnn/conditionals.py`
  - Conditional expectation, variance, mode, sampling.

- `deep-research/source-repos/joint-distribution-neuron/joint-distribution-neuron/hcrnn/network.py`
  - Forward/reverse inference, uncertainty, training objectives, pruning.

## KAN Baseline Repos

- `deep-research/source-repos/pykan/kan/KANLayer.py`
  - Faithful edge-spline KAN behavior.

- `deep-research/source-repos/pykan/kan/spline.py`
  - B-spline basis and coefficient conversion.

- `deep-research/source-repos/efficient-kan/src/efficient_kan/kan.py`
  - Practical B-spline KANLinear baseline.

- `deep-research/source-repos/fast-kan/fastkan/fastkan.py`
  - RBF, LayerNorm, base update, FastKAN attention transform.

- `deep-research/source-repos/kan-gpt/kan_gpt/model.py`
  - KAN replacement for QKV, projection, FFN, and LM head.

- `deep-research/source-repos/kansformers/model.py`
  - minGPT-style all-projection KAN replacement.

- `deep-research/source-repos/kansformers/kan.py`
  - Shape-compatible efficient KANLinear.

- `deep-research/source-repos/kat/katransformer.py`
  - KAT's grouped rational activation MLP replacement inside ViT blocks.

