# Paper Notes: Updating the Standard Neuron Model / IBNN

Local source: `papers/2605.30370v2.pdf`  
Extracted text: `deep-research/papers/2605.30370v2.txt`

## Core Claim

The paper proposes replacing the standard artificial neuron with a cortical-cell-inspired implicit-bias neuron. The resulting networks are called implicit bias neural networks, or IBNNs.

This is related motivation for moving beyond point neurons, but it is not HCR and not KAN.

## Mechanism

The IBNN layer output is implicit: neuron outputs are coupled through a nonlinear bias term. Computing a layer requires solving a coupled fixed-point or optimization problem, rather than applying a standard affine map followed by an activation.

Implementation consequence:

- An IBNN baseline would be a separate architecture family.
- It should not be mixed into HCR terminology unless we explicitly combine implicit dynamics with joint-density coefficients.

## Claimed Benefits

The paper reports and analyzes:

- Better sample efficiency on image datasets.
- Increased robustness to input perturbations and adversarial attacks.
- More expressivity with the same number of trainable parameters.
- Faster learning in terms of epochs to reach a target validation accuracy.
- Less memorization under label corruption, connected to lower input-loss curvature.

Implementation consequence:

- The right comparison metrics include sample efficiency, robustness/corruption, curvature or stability proxies, and label-noise performance.
- These metrics are also relevant for HCR-Bio-LLM because HCR claims should not rest only on next-token loss.

## Methods and Cost

The paper's implementation requires solving implicit layer outputs and handling additional computational burden. It discusses forward fixed-point calculation, custom implementation in PyTorch, and warmup from a standard model surrogate.

Implementation consequence:

- IBNN is likely expensive for an LM prototype.
- If used, start as a small MLP or image/text-classification baseline, not inside every transformer block.

## Relationship to HCR-Bio-LLM

Relevant:

- Biological inspiration for richer neuron models.
- Robustness and sample-efficiency evaluation.
- Same-layer coupling as a different route to richer computation.

Not directly relevant:

- Product-basis density coefficients.
- Conditional density inference.
- KAN-like edge functions.
- Transformer density embeddings.

Bottom line:

IBNN belongs in the "related bio-inspired neuron" section and as a possible future baseline. It should not be used as evidence that HCR or KAN mechanisms are correct.

