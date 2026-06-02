# Deep Research Index

This folder is a grounded research notebook for the HCR-Bio-LLM direction. It is intentionally source-first: the local papers in `papers/`, the requested Wolfram posts, and the requested GitHub repos are treated as primary evidence.

## What is in scope

- HCR / HCRNN as local joint-density modeling with mixed moments, conditional inference, density propagation, and bidirectional inference.
- KAN as learnable univariate edge functions, mostly spline-based in the original paper, with efficient, RBF, GPT, and transformer variants as baselines.
- IBNN / cortical-neuron work as related motivation for replacing standard point neurons, not as an HCR or KAN mechanism.

## Source Map

- `papers/2405-hcrnn-notes.md`: notes from `papers/2405.05097v8.pdf`.
- `papers/2404-kan-notes.md`: notes from `papers/2404.19756v5.pdf`.
- `papers/2605-ibnn-notes.md`: notes from `papers/2605.30370v2.pdf`.
- `sources/source_manifest.md`: exact sources, cloned repo commits, and local extraction files.
- `sources/wolfram-notes.md`: what the public Wolfram pages expose and what they do not expose.
- `sources/joint-distribution-neuron-notes.md`: local code review of the HCRNN prototype repo.
- `sources/kan-family-repo-notes.md`: local code review of pykan, efficient-kan, fast-kan, kan-gpt, kansformers, and kat.
- `snapshots/code_entrypoints.md`: concrete source files to revisit when implementing.
- `synthesis/hcr-vs-kan-vs-ibnn.md`: conceptual comparison and boundaries.
- `synthesis/implementation-gap-audit.md`: what this repo already implements vs what remains.
- `synthesis/architecture-roadmap.md`: recommended design path.
- `synthesis/experiments.md`: experiments and metrics needed to stay honest.
- `synthesis/completion-checklist.md`: source-by-source coverage checklist.

## Current Bottom Line

The faithful HCR path is not "put KAN in a Transformer." It is:

1. Normalize variables into a stable `[0, 1]` coordinate system.
2. Represent a local joint density with product-basis mixed-moment coefficients.
3. Condition by substituting known variables and normalizing.
4. Propagate expected values, conditional distributions, or density moment vectors.
5. Use the same local joint model for reverse inference, uncertainty, and sampling.

KAN is useful as a comparison family because HCR pairwise expected-value propagation can reduce to a KAN-like sum of univariate functions, but KAN itself does not model a local joint density unless we explicitly add that machinery.

