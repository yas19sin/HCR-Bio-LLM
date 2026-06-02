# Source Manifest

Generated for the deep research pass on 2026-06-02.

## Local Papers

The paper PDFs live in the repository `papers/` folder. Extracted text was written under `deep-research/papers/` for line-searchable review.

| PDF | Extracted text | Pages | Characters | Primary use |
| --- | --- | ---: | ---: | --- |
| `papers/2404.19756v5.pdf` | `deep-research/papers/2404.19756v5.txt` | 50 | 135194 | KAN paper grounding |
| `papers/2405.05097v8.pdf` | `deep-research/papers/2405.05097v8.txt` | 11 | 59202 | HCRNN paper grounding |
| `papers/2605.30370v2.pdf` | `deep-research/papers/2605.30370v2.txt` | 26 | 82596 | cortical / IBNN grounding |

Extraction manifest: `deep-research/papers/paper_extract_manifest.json`.

## Web Pages

| Source | URL | What was verified |
| --- | --- | --- |
| Wolfram Community HCR post | https://community.wolfram.com/groups/-/m/t/3017754 | Public page title, author, staff-pick metadata, and notebook attachment. The full notebook body is not exposed in public HTML. |
| Wolfram Community HCRNN post | https://community.wolfram.com/groups/-/m/t/3241700 | Public page title, author, image caption about HCR neuron/neural network with local joint-distribution model, and notebook attachment. The full notebook body is not exposed in public HTML. |

## Cloned Source Repos

All requested GitHub repos were shallow-cloned under `deep-research/source-repos/`. Git safe-directory restrictions prevented normal `git rev-parse`, so commit hashes were read from local `.git/HEAD` and refs.

| Repo | Local path | Commit | Primary use |
| --- | --- | --- | --- |
| https://github.com/vfd-org/joint-distribution-neuron | `deep-research/source-repos/joint-distribution-neuron` | `9298614a59d1256e92e496271daf863abba90cc4` | HCRNN prototype mechanics and tests |
| https://github.com/kindxiaoming/pykan | `deep-research/source-repos/pykan` | `ecde4ec3274d3bef1ad737479cf126aed38ab530` | Original KAN implementation details |
| https://github.com/Blealtan/efficient-kan | `deep-research/source-repos/efficient-kan` | `7b6ce1c87f18c8bc90c208f6b494042344216b11` | Memory-efficient B-spline KAN |
| https://github.com/ZiyaoLi/fast-kan | `deep-research/source-repos/fast-kan` | `17b65401c252334fffb5e63c9852dd8316d29e69` | RBF / FastKAN variant |
| https://github.com/adityang/kan-gpt | `deep-research/source-repos/kan-gpt` | `0c6e4c2582d9a0e23c612c3b695846d4caceac1c` | GPT with KAN projections |
| https://github.com/akaashdash/kansformers | `deep-research/source-repos/kansformers` | `e58d5bc28a5fba09a6996454f6a121529bcf38f3` | minGPT-style KAN replacement |
| https://github.com/Adamdad/kat | `deep-research/source-repos/kat` | `d254de7c14b6c050bd00cac3689b0a5614659a7f` | scalable transformer KAN reference |

## Trust Rules for This Folder

- Papers are treated as primary conceptual sources.
- Local cloned code is treated as primary implementation evidence.
- GitHub README claims are useful but weaker than code paths.
- Wolfram public pages confirm context and attachments, but not full notebook internals unless notebooks are separately downloaded or opened.

