# Gate 0 Design: Multi-Algorithm Residual Geometry in a Tiny Transformer

Date: 2026-09-13

## Purpose

Build the smallest controlled transformer experiment that can test whether several distinct algorithms coexist as geometric modes in a shared residual stream, whether simple linear maps transform one algorithmic mode into another, whether those maps compose, whether the geometry is basis-independent under random orthogonal scrambling, and whether a genuinely held-out rule produces a residual component outside the span of the trained algorithm family.

The experiment must run on ordinary GitHub Actions CPU. Scientific failure is allowed and should be recorded faithfully; only engineering failures should make CI red.

## Core hypothesis

The study will test a deliberately strong but falsifiable geometric picture:

1. Familiar algorithms occupy reproducible task-dependent directions or subspaces in the residual stream.
2. A linear map fitted between the residual states of two algorithms on some inputs may generalize to unseen inputs.
3. Such maps may approximately compose, so that `T_BC @ T_AB` predicts part of `T_AC`.
4. A random orthogonal change of basis should not destroy useful geometry if the representation is genuinely basis-independent.
5. A held-out algorithm inferred from demonstrations may require a residual component outside the span of the trained algorithm modes.

No individual claim is assumed true in advance.

## Gate 0 constraints

- Reference environment: GitHub Actions, Ubuntu, CPU only.
- No external datasets or model downloads.
- No pretrained weights.
- Deterministic synthetic data generated on the fly.
- PyTorch CPU is the only heavy dependency.
- Fixed small sequence lengths and a single shared token vocabulary.
- One tiny decoder-only transformer shared by all tasks.
- Full experiment target: comfortably below the GitHub Actions job timeout and designed for a practical single-job run.
- Tests must stay much cheaper than the full experiment.

## Why in-context episodes rather than task-name tokens

The model should not be told the algorithm name. Each example is a short in-context episode made of demonstrations plus one query:

```text
<x1> <SEP> <y1> <PAIR>
<x2> <SEP> <y2> <PAIR>
...
<xq> <SEP> <QUERY>
```

where all demonstrations in one episode use the same hidden algorithm.

At the query position, the model must infer which computation is active from the demonstrations. This gives a meaningful residual representation of inferred computation and permits a held-out algorithm to be presented without inventing an unseen task token.

## Shared tape and symbol space

Use a common fixed-length tape so the same input can be evaluated under every algorithm.

Default Gate 0 settings:

- Symbol values: `0..7` (`modulus = 8`).
- Tape length: `4`.
- Demonstrations per episode: `3`.
- One query per episode.
- All outputs are also length 4 and use the same symbol vocabulary.
- Special tokens provide separators and episode structure only; they do not name tasks.

The exact constants are configuration values, not scattered literals.

## Trained algorithm family

Train on six distinct fixed-length transformations:

### A. COPY

`y[i] = x[i]`

### B. REVERSE

`y = reversed(x)`

### C. SORT

`y = sorted(x)` in ascending order.

### D. CUMSUM_MOD

`y[i] = sum(x[0:i+1]) mod 8`

### E. PREFIX_PARITY

`y[i] = sum(x[j] mod 2 for j <= i) mod 2`

Outputs therefore use only symbol tokens 0 and 1, but remain in the shared vocabulary.

### F. SWAP_PAIRS

For tape length 4:

`[x0, x1, x2, x3] -> [x1, x0, x3, x2]`

The implementation should generalize pairwise swapping to any even tape length.

These tasks share the same interface while requiring different computations: identity, permutation, ordering, modular accumulation, stateful binary accumulation, and local permutation.

## Held-out algorithm

Use `DELTA_MOD` only at evaluation time:

- `y[0] = x[0]`
- `y[i] = (x[i] - x[i-1]) mod 8` for `i > 0`

The model receives ordinary in-context demonstrations of `DELTA_MOD`, exactly as it receives demonstrations for the trained algorithms, but no parameter update is allowed.

This held-out task is intentionally related enough to the trained arithmetic family that a tiny model has a plausible chance to infer it, while still being a distinct unseen rule. If the model does not solve it above the declared behavioral threshold, geometry for the held-out task is still reported but must be labeled as an unsolved-novelty regime rather than evidence for algorithm invention.

## Model

A tiny decoder-only causal transformer implemented directly with PyTorch modules.

Initial configuration:

- Layers: 4
- Model width: 64
- Attention heads: 4
- MLP hidden width: 128
- Learned token embeddings
- Learned positional embeddings
- Pre-layer normalization
- GELU MLP
- Causal attention mask
- Shared output projection over the vocabulary
- Dropout disabled for determinism and simplicity

The model API must support `return_residuals=True`, returning the residual stream after the embedding stage and after each full transformer block.

If Gate 0 runtime is too high, reduce training steps or batch size before reducing width or depth, because the layer-wise geometry is part of the experiment.

## Training objective

Train autoregressively on complete in-context episodes from a balanced mixture of the six trained algorithms.

Loss is computed only on output-symbol positions, not on demonstration inputs or structural separators. This prevents the model from wasting capacity on copying the prompt format.

Training data is generated from deterministic seeded RNG streams. Evaluation examples use disjoint RNG seeds and are never replayed during training.

Training stops at a fixed step budget. There is no adaptive early stopping in Gate 0 so runs remain easy to reproduce.

## Behavioral evaluation

Before any geometric claim is interpreted, record exact-token output accuracy for each trained algorithm on held-out episodes.

Report:

- per-task exact-sequence accuracy;
- per-token accuracy;
- overall average;
- held-out `DELTA_MOD` exact-sequence and per-token accuracy.

The geometry analysis is always produced, but any task below the configured competence floor must be marked as behaviorally unresolved.

The initial competence floor for a trained task is exact-sequence accuracy >= 0.80. This threshold is a reporting rule, not a CI failure condition.

## Residual extraction

For paired geometric analysis, generate a bank of query inputs `x` and build one episode per trained algorithm using independently generated demonstrations but the exact same query tape.

At every layer, extract the residual vector at the query decision position immediately before generation of the query output. This is the primary task-state representation.

Also support an optional trajectory representation formed by concatenating or stacking residual states at each generated output position. Gate 0 analysis should begin with the single query-state representation and only use the trajectory form as a secondary diagnostic.

All residuals and labels used by analysis are held-out from training.

## Analysis 1: task-mode separability

For every layer:

1. Compute the mean residual vector per trained algorithm.
2. Compute pairwise cosine similarity and Euclidean distance between task means.
3. Train a tiny linear task classifier on a subset of residuals and test it on disjoint residuals.
4. Compute PCA/SVD of centered task means and of all residual samples.

Report task-mode rank and singular spectrum rather than assuming one direction per algorithm.

A shuffled-task-label classifier is the null baseline.

## Analysis 2: inter-algorithm linear maps

For each ordered pair of trained algorithms `(A, B)` and each layer, create paired residual matrices from the same query inputs:

`H_A in R^(n x d)` and `H_B in R^(n x d)`.

Fit an affine ridge map on a fit split:

`H_B ~= H_A @ W_AB + b_AB`

with a small fixed ridge coefficient selected in advance.

Evaluate on unseen paired query inputs.

Primary metrics:

- normalized mean-squared error;
- cosine similarity between predicted and true residuals;
- improvement over predicting the mean residual of B;
- improvement over a random pairing baseline.

The primary scientific result is generalization to unseen inputs, not training fit.

## Analysis 3: map composition

For triples `(A, B, C)`, compare:

`W_AB @ W_BC`

with a directly fitted `W_AC`, using the appropriate affine handling for biases.

Two composition metrics are required:

1. Parameter-space disagreement after normalization.
2. Held-out action disagreement: apply the composed map to unseen A residuals and compare its predictions with true C residuals.

The action metric is primary because two different matrices can act similarly on the occupied residual subspace.

Include a baseline in which one constituent map is replaced by a random map matched in Frobenius norm.

## Analysis 4: random orthogonal basis scramble

Construct a deterministic random orthogonal matrix `Q` from a seeded Gaussian matrix using QR decomposition, correcting the sign convention for reproducibility.

Transform held-out residuals:

`h' = h @ Q`

Then train the same tiny linear task classifier/readout on scrambled residuals and compare with the unscrumbled version.

Because an orthogonal basis change preserves distances and linear separability, this experiment is explicitly a control for accidental coordinate dependence in the analysis pipeline.

Gate 0 does not physically insert `Q` between transformer blocks. A later gate may do so together with compensated read/write matrices.

## Analysis 5: known-task span and held-out orthogonal energy

At each layer, form a known-task subspace from centered trained-task mean vectors. Use SVD with a fixed numerical tolerance to obtain an orthonormal basis `U_known`.

For a residual `h`, define the centered vector `z` relative to the global trained-task mean and compute:

`rho_perp = ||z - U_known U_known^T z||^2 / (||z||^2 + eps)`

Measure `rho_perp` for:

- each trained algorithm on held-out episodes;
- held-out `DELTA_MOD` episodes;
- a shuffled/control grouping.

Report distributions, not only means.

The relevant quantity is the excess orthogonal energy of `DELTA_MOD` relative to the known-task control distribution.

No claim of novel-computation geometry is allowed unless `DELTA_MOD` is also behaviorally solved above the configured held-out competence threshold. For Gate 0, that threshold is exact-sequence accuracy >= 0.50, which is intentionally weaker than the trained-task threshold.

## Analysis 6: relation between behavioral success and geometry

For every episode, preserve whether the model answered the query correctly. Compare geometry for correct and incorrect episodes.

This prevents a misleading conclusion in which an apparently novel residual direction is merely the signature of failure or uncertainty.

For `DELTA_MOD`, report `rho_perp` separately for correct and incorrect episodes whenever both groups contain enough samples.

## Anti-cheating and null controls

Gate 0 must include the following controls:

- no task-name tokens;
- training and evaluation RNG streams separated by seed;
- paired map evaluation uses unseen query tapes;
- random-pairing baseline for linear maps;
- mean-predictor baseline for linear maps;
- shuffled-label baseline for mode classification;
- random norm-matched map baseline for composition;
- orthogonal scramble uses a fixed seed and preserves all residual samples exactly up to numerical tolerance;
- held-out algorithm is absent from training data generation and training task enumeration.

Unit tests should directly assert the last property.

## Outputs

A full run writes a self-contained directory, for example `artifacts/gate0/`, containing:

- `config.json` — exact experiment configuration and seeds;
- `metrics.json` — machine-readable training, behavioral, map, composition, scramble, and novelty metrics;
- `RESULTS.md` — compact human-readable scientific receipt;
- `model.pt` — trained tiny model state dict;
- `task_geometry.csv` — per-layer task mean / span summaries;
- `linear_maps.csv` — per-layer per-pair transfer metrics;
- `composition.csv` — per-layer triple composition metrics;
- `novelty.csv` — per-layer orthogonal-energy results;
- a small set of PNG plots for singular spectra, map transfer by layer, and held-out orthogonal energy.

The workflow uploads this directory as a GitHub Actions artifact.

## Repository layout planned for implementation

```text
TransformerStudy/
  README.md
  pyproject.toml
  transformer_study/
    __init__.py
    tasks.py
    episodes.py
    model.py
    train.py
    residuals.py
    analysis.py
    experiment.py
  tests/
    test_tasks.py
    test_episodes.py
    test_model.py
    test_analysis.py
  .github/workflows/
    ci.yml
    gate0.yml
  docs/superpowers/specs/
    2026-09-13-multi-algorithm-residual-geometry-design.md
```

Keep implementation files narrow enough that task generation, model behavior, training, and analysis can be tested independently.

## GitHub Actions design

### `ci.yml`

Runs on pushes and pull requests.

- Python 3.11
- install CPU PyTorch and project test dependencies
- run unit tests
- run a microscopic smoke training job with very few steps
- verify deterministic output shape and that the experiment command creates a valid metrics receipt

The smoke job tests plumbing only; it does not establish scientific results.

### `gate0.yml`

Runs the real Gate 0 experiment on GitHub-hosted CPU.

Triggers:

- `workflow_dispatch`
- optionally pushes to `main` when experiment code or workflow files change, if runtime remains comfortably small after calibration

Steps:

1. checkout;
2. setup Python 3.11;
3. install CPU PyTorch and project;
4. run `python -m transformer_study.experiment --preset gate0 --output artifacts/gate0`;
5. run receipt validation;
6. upload `artifacts/gate0` regardless of whether the scientific hypothesis is supported.

The experiment command must exit nonzero only for engineering failures such as NaNs, malformed outputs, broken invariants, or insufficiently completed execution. Poor algorithm accuracy or failed linear transfer must be written as results and exit successfully.

## Runtime strategy

The implementation should expose batch size, training steps, analysis sample count, and seed count in configuration.

Initial implementation target:

- one training seed for the first automated Gate 0;
- enough training steps to reach useful competence but with a hard conservative CPU budget;
- hundreds, not tens of thousands, of held-out geometry samples;
- vectorized analysis using NumPy/PyTorch linear algebra;
- no hyperparameter sweep in Gate 0.

If the first Actions run is too slow, reduce analysis sample counts first, then training steps/batch size. Do not introduce paid infrastructure or external services.

## Testing strategy

### Task tests

For fixed hand-written inputs, assert exact outputs for all seven transformations.

### Episode tests

Assert episode parse/format round trips, constant lengths, correct loss-mask positions, and absence of held-out task generation in training mode.

### Model tests

Assert causal shapes, deterministic forward passes in eval mode, residual count equals `layers + 1`, and logits use the shared vocabulary.

### Geometry tests

Use synthetic matrices with known transformations to test:

- affine ridge recovery;
- held-out prediction metrics;
- affine composition;
- orthogonal matrix construction and norm preservation;
- span projection and `rho_perp` edge cases;
- shuffled/random baselines.

### Receipt test

A tiny end-to-end run must create syntactically valid `config.json`, `metrics.json`, and `RESULTS.md`.

## Interpretation rules

The README and generated receipt must distinguish evidence levels.

Allowed examples:

- "Residuals are linearly separable by trained algorithm at layer 3."
- "The fitted A->B map generalizes better than the mean and random-pair baselines."
- "Composition is no better than the random-map control."
- "DELTA_MOD produces higher orthogonal energy, but the model does not solve DELTA_MOD, so this is not evidence of novel-computation geometry."

Disallowed examples unless directly supported:

- "The transformer stores algorithms as vectors."
- "Random transformations explain transformer intelligence."
- "A novel algorithm necessarily occupies an orthogonal subspace."

## Gate 0 completion criteria

Gate 0 is complete when:

1. Unit tests and smoke CI pass on GitHub Actions CPU.
2. The full experiment runs to completion on GitHub Actions CPU without external data or paid compute.
3. The six trained-task behavioral metrics are reported.
4. Layer-wise residual separability is reported.
5. Every ordered algorithm pair has held-out linear-transfer metrics and baselines.
6. Composition metrics and random-map controls are reported.
7. Orthogonal basis-scramble controls are reported.
8. Known-span and held-out `DELTA_MOD` orthogonal-energy measurements are reported.
9. Correct-vs-incorrect novelty geometry is reported where sample counts permit.
10. `RESULTS.md` states what succeeded, what failed, and what Gate 1 should test next without upgrading negative findings into positive claims.

The scientific hypothesis is not a completion criterion.

## Likely Gate 1, deliberately out of scope

Do not implement these in Gate 0:

- physically inserting random basis rotations between transformer blocks;
- compensated Q/K/V/OV basis transformations;
- multiple training seeds or large statistical sweeps;
- larger language models;
- learned nonlinear maps between algorithm modes;
- fine-tuning on the held-out algorithm;
- additional task families such as associative lookup or regression;
- attempting publication-style claims before Gate 0 establishes whether the geometry exists at all.

Gate 0 should answer one question cleanly: when a tiny transformer must infer several different algorithms from demonstrations, what linear geometry actually appears in its shared residual stream?