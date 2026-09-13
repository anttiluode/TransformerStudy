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
- Normal push/PR CI must remain cheap; the full scientific run is manual-only.
- Tests must stay much cheaper than the full experiment.

## Why in-context episodes rather than task-name tokens

The model is not told the algorithm name. Each example is a short in-context episode made of demonstrations plus one query:

```text
<x1> <SEP> <y1> <PAIR>
<x2> <SEP> <y2> <PAIR>
<x3> <SEP> <y3> <PAIR>
<xq> <SEP> <yq>
```

All demonstrations and the query in an episode use the same hidden algorithm. At evaluation time `<yq>` is withheld and generated autoregressively.

The residual at the final query `<SEP>` is therefore the primary pre-answer representation: the model has seen the demonstrations and query input but has not yet seen any query-answer token.

This permits a held-out algorithm to be presented without inventing an unseen task token.

## Shared tape and symbol space

Use a common fixed-length tape so the exact same query input can be evaluated under every algorithm.

Default Gate 0 settings:

- Symbol values: `0..7` (`modulus = 8`).
- Tape length: `4`.
- Demonstrations per episode: `3`.
- One query per episode.
- All outputs are length 4 and use the same symbol vocabulary.
- Special tokens provide separators and episode structure only; they never name tasks.

The exact constants live in one experiment configuration object.

## Trained algorithm family

Train on six distinct fixed-length transformations.

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

Outputs use only symbol tokens 0 and 1 but remain in the shared vocabulary.

### F. SWAP_PAIRS

For tape length 4:

`[x0, x1, x2, x3] -> [x1, x0, x3, x2]`

The implementation generalizes pairwise swapping to any even tape length.

These tasks share one interface while requiring identity, global permutation, ordering, modular accumulation, binary state accumulation, and local permutation.

## Held-out algorithm

Use `DELTA_MOD` only at evaluation time:

- `y[0] = x[0]`
- `y[i] = (x[i] - x[i-1]) mod 8` for `i > 0`

The frozen model receives ordinary in-context demonstrations of `DELTA_MOD`, exactly as it receives demonstrations for trained algorithms. No optimizer step, adapter, or parameter edit is allowed.

This held-out rule is related enough to the arithmetic family that a tiny model has a plausible chance to infer it, while remaining absent from training. If the model does not solve it above the declared behavioral threshold, its geometry is still reported but labeled an unsolved-novelty regime rather than evidence for algorithm invention.

## Model

A tiny decoder-only causal transformer implemented directly with PyTorch modules.

Gate 0 model preset:

- Layers: `3`
- Model width: `48`
- Attention heads: `4`
- MLP hidden width: `96`
- Learned token embeddings
- Learned positional embeddings
- Pre-layer normalization
- GELU MLP
- Causal attention mask
- Shared output projection over the vocabulary
- Dropout: `0`

The model API supports `return_residuals=True`, returning the residual stream after embeddings and after each full transformer block, for `layers + 1` residual tensors.

Layer-wise geometry is part of the experiment, so runtime reductions should first reduce analysis sample count or training steps rather than deleting layers.

## Training objective

Train autoregressively on complete in-context episodes from a balanced mixture of the six trained algorithms.

**Loss is computed only on the four query-output symbol positions.** Demonstration outputs are visible context but are not prediction targets. This makes the optimization objective explicitly meta-learning-like: infer the hidden transformation from demonstrations, then answer the query.

Training data is generated from deterministic seeded RNG streams. Evaluation and geometric-analysis examples use disjoint RNG streams and are never replayed during training.

Gate 0 default training preset:

- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `0`;
- batch size: `32`;
- training steps: `2000`;
- gradient clipping: `1.0`;
- model/training seed: `17`;
- training-data seed stream root: `1000`.

Training uses a fixed step budget rather than adaptive early stopping.

The values are calibration defaults, not scientific thresholds. If the first manual Actions run is too slow, the only allowed runtime calibration before interpreting results is to lower analysis sample counts and then training steps. Any changed preset must be written verbatim to `config.json` and `RESULTS.md`.

## Behavioral evaluation

Before geometric claims are interpreted, record exact output quality for every task.

Gate 0 evaluation uses `192` held-out episodes per task from fixed disjoint seeds.

Report:

- per-task exact-sequence accuracy;
- per-token accuracy;
- overall trained-task average;
- held-out `DELTA_MOD` exact-sequence and per-token accuracy.

The geometry analysis is always produced, but any task below the configured competence floor is marked behaviorally unresolved.

Reporting thresholds:

- trained-task competence: exact-sequence accuracy `>= 0.80`;
- held-out novelty competence: exact-sequence accuracy `>= 0.50`.

These are interpretation rules, never CI pass/fail rules.

## Residual extraction

For paired geometric analysis, generate a bank of query tapes and construct one episode per algorithm for the exact same query tape. Demonstrations are independently generated per algorithm and per episode so the task signal cannot be a shared demonstration identity.

At every layer, extract the residual vector at the final query `<SEP>` immediately before the first query-output token. This is the primary task-state representation.

Gate 0 uses two disjoint paired banks:

- map-fit bank: `192` query tapes;
- map-test bank: `192` query tapes.

An optional trajectory representation may stack residual states at generated output positions, but it is secondary and must not replace the pre-answer query-state analysis in Gate 0.

All residuals used by analysis are held out from training.

## Analysis 1: task-mode separability

For every layer:

1. Compute the mean residual vector per trained algorithm.
2. Compute pairwise cosine similarity and Euclidean distance between task means.
3. Fit a closed-form ridge one-vs-rest task classifier on the fit bank and evaluate it on the test bank.
4. Compute SVD of centered task means and of centered residual samples.

Report task-mode rank and singular spectrum rather than assuming one direction per algorithm.

A shuffled-task-label classifier is the null baseline.

## Analysis 2: inter-algorithm linear maps

For each ordered pair of trained algorithms `(A, B)` and each layer, create paired residual matrices from the same query tapes:

`H_A in R^(n x d)` and `H_B in R^(n x d)`.

Fit an affine ridge map on the fit bank:

`H_B ~= H_A @ W_AB + b_AB`

using a fixed ridge coefficient `lambda = 1e-3` after feature centering/standard numerical conditioning.

Evaluate only on the disjoint test bank.

Primary metrics:

- normalized mean-squared error;
- cosine similarity between predicted and true residuals;
- improvement over predicting the test-set mean residual of B using the fit-set B mean;
- improvement over a deterministic random-pairing baseline.

The primary scientific result is held-out generalization, not fit error.

## Analysis 3: map composition

For row-vector affine maps,

`A -> B: h_B = h_A W_AB + b_AB`

and

`B -> C: h_C = h_B W_BC + b_BC`,

so the composed map is

- `W_comp = W_AB @ W_BC`
- `b_comp = b_AB @ W_BC + b_BC`.

For triples `(A, B, C)`, compare that composition with the directly fitted `A -> C` map.

Two metrics are required:

1. normalized parameter-space disagreement;
2. held-out action error: apply the composed affine map to unseen A residuals and compare with true C residuals.

The action metric is primary because different matrices can act similarly on the occupied residual subspace.

Include a deterministic baseline in which one constituent matrix is replaced by a Gaussian random matrix rescaled to the same Frobenius norm.

## Analysis 4: random orthogonal basis scramble

Construct a deterministic random orthogonal matrix `Q` from a seeded Gaussian matrix using QR decomposition with a deterministic diagonal-sign convention.

Transform held-out residuals:

`h_scrambled = h @ Q`.

Run two fit/test probes on both original and scrambled residuals:

1. the same ridge task classifier from Analysis 1;
2. four per-output-position ridge classifiers predicting the correct query output token from the pre-answer residual.

Retrain only these tiny closed-form probes after scrambling; the transformer remains frozen.

Report the difference between original and scrambled held-out accuracy. Because orthogonal changes of basis preserve linear information, near-equality is expected and functions as a pipeline/control check rather than evidence that random transformers themselves compute the task.

Also assert numerical norm and pairwise-distance preservation in unit tests.

Gate 0 does not physically insert `Q` between transformer blocks. That is reserved for a later gate.

## Analysis 5: known-task span and held-out orthogonal energy

At each layer, form a known-task subspace from centered trained-task mean vectors. Use SVD with numerical tolerance `max(shape) * eps * largest_singular_value` to obtain an orthonormal basis `U_known`.

For a residual `h`, define `z` relative to the global trained-task mean and compute:

`rho_perp = ||z - U_known U_known^T z||^2 / (||z||^2 + eps)`.

Measure `rho_perp` for:

- every trained algorithm on held-out episodes;
- held-out `DELTA_MOD` episodes;
- a shuffled-task grouping as a null diagnostic.

Report median, mean, standard deviation, and fixed quantiles rather than only a single average.

The relevant comparison is excess orthogonal energy of `DELTA_MOD` relative to the trained-task control distribution.

No claim of novel-computation geometry is allowed unless `DELTA_MOD` also exceeds the `0.50` exact-sequence competence threshold.

## Analysis 6: relation between behavioral success and geometry

Preserve whether every evaluation episode was answered exactly correctly.

Compare `rho_perp` for correct and incorrect `DELTA_MOD` episodes. Emit this split only when both groups contain at least `20` episodes; otherwise record that the split was underpowered.

This guards against interpreting a generic failure/uncertainty direction as a novel algorithmic direction.

## Anti-cheating and null controls

Gate 0 includes all of the following:

- no task-name tokens;
- training and evaluation RNG streams separated by seed;
- map fitting and map testing use disjoint query tapes;
- random-pairing baseline for linear maps;
- mean-predictor baseline for linear maps;
- shuffled-label baseline for mode classification;
- random norm-matched map baseline for composition;
- fixed-seed orthogonal scramble with exact numerical invariance tests;
- `DELTA_MOD` absent from the training task enumeration and training sampler;
- unit tests that directly assert the held-out task cannot be sampled in training mode.

## Outputs

A full run writes `artifacts/gate0/` containing:

- `config.json` — exact model, optimizer, sample counts, and seeds;
- `metrics.json` — machine-readable behavioral and geometric metrics;
- `RESULTS.md` — compact human-readable scientific receipt;
- `model.pt` — trained model state dict;
- `task_geometry.csv` — per-layer mode/span summaries;
- `linear_maps.csv` — per-layer per-pair transfer metrics;
- `composition.csv` — per-layer triple composition metrics;
- `novelty.csv` — per-layer orthogonal-energy metrics;
- `scramble.csv` — original-vs-scrambled probe metrics;
- PNG plots for singular spectra, map transfer by layer, and held-out orthogonal energy.

The full workflow uploads the directory as a GitHub Actions artifact even when the scientific hypothesis is unsupported.

## Repository layout planned for implementation

```text
TransformerStudy/
  README.md
  pyproject.toml
  transformer_study/
    __init__.py
    config.py
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
    test_smoke.py
  .github/workflows/
    ci.yml
    gate0.yml
  docs/superpowers/specs/
    2026-09-13-multi-algorithm-residual-geometry-design.md
```

Implementation files remain small and independently testable.

## GitHub Actions design

### `ci.yml`

Triggers on pushes and pull requests.

- Python 3.11;
- install CPU PyTorch and project test dependencies;
- run unit tests;
- run an 8-step microscopic smoke experiment with batch size 4 and tiny analysis banks;
- verify deterministic shapes and a valid metrics receipt.

The smoke run validates plumbing only and makes no scientific claim.

### `gate0.yml`

The full scientific workflow is **manual-only** via `workflow_dispatch` in Gate 0. Normal pushes never start the long experiment.

Steps:

1. checkout;
2. setup Python 3.11;
3. install CPU PyTorch and project;
4. run `python -m transformer_study.experiment --preset gate0 --output artifacts/gate0`;
5. run receipt validation;
6. upload `artifacts/gate0` with `if: always()` while still allowing genuine engineering failures to mark the job red.

The experiment exits nonzero only for engineering failures such as NaNs, malformed outputs, broken invariants, or incomplete execution. Poor task accuracy, poor transfer, failed composition, or absent novelty effects are valid scientific outputs and exit successfully.

## Runtime strategy

The first full preset is exactly:

- model: 3 layers, width 48, 4 heads, MLP 96;
- training: 2000 steps, batch 32;
- behavioral evaluation: 192 episodes/task;
- map fit: 192 paired query tapes;
- map test: 192 paired query tapes;
- one training seed.

No hyperparameter sweep runs in Gate 0.

If the manual Actions run is too slow, calibration happens in this order:

1. reduce fit/test/evaluation sample counts;
2. reduce training steps;
3. reduce batch size.

Every calibration changes the saved preset and receipt. Do not add paid infrastructure or external services.

## Testing strategy

### Task tests

For fixed hand-written inputs, assert exact outputs for all seven transformations.

### Episode tests

Assert exact serialization length, query-output loss-mask positions, evaluation prompt truncation before `<yq>`, and absence of the held-out task from training sampling.

### Model tests

Assert causal shapes, deterministic eval-mode forward passes, residual count equals `layers + 1`, and logits use the shared vocabulary.

### Geometry tests

Use synthetic matrices with known transformations to test:

- affine ridge recovery;
- held-out prediction metrics;
- affine composition including bias composition;
- orthogonal matrix determinism, norm preservation, and distance preservation;
- basis-invariance of ridge probes to numerical tolerance;
- span projection and `rho_perp` edge cases;
- shuffled/random baselines.

### Receipt test

A tiny end-to-end run creates syntactically valid `config.json`, `metrics.json`, and `RESULTS.md` and all required CSV files.

## Interpretation rules

The README and generated receipt distinguish evidence levels.

Allowed examples:

- "Residuals are linearly separable by trained algorithm at layer 2."
- "The fitted A->B map generalizes better than the mean and random-pair baselines."
- "Composition is no better than the random-map control."
- "DELTA_MOD produces higher orthogonal energy, but the model does not solve DELTA_MOD, so this is not evidence of novel-computation geometry."

Disallowed unless directly supported:

- "The transformer stores algorithms as vectors."
- "Random transformations explain transformer intelligence."
- "A novel algorithm necessarily occupies an orthogonal subspace."

## Gate 0 completion criteria

Gate 0 is complete when:

1. Unit tests and smoke CI pass on GitHub Actions CPU.
2. The full manual experiment runs to completion on GitHub Actions CPU without external data or paid compute.
3. The six trained-task behavioral metrics are reported.
4. Layer-wise residual separability is reported.
5. Every ordered trained-algorithm pair has held-out linear-transfer metrics and baselines.
6. Composition metrics and random-map controls are reported.
7. Orthogonal basis-scramble task and output-readout controls are reported.
8. Known-span and held-out `DELTA_MOD` orthogonal-energy measurements are reported.
9. Correct-vs-incorrect novelty geometry is reported when sample counts permit, otherwise explicitly marked underpowered.
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
- publication-style claims before Gate 0 establishes what geometry is actually present.

Gate 0 answers one question cleanly: **when a tiny transformer must infer several different algorithms from demonstrations, what linear geometry actually appears in its shared residual stream?**