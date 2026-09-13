# TransformerStudy

A small controlled study of **algorithms inside one transformer**.

Gate 0 asks a deliberately narrow question: when a tiny decoder-only transformer has to infer several different sequence transformations from in-context demonstrations, what linear geometry appears in its shared residual stream?

The experiment does **not** assume the hypothesis is true. Low task accuracy, poor linear transfer, failed map composition, or no held-out novelty effect are valid scientific results and are written to the receipt rather than treated as CI failures.

## Gate 0 tasks

The model is trained from scratch on six hidden transformations over the same four-symbol tape and the same vocabulary:

- `COPY`
- `REVERSE`
- `SORT`
- `CUMSUM_MOD`
- `PREFIX_PARITY`
- `SWAP_PAIRS`

The model is never given a task-name token. Each episode contains three input/output demonstrations followed by one query input. Training loss applies only to the four query-output symbols.

`DELTA_MOD` is held out completely from training. At evaluation time it is presented only through demonstrations, with the frozen model.

## What Gate 0 measures

After training, the model is frozen. The study extracts the residual vector at the final query separator, immediately before any query-answer token is generated, and measures:

1. layer-wise linear separability of the six trained tasks;
2. held-out affine maps `T_A->B` between paired residual states from the same query tapes;
3. approximate composition `T_B->C(T_A->B(h))` versus direct `T_A->C`;
4. a fixed random orthogonal scramble followed by fresh tiny linear probes;
5. the fraction of residual energy outside the span of trained-task means for `DELTA_MOD`;
6. whether the held-out geometry differs between correct and incorrect `DELTA_MOD` episodes.

Every analysis has a null/control: shuffled task labels, random pairings, mean prediction, norm-matched random maps, or the coordinate-preserving orthogonal scramble.

## Interpretation guardrails

The trained-task exact-sequence threshold of **0.80** and the held-out `DELTA_MOD` threshold of **0.50** are interpretation filters, not test gates.

The orthogonal scramble is a **coordinate-invariance control**. Preserving linear probe performance after an orthogonal rotation is expected linear algebra and is **not evidence for random transformers** computing the algorithms.

Likewise, high `DELTA_MOD` orthogonal energy is not evidence for novel computation if `DELTA_MOD` exact accuracy is below **0.50**. In that regime the receipt explicitly labels the novelty interpretation as ineligible.

Gate 0 can support statements such as “residuals are linearly separable by task” or “an A→B map generalizes better than its random-pair baseline.” It does not by itself justify saying that the transformer literally stores algorithms as vectors.

## Install and test

Python 3.11 is the reference environment.

```bash
python -m pip install -e '.[test]'
pytest -q
```

Run the cheap smoke experiment locally:

```bash
python -m transformer_study.experiment --preset smoke --output artifacts/smoke
python -m transformer_study.experiment --validate artifacts/smoke
```

The smoke preset validates plumbing only. It uses one small layer, eight training steps, and tiny analysis banks; its scientific scores should not be interpreted.

## Full GitHub Actions experiment

The full run is intentionally **manual-only** so normal pushes do not spend a long CPU job.

In GitHub:

1. open **Actions**;
2. select **Gate 0**;
3. choose **Run workflow**;
4. download the `transformer-study-gate0` artifact after completion.

The workflow runs:

```bash
python -m transformer_study.experiment --preset gate0 --output artifacts/gate0
python -m transformer_study.experiment --validate artifacts/gate0
```

The artifact contains the exact configuration, trained state dict, JSON metrics, CSV analysis tables, plots, and `RESULTS.md`.

## Default full preset

- 3 decoder blocks
- width 48
- 4 attention heads
- MLP width 96
- tape length 4, modulus 8
- 3 demonstrations per episode
- AdamW, learning rate `3e-4`
- batch size 32
- 2000 training steps
- 192 behavioral episodes per task
- 192 paired map-fit queries
- 192 disjoint paired map-test queries
- one training seed

If the first manual Actions run is too slow, the approved calibration order is: reduce analysis sample counts, then training steps, then batch size. Any changed preset must be recorded in the receipt before scientific interpretation.

## Output receipt

A full run writes:

- `config.json`
- `metrics.json`
- `RESULTS.md`
- `model.pt`
- `task_geometry.csv`
- `linear_maps.csv`
- `composition.csv`
- `novelty.csv`
- `scramble.csv`
- `singular_spectra.png`
- `map_transfer.png`
- `novelty_orthogonal_energy.png`

Engineering failures such as NaNs or malformed receipts return nonzero. Scientific disappointment does not.

## Gate 1: translation null

Gate 1 asks whether Gate 0's inter-task affine maps do more than add task-dependent mean offsets. It compares identity, translation, rank-1/2/4/8 centered corrections, and full affine ridge on the same disjoint held-out residual banks. The central null is

```text
h_B ~= h_A + (mu_B - mu_A)
```

Translation composition is an exact null because the offsets telescope: `A -> B -> C` gives the same mean offset as direct `A -> C`. Gate 1 therefore measures the extra held-out gain of centered/affine transformations beyond translation rather than treating composition alone as evidence for an algorithm algebra.

The transformer architecture, task suite, episode format, seeds, optimizer, query-output-only loss, and residual extraction point are unchanged. Gate 1 changes only the fixed training budget from 2000 to **8000 steps**. Correct-conditioned metrics are evaluation-only: the maps are still fit on the complete map-fit bank and are never refit on a small successful subset.

Run Gate 1 locally with:

```bash
python -m transformer_study.experiment --preset gate1 --output artifacts/gate1
python -m transformer_study.experiment --validate artifacts/gate1
```

On GitHub, open **Actions**, select **Gate 1**, and choose **Run workflow**. The full 8000-step Gate 1 experiment is manual-only and is **not part of ordinary push/pull-request CI**. Its artifact is named `transformer-study-gate1`.

## Gate 2: causal transport across nonlinear routes

Gate 2 follows only because Gate 1 found residual geometry richer than a task-mean translation. It keeps the Gate 1 architecture, optimizer, task family, seeds, and fixed **8000-step** budget, then preregisters the two Gate 1-competent algorithms as the sole focal pair: `SORT` and `PREFIX_PARITY`. Gate 2 does not search task pairs for a flattering effect.

The new observable is the model's **state-dependent route fingerprint**. At the final query separator, every decoder block records the 96-dimensional `GELU'(z)` vector from the MLP and the per-head attention row. These are smooth routing coefficients, not literal binary branches.

At residual layers 1 and 2 Gate 2 fits held-out affine maps between the focal tasks and then intervenes on only the query-separator residual under the unchanged source-task demonstrations. Five preregistered conditions are compared:

- identity / no intervention;
- task-mean translation;
- fitted full affine transport;
- deterministic norm-matched random displacement;
- the actual paired target-task residual as a state-sufficiency upper-bound control.

The same residual patch is re-applied during each autoregressive recomputation. Gate 2 measures both the generated answer and whether later GELU/attention fingerprints become more target-like. A claim of a changed **routing regime** is eligible only when the focal tasks remain behaviorally competent and affine transport beats both translation and norm-matched random controls in target behavior and downstream route shift.

Gate 2 also includes a small causal pruning experiment. For each block, units are ranked by the absolute difference between the focal tasks' mean GELU-derivative coefficients. Exactly 10% of the MLP hidden units are masked using three fixed strategies: most selective, least selective, and deterministic random. This is an exploratory branch-pruning test, not a claim that derivative selectivity is an optimal pruning algorithm.

Run Gate 2 locally with:

```bash
python -m transformer_study.experiment --preset gate2 --output artifacts/gate2
python -m transformer_study.experiment --validate artifacts/gate2
```

On GitHub, open **Actions**, select **Gate 2**, and choose **Run workflow**. The full run is manual-only and writes the standard Gate 0/1 receipt plus `route_geometry.csv`, `causal_transport.csv`, `route_shift.csv`, `pruning.csv`, `gate2_causal_behavior.png`, and `gate2_route_shift.png`. Its artifact is named `transformer-study-gate2`.
