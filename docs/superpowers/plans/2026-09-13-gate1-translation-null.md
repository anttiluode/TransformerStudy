# Gate 1 Translation-Null Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Gate 0's strong inter-task affine transfer and composition are materially richer than task-dependent translation, using rank-limited centered corrections, correct-only evaluation, and the same transformer trained for a fixed 8000 steps.

**Architecture:** Keep the Gate 0 model, task suite, episode format, seeds, residual extraction point, and optimization settings unchanged. Add a Gate 1 analysis layer that decomposes each inter-task map into translation plus a centered linear correction, evaluates ranks 1/2/4/8 and full affine on the disjoint test bank, conditions evaluation on both-correct rows without refitting, and compares full-affine composition against the exactly compositional translation null. Extend the existing experiment/receipt pipeline and add a manual-only GitHub Actions CPU workflow.

**Tech Stack:** Python 3.11, PyTorch CPU, NumPy, matplotlib, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-gate1-translation-null-design.md`

## Global Constraints

- Gate 1 stays CPU-only on ordinary GitHub Actions.
- Transformer architecture is unchanged: 3 decoder blocks, `d_model=48`, 4 heads, MLP width 96, dropout 0.
- Episode format is unchanged: modulus 8, tape length 4, 3 demonstrations plus 1 query.
- Training tasks remain COPY, REVERSE, SORT, CUMSUM_MOD, PREFIX_PARITY, SWAP_PAIRS.
- DELTA_MOD remains held out.
- Training loss remains query-output-only.
- Residual extraction remains the final query `<SEP>` before query output generation.
- All Gate 0 seeds remain unchanged.
- Gate 1 changes only the training budget to exactly `8000` steps; batch size remains 32, AdamW lr remains `3e-4`, weight decay remains 0, grad clip remains 1.0.
- Map-fit and map-test banks remain disjoint and use the existing deterministic roots.
- Ridge lambda remains `1e-3`.
- Correct-conditioned rows are emitted only when both source and target are exactly correct on at least 20 paired test rows; maps are still fit on the complete fit bank.
- Trained-task competence remains exact accuracy `>= 0.80`; DELTA_MOD novelty threshold remains `>= 0.50`; neither is a CI gate.
- `schema_version` remains 1.
- Scientific weakness is recorded, never raised as a process failure.
- Gate 1 full training is manual-only; normal push/PR CI continues to run only tests plus the existing smoke preset.

## File Structure

- Modify `transformer_study/config.py` — add the fixed `gate1_config()` preset.
- Modify `transformer_study/analysis_core.py` — add translation/centered-map/low-rank primitives.
- Create `transformer_study/gate1_analysis.py` — pairwise translation controls, correct-conditioned evaluation, and composition controls.
- Modify `transformer_study/analysis.py` — re-export the Gate 1 analysis API.
- Modify `transformer_study/experiment.py` — accept `gate1`, run Gate 1 analyses, write the three new CSVs and metric keys.
- Modify `transformer_study/receipt.py` — preset-aware validation and Gate 1 interpretation prose.
- Modify `transformer_study/plots.py` — add the two Gate 1 translation-vs-affine plots.
- Modify `tests/test_tasks.py` — freeze the Gate 1 8000-step preset without duplicating architecture constants elsewhere.
- Modify `tests/test_analysis.py` — translation telescoping, low-rank recovery, disjoint-bank behavior, and correct-conditioned guards.
- Modify `tests/test_smoke.py` — keep smoke compatibility and test preset-aware Gate 1 receipt validation without 8000-step training.
- Modify `tests/test_workflows.py` — require a manual-only Gate 1 workflow and ensure CI never runs Gate 1.
- Create `.github/workflows/gate1.yml` — manual CPU-only full run with artifact upload.
- Modify `README.md` — document the Gate 1 null and manual workflow.

---

### Task 1: Freeze the Gate 1 Preset

**Files:**
- Modify: `transformer_study/config.py`
- Modify: `tests/test_tasks.py`

**Interfaces:**
- Consumes: existing `ExperimentConfig`, `gate0_config()`.
- Produces: `gate1_config() -> ExperimentConfig` with all Gate 0 fields unchanged except `steps=8000`.

- [ ] **Step 1: Write the failing preset test**

Append to `tests/test_tasks.py`:

```python
from transformer_study.config import gate0_config, gate1_config


def test_gate1_changes_only_training_steps():
    gate0 = gate0_config().to_dict()
    gate1 = gate1_config().to_dict()
    differing = {key for key in gate0 if gate0[key] != gate1[key]}
    assert differing == {"steps"}
    assert gate1["steps"] == 8000
```

- [ ] **Step 2: Run the focused test and verify the red state**

Run:

```bash
pytest tests/test_tasks.py::test_gate1_changes_only_training_steps -q
```

Expected: import failure because `gate1_config` does not exist.

- [ ] **Step 3: Implement the preset minimally**

Add to `transformer_study/config.py`:

```python
def gate1_config() -> ExperimentConfig:
    return replace(ExperimentConfig(), steps=8000)
```

Do not change any default field in `ExperimentConfig`.

- [ ] **Step 4: Verify Task 1**

Run:

```bash
pytest tests/test_tasks.py -q
```

Expected: all task/config tests pass.

- [ ] **Step 5: Commit**

```bash
git add transformer_study/config.py tests/test_tasks.py
git commit -m "feat: add fixed Gate 1 training preset"
```

---

### Task 2: Add Translation and Low-Rank Centered Map Primitives

**Files:**
- Modify: `transformer_study/analysis_core.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Produces:
  - `fit_centered_ridge(x, y, ridge=1e-3) -> tuple[mu_x, mu_y, matrix]`
  - `translation_predict(x, mu_x, mu_y) -> np.ndarray`
  - `truncate_centered_correction(matrix, rank) -> np.ndarray`
  - `centered_predict(x, mu_x, mu_y, matrix) -> np.ndarray`
  - `rank_for_fraction(translation_nmse, rank_nmse, affine_nmse, fraction=0.9) -> int | None`
- Row-vector convention remains `x @ M`.

- [ ] **Step 1: Add failing mathematical invariants**

Append to `tests/test_analysis.py`:

```python
from transformer_study.analysis import (
    centered_predict,
    fit_centered_ridge,
    rank_for_fraction,
    translation_predict,
    truncate_centered_correction,
)


def test_translation_offsets_telescope_exactly():
    rng = np.random.default_rng(101)
    x = rng.normal(size=(32, 6))
    mu_a = rng.normal(size=6)
    mu_b = rng.normal(size=6)
    mu_c = rng.normal(size=6)
    via_b = translation_predict(
        translation_predict(x, mu_a, mu_b), mu_b, mu_c
    )
    direct = translation_predict(x, mu_a, mu_c)
    assert np.allclose(via_b, direct, atol=1e-12)


def test_rank_two_centered_correction_recovers_rank_two_transform():
    rng = np.random.default_rng(102)
    x = rng.normal(size=(512, 8))
    u = rng.normal(size=(8, 2))
    v = rng.normal(size=(2, 8))
    correction = u @ v
    mu_x = rng.normal(size=8)
    mu_y = rng.normal(size=8)
    xs = x + mu_x
    ys = mu_y + x @ (np.eye(8) + correction)
    fit_mu_x, fit_mu_y, matrix = fit_centered_ridge(xs, ys, ridge=1e-8)
    rank2 = truncate_centered_correction(matrix, rank=2)
    pred = centered_predict(xs, fit_mu_x, fit_mu_y, rank2)
    assert np.mean((pred - ys) ** 2) < 1e-8


def test_rank_fraction_reports_smallest_rank_reaching_target_gain():
    rank_nmse = {1: 0.70, 2: 0.35, 4: 0.21, 8: 0.20}
    assert rank_for_fraction(1.0, rank_nmse, 0.20, fraction=0.90) == 4
    assert rank_for_fraction(0.20, rank_nmse, 0.20, fraction=0.90) is None
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest tests/test_analysis.py -k "translation_offsets or rank_two_centered or rank_fraction" -q
```

Expected: import failures for the new functions.

- [ ] **Step 3: Implement centered-map primitives**

Add to `transformer_study/analysis_core.py`:

```python
def fit_centered_ridge(
    x: np.ndarray, y: np.ndarray, ridge: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be matrices with identical shape")
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    xc = x - mu_x
    yc = y - mu_y
    gram = xc.T @ xc + np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    rhs = xc.T @ yc
    try:
        matrix = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        matrix = np.linalg.pinv(gram) @ rhs
    return mu_x, mu_y, matrix


def translation_predict(x: np.ndarray, mu_x: np.ndarray, mu_y: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) + np.asarray(mu_y) - np.asarray(mu_x)


def centered_predict(
    x: np.ndarray, mu_x: np.ndarray, mu_y: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    return np.asarray(mu_y) + (np.asarray(x, dtype=np.float64) - np.asarray(mu_x)) @ np.asarray(matrix)


def truncate_centered_correction(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if rank < 0 or rank > matrix.shape[0]:
        raise ValueError("rank out of range")
    correction = matrix - np.eye(matrix.shape[0], dtype=np.float64)
    u, s, vt = np.linalg.svd(correction, full_matrices=False)
    kept = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return np.eye(matrix.shape[0], dtype=np.float64) + kept


def rank_for_fraction(
    translation_nmse: float,
    rank_nmse: dict[int, float],
    affine_nmse: float,
    fraction: float = 0.90,
) -> int | None:
    full_gain = float(translation_nmse) - float(affine_nmse)
    if full_gain <= 0:
        return None
    target = fraction * full_gain
    for rank in sorted(rank_nmse):
        if float(translation_nmse) - float(rank_nmse[rank]) >= target:
            return int(rank)
    return None
```

- [ ] **Step 4: Re-export the primitives**

Modify `transformer_study/analysis.py` so the five new names are imported from `analysis_core` and included in its public surface alongside the existing Gate 0 primitives.

- [ ] **Step 5: Verify Task 2**

Run:

```bash
pytest tests/test_analysis.py -q
```

Expected: all analysis tests pass.

- [ ] **Step 6: Commit**

```bash
git add transformer_study/analysis_core.py transformer_study/analysis.py tests/test_analysis.py
git commit -m "feat: add translation and centered-map primitives"
```

---

### Task 3: Implement Pairwise Translation Controls and Correct-Only Evaluation

**Files:**
- Create: `transformer_study/gate1_analysis.py`
- Modify: `transformer_study/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `ResidualBank`, `TRAIN_TASKS`, `fit_affine_ridge`, `fit_centered_ridge`, `translation_predict`, `centered_predict`, `truncate_centered_correction`, `normalized_mse`, `mean_cosine`, `rank_for_fraction`.
- Produces:
  - `analyze_translation_controls(fit_bank, test_bank, ridge, ranks=(1,2,4,8)) -> tuple[list[dict], dict]`
  - `analyze_correct_conditioned(fit_bank, test_bank, ridge, min_n=20, ranks=(1,2,4,8)) -> list[dict]`
- Returned map bundle contains full affine and centered maps keyed by `(layer, source_task, target_task)` for Task 4.

- [ ] **Step 1: Add a failing translation-vs-affine synthetic test**

Append to `tests/test_analysis.py`:

```python
from transformer_study.analysis import (
    analyze_correct_conditioned,
    analyze_translation_controls,
)


def _shared_bank(base: np.ndarray, transforms: dict, correct_value: bool = True):
    states = {0: {}}
    targets = {}
    correct = {}
    for task in TRAIN_TASKS:
        matrix, offset = transforms[task]
        states[0][task] = base @ matrix + offset
        targets[task] = np.zeros((len(base), 4), dtype=np.int64)
        correct[task] = np.full(len(base), correct_value, dtype=bool)
    return ResidualBank(states, np.zeros((len(base), 4), dtype=np.int64), targets, correct)


def test_translation_controls_detect_non_translation_geometry_on_test_bank():
    rng = np.random.default_rng(103)
    fit_latent = rng.normal(size=(192, 6))
    test_latent = rng.normal(size=(192, 6)) + 3.0
    transforms = {}
    for i, task in enumerate(TRAIN_TASKS):
        matrix = np.eye(6)
        if task.value == "reverse":
            matrix = matrix.copy()
            matrix[0, 1] = 0.7
        transforms[task] = (matrix, np.full(6, i * 0.2))
    fit = _shared_bank(fit_latent, transforms)
    test = _shared_bank(test_latent, transforms)
    rows, _ = analyze_translation_controls(fit, test, ridge=1e-8)
    row = next(r for r in rows if r["source"] == "copy" and r["target"] == "reverse")
    assert row["translation_nmse"] > row["affine_nmse"]
    assert row["absolute_gain"] > 0
    assert row["centered_identity_nmse"] > row["full_centered_nmse"]
```

The shifted `test_latent` is deliberate: a mistaken implementation that evaluates on the fit bank cannot satisfy the intended disjoint-bank invariant reliably.

- [ ] **Step 2: Add failing correct-conditioned threshold tests**

Append:

```python
def test_correct_conditioned_uses_full_fit_but_requires_twenty_test_rows():
    rng = np.random.default_rng(104)
    latent = rng.normal(size=(64, 5))
    transforms = {task: (np.eye(5), np.full(5, i * 0.1)) for i, task in enumerate(TRAIN_TASKS)}
    fit = _shared_bank(latent, transforms)
    test = _shared_bank(latent + 0.5, transforms)
    source, target = TRAIN_TASKS[0], TRAIN_TASKS[1]
    test.correct[source][:] = False
    test.correct[target][:] = False
    test.correct[source][:19] = True
    test.correct[target][:19] = True
    rows = analyze_correct_conditioned(fit, test, ridge=1e-6, min_n=20)
    row = next(r for r in rows if r["source"] == source.value and r["target"] == target.value)
    assert row["n"] == 19
    assert row["status"] == "underpowered"
    assert row["translation_nmse"] is None
    test.correct[source][19] = True
    test.correct[target][19] = True
    rows = analyze_correct_conditioned(fit, test, ridge=1e-6, min_n=20)
    row = next(r for r in rows if r["source"] == source.value and r["target"] == target.value)
    assert row["n"] == 20
    assert row["status"] == "eligible"
    assert row["translation_nmse"] is not None
```

- [ ] **Step 3: Run the new tests and verify failure**

Run:

```bash
pytest tests/test_analysis.py -k "translation_controls_detect or correct_conditioned" -q
```

Expected: import failures because `gate1_analysis.py` does not exist.

- [ ] **Step 4: Implement pair fitting in one private helper**

Create `transformer_study/gate1_analysis.py` with a private `_fit_pair()` that computes from the complete fit bank only:

```python
def _fit_pair(x_fit, y_fit, ridge, ranks):
    mu_x, mu_y, centered_matrix = fit_centered_ridge(x_fit, y_fit, ridge=ridge)
    affine_w, affine_b = fit_affine_ridge(x_fit, y_fit, ridge=ridge)
    rank_matrices = {
        int(rank): truncate_centered_correction(centered_matrix, int(rank))
        for rank in ranks
    }
    return {
        "mu_x": mu_x,
        "mu_y": mu_y,
        "centered_matrix": centered_matrix,
        "affine_w": affine_w,
        "affine_b": affine_b,
        "rank_matrices": rank_matrices,
    }
```

This helper must never receive a test-bank subset.

- [ ] **Step 5: Implement `analyze_translation_controls()`**

For every layer and ordered pair of distinct training tasks:

1. fit `_fit_pair()` on the complete fit-bank arrays;
2. evaluate only on the disjoint test-bank arrays;
3. compute identity, translation, ranks 1/2/4/8, full affine, centered identity, and full centered NMSE;
4. compute translation/full-affine cosine;
5. compute absolute/relative affine gain;
6. compute `rank90` using `rank_for_fraction()`.

Each row must contain exactly these scientific fields in addition to layer/source/target:

```python
{
    "identity_nmse": ...,
    "translation_nmse": ...,
    "rank1_nmse": ...,
    "rank2_nmse": ...,
    "rank4_nmse": ...,
    "rank8_nmse": ...,
    "affine_nmse": ...,
    "translation_cosine": ...,
    "affine_cosine": ...,
    "absolute_gain": ...,
    "relative_gain": ...,
    "rank90": int_or_none,
    "centered_identity_nmse": ...,
    "full_centered_nmse": ...,
}
```

Use denominator protection `max(abs(translation_nmse), 1e-12)` for relative gain.

Return a second object:

```python
{
    "affine": {(layer, source, target): (w, b), ...},
    "centered": {(layer, source, target): matrix, ...},
    "means": {(layer, task): mean, ...},
}
```

- [ ] **Step 6: Implement `analyze_correct_conditioned()`**

Fit all maps on the complete fit bank, then make the test mask:

```python
mask = np.asarray(test_bank.correct[source], dtype=bool) & np.asarray(
    test_bank.correct[target], dtype=bool
)
```

For `n < min_n`, emit:

```python
{
    "layer": layer,
    "source": source.value,
    "target": target.value,
    "n": n,
    "status": "underpowered",
    "translation_nmse": None,
    "affine_nmse": None,
    "absolute_gain": None,
    "relative_gain": None,
    "centered_identity_nmse": None,
    "full_centered_nmse": None,
}
```

For `n >= min_n`, evaluate the already fitted maps on the masked test rows and emit `status="eligible"` with the same six metrics populated.

- [ ] **Step 7: Re-export and verify Task 3**

Re-export the two analysis functions from `transformer_study/analysis.py`.

Run:

```bash
pytest tests/test_analysis.py -q
```

Expected: all analysis tests pass.

- [ ] **Step 8: Commit**

```bash
git add transformer_study/gate1_analysis.py transformer_study/analysis.py tests/test_analysis.py
git commit -m "feat: add Gate 1 translation controls"
```

---

### Task 4: Add the Exact Translation Composition Null and Centered Composition

**Files:**
- Modify: `transformer_study/gate1_analysis.py`
- Modify: `transformer_study/analysis.py`
- Modify: `tests/test_analysis.py`

**Interfaces:**
- Produces: `analyze_composition_controls(fit_bank, test_bank, map_bundle, ridge) -> list[dict]`.
- `map_bundle` is the object returned by `analyze_translation_controls()`.

- [ ] **Step 1: Write the failing composition-control test**

Append to `tests/test_analysis.py`:

```python
from transformer_study.analysis import analyze_composition_controls


def test_translation_composition_matches_direct_translation_to_roundoff():
    rng = np.random.default_rng(105)
    fit_latent = rng.normal(size=(128, 5))
    test_latent = rng.normal(size=(128, 5))
    transforms = {task: (np.eye(5), np.full(5, i * 0.25)) for i, task in enumerate(TRAIN_TASKS)}
    fit = _shared_bank(fit_latent, transforms)
    test = _shared_bank(test_latent, transforms)
    _, bundle = analyze_translation_controls(fit, test, ridge=1e-8)
    rows = analyze_composition_controls(fit, test, bundle, ridge=1e-8)
    assert rows
    assert max(abs(r["translation_composed_nmse"] - r["translation_direct_nmse"]) for r in rows) < 1e-12
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_analysis.py::test_translation_composition_matches_direct_translation_to_roundoff -q
```

Expected: import failure for `analyze_composition_controls`.

- [ ] **Step 3: Implement composition controls**

For every layer and ordered distinct triple `(A, B, C)`:

- direct affine prediction uses stored `A -> C` affine map;
- composed affine prediction uses `compose_affine(A->B, B->C)`;
- direct translation uses `mu_C - mu_A`;
- composed translation applies `A->B` offset then `B->C` offset explicitly;
- direct centered prediction uses `M_AC` on `H_A_test - mu_A` and compares against `H_C_test - mu_C`;
- composed centered prediction uses `M_AB @ M_BC` on centered A states and compares against centered C states.

Each row must contain:

```python
{
    "layer": layer,
    "a": a.value,
    "b": b.value,
    "c": c.value,
    "affine_direct_nmse": ...,
    "affine_composed_nmse": ...,
    "translation_direct_nmse": ...,
    "translation_composed_nmse": ...,
    "translation_action_difference": ...,
    "centered_direct_nmse": ...,
    "centered_composed_nmse": ...,
    "centered_parameter_disagreement": ...,
}
```

`translation_action_difference` is the maximum absolute elementwise difference between the two translation predictions; it must be approximately machine precision on deterministic synthetic tests.

- [ ] **Step 4: Re-export and verify Task 4**

Run:

```bash
pytest tests/test_analysis.py -q
```

Expected: all analysis tests pass.

- [ ] **Step 5: Commit**

```bash
git add transformer_study/gate1_analysis.py transformer_study/analysis.py tests/test_analysis.py
git commit -m "feat: add Gate 1 composition null controls"
```

---

### Task 5: Integrate Gate 1 into the Experiment and Receipt Schema

**Files:**
- Modify: `transformer_study/experiment.py`
- Modify: `transformer_study/receipt.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- CLI accepts `--preset gate1`.
- Gate 1 receipt adds:
  - `translation_controls.csv`
  - `correct_conditioned.csv`
  - `composition_controls.csv`
  - `metrics["translation_controls"]`
  - `metrics["correct_conditioned"]`
  - `metrics["composition_controls"]`
- Gate 0 and smoke receipt validation remains backward-compatible.

- [ ] **Step 1: Write failing preset-aware receipt tests without running 8000 steps**

Append to `tests/test_smoke.py`:

```python
def test_gate1_receipt_requires_gate1_extensions_but_smoke_does_not(tmp_path: Path):
    out = tmp_path / "receipt"
    run_experiment("smoke", out)
    validate_receipt(out)

    config = json.loads((out / "config.json").read_text())
    config["preset"] = "gate1"
    (out / "config.json").write_text(json.dumps(config))
    try:
        validate_receipt(out)
    except ValueError as exc:
        assert "Gate 1" in str(exc) or "gate1" in str(exc)
    else:
        raise AssertionError("Gate 1 receipt unexpectedly validated without extensions")

    for filename in ("translation_controls.csv", "correct_conditioned.csv", "composition_controls.csv"):
        (out / filename).write_text("layer\n0\n")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["translation_controls"] = []
    metrics["correct_conditioned"] = []
    metrics["composition_controls"] = []
    (out / "metrics.json").write_text(json.dumps(metrics))
    validate_receipt(out)
```

Also add:

```python
from transformer_study.experiment import _preset


def test_gate1_cli_preset_is_fixed_at_8000_steps():
    assert _preset("gate1").steps == 8000
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_smoke.py -q
```

Expected: failure because `gate1` is not a recognized preset and Gate 1 validation does not exist.

- [ ] **Step 3: Extend preset resolution and orchestration**

Modify `transformer_study/experiment.py`:

```python
from .config import ExperimentConfig, gate0_config, gate1_config, smoke_config
```

and add to `_preset()`:

```python
if name == "gate1":
    return gate1_config()
```

Add `gate1` to the argparse choices.

After the existing Gate 0 analyses, initialize Gate 1 lists to `None`. When `preset_name == "gate1"`, run:

```python
translation_controls, gate1_maps = analyze_translation_controls(
    fit_bank, test_bank, cfg.ridge_lambda
)
correct_conditioned = analyze_correct_conditioned(
    fit_bank, test_bank, cfg.ridge_lambda, min_n=cfg.correct_split_min_n
)
composition_controls = analyze_composition_controls(
    fit_bank, test_bank, gate1_maps, cfg.ridge_lambda
)
```

Add those three lists to `metrics` only for Gate 1, and write the three CSV files only for Gate 1.

- [ ] **Step 4: Make receipt validation preset-aware**

In `transformer_study/receipt.py`, read `config.json` after confirming the base Gate 0 files exist. If `config["preset"] == "gate1"`, additionally require:

```python
GATE1_FILES = {
    "translation_controls.csv",
    "correct_conditioned.csv",
    "composition_controls.csv",
}
GATE1_KEYS = {
    "translation_controls",
    "correct_conditioned",
    "composition_controls",
}
```

Raise an explicit `ValueError` naming missing Gate 1 files or keys. Do not require these for `smoke` or `gate0`.

- [ ] **Step 5: Verify Task 5**

Run:

```bash
pytest tests/test_smoke.py -q
pytest -q
```

Expected: the full suite passes without running an 8000-step training job.

- [ ] **Step 6: Commit**

```bash
git add transformer_study/experiment.py transformer_study/receipt.py tests/test_smoke.py
git commit -m "feat: integrate Gate 1 receipt pipeline"
```

---

### Task 6: Add Gate 1 Interpretation and Plots

**Files:**
- Modify: `transformer_study/receipt.py`
- Modify: `transformer_study/plots.py`
- Modify: `transformer_study/experiment.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Produces:
  - `render_gate1_plots(out, translation_controls)`
  - `RESULTS.md` Gate 1 section answering the six ordered questions from the spec.
  - `translation_vs_affine.png`
  - `affine_gain_over_translation.png`

- [ ] **Step 1: Add a fast result-rendering test**

Extend `tests/test_smoke.py` with a direct unit test of the prose renderer using synthetic rows rather than training:

```python
from transformer_study.receipt import render_gate1_summary


def test_gate1_summary_states_translation_null_and_competence_level():
    behavior = {
        "tasks": {
            "copy": {"exact": 0.9},
            "reverse": {"exact": 0.85},
            "sort": {"exact": 0.2},
            "cumsum_mod": {"exact": 0.2},
            "prefix_parity": {"exact": 1.0},
            "swap_pairs": {"exact": 0.2},
        }
    }
    controls = [
        {"layer": 3, "source": "copy", "target": "reverse", "translation_nmse": 0.20,
         "affine_nmse": 0.10, "absolute_gain": 0.10, "relative_gain": 0.50, "rank90": 2}
    ]
    correct = [
        {"layer": 3, "source": "copy", "target": "reverse", "n": 32, "status": "eligible",
         "translation_nmse": 0.22, "affine_nmse": 0.11, "absolute_gain": 0.11, "relative_gain": 0.50,
         "centered_identity_nmse": 0.22, "full_centered_nmse": 0.11}
    ]
    composition = [
        {"layer": 3, "a": "copy", "b": "reverse", "c": "prefix_parity",
         "centered_direct_nmse": 0.12, "centered_composed_nmse": 0.13,
         "translation_direct_nmse": 0.20, "translation_composed_nmse": 0.20}
    ]
    text = render_gate1_summary(behavior, controls, correct, composition)
    assert "3 trained tasks" in text
    assert "translation" in text.lower()
    assert "full affine" in text.lower()
    assert "rank" in text.lower()
    assert "both-correct" in text.lower()
    assert "centered composition" in text.lower()
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_smoke.py::test_gate1_summary_states_translation_null_and_competence_level -q
```

Expected: import failure for `render_gate1_summary`.

- [ ] **Step 3: Implement `render_gate1_summary()`**

Add to `transformer_study/receipt.py` a pure function that returns Markdown text and answers, in this exact order:

1. number and names of trained tasks with exact accuracy `>= 0.80`;
2. whether translation explains most of the map accuracy, based on mean final-layer translation and affine NMSE;
3. mean absolute and relative full-affine gain at the final layer;
4. distribution of non-null `rank90` values at the final layer;
5. whether eligible both-correct rows retain positive mean affine gain;
6. centered direct vs centered composed NMSE at the final layer and the explicit reminder that translation composes exactly by construction.

The prose must obey competence rules:

```python
if competent_count < 2:
    eligibility = "representation geometry only; fewer than two trained tasks are competent"
elif competent_count < 3:
    eligibility = "pairwise algorithmic interpretation eligible; centered triple composition not yet eligible"
else:
    eligibility = "pairwise and centered triple-composition interpretation eligible"
```

Do not emit a claim that algorithms are stored as vectors or that an operator algebra exists.

- [ ] **Step 4: Integrate Gate 1 summary into `RESULTS.md`**

Extend `render_results()` with optional `translation_controls=None`, `correct_conditioned=None`, `composition_controls=None`. When `preset_name == "gate1"`, append:

```markdown
## Gate 1: Translation null

<render_gate1_summary(...)>
```

All Gate 0 prose and guardrails remain present.

- [ ] **Step 5: Implement the two Gate 1 plots**

Add to `transformer_study/plots.py`:

```python
def render_gate1_plots(out: Path, rows: list[dict]) -> None:
    layers = sorted({r["layer"] for r in rows})
    translation = [np.mean([r["translation_nmse"] for r in rows if r["layer"] == layer]) for layer in layers]
    affine = [np.mean([r["affine_nmse"] for r in rows if r["layer"] == layer]) for layer in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, translation, marker="o", label="translation")
    ax.plot(layers, affine, marker="o", label="full affine")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean NMSE")
    ax.set_title("Gate 1: translation vs full affine")
    if layers:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "translation_vs_affine.png", dpi=120)
    plt.close(fig)

    gain = [np.mean([r["relative_gain"] for r in rows if r["layer"] == layer]) for layer in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, gain, marker="o")
    ax.axhline(0.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean relative affine gain")
    ax.set_title("Gate 1: gain beyond translation")
    fig.tight_layout()
    fig.savefig(out / "affine_gain_over_translation.png", dpi=120)
    plt.close(fig)
```

Use matplotlib defaults only; do not set explicit colors or styles.

- [ ] **Step 6: Call the renderer only for Gate 1**

In `experiment.py`, after `render_plots(...)`, call `render_gate1_plots(out, translation_controls)` only when `preset_name == "gate1"`.

- [ ] **Step 7: Verify Task 6**

Run:

```bash
pytest tests/test_smoke.py -q
pytest -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add transformer_study/receipt.py transformer_study/plots.py transformer_study/experiment.py tests/test_smoke.py
git commit -m "feat: report Gate 1 translation-null interpretation"
```

---

### Task 7: Add Manual Gate 1 GitHub Workflow and Documentation

**Files:**
- Create: `.github/workflows/gate1.yml`
- Modify: `tests/test_workflows.py`
- Modify: `README.md`

**Interfaces:**
- Full command: `python -m transformer_study.experiment --preset gate1 --output artifacts/gate1`.
- Validation command: `python -m transformer_study.experiment --validate artifacts/gate1`.

- [ ] **Step 1: Write failing workflow invariants**

Extend `tests/test_workflows.py`:

```python
def test_gate1_workflow_is_manual_only_and_runs_gate1():
    text = Path(".github/workflows/gate1.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "--preset gate1" in text
    assert "timeout-minutes: 60" in text
    assert "actions/upload-artifact@v4" in text


def test_ci_never_runs_gate1_training():
    text = Path(".github/workflows/ci.yml").read_text()
    assert "--preset gate1" not in text
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest tests/test_workflows.py -q
```

Expected: failure because `.github/workflows/gate1.yml` does not exist.

- [ ] **Step 3: Create the workflow**

Create `.github/workflows/gate1.yml`:

```yaml
name: Gate 1
on:
  workflow_dispatch:
jobs:
  experiment:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[test]'
      - run: python -m transformer_study.experiment --preset gate1 --output artifacts/gate1
      - run: python -m transformer_study.experiment --validate artifacts/gate1
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: transformer-study-gate1
          path: artifacts/gate1
```

- [ ] **Step 4: Update README**

Add a Gate 1 section explaining:

```text
Gate 1 asks whether Gate 0's inter-task affine maps do more than add task-dependent mean offsets. It compares identity, translation, rank-1/2/4/8 centered corrections, and full affine ridge on disjoint held-out residual banks. Translation composition is an exact null because the task offsets telescope.
```

Document the manual GitHub Actions workflow and the local command. State explicitly that Gate 1's 8000-step run is not part of ordinary CI.

- [ ] **Step 5: Verify Task 7**

Run:

```bash
pytest tests/test_workflows.py -q
pytest -q
python -m transformer_study.experiment --preset smoke --output /tmp/transformer-study-gate1-smoke
python -m transformer_study.experiment --validate /tmp/transformer-study-gate1-smoke
```

Expected: all commands exit 0; smoke remains compatible and cheap.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/gate1.yml tests/test_workflows.py README.md
git commit -m "ci: add manual Gate 1 CPU experiment"
```

---

### Task 8: Final Verification, PR, Merge, and Full GitHub CPU Run

**Files:**
- No new source files expected.
- The full run produces artifacts only in GitHub Actions, not committed generated results.

**Interfaces:**
- Feature branch: `gate1`.
- Base branch: `main`.
- Full scientific artifact name: `transformer-study-gate1`.

- [ ] **Step 1: Run fresh local verification on the exact branch tip**

Run:

```bash
python -m pip install -e '.[test]'
pytest -q
python -m transformer_study.experiment --preset smoke --output /tmp/transformer-study-gate1-smoke
python -m transformer_study.experiment --validate /tmp/transformer-study-gate1-smoke
```

If the container is offline and build isolation attempts network access, use the environment-equivalent local verification command only for this container:

```bash
python -m pip install -e '.[test]' --no-build-isolation --no-deps
```

GitHub Actions must still use the ordinary install command.

- [ ] **Step 2: Open a PR against `main`**

PR description must state:

- Gate 1 keeps the transformer/task setup fixed and changes only training steps from 2000 to 8000;
- translation is the primary null;
- rank-limited corrections measure complexity beyond translation;
- correct-only metrics are evaluation-only and never refit on small subsets;
- translation composition is expected to telescope exactly;
- scientific weakness is not a CI failure.

- [ ] **Step 3: Wait for PR CI and inspect the actual job**

Do not merge while CI is pending or red. Confirm the Python 3.11 Ubuntu job completes install, pytest, smoke experiment, and smoke receipt validation successfully.

- [ ] **Step 4: Merge only after green CI and user authorization**

Merge the PR to `main` using the repository's merge method. Verify `main` points at the resulting merge commit before running Gate 1.

- [ ] **Step 5: Run the full Gate 1 workflow**

Preferred path: dispatch `.github/workflows/gate1.yml` manually on `main`.

If the connector still lacks workflow-dispatch support, create a one-off branch from the verified merge commit, change only `gate1.yml` trigger from `workflow_dispatch` to a branch-specific push trigger, push that single workflow-only change, and use the resulting Gate 1 job. The scientific code/config must remain byte-for-byte identical to merged `main`.

- [ ] **Step 6: Verify the completed job and receipt**

Require the workflow job to show success for:

```text
python -m pip install -e '.[test]'
python -m transformer_study.experiment --preset gate1 --output artifacts/gate1
python -m transformer_study.experiment --validate artifacts/gate1
artifact upload
```

Download `transformer-study-gate1`, inspect `RESULTS.md`, `metrics.json`, and the three Gate 1 CSVs.

- [ ] **Step 7: Report the scientific outcome in the spec's order**

Report:

1. competent task count and names;
2. final-layer mean translation NMSE vs mean full-affine NMSE;
3. mean absolute and relative affine gain;
4. rank90 distribution;
5. both-correct eligible-row gain, if powered;
6. centered direct vs composed map behavior, explicitly comparing it to the exactly telescoping translation null.

If fewer than two tasks are competent, label conclusions as representation geometry only. If at least two are competent, allow pairwise algorithmic interpretation only for competent/both-correct pairs. If at least three are competent, centered triple-composition among competent tasks becomes eligible.

## Plan Self-Review

- **Spec coverage:** fixed 8000-step budget, unchanged architecture/task/seeds, translation null, ranks 1/2/4/8, centered diagnostic, correct-only evaluation, explicit translation composition, centered composition, competence interpretation, three CSVs, three metrics keys, plots, schema compatibility, manual workflow, and no causal injection are all assigned to explicit tasks.
- **Placeholder scan:** no TBD/TODO/"similar to" implementation gaps remain; every new function is named before later tasks consume it.
- **Type consistency:** all residual maps use row-vector orientation `x @ M`; map-bundle keys use `(layer, TaskName, TaskName)`; CSV/metrics fields are fixed across analysis, receipt, tests, and reporting.
- **Scope:** causal residual injection, architecture changes, task-suite tuning, multiple seeds, hyperparameter search, and Gate 2 remain explicitly out of scope.
