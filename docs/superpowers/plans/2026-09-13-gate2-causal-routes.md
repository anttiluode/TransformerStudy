# Gate 2 Causal Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal Gate 2 that tests whether affine residual transport moves the tiny transformer into a target-like nonlinear routing regime and whether route-selective MLP pruning has causal effects.

**Architecture:** Preserve the existing transformer and Gate 0/1 APIs by adding opt-in diagnostics/interventions to `model.py`, isolate Gate 2 scientific logic in a new `gate2_analysis.py`, and integrate it through the existing experiment/receipt/plot pipeline. Gate 2 retrains the frozen Gate 1 architecture for 8000 steps, uses the preregistered SORT <-> PREFIX_PARITY pair, and writes a manual-only GitHub Actions artifact.

**Tech Stack:** Python 3.11, PyTorch, NumPy, matplotlib, pytest, GitHub Actions CPU.

**Spec:** `docs/superpowers/specs/2026-09-13-gate2-causal-routes-design.md`

## Global Constraints

- Keep model architecture at 3 blocks, d_model 48, 4 heads, MLP width 96, GELU, pre-LayerNorm, dropout 0.
- Keep Gate 2 training at exactly 8000 steps, AdamW lr 3e-4, batch 32, model seed 17.
- Keep the focal pair fixed as SORT and PREFIX_PARITY.
- Fit maps only on the affine fit bank; causal patch rows use a disjoint seed range.
- Scientific disappointment never returns nonzero; engineering failures do.
- Ordinary push/PR CI must not run the 8000-step Gate 2 experiment.
- Preserve Gate 0 and Gate 1 receipt compatibility with schema_version 1.

---

### Task 1: Instrument nonlinear routes and add intervention hooks

**Files:**
- Modify: `transformer_study/model.py`
- Modify: `transformer_study/train.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Produces `gelu_derivative(x: torch.Tensor) -> torch.Tensor`.
- Extends `TinyTransformer.forward(tokens, return_residuals=False, return_diagnostics=False, residual_patch=None, mlp_masks=None)` without changing default output.
- `residual_patch` is `None` or a dict containing `layer`, `position`, and `value`; layer indexes residual depth after a block.
- `mlp_masks` is `None` or a mapping from zero-based block index to a length-`mlp_hidden` tensor mask.
- Diagnostics contain one entry per block with `mlp_preactivation`, `gelu_derivative`, and `attention_weights` tensors.
- Extends `greedy_query_output(..., residual_patch=None, mlp_masks=None)` so the same patch/mask is re-applied on each autoregressive recomputation.

- [ ] **Step 1: Write failing model tests**

Add tests equivalent to:

```python
def test_gelu_derivative_matches_autograd():
    x = torch.tensor([-2.0, -0.2, 0.0, 0.7, 2.0], requires_grad=True)
    torch.nn.functional.gelu(x).sum().backward()
    assert torch.allclose(gelu_derivative(x.detach()), x.grad, atol=1e-6)


def test_diagnostics_do_not_change_logits():
    plain = model(x)
    diag_logits, diagnostics = model(x, return_diagnostics=True)
    assert torch.allclose(plain, diag_logits, atol=1e-6)
    assert len(diagnostics) == cfg.layers


def test_identity_residual_patch_is_noop():
    logits, residuals = model(x, return_residuals=True)
    patch = {"layer": 1, "position": x.shape[1] - 1, "value": residuals[1][:, -1, :]}
    patched = model(x, residual_patch=patch)
    assert torch.allclose(logits, patched, atol=1e-6)


def test_all_ones_mlp_mask_is_noop():
    mask = {0: torch.ones(cfg.mlp_hidden)}
    assert torch.allclose(model(x), model(x, mlp_masks=mask), atol=1e-6)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run `pytest -q tests/test_model.py` and confirm the new symbols/options are missing.

- [ ] **Step 3: Refactor DecoderBlock minimally**

Replace the opaque sequential MLP with named `mlp_in`, GELU application, and `mlp_out` so the 96-dimensional hidden route can be observed and masked. Keep parameter shapes and initialization behavior equivalent to the existing `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)` construction.

- [ ] **Step 4: Implement analytic GELU derivative and diagnostic capture**

Use PyTorch's exact tanh-free GELU convention:

```python
def gelu_derivative(x):
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(x * inv_sqrt_2)) + x * torch.exp(-0.5 * x.square()) * inv_sqrt_2pi
```

Request per-head attention weights only when diagnostics are enabled (`average_attn_weights=False`).

- [ ] **Step 5: Implement residual patching and MLP masks**

Patch only the declared sequence position after appending the selected residual depth. Validate layer, position, replacement shape, block index, and mask width; reject malformed interventions with `ValueError`.

- [ ] **Step 6: Extend autoregressive generation**

Pass the optional patch/masks through each forward call in `greedy_query_output`; do not change callers that omit them.

- [ ] **Step 7: Run model/training tests**

Run `pytest -q tests/test_model.py tests/test_training.py` and require PASS.

- [ ] **Step 8: Commit**

Commit message: `feat: expose nonlinear route diagnostics and interventions`.

---

### Task 2: Implement Gate 2 route banks and scientific controls

**Files:**
- Create: `transformer_study/gate2_analysis.py`
- Create: `tests/test_gate2.py`
- Modify: `transformer_study/config.py`

**Interfaces:**
- Add `gate2_config() -> ExperimentConfig` with `steps=8000` plus Gate 2 seed/count fields.
- Produce `extract_route_bank(...)`, `fit_focal_maps(...)`, `analyze_route_geometry(...)`, `analyze_causal_transport(...)`, and `analyze_pruning(...)`.
- Focal tasks are constants `FOCAL_A = TaskName.SORT`, `FOCAL_B = TaskName.PREFIX_PARITY` (using the repository's exact enum values).

- [ ] **Step 1: Write failing Gate 2 tests**

Cover:

```python
def test_random_norm_matches_affine_displacement():
    random = norm_matched_random(source, affine, seed=7)
    assert np.allclose(np.linalg.norm(random-source, axis=1), np.linalg.norm(affine-source, axis=1))


def test_pruning_indices_are_exact_and_deterministic():
    first = pruning_indices(selectivity, width=96, count=10, seed=9)
    second = pruning_indices(selectivity, width=96, count=10, seed=9)
    assert first == second
    assert all(len(v) == 10 for v in first.values())


def test_gate2_seed_ranges_are_disjoint():
    cfg = gate2_config()
    ranges = gate2_seed_ranges(cfg)
    assert all(a.isdisjoint(b) for each distinct pair a,b in ranges)
```

Also include synthetic route-similarity and affine-map tests so target-minus-source route shift has known sign.

- [ ] **Step 2: Run focused test and confirm failure**

Run `pytest -q tests/test_gate2.py`.

- [ ] **Step 3: Add Gate 2 configuration**

Add deterministic roots for route reference, causal patch, and pruning banks and counts `route_reference=192`, `causal_patch=96`, `pruning_eval=192`; smoke configuration reduces them to single-digit values.

- [ ] **Step 4: Implement route extraction**

For paired focal prompts sharing the same query tape, capture residual states plus query-separator GELU derivative vectors and flattened per-head attention rows for each block. Keep data as NumPy float64 for analysis.

- [ ] **Step 5: Implement affine/translation/random/true-target intervention construction**

Fit affine and task means on the fit bank only at residual layers 1 and 2. Generate deterministic row-wise norm-matched random controls with an epsilon-safe zero-displacement path.

- [ ] **Step 6: Implement causal generation and route-shift metrics**

For each source/target direction, layer, and intervention, generate the four output symbols, score against source and target targets, then capture downstream diagnostics on the patched prompt. Aggregate behavior into `causal_transport` and route similarities into `route_shift`.

- [ ] **Step 7: Implement route separability**

Deterministically split route reference rows in half, fit closed-form ridge binary classifiers for GELU and attention fingerprints, and compare against shuffled-label controls.

- [ ] **Step 8: Implement route-selective pruning**

Compute absolute mean GELU-derivative task difference per unit. For each block mask exactly 10 units using most-selective, least-selective, and deterministic-random strategies and evaluate both focal tasks without retraining.

- [ ] **Step 9: Run Gate 2 unit tests**

Run `pytest -q tests/test_gate2.py` and require PASS.

- [ ] **Step 10: Commit**

Commit message: `feat: add Gate 2 causal route analysis`.

---

### Task 3: Integrate Gate 2 with experiment, receipts, and plots

**Files:**
- Modify: `transformer_study/experiment.py`
- Modify: `transformer_study/receipt.py`
- Modify: `transformer_study/plots.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- CLI accepts `--preset gate2`.
- Gate 2 metrics keys: `route_geometry`, `causal_transport`, `route_shift`, `pruning`.
- Gate 2 required files: `route_geometry.csv`, `causal_transport.csv`, `route_shift.csv`, `pruning.csv`, `gate2_route_shift.png`, `gate2_causal_behavior.png`.

- [ ] **Step 1: Write failing receipt/smoke tests**

Add a cheap Gate 2 smoke-path test using a reduced config or direct analysis helpers; add receipt validation tests proving Gate 0/1 do not require Gate 2 files and Gate 2 does.

- [ ] **Step 2: Run focused smoke tests and confirm failure**

Run `pytest -q tests/test_smoke.py`.

- [ ] **Step 3: Integrate the Gate 2 preset**

Keep the standard Gate 0 analyses for continuity, run Gate 1 translation controls for Gate 2 as well, then run the focal Gate 2 analyses and add the four metric sections.

- [ ] **Step 4: Write Gate 2 CSVs and plots**

Create compact plots comparing target exact accuracy across intervention methods and target-minus-source route shift across methods/layers.

- [ ] **Step 5: Extend RESULTS.md**

Add `render_gate2_summary(...)` answering the six interpretation questions in the spec. Explicitly mark causal interpretation ineligible when SORT or PREFIX_PARITY is below 0.80.

- [ ] **Step 6: Extend receipt validation**

Require Gate 2 files/keys only when `config.json` has `preset == "gate2"`; keep schema_version 1.

- [ ] **Step 7: Run focused tests**

Run `pytest -q tests/test_smoke.py tests/test_analysis.py`.

- [ ] **Step 8: Commit**

Commit message: `feat: integrate Gate 2 experiment receipt`.

---

### Task 4: Add the manual Gate 2 workflow and user-facing docs

**Files:**
- Create: `.github/workflows/gate2.yml`
- Modify: `tests/test_workflows.py`
- Modify: `README.md`

**Interfaces:**
- Workflow name `Gate 2`.
- Artifact name `transformer-study-gate2`.
- Manual `workflow_dispatch` only.

- [ ] **Step 1: Write failing workflow invariant test**

Assert `.github/workflows/gate2.yml` contains `workflow_dispatch`, 60-minute timeout or less, `--preset gate2`, `--validate artifacts/gate2`, and artifact name `transformer-study-gate2`.

- [ ] **Step 2: Run workflow tests and confirm failure**

Run `pytest -q tests/test_workflows.py`.

- [ ] **Step 3: Add workflow**

Mirror Gate 1 dependency installation and artifact upload, but run the Gate 2 preset/output paths.

- [ ] **Step 4: Update README**

Document the causal question, preregistered focal pair, five patch controls, route-shift criterion, and exploratory 10-of-96 pruning test without claiming a positive result before the workflow runs.

- [ ] **Step 5: Run workflow tests**

Run `pytest -q tests/test_workflows.py` and require PASS.

- [ ] **Step 6: Commit**

Commit message: `ci: add manual Gate 2 experiment`.

---

### Task 5: Full verification, merge, workflow run, and artifact inspection

**Files:**
- No new source files unless verification exposes a bug.

- [ ] **Step 1: Run full unit suite**

Run `pytest -q`; require all tests pass.

- [ ] **Step 2: Run smoke experiment and validate receipt**

Run:

```bash
python -m transformer_study.experiment --preset smoke --output artifacts/smoke
python -m transformer_study.experiment --validate artifacts/smoke
```

Require both exit 0.

- [ ] **Step 3: Review branch diff**

Confirm changes are scoped to Gate 2/instrumentation/docs and no ordinary CI path launches 8000-step training.

- [ ] **Step 4: Merge Gate 2 to main**

Create a pull request, ensure GitHub CI is green, then squash/merge or merge according to repository convention.

- [ ] **Step 5: Dispatch manual Gate 2 workflow on main**

Run `Gate 2` with `workflow_dispatch` and wait for terminal status.

- [ ] **Step 6: Download and validate artifact**

Inspect `RESULTS.md`, `metrics.json`, and the four Gate 2 CSVs. Verify artifact validation succeeded in the workflow.

- [ ] **Step 7: Report scientific result without overclaiming**

Report competence first, then route separability, true-target state sufficiency, affine-vs-control behavioral shift, downstream route shift, and pruning selectivity. A null result is reported as a null result.
