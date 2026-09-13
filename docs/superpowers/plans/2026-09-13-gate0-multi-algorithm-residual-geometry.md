# Gate 0 Multi-Algorithm Residual Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiny CPU-only decoder transformer that infers one of six hidden sequence algorithms from demonstrations, then measures residual-stream separability, inter-algorithm linear maps, map composition, orthogonal-basis invariance, and held-out DELTA_MOD orthogonal energy.

**Architecture:** Synthetic fixed-length in-context episodes feed one shared 3-layer decoder-only transformer. Training optimizes only the four query-output symbols. A frozen post-training analysis pipeline extracts the residual at the final query separator, fits only closed-form ridge probes/maps on held-out banks, and writes a reproducible scientific receipt. Normal CI stays cheap; the full Gate 0 run is manual GitHub Actions CPU only.

**Tech Stack:** Python 3.11, PyTorch CPU, NumPy, matplotlib, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-multi-algorithm-residual-geometry-design.md`

## Global Constraints

- Reference environment: GitHub Actions Ubuntu CPU only.
- No external datasets, model downloads, pretrained weights, paid compute, or external services.
- Python 3.11.
- Gate 0 model: 3 layers, width 48, 4 attention heads, MLP width 96, dropout 0.
- Symbol vocabulary: values 0..7, tape length 4, 3 demonstrations plus 1 query.
- Training tasks only: COPY, REVERSE, SORT, CUMSUM_MOD, PREFIX_PARITY, SWAP_PAIRS.
- Held-out task: DELTA_MOD; it must be impossible to sample in training mode.
- Training preset: AdamW, lr 3e-4, weight decay 0, batch 32, 2000 steps, grad clip 1.0, model seed 17, training-data root seed 1000.
- Training loss applies only to the four query-output symbol positions.
- Behavioral evaluation: 192 held-out episodes per task.
- Residual-map banks: 192 fit query tapes and 192 disjoint test query tapes.
- Affine ridge lambda: 1e-3.
- Trained-task interpretation threshold: exact-sequence accuracy >= 0.80; DELTA_MOD novelty threshold: >= 0.50. Neither threshold can fail CI.
- Full scientific workflow is manual-only (`workflow_dispatch`).
- Scientific failure is recorded, not raised; only engineering failures return nonzero.

## File Structure

- `pyproject.toml` — packaging, dependencies, pytest configuration.
- `README.md` — experiment purpose, local commands, interpretation rules.
- `transformer_study/__init__.py` — package version/export surface.
- `transformer_study/config.py` — frozen experiment dataclass and presets.
- `transformer_study/tasks.py` — seven pure sequence transformations and task enumerations.
- `transformer_study/episodes.py` — vocabulary, episode serialization, masks, seeded batch generation.
- `transformer_study/model.py` — tiny causal transformer with residual capture.
- `transformer_study/train.py` — masked training, greedy query generation, behavioral evaluation.
- `transformer_study/residuals.py` — paired residual-bank extraction and correctness metadata.
- `transformer_study/analysis.py` — ridge probes/maps, composition, scramble, span/novelty metrics.
- `transformer_study/experiment.py` — CLI orchestration, CSV/JSON/Markdown receipts, plots, validation.
- `tests/test_tasks.py` — exact transformation tests and training/held-out separation.
- `tests/test_episodes.py` — serialization/masking/seed invariants.
- `tests/test_model.py` — causality, shape, residual-count, determinism tests.
- `tests/test_training.py` — masked-loss and greedy-generation plumbing tests.
- `tests/test_analysis.py` — synthetic recovery/control tests.
- `tests/test_smoke.py` — tiny end-to-end receipt test.
- `.github/workflows/ci.yml` — cheap push/PR tests and 8-step smoke run.
- `.github/workflows/gate0.yml` — manual full experiment and artifact upload.

---

### Task 1: Project Skeleton, Configuration, and Algorithm Suite

**Files:**
- Create: `pyproject.toml`
- Create: `transformer_study/__init__.py`
- Create: `transformer_study/config.py`
- Create: `transformer_study/tasks.py`
- Create: `tests/test_tasks.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `gate0_config()`, `smoke_config()`, `TaskName`, `TRAIN_TASKS`, `HELD_OUT_TASK`, `apply_task(task, tape, modulus)`.
- Later tasks consume these exact names.

- [ ] **Step 1: Write the failing task/config tests**

```python
# tests/test_tasks.py
import pytest

from transformer_study.config import gate0_config
from transformer_study.tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName, apply_task


def test_gate0_constants_are_frozen():
    cfg = gate0_config()
    assert (cfg.modulus, cfg.tape_len, cfg.demos) == (8, 4, 3)
    assert (cfg.layers, cfg.d_model, cfg.heads, cfg.mlp_hidden) == (3, 48, 4, 96)
    assert (cfg.steps, cfg.batch_size, cfg.lr) == (2000, 32, 3e-4)
    assert (cfg.eval_episodes, cfg.map_fit, cfg.map_test) == (192, 192, 192)


def test_exact_algorithms():
    x = [5, 1, 7, 2]
    assert apply_task(TaskName.COPY, x, 8) == [5, 1, 7, 2]
    assert apply_task(TaskName.REVERSE, x, 8) == [2, 7, 1, 5]
    assert apply_task(TaskName.SORT, x, 8) == [1, 2, 5, 7]
    assert apply_task(TaskName.CUMSUM_MOD, x, 8) == [5, 6, 5, 7]
    assert apply_task(TaskName.PREFIX_PARITY, x, 8) == [1, 0, 1, 1]
    assert apply_task(TaskName.SWAP_PAIRS, x, 8) == [1, 5, 2, 7]
    assert apply_task(TaskName.DELTA_MOD, x, 8) == [5, 4, 6, 3]


def test_delta_mod_is_never_a_training_task():
    assert HELD_OUT_TASK is TaskName.DELTA_MOD
    assert HELD_OUT_TASK not in TRAIN_TASKS
    assert set(TRAIN_TASKS) == {
        TaskName.COPY,
        TaskName.REVERSE,
        TaskName.SORT,
        TaskName.CUMSUM_MOD,
        TaskName.PREFIX_PARITY,
        TaskName.SWAP_PAIRS,
    }


def test_swap_pairs_requires_even_length():
    with pytest.raises(ValueError, match="even"):
        apply_task(TaskName.SWAP_PAIRS, [1, 2, 3], 8)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_tasks.py -q`

Expected: import failure because package/config/tasks do not exist.

- [ ] **Step 3: Implement configuration and pure task functions**

```python
# transformer_study/config.py
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ExperimentConfig:
    modulus: int = 8
    tape_len: int = 4
    demos: int = 3
    layers: int = 3
    d_model: int = 48
    heads: int = 4
    mlp_hidden: int = 96
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    steps: int = 2000
    grad_clip: float = 1.0
    model_seed: int = 17
    train_seed_root: int = 1000
    eval_seed_root: int = 2000
    map_fit_seed_root: int = 3000
    map_test_seed_root: int = 4000
    scramble_seed: int = 5000
    ridge_lambda: float = 1e-3
    eval_episodes: int = 192
    map_fit: int = 192
    map_test: int = 192
    trained_competence: float = 0.80
    novel_competence: float = 0.50
    correct_split_min_n: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


def gate0_config() -> ExperimentConfig:
    return ExperimentConfig()


def smoke_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        layers=1,
        d_model=24,
        heads=4,
        mlp_hidden=48,
        batch_size=4,
        steps=8,
        eval_episodes=8,
        map_fit=8,
        map_test=8,
    )
```

```python
# transformer_study/tasks.py
from enum import Enum


class TaskName(str, Enum):
    COPY = "copy"
    REVERSE = "reverse"
    SORT = "sort"
    CUMSUM_MOD = "cumsum_mod"
    PREFIX_PARITY = "prefix_parity"
    SWAP_PAIRS = "swap_pairs"
    DELTA_MOD = "delta_mod"


TRAIN_TASKS = (
    TaskName.COPY,
    TaskName.REVERSE,
    TaskName.SORT,
    TaskName.CUMSUM_MOD,
    TaskName.PREFIX_PARITY,
    TaskName.SWAP_PAIRS,
)
HELD_OUT_TASK = TaskName.DELTA_MOD


def apply_task(task: TaskName, tape: list[int], modulus: int) -> list[int]:
    x = list(tape)
    if task is TaskName.COPY:
        return x
    if task is TaskName.REVERSE:
        return x[::-1]
    if task is TaskName.SORT:
        return sorted(x)
    if task is TaskName.CUMSUM_MOD:
        total, out = 0, []
        for value in x:
            total = (total + value) % modulus
            out.append(total)
        return out
    if task is TaskName.PREFIX_PARITY:
        total, out = 0, []
        for value in x:
            total = (total + (value & 1)) & 1
            out.append(total)
        return out
    if task is TaskName.SWAP_PAIRS:
        if len(x) % 2:
            raise ValueError("swap_pairs requires even tape length")
        out = x[:]
        for i in range(0, len(x), 2):
            out[i], out[i + 1] = x[i + 1], x[i]
        return out
    if task is TaskName.DELTA_MOD:
        return [x[0]] + [(x[i] - x[i - 1]) % modulus for i in range(1, len(x))]
    raise ValueError(f"unknown task: {task}")
```

- [ ] **Step 4: Add packaging metadata and run tests**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "transformer-study"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.2", "matplotlib>=3.8"]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run: `python -m pip install -e '.[test]' && pytest tests/test_tasks.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml transformer_study tests/test_tasks.py
git commit -m "feat: add Gate 0 config and algorithm suite"
```

---

### Task 2: Episode Vocabulary, Serialization, and Seeded Sampling

**Files:**
- Create: `transformer_study/episodes.py`
- Create: `tests/test_episodes.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `TaskName`, `TRAIN_TASKS`, `apply_task`.
- Produces: `Vocabulary`, `Episode`, `make_episode()`, `sample_training_batch()`, `make_eval_prompt()`, `episode_length()`.

- [ ] **Step 1: Write failing episode invariants**

```python
# tests/test_episodes.py
import numpy as np

from transformer_study.config import gate0_config
from transformer_study.episodes import Vocabulary, episode_length, make_episode, sample_training_batch
from transformer_study.tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName


def test_episode_has_fixed_length_and_only_query_targets():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    ep = make_episode(TaskName.REVERSE, cfg, np.random.default_rng(123), vocab)
    assert len(ep.tokens) == episode_length(cfg)
    assert len(ep.loss_mask) == len(ep.tokens)
    assert sum(ep.loss_mask) == cfg.tape_len
    assert ep.query_sep_index < ep.query_output_start
    assert ep.query_output_start == ep.query_sep_index + 1
    assert ep.loss_mask[ep.query_output_start:ep.query_output_start + cfg.tape_len] == [1] * cfg.tape_len


def test_training_sampler_is_balanced_and_never_samples_delta():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    tasks = []
    for step in range(12):
        batch = sample_training_batch(cfg, vocab, step=step, batch_size=6)
        tasks.extend(batch.task_names)
    assert HELD_OUT_TASK not in tasks
    assert set(tasks) == set(TRAIN_TASKS)


def test_eval_prompt_stops_at_query_separator():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    ep = make_episode(TaskName.COPY, cfg, np.random.default_rng(3), vocab)
    prompt = ep.tokens[: ep.query_output_start]
    assert prompt[-1] == vocab.sep
    assert len(prompt) == ep.query_output_start
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_episodes.py -q`

Expected: import failure for `episodes`.

- [ ] **Step 3: Implement immutable vocabulary and episode serialization**

```python
# transformer_study/episodes.py
from dataclasses import dataclass
import numpy as np
import torch

from .config import ExperimentConfig
from .tasks import TRAIN_TASKS, TaskName, apply_task


@dataclass(frozen=True)
class Vocabulary:
    modulus: int

    @property
    def sep(self) -> int: return self.modulus
    @property
    def pair(self) -> int: return self.modulus + 1
    @property
    def bos(self) -> int: return self.modulus + 2
    @property
    def size(self) -> int: return self.modulus + 3


@dataclass
class Episode:
    tokens: list[int]
    loss_mask: list[int]
    task_name: TaskName
    query_input: list[int]
    query_target: list[int]
    query_sep_index: int
    query_output_start: int


def episode_length(cfg: ExperimentConfig) -> int:
    # BOS + 3 * (x4 SEP y4 PAIR) + (query x4 SEP y4)
    return 1 + cfg.demos * (2 * cfg.tape_len + 2) + (2 * cfg.tape_len + 1)


def _draw_tape(cfg: ExperimentConfig, rng: np.random.Generator) -> list[int]:
    return rng.integers(0, cfg.modulus, size=cfg.tape_len).tolist()


def make_episode(task: TaskName, cfg: ExperimentConfig, rng: np.random.Generator, vocab: Vocabulary,
                 query_input: list[int] | None = None) -> Episode:
    tokens = [vocab.bos]
    for _ in range(cfg.demos):
        x = _draw_tape(cfg, rng)
        y = apply_task(task, x, cfg.modulus)
        tokens += x + [vocab.sep] + y + [vocab.pair]
    qx = list(query_input) if query_input is not None else _draw_tape(cfg, rng)
    qy = apply_task(task, qx, cfg.modulus)
    tokens += qx + [vocab.sep]
    query_sep_index = len(tokens) - 1
    query_output_start = len(tokens)
    tokens += qy
    mask = [0] * len(tokens)
    mask[query_output_start: query_output_start + cfg.tape_len] = [1] * cfg.tape_len
    return Episode(tokens, mask, task, qx, qy, query_sep_index, query_output_start)
```

Implement `sample_training_batch()` by cycling tasks via `(step * batch_size + row) % len(TRAIN_TASKS)` and seeding each row from `train_seed_root + step * batch_size + row`; return tensors plus `task_names`. This gives exact balance without relying on stochastic balancing.

- [ ] **Step 4: Run episode tests**

Run: `pytest tests/test_episodes.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transformer_study/episodes.py tests/test_episodes.py
git commit -m "feat: add in-context episode generator"
```

---

### Task 3: Tiny Causal Transformer with Residual Capture

**Files:**
- Create: `transformer_study/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `Vocabulary.size`.
- Produces: `TinyTransformer(config, vocab_size)`, `forward(tokens, return_residuals=False)` returning logits or `(logits, residuals)`.

- [ ] **Step 1: Write failing model tests**

```python
# tests/test_model.py
import torch

from transformer_study.config import smoke_config
from transformer_study.episodes import Vocabulary
from transformer_study.model import TinyTransformer


def test_forward_shapes_and_residual_count():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    model = TinyTransformer(cfg, vocab.size)
    x = torch.zeros((2, 12), dtype=torch.long)
    logits, residuals = model(x, return_residuals=True)
    assert logits.shape == (2, 12, vocab.size)
    assert len(residuals) == cfg.layers + 1
    assert all(r.shape == (2, 12, cfg.d_model) for r in residuals)


def test_eval_forward_is_deterministic():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(1)
    model = TinyTransformer(cfg, vocab.size).eval()
    x = torch.randint(0, vocab.size, (2, 12))
    assert torch.equal(model(x), model(x))


def test_future_token_cannot_change_past_logits():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(2)
    model = TinyTransformer(cfg, vocab.size).eval()
    a = torch.randint(0, vocab.size, (1, 10))
    b = a.clone(); b[0, 9] = (b[0, 9] + 1) % vocab.size
    assert torch.allclose(model(a)[:, :9], model(b)[:, :9], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model.py -q`

Expected: import failure.

- [ ] **Step 3: Implement pre-LN decoder blocks**

Use `torch.nn.MultiheadAttention(batch_first=True)` with an upper-triangular boolean causal mask. Each block performs:

```python
h = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal, need_weights=False)[0]
x = h + self.mlp(self.ln2(h))
```

`TinyTransformer.forward()` must append `x` after token+position embeddings and after every complete block. Position embeddings need capacity exactly `episode_length(gate0_config())`; the constructor should derive `max_seq_len` from config rather than a magic constant.

- [ ] **Step 4: Run model tests**

Run: `pytest tests/test_model.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transformer_study/model.py tests/test_model.py
git commit -m "feat: add residual-capturing causal transformer"
```

---

### Task 4: Masked Training, Greedy Query Generation, and Behavioral Metrics

**Files:**
- Create: `transformer_study/train.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: `TinyTransformer`, episode batches, `ExperimentConfig`.
- Produces: `masked_next_token_loss()`, `train_model()`, `greedy_query_output()`, `evaluate_task()`, `BehaviorMetrics`.

- [ ] **Step 1: Write failing masked-loss and generation tests**

```python
# tests/test_training.py
import torch

from transformer_study.train import masked_next_token_loss


def test_masked_loss_ignores_unmasked_positions():
    logits = torch.zeros((1, 5, 4))
    tokens = torch.tensor([[0, 1, 2, 3, 0]])
    mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)
    loss1 = masked_next_token_loss(logits, tokens, mask)
    logits[:, 0:2, :] = 1000.0
    loss2 = masked_next_token_loss(logits, tokens, mask)
    assert torch.allclose(loss1, loss2)
```

The implementation must respect autoregressive shifting: logits at position `i-1` predict token `i`, so a query-output mask on token positions must be shifted when selecting logits.

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_training.py -q`

Expected: import failure.

- [ ] **Step 3: Implement training and greedy decoding**

Core loss:

```python
def masked_next_token_loss(logits, tokens, token_mask):
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target = tokens[:, 1:].reshape(-1)
    keep = token_mask[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(pred[keep], target[keep])
```

`train_model()` must:
- set Python/NumPy/PyTorch seeds from `model_seed`;
- create AdamW with exact config lr/weight decay;
- generate a deterministic balanced batch per step;
- clip gradient norm at 1.0;
- reject non-finite loss immediately;
- return the trained model plus a compact loss history.

`greedy_query_output()` must feed the prompt ending at query `<SEP>`, append exactly four argmax symbol tokens, and raise if a structural special token is selected; this is an engineering failure because behavioral accuracy would otherwise be ambiguous.

`evaluate_task()` must create fixed held-out episodes from `eval_seed_root + task_index * 100_000 + episode_index` and return exact-sequence accuracy, token accuracy, and per-episode correctness.

- [ ] **Step 4: Run training tests plus a 2-step local smoke**

Run: `pytest tests/test_training.py -q`

Then run a short Python snippet constructing `smoke_config()`, training for 2 steps via `dataclasses.replace`, and confirming loss is finite.

Expected: tests pass; finite loss.

- [ ] **Step 5: Commit**

```bash
git add transformer_study/train.py tests/test_training.py
git commit -m "feat: add masked meta-learning training loop"
```

---

### Task 5: Residual Banks and Closed-Form Geometry Analysis

**Files:**
- Create: `transformer_study/residuals.py`
- Create: `transformer_study/analysis.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Produces: `ResidualBank`, `extract_paired_bank()`, `fit_affine_ridge()`, `apply_affine()`, `compose_affine()`, `ridge_classifier()`, `orthogonal_matrix()`, `known_task_basis()`, `rho_perp()`, and layer-level analysis functions returning plain dict/list records.

- [ ] **Step 1: Write synthetic failing tests for the mathematics before connecting model residuals**

```python
# tests/test_analysis.py
import numpy as np

from transformer_study.analysis import (
    apply_affine, compose_affine, fit_affine_ridge,
    known_task_basis, orthogonal_matrix, rho_perp,
)


def test_affine_ridge_recovers_known_map():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 5))
    w = rng.normal(size=(5, 5))
    b = rng.normal(size=(5,))
    y = x @ w + b
    wh, bh = fit_affine_ridge(x, y, ridge=1e-8)
    assert np.mean((apply_affine(x, wh, bh) - y) ** 2) < 1e-8


def test_affine_composition_includes_bias():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 4))
    w1, w2 = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    b1, b2 = rng.normal(size=4), rng.normal(size=4)
    wc, bc = compose_affine(w1, b1, w2, b2)
    direct = (x @ w1 + b1) @ w2 + b2
    assert np.allclose(x @ wc + bc, direct)


def test_seeded_orthogonal_scramble_preserves_geometry():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 12))
    q = orthogonal_matrix(12, seed=5000)
    assert np.allclose(q.T @ q, np.eye(12), atol=1e-10)
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(x @ q, axis=1), atol=1e-10)
    assert np.allclose(np.linalg.norm(x[:, None] - x[None, :], axis=2),
                       np.linalg.norm((x @ q)[:, None] - (x @ q)[None, :], axis=2), atol=1e-10)


def test_known_span_gives_zero_and_orthogonal_gives_one():
    means = np.array([[1., 0., 0.], [-1., 0., 0.]])
    center, basis = known_task_basis(means)
    assert rho_perp(np.array([2., 0., 0.]), center, basis) < 1e-10
    assert rho_perp(np.array([0., 1., 0.]), center, basis) > 0.999999
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_analysis.py -q`

Expected: import failure.

- [ ] **Step 3: Implement numerically explicit ridge / projection primitives**

Use augmented coordinates for affine ridge but do not regularize the bias column:

```python
def fit_affine_ridge(x, y, ridge=1e-3):
    xa = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    reg = np.eye(xa.shape[1]) * ridge
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return theta[:-1], theta[-1]
```

`orthogonal_matrix()` must QR a seeded Gaussian matrix and multiply columns by the sign of `diag(R)`, treating zeros as +1, so repeated platforms use the same convention.

`known_task_basis()` centers task means by their global mean, runs SVD, and keeps singular vectors above `max(shape) * np.finfo(dtype).eps * s[0]`.

- [ ] **Step 4: Implement residual extraction and six analyses**

`ResidualBank` stores, per layer and task:
- `states: np.ndarray[n, d]`;
- `query_inputs`;
- `targets`;
- `correct: np.ndarray[n, bool]`.

`extract_paired_bank()` must use a shared query tape for all tasks at each row but separate demonstration RNG seeds derived from `(bank_seed_root, row, task_index)`.

Analysis records must include at minimum:
- separability: layer, classifier accuracy, shuffled accuracy, rank, singular values;
- map rows: layer, source, target, nmse, cosine, mean-baseline nmse, random-pair nmse;
- composition rows: layer, A, B, C, action nmse, direct-map nmse, parameter disagreement, random-map action nmse;
- scramble rows: layer, probe type, original accuracy, scrambled accuracy, difference;
- novelty rows: layer, task, mean, median, std, q10, q25, q75, q90, n, competence flag;
- correct/incorrect DELTA split only when both groups have at least 20 samples, else `split_status="underpowered"`.

- [ ] **Step 5: Run mathematical and residual-analysis tests**

Add a fake residual bank with known labels to verify shuffled-label accuracy is lower than true-label accuracy and random-pair map error exceeds correctly paired map error.

Run: `pytest tests/test_analysis.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add transformer_study/residuals.py transformer_study/analysis.py tests/test_analysis.py
git commit -m "feat: add residual geometry analyses and null controls"
```

---

### Task 6: Experiment CLI, Metrics Schema, Receipt, CSVs, and Plots

**Files:**
- Create: `transformer_study/experiment.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces CLI: `python -m transformer_study.experiment --preset {smoke,gate0} --output PATH`
- Produces CLI: `python -m transformer_study.experiment --validate PATH`
- Produces exact artifact set required by the spec.

- [ ] **Step 1: Write the failing receipt-schema smoke test**

```python
# tests/test_smoke.py
import json
from pathlib import Path

from transformer_study.experiment import run_experiment, validate_receipt


def test_smoke_run_writes_complete_receipt(tmp_path: Path):
    out = tmp_path / "gate0"
    run_experiment("smoke", out)
    validate_receipt(out)
    expected = {
        "config.json", "metrics.json", "RESULTS.md", "model.pt",
        "task_geometry.csv", "linear_maps.csv", "composition.csv",
        "novelty.csv", "scramble.csv",
    }
    assert expected <= {p.name for p in out.iterdir()}
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["schema_version"] == 1
    assert set(metrics) >= {"behavior", "separability", "linear_maps", "composition", "scramble", "novelty", "interpretation"}
```

- [ ] **Step 2: Run smoke test to verify failure**

Run: `pytest tests/test_smoke.py -q`

Expected: import failure.

- [ ] **Step 3: Implement the versioned metrics schema**

`metrics.json` top-level shape is fixed to:

```json
{
  "schema_version": 1,
  "status": "complete",
  "behavior": {
    "trained_average_exact": 0.0,
    "tasks": {
      "copy": {"exact": 0.0, "token": 0.0, "competent": false}
    },
    "delta_mod": {"exact": 0.0, "token": 0.0, "competent": false}
  },
  "separability": [],
  "linear_maps": [],
  "composition": [],
  "scramble": [],
  "novelty": [],
  "interpretation": {
    "delta_mode": "solved|unsolved",
    "notes": []
  }
}
```

`validate_receipt()` must reject:
- missing required files;
- missing required top-level keys;
- any NaN/Inf recursively in `metrics.json`;
- `status != "complete"`;
- wrong `schema_version`;
- missing all trained task behavior rows;
- missing DELTA_MOD behavior row.

It must not reject low scientific scores.

- [ ] **Step 4: Implement orchestration and deterministic output tables**

`run_experiment()` sequence:
1. load preset and write `config.json` immediately;
2. train model;
3. evaluate six trained tasks plus DELTA_MOD;
4. extract fit/test paired banks;
5. run all six analyses;
6. save `model.pt`;
7. write sorted CSVs with stable column order;
8. write `metrics.json` atomically via temporary file + rename;
9. render `RESULTS.md` with thresholds and caveats;
10. render PNG plots using matplotlib default colors/styles only;
11. validate the receipt.

The receipt must explicitly emit one of these lines for DELTA_MOD:
- `Novelty interpretation: eligible (DELTA_MOD exact >= 0.50).`
- `Novelty interpretation: NOT ELIGIBLE; DELTA_MOD is behaviorally unsolved.`

No result may say that the transformer "stores algorithms as vectors" unless a later gate adds evidence beyond this spec.

- [ ] **Step 5: Add CLI and run the smoke experiment**

CLI behavior:

```bash
python -m transformer_study.experiment --preset smoke --output /tmp/transformer-study-smoke
python -m transformer_study.experiment --validate /tmp/transformer-study-smoke
```

Expected: both commands exit 0 and produce the complete receipt even if all accuracies are poor.

- [ ] **Step 6: Run the full local test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add transformer_study/experiment.py tests/test_smoke.py
git commit -m "feat: add Gate 0 experiment receipt and CLI"
```

---

### Task 7: GitHub Actions, README, and Final Verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/gate0.yml`
- Create: `README.md`

**Interfaces:**
- CI command: `pytest -q` followed by an 8-step smoke experiment.
- Full workflow command: `python -m transformer_study.experiment --preset gate0 --output artifacts/gate0`.

- [ ] **Step 1: Add cheap push/PR CI**

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[test]'
      - run: pytest -q
      - run: python -m transformer_study.experiment --preset smoke --output artifacts/smoke
      - run: python -m transformer_study.experiment --validate artifacts/smoke
```

- [ ] **Step 2: Add manual-only full Gate 0 workflow**

```yaml
# .github/workflows/gate0.yml
name: Gate 0
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
      - run: python -m transformer_study.experiment --preset gate0 --output artifacts/gate0
      - run: python -m transformer_study.experiment --validate artifacts/gate0
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: transformer-study-gate0
          path: artifacts/gate0
```

The experiment step itself remains allowed to fail for engineering errors; `if: always()` only guarantees artifact upload for postmortem.

- [ ] **Step 3: Write README focused on reproducibility and interpretation**

README must contain:
- exact scientific question;
- six trained tasks and one held-out task;
- `pip install -e '.[test]'`;
- `pytest -q`;
- smoke CLI;
- manual GitHub Actions Gate 0 instructions;
- explanation that 0.80/0.50 are interpretation filters, not test gates;
- explicit warning that orthogonal scramble is a coordinate-invariance control, not evidence for random transformers;
- explicit warning that high DELTA_MOD orthogonal energy is uninterpretable as novel computation if DELTA_MOD exact accuracy < 0.50.

- [ ] **Step 4: Run verification before claiming completion**

Run locally:

```bash
python -m pip install -e '.[test]'
pytest -q
python -m transformer_study.experiment --preset smoke --output /tmp/transformer-study-smoke
python -m transformer_study.experiment --validate /tmp/transformer-study-smoke
```

Expected: all exit 0.

Then inspect `/tmp/transformer-study-smoke/RESULTS.md` and confirm low/negative scientific results are rendered as results rather than exceptions.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows README.md
git commit -m "ci: add CPU Gate 0 workflows and study guide"
```

- [ ] **Step 6: Push branch / open PR and let CI prove the reference environment**

The implementation branch should target `main`. Do not merge before the CI workflow is green. After merge, manually dispatch `Gate 0`; use its actual runtime and receipt to decide whether the approved calibration order is needed.

## Plan Self-Review

- Spec coverage: all trained/held-out tasks, query-only loss, model dimensions, deterministic seeds, two disjoint paired banks, six analyses, controls, thresholds, receipt files, CI philosophy, and manual full workflow are assigned to explicit tasks.
- Placeholder scan: no TBD/TODO/"similar to" implementation gaps remain; implementation details that are not pasted verbatim are constrained by exact interfaces and invariants.
- Type consistency: `ExperimentConfig`, `TaskName`, `Episode`, `TinyTransformer`, residual banks, affine map orientation (`row @ W + b`), and CLI names are consistent across tasks.
- Scope: physical inter-block Q insertion, multiple training seeds, nonlinear maps, larger models, and fine-tuning on DELTA_MOD remain intentionally outside Gate 0.