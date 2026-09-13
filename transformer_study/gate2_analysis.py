from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch

from .analysis_core import apply_affine, fit_affine_ridge, ridge_classifier
from .config import ExperimentConfig
from .episodes import Vocabulary, make_episode, make_eval_prompt
from .tasks import TaskName
from .train import evaluate_task, greedy_query_output


FOCAL_TASKS = (TaskName.SORT, TaskName.PREFIX_PARITY)


@dataclass
class RouteBank:
    residuals: dict[int, dict[TaskName, np.ndarray]]
    gelu: dict[int, dict[TaskName, np.ndarray]]
    attention: dict[int, dict[TaskName, np.ndarray]]
    prompts: dict[TaskName, list[list[int]]]
    targets: dict[TaskName, np.ndarray]
    query_inputs: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.query_inputs))


def gate2_seed_ranges(cfg: ExperimentConfig) -> dict[str, set[int]]:
    """Return the primary row-seed ranges used by Gate 2 analysis banks."""
    return {
        "route_reference": set(
            range(cfg.route_reference_seed_root, cfg.route_reference_seed_root + cfg.route_reference)
        ),
        "causal_patch": set(
            range(cfg.causal_patch_seed_root, cfg.causal_patch_seed_root + cfg.causal_patch)
        ),
        "pruning_eval": set(
            range(cfg.pruning_seed_root, cfg.pruning_seed_root + cfg.pruning_eval)
        ),
    }


def norm_matched_random(source: np.ndarray, target_like: np.ndarray, seed: int) -> np.ndarray:
    """Move each source row in a random direction by the target-like displacement norm."""
    source = np.asarray(source, dtype=np.float64)
    target_like = np.asarray(target_like, dtype=np.float64)
    if source.shape != target_like.shape or source.ndim != 2:
        raise ValueError("source and target_like must be matrices with identical shape")
    displacement = target_like - source
    lengths = np.linalg.norm(displacement, axis=1)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=source.shape)
    direction_norms = np.linalg.norm(directions, axis=1)
    direction_norms[direction_norms == 0.0] = 1.0
    directions = directions / direction_norms[:, None]
    out = source + directions * lengths[:, None]
    zero = lengths == 0.0
    if np.any(zero):
        out[zero] = source[zero]
    return out


def pruning_indices(
    selectivity: np.ndarray,
    *,
    width: int,
    count: int,
    seed: int,
) -> dict[str, list[int]]:
    selectivity = np.asarray(selectivity, dtype=np.float64)
    if selectivity.ndim != 1 or len(selectivity) != int(width):
        raise ValueError("selectivity must be a vector matching width")
    if not 0 < int(count) <= int(width):
        raise ValueError("count must be between 1 and width")
    most = np.argsort(-selectivity, kind="stable")[:count].astype(int).tolist()
    least = np.argsort(selectivity, kind="stable")[:count].astype(int).tolist()
    rng = np.random.default_rng(seed)
    random = rng.choice(width, size=count, replace=False).astype(int).tolist()
    return {
        "most_selective": most,
        "least_selective": least,
        "random": random,
    }


def _row_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("cosine inputs must be matrices with identical shape")
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return numerator / denominator


def route_target_shift(
    patched: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    """Positive means the patched route is more target-like than source-like."""
    return float(np.mean(_row_cosine(patched, target) - _row_cosine(patched, source)))


def _normalized_row_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = np.linalg.norm(a - b, axis=1)
    denominator = np.linalg.norm(b, axis=1) + eps
    return numerator / denominator


@torch.no_grad()
def extract_route_bank(
    model,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    n: int,
    seed_root: int,
    tasks: tuple[TaskName, ...] | list[TaskName] = FOCAL_TASKS,
    device: str | torch.device = "cpu",
) -> RouteBank:
    """Capture paired residual and nonlinear-route fingerprints at the query separator."""
    device = torch.device(device)
    tasks = list(tasks)
    residual_lists = {
        layer: {task: [] for task in tasks} for layer in range(cfg.layers + 1)
    }
    gelu_lists = {
        block: {task: [] for task in tasks} for block in range(cfg.layers)
    }
    attention_lists = {
        block: {task: [] for task in tasks} for block in range(cfg.layers)
    }
    prompts = {task: [] for task in tasks}
    targets = {task: [] for task in tasks}
    queries: list[list[int]] = []
    model.eval()

    for row in range(n):
        query_rng = np.random.default_rng(seed_root + row)
        query = query_rng.integers(0, cfg.modulus, size=cfg.tape_len).tolist()
        queries.append(query)
        for task_index, task in enumerate(tasks):
            demo_seed = seed_root + 1_000_000 + row * 10_000 + task_index * 101
            demo_rng = np.random.default_rng(demo_seed)
            episode = make_episode(task, cfg, demo_rng, vocab, query_input=query)
            prompt = make_eval_prompt(episode)
            prompts[task].append(prompt)
            targets[task].append(episode.query_target)
            x = torch.tensor([prompt], dtype=torch.long, device=device)
            _, residuals, diagnostics = model(
                x,
                return_residuals=True,
                return_diagnostics=True,
            )
            for layer, residual in enumerate(residuals):
                residual_lists[layer][task].append(
                    residual[0, -1].detach().cpu().numpy().astype(np.float64, copy=True)
                )
            for block, diag in enumerate(diagnostics):
                gelu_lists[block][task].append(
                    diag["gelu_derivative"][0, -1]
                    .detach().cpu().numpy().astype(np.float64, copy=True)
                )
                attention_lists[block][task].append(
                    diag["attention_weights"][0, :, -1, :]
                    .reshape(-1).detach().cpu().numpy().astype(np.float64, copy=True)
                )

    residuals = {
        layer: {task: np.stack(values, axis=0) for task, values in task_map.items()}
        for layer, task_map in residual_lists.items()
    }
    gelu = {
        block: {task: np.stack(values, axis=0) for task, values in task_map.items()}
        for block, task_map in gelu_lists.items()
    }
    attention = {
        block: {task: np.stack(values, axis=0) for task, values in task_map.items()}
        for block, task_map in attention_lists.items()
    }
    return RouteBank(
        residuals=residuals,
        gelu=gelu,
        attention=attention,
        prompts=prompts,
        targets={task: np.asarray(values, dtype=np.int64) for task, values in targets.items()},
        query_inputs=np.asarray(queries, dtype=np.int64),
    )


def fit_focal_maps(
    fit_bank,
    ridge: float,
    patch_layers: tuple[int, ...] = (1, 2),
) -> dict:
    """Fit only the preregistered SORT/PREFIX_PARITY bidirectional maps."""
    affine: dict[tuple[int, TaskName, TaskName], tuple[np.ndarray, np.ndarray]] = {}
    means: dict[tuple[int, TaskName], np.ndarray] = {}
    for layer in patch_layers:
        if layer not in fit_bank.states:
            raise ValueError(f"patch layer {layer} is not present in fit bank")
        for task in FOCAL_TASKS:
            means[(layer, task)] = np.asarray(
                fit_bank.states[layer][task], dtype=np.float64
            ).mean(axis=0)
        for source, target in permutations(FOCAL_TASKS, 2):
            affine[(layer, source, target)] = fit_affine_ridge(
                fit_bank.states[layer][source],
                fit_bank.states[layer][target],
                ridge=ridge,
            )
    return {"affine": affine, "means": means}


def _mean_pairwise_cosine_distance(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    unit = x / norms
    cosine = unit @ unit.T
    upper = np.triu_indices(len(x), k=1)
    return float(np.mean(1.0 - cosine[upper]))


def _mean_between_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    au = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bu = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(1.0 - au @ bu.T))


def analyze_route_geometry(bank: RouteBank, ridge: float, seed: int) -> list[dict]:
    rows: list[dict] = []
    if bank.n < 4:
        raise ValueError("route geometry needs at least four paired rows")
    split = max(2, bank.n // 2)
    if split >= bank.n:
        split = bank.n - 1
    rng = np.random.default_rng(seed)

    for block in sorted(bank.gelu):
        for feature, store in (("gelu", bank.gelu), ("attention", bank.attention)):
            a = np.asarray(store[block][FOCAL_TASKS[0]], dtype=np.float64)
            b = np.asarray(store[block][FOCAL_TASKS[1]], dtype=np.float64)
            x_fit = np.concatenate([a[:split], b[:split]], axis=0)
            y_fit = np.concatenate([
                np.zeros(split, dtype=np.int64),
                np.ones(split, dtype=np.int64),
            ])
            x_test = np.concatenate([a[split:], b[split:]], axis=0)
            y_test = np.concatenate([
                np.zeros(bank.n - split, dtype=np.int64),
                np.ones(bank.n - split, dtype=np.int64),
            ])
            shuffled = y_fit.copy()
            rng.shuffle(shuffled)
            rows.append({
                "block": int(block),
                "feature": feature,
                "classifier_accuracy": ridge_classifier(
                    x_fit, y_fit, x_test, y_test, ridge=ridge
                ),
                "shuffled_accuracy": ridge_classifier(
                    x_fit, shuffled, x_test, y_test, ridge=ridge
                ),
                "within_sort_distance": _mean_pairwise_cosine_distance(a),
                "within_prefix_parity_distance": _mean_pairwise_cosine_distance(b),
                "between_distance": _mean_between_cosine_distance(a, b),
            })
    return rows


def _patch_values_for_direction(
    bank: RouteBank,
    maps: dict,
    layer: int,
    source: TaskName,
    target: TaskName,
    seed: int,
) -> dict[str, np.ndarray | None]:
    source_states = np.asarray(bank.residuals[layer][source], dtype=np.float64)
    target_states = np.asarray(bank.residuals[layer][target], dtype=np.float64)
    mu_source = maps["means"][(layer, source)]
    mu_target = maps["means"][(layer, target)]
    translation = source_states + mu_target - mu_source
    w, b = maps["affine"][(layer, source, target)]
    affine = apply_affine(source_states, w, b)
    return {
        "identity": None,
        "translation": translation,
        "affine": affine,
        "random_norm": norm_matched_random(source_states, affine, seed=seed),
        "true_target": target_states,
    }


def _score_predictions(predictions: list[list[int]], targets: np.ndarray) -> tuple[float, float]:
    target_rows = np.asarray(targets, dtype=np.int64)
    pred_rows = np.asarray(predictions, dtype=np.int64)
    exact = float(np.mean(np.all(pred_rows == target_rows, axis=1)))
    token = float(np.mean(pred_rows == target_rows))
    return exact, token


@torch.no_grad()
def analyze_causal_transport(
    model,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    bank: RouteBank,
    maps: dict,
    patch_layers: tuple[int, ...] = (1, 2),
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[list[dict], list[dict]]:
    """Patch one query-separator residual and measure behavior plus downstream routing."""
    device = torch.device(device)
    behavior_rows: list[dict] = []
    route_rows: list[dict] = []
    model.eval()

    for direction_index, (source, target) in enumerate(permutations(FOCAL_TASKS, 2)):
        for layer in patch_layers:
            method_values = _patch_values_for_direction(
                bank,
                maps,
                layer,
                source,
                target,
                seed=seed + direction_index * 1000 + layer * 17,
            )
            method_scores: dict[str, dict[str, float]] = {}
            method_route_accumulator: dict[tuple[str, int, str], list[np.ndarray]] = {}

            for method, values in method_values.items():
                predictions: list[list[int]] = []
                for row in range(bank.n):
                    prompt = bank.prompts[source][row]
                    patch = None
                    if values is not None:
                        patch = {
                            "layer": int(layer),
                            "position": len(prompt) - 1,
                            "value": np.asarray(values[row], dtype=np.float32),
                        }
                    predictions.append(
                        greedy_query_output(
                            model,
                            prompt,
                            vocab,
                            cfg.tape_len,
                            device=device,
                            residual_patch=patch,
                        )
                    )

                    x = torch.tensor([prompt], dtype=torch.long, device=device)
                    _, diagnostics = model(
                        x,
                        return_diagnostics=True,
                        residual_patch=patch,
                    )
                    for block in range(layer, cfg.layers):
                        method_route_accumulator.setdefault(
                            (method, block, "gelu"), []
                        ).append(
                            diagnostics[block]["gelu_derivative"][0, -1]
                            .detach().cpu().numpy().astype(np.float64, copy=True)
                        )
                        method_route_accumulator.setdefault(
                            (method, block, "attention"), []
                        ).append(
                            diagnostics[block]["attention_weights"][0, :, -1, :]
                            .reshape(-1).detach().cpu().numpy().astype(np.float64, copy=True)
                        )

                source_exact, source_token = _score_predictions(
                    predictions, bank.targets[source]
                )
                target_exact, target_token = _score_predictions(
                    predictions, bank.targets[target]
                )
                method_scores[method] = {
                    "source_exact": source_exact,
                    "source_token": source_token,
                    "target_exact": target_exact,
                    "target_token": target_token,
                }

            identity = method_scores["identity"]
            for method in ("identity", "translation", "affine", "random_norm", "true_target"):
                scores = method_scores[method]
                behavior_rows.append({
                    "source": source.value,
                    "target": target.value,
                    "patch_layer": int(layer),
                    "method": method,
                    "n": int(bank.n),
                    **scores,
                    "target_exact_shift_vs_identity": float(
                        scores["target_exact"] - identity["target_exact"]
                    ),
                    "target_token_shift_vs_identity": float(
                        scores["target_token"] - identity["target_token"]
                    ),
                    "source_exact_shift_vs_identity": float(
                        scores["source_exact"] - identity["source_exact"]
                    ),
                    "source_token_shift_vs_identity": float(
                        scores["source_token"] - identity["source_token"]
                    ),
                })

                for block in range(layer, cfg.layers):
                    for feature, store in (("gelu", bank.gelu), ("attention", bank.attention)):
                        patched = np.stack(
                            method_route_accumulator[(method, block, feature)], axis=0
                        )
                        source_route = np.asarray(store[block][source], dtype=np.float64)
                        target_route = np.asarray(store[block][target], dtype=np.float64)
                        source_cos = _row_cosine(patched, source_route)
                        target_cos = _row_cosine(patched, target_route)
                        route_rows.append({
                            "source": source.value,
                            "target": target.value,
                            "patch_layer": int(layer),
                            "method": method,
                            "downstream_block": int(block),
                            "feature": feature,
                            "n": int(bank.n),
                            "source_cosine": float(np.mean(source_cos)),
                            "target_cosine": float(np.mean(target_cos)),
                            "target_minus_source_cosine": float(
                                np.mean(target_cos - source_cos)
                            ),
                            "source_normalized_distance": float(
                                np.mean(_normalized_row_distance(patched, source_route))
                            ),
                            "target_normalized_distance": float(
                                np.mean(_normalized_row_distance(patched, target_route))
                            ),
                        })
    return behavior_rows, route_rows


def analyze_pruning(
    model,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    route_bank: RouteBank,
    seed: int,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Mask exactly 10% of one block's MLP units using three preregistered strategies."""
    device = torch.device(device)
    count = max(1, int(round(cfg.mlp_hidden * 0.10)))
    baseline = {
        task: evaluate_task(
            model,
            task,
            cfg,
            vocab,
            n=cfg.pruning_eval,
            seed_root=cfg.pruning_seed_root,
            device=device,
        )
        for task in FOCAL_TASKS
    }
    rows: list[dict] = []

    for block in range(cfg.layers):
        selectivity = np.abs(
            np.mean(route_bank.gelu[block][FOCAL_TASKS[0]], axis=0)
            - np.mean(route_bank.gelu[block][FOCAL_TASKS[1]], axis=0)
        )
        strategies = pruning_indices(
            selectivity,
            width=cfg.mlp_hidden,
            count=count,
            seed=seed + block * 101,
        )
        for strategy, indices in strategies.items():
            mask = torch.ones(cfg.mlp_hidden, dtype=torch.float32, device=device)
            mask[torch.tensor(indices, dtype=torch.long, device=device)] = 0.0
            for task in FOCAL_TASKS:
                result = evaluate_task(
                    model,
                    task,
                    cfg,
                    vocab,
                    n=cfg.pruning_eval,
                    seed_root=cfg.pruning_seed_root,
                    device=device,
                    mlp_masks={block: mask},
                )
                rows.append({
                    "block": int(block),
                    "strategy": strategy,
                    "task": task.value,
                    "masked_count": int(count),
                    "masked_indices": indices,
                    "baseline_exact": float(baseline[task].exact_accuracy),
                    "exact": float(result.exact_accuracy),
                    "exact_delta": float(
                        result.exact_accuracy - baseline[task].exact_accuracy
                    ),
                    "baseline_token": float(baseline[task].token_accuracy),
                    "token": float(result.token_accuracy),
                    "token_delta": float(
                        result.token_accuracy - baseline[task].token_accuracy
                    ),
                })
    return rows
