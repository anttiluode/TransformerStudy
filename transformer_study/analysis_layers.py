from itertools import permutations
from typing import Iterable

import numpy as np

from .analysis_core import (
    apply_affine, compose_affine, fit_affine_ridge, known_task_basis,
    mean_cosine, normalized_mse, orthogonal_matrix, rho_perp, ridge_classifier,
)
from .tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName


def _task_stack(bank, layer: int, tasks: Iterable[TaskName]):
    xs, ys = [], []
    for label, task in enumerate(tasks):
        states = bank.states[layer][task]
        xs.append(states)
        ys.append(np.full(len(states), label, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def analyze_separability(fit_bank, test_bank, ridge: float, seed: int = 0) -> list[dict]:
    rows: list[dict] = []
    tasks = list(TRAIN_TASKS)
    rng = np.random.default_rng(seed)
    for layer in sorted(fit_bank.states):
        x_fit, y_fit = _task_stack(fit_bank, layer, tasks)
        x_test, y_test = _task_stack(test_bank, layer, tasks)
        acc = ridge_classifier(x_fit, y_fit, x_test, y_test, ridge=ridge)
        shuffled = ridge_classifier(x_fit, rng.permutation(y_fit), x_test, y_test, ridge=ridge)
        means = np.stack([fit_bank.states[layer][task].mean(axis=0) for task in tasks])
        centered_means = means - means.mean(axis=0)
        s = np.linalg.svd(centered_means, compute_uv=False)
        tol = max(centered_means.shape) * np.finfo(np.float64).eps * (s[0] if len(s) else 0.0)
        rank = int(np.sum(s > tol))
        rows.append(
            {
                "layer": int(layer),
                "classifier_accuracy": acc,
                "shuffled_accuracy": shuffled,
                "rank": rank,
                "singular_values": [float(v) for v in s],
            }
        )
    return rows


def analyze_linear_maps(fit_bank, test_bank, ridge: float, seed: int = 0) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    maps: dict[tuple[int, TaskName, TaskName], tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(seed)
    tasks = list(TRAIN_TASKS)
    for layer in sorted(fit_bank.states):
        for source, target in permutations(tasks, 2):
            x_fit = fit_bank.states[layer][source]
            y_fit = fit_bank.states[layer][target]
            x_test = test_bank.states[layer][source]
            y_test = test_bank.states[layer][target]
            w, b = fit_affine_ridge(x_fit, y_fit, ridge=ridge)
            maps[(layer, source, target)] = (w, b)
            pred = apply_affine(x_test, w, b)
            mean_pred = np.repeat(y_fit.mean(axis=0, keepdims=True), len(y_test), axis=0)
            perm = rng.permutation(len(y_fit))
            wr, br = fit_affine_ridge(x_fit, y_fit[perm], ridge=ridge)
            random_pred = apply_affine(x_test, wr, br)
            rows.append(
                {
                    "layer": int(layer),
                    "source": source.value,
                    "target": target.value,
                    "nmse": normalized_mse(pred, y_test),
                    "cosine": mean_cosine(pred, y_test),
                    "mean_baseline_nmse": normalized_mse(mean_pred, y_test),
                    "random_pair_nmse": normalized_mse(random_pred, y_test),
                }
            )
    return rows, maps


def analyze_composition(fit_bank, test_bank, maps: dict, ridge: float, seed: int = 0) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    tasks = list(TRAIN_TASKS)
    for layer in sorted(fit_bank.states):
        for a, b, c in permutations(tasks, 3):
            w_ab, b_ab = maps[(layer, a, b)]
            w_bc, b_bc = maps[(layer, b, c)]
            w_ac, b_ac = maps[(layer, a, c)]
            wc, bc = compose_affine(w_ab, b_ab, w_bc, b_bc)
            x_test = test_bank.states[layer][a]
            y_test = test_bank.states[layer][c]
            comp_pred = apply_affine(x_test, wc, bc)
            direct_pred = apply_affine(x_test, w_ac, b_ac)
            denom = np.linalg.norm(w_ac) + 1e-12
            parameter_disagreement = float(np.linalg.norm(wc - w_ac) / denom)
            random_w = rng.normal(size=w_bc.shape)
            random_w *= np.linalg.norm(w_bc) / (np.linalg.norm(random_w) + 1e-12)
            random_wc, random_bc = compose_affine(w_ab, b_ab, random_w, b_bc)
            random_pred = apply_affine(x_test, random_wc, random_bc)
            rows.append(
                {
                    "layer": int(layer),
                    "a": a.value,
                    "b": b.value,
                    "c": c.value,
                    "action_nmse": normalized_mse(comp_pred, y_test),
                    "direct_map_nmse": normalized_mse(direct_pred, y_test),
                    "parameter_disagreement": parameter_disagreement,
                    "random_map_action_nmse": normalized_mse(random_pred, y_test),
                }
            )
    return rows


def analyze_scramble(fit_bank, test_bank, ridge: float, seed: int, modulus: int) -> list[dict]:
    rows: list[dict] = []
    tasks = list(TRAIN_TASKS)
    for layer in sorted(fit_bank.states):
        x_fit, y_fit = _task_stack(fit_bank, layer, tasks)
        x_test, y_test = _task_stack(test_bank, layer, tasks)
        q = orthogonal_matrix(x_fit.shape[1], seed + layer)
        original = ridge_classifier(x_fit, y_fit, x_test, y_test, ridge=ridge)
        scrambled = ridge_classifier(x_fit @ q, y_fit, x_test @ q, y_test, ridge=ridge)
        rows.append(
            {
                "layer": int(layer),
                "probe_type": "task",
                "original_accuracy": original,
                "scrambled_accuracy": scrambled,
                "difference": scrambled - original,
            }
        )
        target_fit = np.concatenate([fit_bank.targets[t] for t in tasks])
        target_test = np.concatenate([test_bank.targets[t] for t in tasks])
        for pos in range(target_fit.shape[1]):
            original = ridge_classifier(x_fit, target_fit[:, pos], x_test, target_test[:, pos], ridge=ridge)
            scrambled = ridge_classifier(
                x_fit @ q, target_fit[:, pos], x_test @ q, target_test[:, pos], ridge=ridge
            )
            rows.append(
                {
                    "layer": int(layer),
                    "probe_type": f"output_{pos}",
                    "original_accuracy": original,
                    "scrambled_accuracy": scrambled,
                    "difference": scrambled - original,
                }
            )
    return rows


def _distribution_record(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "n": int(len(values)),
    }


def analyze_novelty(bank, trained_competence: dict[TaskName, bool], delta_competent: bool, min_split_n: int) -> list[dict]:
    rows: list[dict] = []
    tasks = list(TRAIN_TASKS)
    for layer in sorted(bank.states):
        means = np.stack([bank.states[layer][task].mean(axis=0) for task in tasks])
        center, basis = known_task_basis(means)
        for task in tasks + [HELD_OUT_TASK]:
            values = np.asarray(rho_perp(bank.states[layer][task], center, basis))
            rec = {
                "layer": int(layer),
                "task": task.value,
                **_distribution_record(values),
                "competent": bool(delta_competent if task is HELD_OUT_TASK else trained_competence[task]),
                "split_status": "all",
            }
            rows.append(rec)
        delta_values = np.asarray(rho_perp(bank.states[layer][HELD_OUT_TASK], center, basis))
        correct = np.asarray(bank.correct[HELD_OUT_TASK], dtype=bool)
        if int(correct.sum()) >= min_split_n and int((~correct).sum()) >= min_split_n:
            for label, mask in (("correct", correct), ("incorrect", ~correct)):
                rows.append(
                    {
                        "layer": int(layer),
                        "task": f"{HELD_OUT_TASK.value}_{label}",
                        **_distribution_record(delta_values[mask]),
                        "competent": bool(delta_competent),
                        "split_status": label,
                    }
                )
        else:
            rows.append(
                {
                    "layer": int(layer),
                    "task": f"{HELD_OUT_TASK.value}_correctness_split",
                    **_distribution_record(delta_values),
                    "competent": bool(delta_competent),
                    "split_status": "underpowered",
                }
            )
    return rows
