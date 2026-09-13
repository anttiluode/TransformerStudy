from __future__ import annotations

from itertools import permutations

import numpy as np

from .analysis_core import (
    apply_affine,
    centered_predict,
    compose_affine,
    fit_affine_ridge,
    fit_centered_ridge,
    mean_cosine,
    normalized_mse,
    rank_for_fraction,
    translation_predict,
    truncate_centered_correction,
)
from .tasks import TRAIN_TASKS, TaskName


def _fit_pair(x_fit: np.ndarray, y_fit: np.ndarray, ridge: float, ranks: tuple[int, ...]):
    mu_x, mu_y, centered_matrix = fit_centered_ridge(x_fit, y_fit, ridge=ridge)
    affine_w, affine_b = fit_affine_ridge(x_fit, y_fit, ridge=ridge)
    dim = centered_matrix.shape[0]
    rank_matrices = {
        int(rank): truncate_centered_correction(centered_matrix, min(int(rank), dim))
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


def _evaluate_pair(fit: dict, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    identity = np.asarray(x_test, dtype=np.float64)
    translation = translation_predict(x_test, fit["mu_x"], fit["mu_y"])
    affine = apply_affine(x_test, fit["affine_w"], fit["affine_b"])
    rank_nmse: dict[int, float] = {}
    for rank, matrix in fit["rank_matrices"].items():
        rank_pred = centered_predict(x_test, fit["mu_x"], fit["mu_y"], matrix)
        rank_nmse[rank] = normalized_mse(rank_pred, y_test)
    translation_nmse = normalized_mse(translation, y_test)
    affine_nmse = normalized_mse(affine, y_test)
    centered_true = np.asarray(y_test, dtype=np.float64) - fit["mu_y"]
    centered_source = np.asarray(x_test, dtype=np.float64) - fit["mu_x"]
    centered_full = centered_source @ fit["centered_matrix"]
    absolute_gain = float(translation_nmse - affine_nmse)
    return {
        "identity_nmse": normalized_mse(identity, y_test),
        "translation_nmse": translation_nmse,
        **{f"rank{rank}_nmse": value for rank, value in rank_nmse.items()},
        "affine_nmse": affine_nmse,
        "translation_cosine": mean_cosine(translation, y_test),
        "affine_cosine": mean_cosine(affine, y_test),
        "absolute_gain": absolute_gain,
        "relative_gain": float(absolute_gain / max(abs(translation_nmse), 1e-12)),
        "rank90": rank_for_fraction(translation_nmse, rank_nmse, affine_nmse),
        "centered_identity_nmse": normalized_mse(centered_source, centered_true),
        "full_centered_nmse": normalized_mse(centered_full, centered_true),
    }


def analyze_translation_controls(
    fit_bank,
    test_bank,
    ridge: float,
    ranks: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    affine_maps: dict[tuple[int, TaskName, TaskName], tuple[np.ndarray, np.ndarray]] = {}
    centered_maps: dict[tuple[int, TaskName, TaskName], np.ndarray] = {}
    means: dict[tuple[int, TaskName], np.ndarray] = {}
    tasks = list(TRAIN_TASKS)
    for layer in sorted(fit_bank.states):
        for task in tasks:
            means[(layer, task)] = fit_bank.states[layer][task].mean(axis=0)
        for source, target in permutations(tasks, 2):
            fitted = _fit_pair(
                fit_bank.states[layer][source],
                fit_bank.states[layer][target],
                ridge,
                ranks,
            )
            affine_maps[(layer, source, target)] = (fitted["affine_w"], fitted["affine_b"])
            centered_maps[(layer, source, target)] = fitted["centered_matrix"]
            metrics = _evaluate_pair(
                fitted,
                test_bank.states[layer][source],
                test_bank.states[layer][target],
            )
            rows.append({
                "layer": int(layer),
                "source": source.value,
                "target": target.value,
                **metrics,
            })
    return rows, {"affine": affine_maps, "centered": centered_maps, "means": means}


def analyze_correct_conditioned(
    fit_bank,
    test_bank,
    ridge: float,
    min_n: int = 20,
    ranks: tuple[int, ...] = (1, 2, 4, 8),
) -> list[dict]:
    rows: list[dict] = []
    tasks = list(TRAIN_TASKS)
    for layer in sorted(fit_bank.states):
        for source, target in permutations(tasks, 2):
            fitted = _fit_pair(
                fit_bank.states[layer][source],
                fit_bank.states[layer][target],
                ridge,
                ranks,
            )
            mask = np.asarray(test_bank.correct[source], dtype=bool) & np.asarray(
                test_bank.correct[target], dtype=bool
            )
            n = int(mask.sum())
            prefix = {
                "layer": int(layer),
                "source": source.value,
                "target": target.value,
                "n": n,
            }
            if n < min_n:
                rows.append({
                    **prefix,
                    "status": "underpowered",
                    "translation_nmse": None,
                    "affine_nmse": None,
                    "absolute_gain": None,
                    "relative_gain": None,
                    "centered_identity_nmse": None,
                    "full_centered_nmse": None,
                })
                continue
            metrics = _evaluate_pair(
                fitted,
                test_bank.states[layer][source][mask],
                test_bank.states[layer][target][mask],
            )
            rows.append({
                **prefix,
                "status": "eligible",
                "translation_nmse": metrics["translation_nmse"],
                "affine_nmse": metrics["affine_nmse"],
                "absolute_gain": metrics["absolute_gain"],
                "relative_gain": metrics["relative_gain"],
                "centered_identity_nmse": metrics["centered_identity_nmse"],
                "full_centered_nmse": metrics["full_centered_nmse"],
            })
    return rows


def analyze_composition_controls(
    fit_bank,
    test_bank,
    map_bundle: dict,
    ridge: float,
) -> list[dict]:
    del fit_bank, ridge
    rows: list[dict] = []
    tasks = list(TRAIN_TASKS)
    affine_maps = map_bundle["affine"]
    centered_maps = map_bundle["centered"]
    means = map_bundle["means"]
    for layer in sorted(test_bank.states):
        for a, b, c in permutations(tasks, 3):
            x = np.asarray(test_bank.states[layer][a], dtype=np.float64)
            y = np.asarray(test_bank.states[layer][c], dtype=np.float64)

            w_ab, b_ab = affine_maps[(layer, a, b)]
            w_bc, b_bc = affine_maps[(layer, b, c)]
            w_ac, b_ac = affine_maps[(layer, a, c)]
            w_comp, b_comp = compose_affine(w_ab, b_ab, w_bc, b_bc)
            affine_direct = apply_affine(x, w_ac, b_ac)
            affine_composed = apply_affine(x, w_comp, b_comp)

            mu_a = means[(layer, a)]
            mu_b = means[(layer, b)]
            mu_c = means[(layer, c)]
            translation_direct = translation_predict(x, mu_a, mu_c)
            via_b = translation_predict(x, mu_a, mu_b)
            translation_composed = translation_predict(via_b, mu_b, mu_c)

            m_ab = centered_maps[(layer, a, b)]
            m_bc = centered_maps[(layer, b, c)]
            m_ac = centered_maps[(layer, a, c)]
            x_centered = x - mu_a
            y_centered = y - mu_c
            centered_direct = x_centered @ m_ac
            centered_composed = x_centered @ (m_ab @ m_bc)
            denom = np.linalg.norm(m_ac) + 1e-12

            rows.append({
                "layer": int(layer),
                "a": a.value,
                "b": b.value,
                "c": c.value,
                "affine_direct_nmse": normalized_mse(affine_direct, y),
                "affine_composed_nmse": normalized_mse(affine_composed, y),
                "translation_direct_nmse": normalized_mse(translation_direct, y),
                "translation_composed_nmse": normalized_mse(translation_composed, y),
                "translation_action_difference": float(np.max(np.abs(translation_direct - translation_composed))),
                "centered_direct_nmse": normalized_mse(centered_direct, y_centered),
                "centered_composed_nmse": normalized_mse(centered_composed, y_centered),
                "centered_parameter_disagreement": float(np.linalg.norm(m_ab @ m_bc - m_ac) / denom),
            })
    return rows
