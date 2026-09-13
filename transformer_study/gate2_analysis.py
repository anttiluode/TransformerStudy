from __future__ import annotations

import numpy as np

from .config import ExperimentConfig


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
