import numpy as np

from transformer_study.config import gate2_config
from transformer_study.gate2_analysis import (
    gate2_seed_ranges,
    norm_matched_random,
    pruning_indices,
    route_target_shift,
)


def test_random_norm_matches_affine_displacement():
    source = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    affine = np.array([[2.0, 2.0, 3.0], [0.5, 2.0, 2.0]])
    random = norm_matched_random(source, affine, seed=7)
    expected = np.linalg.norm(affine - source, axis=1)
    actual = np.linalg.norm(random - source, axis=1)
    assert np.allclose(actual, expected, atol=1e-10)


def test_random_norm_preserves_zero_displacement():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    random = norm_matched_random(source, source.copy(), seed=11)
    assert np.array_equal(random, source)


def test_pruning_indices_are_exact_and_deterministic():
    selectivity = np.linspace(0.0, 1.0, 96)
    first = pruning_indices(selectivity, width=96, count=10, seed=9)
    second = pruning_indices(selectivity, width=96, count=10, seed=9)
    assert first == second
    assert set(first) == {"most_selective", "least_selective", "random"}
    assert all(len(indices) == 10 for indices in first.values())
    assert len(set(first["random"])) == 10
    assert first["most_selective"] == list(range(95, 85, -1))
    assert first["least_selective"] == list(range(10))


def test_gate2_seed_ranges_are_disjoint():
    cfg = gate2_config()
    ranges = gate2_seed_ranges(cfg)
    values = list(ranges.values())
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            assert left.isdisjoint(right)


def test_route_target_shift_has_expected_sign():
    patched = np.array([[1.0, 0.0], [0.9, 0.1]])
    source = np.array([[0.0, 1.0], [0.1, 0.9]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])
    shift = route_target_shift(patched, source, target)
    assert shift > 0.0
