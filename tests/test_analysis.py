import numpy as np

from transformer_study.analysis import (
    apply_affine,
    compose_affine,
    fit_affine_ridge,
    known_task_basis,
    orthogonal_matrix,
    rho_perp,
    ridge_classifier,
)


def test_affine_ridge_recovers_known_map():
    rng = np.random.default_rng(1); x = rng.normal(size=(256, 5)); w = rng.normal(size=(5, 5)); b = rng.normal(size=(5,)); y = x @ w + b
    wh, bh = fit_affine_ridge(x, y, ridge=1e-8)
    assert np.mean((apply_affine(x, wh, bh) - y) ** 2) < 1e-8


def test_affine_composition_includes_bias():
    rng = np.random.default_rng(2); x = rng.normal(size=(32, 4)); w1, w2 = rng.normal(size=(4, 4)), rng.normal(size=(4, 4)); b1, b2 = rng.normal(size=4), rng.normal(size=4)
    wc, bc = compose_affine(w1, b1, w2, b2); direct = (x @ w1 + b1) @ w2 + b2
    assert np.allclose(x @ wc + bc, direct)


def test_seeded_orthogonal_scramble_preserves_geometry():
    rng = np.random.default_rng(3); x = rng.normal(size=(64, 12)); q = orthogonal_matrix(12, seed=5000)
    assert np.allclose(q.T @ q, np.eye(12), atol=1e-10)
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(x @ q, axis=1), atol=1e-10)
    assert np.allclose(np.linalg.norm(x[:, None] - x[None, :], axis=2), np.linalg.norm((x @ q)[:, None] - (x @ q)[None, :], axis=2), atol=1e-10)


def test_known_span_gives_zero_and_orthogonal_gives_one():
    means = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    center, basis = known_task_basis(means)
    assert rho_perp(np.array([2.0, 0.0, 0.0]), center, basis) < 1e-10
    assert rho_perp(np.array([0.0, 1.0, 0.0]), center, basis) > 0.999999


def test_ridge_classifier_finds_real_signal_not_shuffled_labels():
    rng = np.random.default_rng(5); x0 = rng.normal(loc=-2.0, scale=0.2, size=(64, 4)); x1 = rng.normal(loc=2.0, scale=0.2, size=(64, 4)); x = np.concatenate([x0, x1]); y = np.array([0] * 64 + [1] * 64)
    acc = ridge_classifier(x, y, x, y, ridge=1e-3); shuffled = ridge_classifier(x, rng.permutation(y), x, y, ridge=1e-3)
    assert acc > 0.99
    assert shuffled < 0.75


from transformer_study.analysis import analyze_linear_maps
from transformer_study.residuals import ResidualBank
from transformer_study.tasks import TRAIN_TASKS


def test_linear_map_analysis_beats_random_pairing_on_known_shared_latent():
    rng = np.random.default_rng(7); base_fit = rng.normal(size=(96, 4)); base_test = rng.normal(size=(96, 4)); states_fit = {0: {}}; states_test = {0: {}}; targets = {}; correct = {}
    for i, task in enumerate(TRAIN_TASKS):
        scale = 1.0 + 0.1 * i; offset = np.full(4, i * 0.3)
        states_fit[0][task] = base_fit * scale + offset; states_test[0][task] = base_test * scale + offset
        targets[task] = np.zeros((96, 4), dtype=np.int64); correct[task] = np.ones(96, dtype=bool)
    fit = ResidualBank(states_fit, np.zeros((96, 4), dtype=np.int64), targets, correct)
    test = ResidualBank(states_test, np.zeros((96, 4), dtype=np.int64), targets, correct)
    rows, _ = analyze_linear_maps(fit, test, ridge=1e-6, seed=1)
    row = next(r for r in rows if r["source"] == "copy" and r["target"] == "reverse")
    assert row["nmse"] < 1e-6
    assert row["random_pair_nmse"] > row["nmse"]

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
