from dataclasses import replace

import numpy as np
import torch

from transformer_study.config import gate2_config, smoke_config
from transformer_study.episodes import Vocabulary, make_episode, make_eval_prompt
from transformer_study.gate2_analysis import (
    analyze_causal_transport,
    analyze_pruning,
    analyze_route_geometry,
    extract_route_bank,
    fit_focal_maps,
    gate2_seed_ranges,
    norm_matched_random,
    pruning_indices,
    route_target_shift,
)
from transformer_study.model import TinyTransformer
from transformer_study.residuals import extract_paired_bank
from transformer_study.tasks import TaskName
from transformer_study.train import greedy_query_output


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


def _tiny_two_layer_model():
    cfg = replace(
        smoke_config(),
        layers=2,
        route_reference=6,
        causal_patch=2,
        pruning_eval=2,
        map_fit=6,
    )
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(23)
    return cfg, vocab, TinyTransformer(cfg, vocab.size).eval()


def test_extract_route_bank_records_expected_shapes():
    cfg, vocab, model = _tiny_two_layer_model()
    bank = extract_route_bank(
        model,
        cfg,
        vocab,
        n=3,
        seed_root=9_000_000,
    )
    assert bank.n == 3
    for task in (TaskName.SORT, TaskName.PREFIX_PARITY):
        assert bank.residuals[1][task].shape == (3, cfg.d_model)
        assert bank.gelu[0][task].shape == (3, cfg.mlp_hidden)
        assert bank.attention[0][task].shape[0] == 3
        assert len(bank.prompts[task]) == 3
        assert bank.targets[task].shape == (3, cfg.tape_len)


def test_greedy_identity_patch_matches_unpatched():
    cfg, vocab, model = _tiny_two_layer_model()
    rng = np.random.default_rng(7)
    episode = make_episode(TaskName.SORT, cfg, rng, vocab)
    prompt = make_eval_prompt(episode)
    x = torch.tensor([prompt], dtype=torch.long)
    _, residuals = model(x, return_residuals=True)
    baseline = greedy_query_output(model, prompt, vocab, cfg.tape_len)
    patched = greedy_query_output(
        model,
        prompt,
        vocab,
        cfg.tape_len,
        residual_patch={
            "layer": 1,
            "position": len(prompt) - 1,
            "value": residuals[1][0, -1].clone(),
        },
    )
    assert patched == baseline


def test_route_geometry_reports_both_router_families():
    cfg, vocab, model = _tiny_two_layer_model()
    bank = extract_route_bank(model, cfg, vocab, n=6, seed_root=9_100_000)
    rows = analyze_route_geometry(bank, cfg.ridge_lambda, seed=31)
    assert {row["feature"] for row in rows} == {"gelu", "attention"}
    assert {row["block"] for row in rows} == {0, 1}
    assert all(0.0 <= row["classifier_accuracy"] <= 1.0 for row in rows)
    assert all(0.0 <= row["shuffled_accuracy"] <= 1.0 for row in rows)


def test_causal_transport_emits_preregistered_controls_and_route_shifts():
    cfg, vocab, model = _tiny_two_layer_model()
    fit_bank = extract_paired_bank(
        model,
        cfg,
        vocab,
        n=cfg.map_fit,
        bank_seed_root=9_200_000,
        tasks=[TaskName.SORT, TaskName.PREFIX_PARITY],
    )
    causal_bank = extract_route_bank(
        model,
        cfg,
        vocab,
        n=cfg.causal_patch,
        seed_root=9_300_000,
    )
    maps = fit_focal_maps(fit_bank, cfg.ridge_lambda, patch_layers=(1,))
    behavior, shifts = analyze_causal_transport(
        model,
        cfg,
        vocab,
        causal_bank,
        maps,
        patch_layers=(1,),
        seed=37,
    )
    assert {row["method"] for row in behavior} == {
        "identity", "translation", "affine", "random_norm", "true_target"
    }
    assert {(row["source"], row["target"]) for row in behavior} == {
        ("sort", "prefix_parity"), ("prefix_parity", "sort")
    }
    assert {row["feature"] for row in shifts} == {"gelu", "attention"}
    assert all(row["downstream_block"] >= row["patch_layer"] for row in shifts)


def test_pruning_analysis_masks_fixed_ten_percent_per_block():
    cfg, vocab, model = _tiny_two_layer_model()
    route_bank = extract_route_bank(model, cfg, vocab, n=6, seed_root=9_400_000)
    rows = analyze_pruning(model, cfg, vocab, route_bank, seed=41)
    assert {row["strategy"] for row in rows} == {
        "most_selective", "least_selective", "random"
    }
    assert all(row["masked_count"] == 5 for row in rows)
    assert {row["block"] for row in rows} == {0, 1}
    assert {row["task"] for row in rows} == {"sort", "prefix_parity"}
