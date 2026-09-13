import json
from pathlib import Path

from transformer_study.experiment import _preset, run_experiment, validate_receipt
from transformer_study.receipt import render_gate1_summary, render_gate2_summary


def test_smoke_run_writes_complete_receipt(tmp_path: Path):
    out = tmp_path / "gate0"
    run_experiment("smoke", out)
    validate_receipt(out)
    expected = {"config.json", "metrics.json", "RESULTS.md", "model.pt", "task_geometry.csv",
                "linear_maps.csv", "composition.csv", "novelty.csv", "scramble.csv"}
    assert expected <= {p.name for p in out.iterdir()}
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["schema_version"] == 1
    assert set(metrics) >= {"behavior", "separability", "linear_maps", "composition", "scramble", "novelty", "interpretation"}


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


def test_gate2_receipt_requires_gate1_and_gate2_extensions(tmp_path: Path):
    out = tmp_path / "gate2-receipt"
    run_experiment("smoke", out)
    config = json.loads((out / "config.json").read_text())
    config["preset"] = "gate2"
    (out / "config.json").write_text(json.dumps(config))

    try:
        validate_receipt(out)
    except ValueError as exc:
        assert "Gate 1" in str(exc) or "Gate 2" in str(exc) or "gate2" in str(exc)
    else:
        raise AssertionError("Gate 2 receipt unexpectedly validated without extensions")

    for filename in (
        "translation_controls.csv", "correct_conditioned.csv", "composition_controls.csv",
        "route_geometry.csv", "causal_transport.csv", "route_shift.csv", "pruning.csv",
        "gate2_route_shift.png", "gate2_causal_behavior.png",
    ):
        (out / filename).write_text("placeholder\n")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics.update({
        "translation_controls": [],
        "correct_conditioned": [],
        "composition_controls": [],
        "route_geometry": [],
        "causal_transport": [],
        "route_shift": [],
        "pruning": [],
    })
    (out / "metrics.json").write_text(json.dumps(metrics))
    validate_receipt(out)


def test_gate1_cli_preset_is_fixed_at_8000_steps():
    assert _preset("gate1").steps == 8000


def test_gate2_cli_preset_is_fixed_at_8000_steps():
    assert _preset("gate2").steps == 8000


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


def test_gate2_summary_reports_competence_causality_routing_and_pruning():
    behavior = {"tasks": {"sort": {"exact": 0.90}, "prefix_parity": {"exact": 1.0}}}
    route_geometry = [
        {"block": 1, "feature": "gelu", "classifier_accuracy": 0.9, "shuffled_accuracy": 0.5},
        {"block": 1, "feature": "attention", "classifier_accuracy": 0.8, "shuffled_accuracy": 0.5},
    ]
    causal = [
        {"source": "sort", "target": "prefix_parity", "patch_layer": 1, "method": "identity",
         "target_exact": 0.0, "target_exact_shift_vs_identity": 0.0},
        {"source": "sort", "target": "prefix_parity", "patch_layer": 1, "method": "translation",
         "target_exact": 0.1, "target_exact_shift_vs_identity": 0.1},
        {"source": "sort", "target": "prefix_parity", "patch_layer": 1, "method": "affine",
         "target_exact": 0.6, "target_exact_shift_vs_identity": 0.6},
        {"source": "sort", "target": "prefix_parity", "patch_layer": 1, "method": "random_norm",
         "target_exact": 0.1, "target_exact_shift_vs_identity": 0.1},
        {"source": "sort", "target": "prefix_parity", "patch_layer": 1, "method": "true_target",
         "target_exact": 0.8, "target_exact_shift_vs_identity": 0.8},
    ]
    shifts = [
        {"method": "affine", "feature": "gelu", "target_minus_source_cosine": 0.3},
        {"method": "translation", "feature": "gelu", "target_minus_source_cosine": 0.0},
        {"method": "random_norm", "feature": "gelu", "target_minus_source_cosine": -0.1},
    ]
    pruning = [
        {"strategy": "most_selective", "task": "sort", "exact_delta": -0.2},
        {"strategy": "least_selective", "task": "sort", "exact_delta": 0.0},
        {"strategy": "random", "task": "sort", "exact_delta": -0.05},
    ]
    text = render_gate2_summary(behavior, route_geometry, causal, shifts, pruning)
    lower = text.lower()
    assert "competent" in lower
    assert "route" in lower
    assert "true-target" in lower or "true target" in lower
    assert "affine" in lower
    assert "prun" in lower
