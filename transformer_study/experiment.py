from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import (
    analyze_composition,
    analyze_composition_controls,
    analyze_correct_conditioned,
    analyze_linear_maps,
    analyze_novelty,
    analyze_scramble,
    analyze_separability,
    analyze_translation_controls,
)
from .config import ExperimentConfig, gate0_config, gate1_config, gate2_config, smoke_config
from .episodes import Vocabulary
from .gate2_analysis import (
    analyze_causal_transport,
    analyze_pruning,
    analyze_route_geometry,
    extract_route_bank,
    fit_focal_maps,
)
from .plots import render_gate1_plots, render_gate2_plots, render_plots
from .receipt import json_safe, render_results, validate_finite, validate_receipt, write_csv
from .residuals import extract_paired_bank
from .tasks import HELD_OUT_TASK, TRAIN_TASKS
from .train import evaluate_task, train_model


def _preset(name: str) -> ExperimentConfig:
    if name == "gate0":
        return gate0_config()
    if name == "gate1":
        return gate1_config()
    if name == "gate2":
        return gate2_config()
    if name == "smoke":
        return smoke_config()
    raise ValueError(f"unknown preset: {name}")


def run_experiment(preset_name: str, output: str | Path) -> dict:
    cfg = _preset(preset_name)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"preset": preset_name, **cfg.to_dict()}, indent=2, sort_keys=True), encoding="utf-8"
    )

    torch.set_num_threads(1)
    vocab = Vocabulary(cfg.modulus)
    model, history = train_model(cfg, vocab)

    behavior_metrics = {}
    competence = {}
    exacts = []
    for task in TRAIN_TASKS:
        result = evaluate_task(model, task, cfg, vocab)
        row = {
            "exact": result.exact_accuracy,
            "token": result.token_accuracy,
            "competent": bool(result.exact_accuracy >= cfg.trained_competence),
        }
        behavior_metrics[task.value] = row
        competence[task] = row["competent"]
        exacts.append(result.exact_accuracy)
    delta_result = evaluate_task(model, HELD_OUT_TASK, cfg, vocab)
    delta_row = {
        "exact": delta_result.exact_accuracy,
        "token": delta_result.token_accuracy,
        "competent": bool(delta_result.exact_accuracy >= cfg.novel_competence),
    }
    behavior = {
        "trained_average_exact": float(np.mean(exacts)),
        "tasks": behavior_metrics,
        "delta_mod": delta_row,
    }

    fit_bank = extract_paired_bank(model, cfg, vocab, cfg.map_fit, cfg.map_fit_seed_root)
    test_bank = extract_paired_bank(model, cfg, vocab, cfg.map_test, cfg.map_test_seed_root)
    separability = analyze_separability(fit_bank, test_bank, cfg.ridge_lambda, seed=cfg.scramble_seed + 10)
    linear_maps, maps = analyze_linear_maps(fit_bank, test_bank, cfg.ridge_lambda, seed=cfg.scramble_seed + 20)
    composition = analyze_composition(fit_bank, test_bank, maps, cfg.ridge_lambda, seed=cfg.scramble_seed + 30)
    scramble = analyze_scramble(fit_bank, test_bank, cfg.ridge_lambda, cfg.scramble_seed, cfg.modulus)
    novelty = analyze_novelty(
        test_bank,
        trained_competence=competence,
        delta_competent=delta_row["competent"],
        min_split_n=cfg.correct_split_min_n,
    )

    translation_controls = None
    correct_conditioned = None
    composition_controls = None
    if preset_name in {"gate1", "gate2"}:
        translation_controls, gate1_maps = analyze_translation_controls(
            fit_bank, test_bank, cfg.ridge_lambda
        )
        correct_conditioned = analyze_correct_conditioned(
            fit_bank, test_bank, cfg.ridge_lambda, min_n=cfg.correct_split_min_n
        )
        composition_controls = analyze_composition_controls(
            fit_bank, test_bank, gate1_maps, cfg.ridge_lambda
        )

    route_geometry = None
    causal_transport = None
    route_shift = None
    pruning = None
    if preset_name == "gate2":
        route_bank = extract_route_bank(
            model,
            cfg,
            vocab,
            n=cfg.route_reference,
            seed_root=cfg.route_reference_seed_root,
        )
        causal_bank = extract_route_bank(
            model,
            cfg,
            vocab,
            n=cfg.causal_patch,
            seed_root=cfg.causal_patch_seed_root,
        )
        patch_layers = (1, 2)
        focal_maps = fit_focal_maps(
            fit_bank,
            cfg.ridge_lambda,
            patch_layers=patch_layers,
        )
        route_geometry = analyze_route_geometry(
            route_bank,
            cfg.ridge_lambda,
            seed=cfg.scramble_seed + 60,
        )
        causal_transport, route_shift = analyze_causal_transport(
            model,
            cfg,
            vocab,
            causal_bank,
            focal_maps,
            patch_layers=patch_layers,
            seed=cfg.scramble_seed + 70,
        )
        pruning = analyze_pruning(
            model,
            cfg,
            vocab,
            route_bank,
            seed=cfg.scramble_seed + 80,
        )

    metrics = {
        "schema_version": 1,
        "status": "complete",
        "behavior": behavior,
        "separability": separability,
        "linear_maps": linear_maps,
        "composition": composition,
        "scramble": scramble,
        "novelty": novelty,
        "interpretation": {
            "delta_mode": "solved" if delta_row["competent"] else "unsolved",
            "notes": [
                "Scientific scores never control process exit status.",
                "Orthogonal scramble is a coordinate-invariance control, not a random-transformer claim.",
                "Gate 2 uses a preregistered SORT/PREFIX_PARITY pair and causal controls; scientific weakness remains a valid result.",
            ],
        },
        "training": {
            "initial_loss": float(history[0]) if history else None,
            "final_loss": float(history[-1]) if history else None,
            "steps": len(history),
        },
    }
    if preset_name in {"gate1", "gate2"}:
        metrics["translation_controls"] = translation_controls
        metrics["correct_conditioned"] = correct_conditioned
        metrics["composition_controls"] = composition_controls
    if preset_name == "gate2":
        metrics["route_geometry"] = route_geometry
        metrics["causal_transport"] = causal_transport
        metrics["route_shift"] = route_shift
        metrics["pruning"] = pruning
    metrics = json_safe(metrics)
    validate_finite(metrics)

    torch.save(model.state_dict(), out / "model.pt")
    write_csv(out / "task_geometry.csv", separability)
    write_csv(out / "linear_maps.csv", linear_maps)
    write_csv(out / "composition.csv", composition)
    write_csv(out / "novelty.csv", novelty)
    write_csv(out / "scramble.csv", scramble)
    if preset_name in {"gate1", "gate2"}:
        write_csv(out / "translation_controls.csv", translation_controls or [])
        write_csv(out / "correct_conditioned.csv", correct_conditioned or [])
        write_csv(out / "composition_controls.csv", composition_controls or [])
    if preset_name == "gate2":
        write_csv(out / "route_geometry.csv", route_geometry or [])
        write_csv(out / "causal_transport.csv", causal_transport or [])
        write_csv(out / "route_shift.csv", route_shift or [])
        write_csv(out / "pruning.csv", pruning or [])

    render_results(
        out, preset_name, cfg, history, behavior, separability, linear_maps, composition, novelty,
        translation_controls=translation_controls,
        correct_conditioned=correct_conditioned,
        composition_controls=composition_controls,
        route_geometry=route_geometry,
        causal_transport=causal_transport,
        route_shift=route_shift,
        pruning=pruning,
    )
    render_plots(out, separability, linear_maps, novelty)
    if preset_name in {"gate1", "gate2"}:
        render_gate1_plots(out, translation_controls or [])
    if preset_name == "gate2":
        render_gate2_plots(out, causal_transport or [], route_shift or [])

    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out / "metrics.json")
    validate_receipt(out)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TransformerStudy experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=["smoke", "gate0", "gate1", "gate2"])
    group.add_argument("--validate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        validate_receipt(args.validate)
        return 0
    if args.output is None:
        parser.error("--output is required with --preset")
    run_experiment(args.preset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
