from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .tasks import HELD_OUT_TASK, TRAIN_TASKS


REQUIRED_FILES = {
    "config.json", "metrics.json", "RESULTS.md", "model.pt",
    "task_geometry.csv", "linear_maps.csv", "composition.csv",
    "novelty.csv", "scramble.csv",
}
GATE1_FILES = {
    "translation_controls.csv", "correct_conditioned.csv", "composition_controls.csv",
}
GATE1_KEYS = {
    "translation_controls", "correct_conditioned", "composition_controls",
}


def json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(json_safe(value), sort_keys=True)
                cooked[key] = value
            writer.writerow(cooked)


def validate_finite(value: Any, path: str = "metrics") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            validate_finite(child, f"{path}[{i}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value at {path}: {value}")


def validate_receipt(output: str | Path) -> None:
    out = Path(output)
    missing = REQUIRED_FILES - {p.name for p in out.iterdir()} if out.exists() else REQUIRED_FILES
    if missing:
        raise ValueError(f"missing receipt files: {sorted(missing)}")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config.get("preset") == "gate1":
        missing_gate1_files = GATE1_FILES - {p.name for p in out.iterdir()}
        if missing_gate1_files:
            raise ValueError(f"missing Gate 1 receipt files: {sorted(missing_gate1_files)}")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if config.get("preset") == "gate1":
        missing_gate1_keys = GATE1_KEYS - set(metrics)
        if missing_gate1_keys:
            raise ValueError(f"missing Gate 1 metrics keys: {sorted(missing_gate1_keys)}")
    if metrics.get("schema_version") != 1:
        raise ValueError("metrics schema_version must be 1")
    if metrics.get("status") != "complete":
        raise ValueError("metrics status must be complete")
    required_keys = {"behavior", "separability", "linear_maps", "composition", "scramble", "novelty", "interpretation"}
    missing_keys = required_keys - set(metrics)
    if missing_keys:
        raise ValueError(f"missing metrics keys: {sorted(missing_keys)}")
    behavior = metrics["behavior"]
    expected_tasks = {task.value for task in TRAIN_TASKS}
    if set(behavior.get("tasks", {})) != expected_tasks:
        raise ValueError("behavior tasks do not contain exactly all trained tasks")
    if "delta_mod" not in behavior:
        raise ValueError("missing DELTA_MOD behavior row")
    validate_finite(metrics)


def render_gate1_summary(
    behavior: dict,
    translation_controls: list[dict],
    correct_conditioned: list[dict],
    composition_controls: list[dict],
) -> str:
    competent = [
        name for name, row in behavior.get("tasks", {}).items()
        if float(row.get("exact", 0.0)) >= 0.80
    ]
    count = len(competent)
    if count < 2:
        eligibility = "representation geometry only; fewer than two trained tasks are competent"
    elif count < 3:
        eligibility = "pairwise algorithmic interpretation eligible; centered triple composition not yet eligible"
    else:
        eligibility = "pairwise and centered triple-composition interpretation eligible"

    names = ", ".join(competent) if competent else "none"
    lines = [
        f"1. **Competence:** {count} trained tasks are competent (exact >= 0.80): {names}. {eligibility}.",
    ]

    if translation_controls:
        final_layer = max(int(r["layer"]) for r in translation_controls)
        final_rows = [r for r in translation_controls if int(r["layer"]) == final_layer]
        translation = float(np.mean([r["translation_nmse"] for r in final_rows]))
        affine = float(np.mean([r["affine_nmse"] for r in final_rows]))
        absolute = float(np.mean([r["absolute_gain"] for r in final_rows]))
        relative = float(np.mean([r["relative_gain"] for r in final_rows]))
        lines.append(
            f"2. **Translation null:** at final layer {final_layer}, mean translation NMSE is {translation:.4f} "
            f"versus full affine {affine:.4f}. No binary 'most' threshold was preregistered; the measured gain below is the decision statistic."
        )
        lines.append(
            f"3. **Full affine gain:** mean absolute gain over translation is {absolute:.4f}; "
            f"mean relative gain is {relative:.1%}."
        )
        ranks = [int(r["rank90"]) for r in final_rows if r.get("rank90") is not None]
        if ranks:
            counts = {rank: ranks.count(rank) for rank in sorted(set(ranks))}
            rank_text = ", ".join(f"rank {rank}: {n}" for rank, n in counts.items())
        else:
            rank_text = "no pair had a positive affine improvement with a measured rank90"
        lines.append(f"4. **Rank beyond translation:** {rank_text}.")
    else:
        final_layer = None
        lines += [
            "2. **Translation null:** no Gate 1 translation-control rows were available.",
            "3. **Full affine gain:** unavailable.",
            "4. **Rank beyond translation:** unavailable.",
        ]

    eligible = [
        r for r in correct_conditioned
        if r.get("status") == "eligible" and (final_layer is None or int(r["layer"]) == final_layer)
    ]
    if eligible:
        correct_gain = float(np.mean([r["absolute_gain"] for r in eligible]))
        correct_relative = float(np.mean([r["relative_gain"] for r in eligible]))
        lines.append(
            f"5. **Both-correct evaluation:** {len(eligible)} eligible final-layer rows retain mean affine gain "
            f"{correct_gain:.4f} ({correct_relative:.1%} relative)."
        )
    else:
        lines.append("5. **Both-correct evaluation:** no final-layer pair had at least 20 both-correct test rows; this diagnostic is underpowered.")

    comp_rows = [
        r for r in composition_controls
        if final_layer is None or int(r["layer"]) == final_layer
    ]
    if comp_rows:
        direct = float(np.mean([r["centered_direct_nmse"] for r in comp_rows]))
        composed = float(np.mean([r["centered_composed_nmse"] for r in comp_rows]))
        lines.append(
            f"6. **Centered composition:** mean centered direct NMSE is {direct:.4f}; composed centered NMSE is {composed:.4f}. "
            "Translation composition is an exact null by construction because task-mean offsets telescope."
        )
    else:
        lines.append(
            "6. **Centered composition:** unavailable. Translation composition remains an exact null by construction because task-mean offsets telescope."
        )
    return "\n\n".join(lines)


def render_results(
    out: Path,
    preset_name: str,
    cfg: ExperimentConfig,
    history: list[float],
    behavior: dict,
    separability: list[dict],
    linear_maps: list[dict],
    composition: list[dict],
    novelty: list[dict],
    translation_controls: list[dict] | None = None,
    correct_conditioned: list[dict] | None = None,
    composition_controls: list[dict] | None = None,
) -> None:
    lines = [
        "# Gate 0 Results", "", f"Preset: `{preset_name}`",
        f"Training steps: {cfg.steps}",
        f"Final training loss: {history[-1]:.6f}" if history else "Final training loss: n/a",
        "", "## Behavior", "",
        "| Task | Exact | Token | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for task in TRAIN_TASKS:
        row = behavior["tasks"][task.value]
        state = "competent" if row["competent"] else "below 0.80 interpretation threshold"
        lines.append(f"| {task.value} | {row['exact']:.3f} | {row['token']:.3f} | {state} |")
    delta = behavior["delta_mod"]
    delta_state = "competent" if delta["competent"] else "below 0.50 novelty threshold"
    lines.append(f"| delta_mod (held out) | {delta['exact']:.3f} | {delta['token']:.3f} | {delta_state} |")
    lines += ["", f"Trained-task mean exact accuracy: {behavior['trained_average_exact']:.3f}.", ""]
    if delta["competent"]:
        lines.append("Novelty interpretation: eligible (DELTA_MOD exact >= 0.50).")
    else:
        lines.append("Novelty interpretation: NOT ELIGIBLE; DELTA_MOD is behaviorally unsolved.")
    lines += ["", "## Geometry summary", ""]
    if separability:
        best = max(separability, key=lambda r: r["classifier_accuracy"])
        lines.append(f"Best trained-task linear separability: {best['classifier_accuracy']:.3f} at layer {best['layer']} (shuffled control {best['shuffled_accuracy']:.3f}).")
    if linear_maps:
        mean_nmse = float(np.mean([r["nmse"] for r in linear_maps]))
        random_nmse = float(np.mean([r["random_pair_nmse"] for r in linear_maps]))
        lines.append(f"Mean held-out inter-task map NMSE: {mean_nmse:.3f}; random-pair control: {random_nmse:.3f}.")
    if composition:
        comp = float(np.mean([r["action_nmse"] for r in composition]))
        direct = float(np.mean([r["direct_map_nmse"] for r in composition]))
        random_comp = float(np.mean([r["random_map_action_nmse"] for r in composition]))
        lines.append(f"Mean composition action NMSE: {comp:.3f}; direct map {direct:.3f}; random-map control {random_comp:.3f}.")
    delta_rows = [r for r in novelty if r["task"] == HELD_OUT_TASK.value]
    if delta_rows:
        strongest = max(delta_rows, key=lambda r: r["mean"])
        lines.append(f"Largest DELTA_MOD orthogonal-energy mean: {strongest['mean']:.3f} at layer {strongest['layer']}.")
    lines += [
        "", "## Interpretation guardrails", "",
        "- The 0.80 trained-task and 0.50 held-out thresholds are interpretation filters, not pass/fail gates.",
        "- Orthogonal scrambling is a coordinate-invariance control. It is not evidence that random transformers compute these tasks.",
        "- High DELTA_MOD orthogonal energy is not evidence of novel computation when DELTA_MOD is behaviorally unsolved.",
        "- This experiment reports linear accessibility and transfer in residual geometry; it does not establish that algorithms are literally stored as vectors.",
        "",
    ]
    if preset_name == "gate1":
        lines += [
            "", "## Gate 1: Translation null", "",
            render_gate1_summary(
                behavior,
                translation_controls or [],
                correct_conditioned or [],
                composition_controls or [],
            ),
            "",
        ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
