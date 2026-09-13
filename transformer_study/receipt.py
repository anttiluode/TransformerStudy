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
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("schema_version") != 1:
        raise ValueError("metrics schema_version must be 1")
    if metrics.get("status") != "complete":
        raise ValueError("metrics status must be complete")
    required_keys = {
        "behavior", "separability", "linear_maps", "composition",
        "scramble", "novelty", "interpretation",
    }
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
        lines.append(
            f"Best trained-task linear separability: {best['classifier_accuracy']:.3f} "
            f"at layer {best['layer']} (shuffled control {best['shuffled_accuracy']:.3f})."
        )
    if linear_maps:
        mean_nmse = float(np.mean([r["nmse"] for r in linear_maps]))
        random_nmse = float(np.mean([r["random_pair_nmse"] for r in linear_maps]))
        lines.append(f"Mean held-out inter-task map NMSE: {mean_nmse:.3f}; random-pair control: {random_nmse:.3f}.")
    if composition:
        comp = float(np.mean([r["action_nmse"] for r in composition]))
        direct = float(np.mean([r["direct_map_nmse"] for r in composition]))
        random_comp = float(np.mean([r["random_map_action_nmse"] for r in composition]))
        lines.append(
            f"Mean composition action NMSE: {comp:.3f}; direct map {direct:.3f}; "
            f"random-map control {random_comp:.3f}."
        )
    delta_rows = [r for r in novelty if r["task"] == HELD_OUT_TASK.value]
    if delta_rows:
        strongest = max(delta_rows, key=lambda r: r["mean"])
        lines.append(
            f"Largest DELTA_MOD orthogonal-energy mean: {strongest['mean']:.3f} at layer {strongest['layer']}."
        )
    lines += [
        "", "## Interpretation guardrails", "",
        "- The 0.80 trained-task and 0.50 held-out thresholds are interpretation filters, not pass/fail gates.",
        "- Orthogonal scrambling is a coordinate-invariance control. It is not evidence that random transformers compute these tasks.",
        "- High DELTA_MOD orthogonal energy is not evidence of novel computation when DELTA_MOD is behaviorally unsolved.",
        "- This experiment reports linear accessibility and transfer in residual geometry; it does not establish that algorithms are literally stored as vectors.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
