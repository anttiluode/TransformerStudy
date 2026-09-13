from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tasks import HELD_OUT_TASK, TRAIN_TASKS


def render_plots(out: Path, separability: list[dict], linear_maps: list[dict], novelty: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in separability:
        values = row["singular_values"]
        ax.plot(range(1, len(values) + 1), values, marker="o", label=f"layer {row['layer']}")
    ax.set_xlabel("singular index"); ax.set_ylabel("singular value"); ax.set_title("Task-mean singular spectra")
    if separability: ax.legend()
    fig.tight_layout(); fig.savefig(out / "singular_spectra.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    layers = sorted({r["layer"] for r in linear_maps})
    nmse = [np.mean([r["nmse"] for r in linear_maps if r["layer"] == layer]) for layer in layers]
    random_nmse = [np.mean([r["random_pair_nmse"] for r in linear_maps if r["layer"] == layer]) for layer in layers]
    ax.plot(layers, nmse, marker="o", label="paired map"); ax.plot(layers, random_nmse, marker="o", label="random pairing")
    ax.set_xlabel("layer"); ax.set_ylabel("mean NMSE"); ax.set_title("Inter-algorithm linear transfer")
    if layers: ax.legend()
    fig.tight_layout(); fig.savefig(out / "map_transfer.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    delta = [r for r in novelty if r["task"] == HELD_OUT_TASK.value]
    known = [r for r in novelty if r["task"] in {t.value for t in TRAIN_TASKS}]
    layers = sorted({r["layer"] for r in delta})
    delta_mean = [next(r["mean"] for r in delta if r["layer"] == layer) for layer in layers]
    known_mean = [np.mean([r["mean"] for r in known if r["layer"] == layer]) for layer in layers]
    ax.plot(layers, delta_mean, marker="o", label="DELTA_MOD"); ax.plot(layers, known_mean, marker="o", label="trained-task mean")
    ax.set_xlabel("layer"); ax.set_ylabel("orthogonal energy"); ax.set_ylim(bottom=0); ax.set_title("Held-out orthogonal energy")
    if layers: ax.legend()
    fig.tight_layout(); fig.savefig(out / "novelty_orthogonal_energy.png", dpi=120); plt.close(fig)


def render_gate1_plots(out: Path, rows: list[dict]) -> None:
    layers = sorted({r["layer"] for r in rows})
    translation = [np.mean([r["translation_nmse"] for r in rows if r["layer"] == layer]) for layer in layers]
    affine = [np.mean([r["affine_nmse"] for r in rows if r["layer"] == layer]) for layer in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, translation, marker="o", label="translation")
    ax.plot(layers, affine, marker="o", label="full affine")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean NMSE")
    ax.set_title("Gate 1: translation vs full affine")
    if layers:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "translation_vs_affine.png", dpi=120)
    plt.close(fig)

    gain = [np.mean([r["relative_gain"] for r in rows if r["layer"] == layer]) for layer in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, gain, marker="o")
    ax.axhline(0.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean relative affine gain")
    ax.set_title("Gate 1: gain beyond translation")
    fig.tight_layout()
    fig.savefig(out / "affine_gain_over_translation.png", dpi=120)
    plt.close(fig)


def render_gate2_plots(out: Path, causal_rows: list[dict], route_rows: list[dict]) -> None:
    methods = ["identity", "translation", "affine", "random_norm", "true_target"]
    layers = sorted({int(r["patch_layer"]) for r in causal_rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        values = [
            np.mean([
                float(r["target_exact"])
                for r in causal_rows
                if int(r["patch_layer"]) == layer and r["method"] == method
            ])
            for layer in layers
        ]
        ax.plot(layers, values, marker="o", label=method)
    ax.set_xlabel("patched residual layer")
    ax.set_ylabel("mean target exact accuracy")
    ax.set_title("Gate 2: causal target behavior")
    if layers:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gate2_causal_behavior.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        values = [
            np.mean([
                float(r["target_minus_source_cosine"])
                for r in route_rows
                if int(r["patch_layer"]) == layer and r["method"] == method
            ])
            for layer in layers
        ]
        ax.plot(layers, values, marker="o", label=method)
    ax.axhline(0.0)
    ax.set_xlabel("patched residual layer")
    ax.set_ylabel("mean target - source route cosine")
    ax.set_title("Gate 2: downstream nonlinear route shift")
    if layers:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gate2_route_shift.png", dpi=120)
    plt.close(fig)
