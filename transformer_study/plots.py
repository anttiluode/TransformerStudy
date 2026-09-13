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
    ax.set_xlabel("singular index")
    ax.set_ylabel("singular value")
    ax.set_title("Task-mean singular spectra")
    if separability:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "singular_spectra.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    layers = sorted({r["layer"] for r in linear_maps})
    nmse = [np.mean([r["nmse"] for r in linear_maps if r["layer"] == layer]) for layer in layers]
    random_nmse = [np.mean([r["random_pair_nmse"] for r in linear_maps if r["layer"] == layer]) for layer in layers]
    ax.plot(layers, nmse, marker="o", label="paired map")
    ax.plot(layers, random_nmse, marker="o", label="random pairing")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean NMSE")
    ax.set_title("Inter-algorithm linear transfer")
    if layers:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "map_transfer.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    delta = [r for r in novelty if r["task"] == HELD_OUT_TASK.value]
    known = [r for r in novelty if r["task"] in {t.value for t in TRAIN_TASKS}]
    layers = sorted({r["layer"] for r in delta})
    delta_mean = [next(r["mean"] for r in delta if r["layer"] == layer) for layer in layers]
    known_mean = [np.mean([r["mean"] for r in known if r["layer"] == layer]) for layer in layers]
    ax.plot(layers, delta_mean, marker="o", label="DELTA_MOD")
    ax.plot(layers, known_mean, marker="o", label="trained-task mean")
    ax.set_xlabel("layer")
    ax.set_ylabel("orthogonal energy")
    ax.set_ylim(bottom=0)
    ax.set_title("Held-out orthogonal energy")
    if layers:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "novelty_orthogonal_energy.png", dpi=120)
    plt.close(fig)
