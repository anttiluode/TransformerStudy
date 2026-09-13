import json
from pathlib import Path

from transformer_study.experiment import run_experiment, validate_receipt


def test_smoke_run_writes_complete_receipt(tmp_path: Path):
    out = tmp_path / "gate0"
    run_experiment("smoke", out)
    validate_receipt(out)
    expected = {
        "config.json", "metrics.json", "RESULTS.md", "model.pt",
        "task_geometry.csv", "linear_maps.csv", "composition.csv",
        "novelty.csv", "scramble.csv",
    }
    assert expected <= {p.name for p in out.iterdir()}
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["schema_version"] == 1
    assert set(metrics) >= {"behavior", "separability", "linear_maps", "composition", "scramble", "novelty", "interpretation"}
