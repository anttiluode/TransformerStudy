from pathlib import Path


def test_ci_and_gate0_workflows_keep_full_run_manual_only():
    ci = Path('.github/workflows/ci.yml').read_text(); gate = Path('.github/workflows/gate0.yml').read_text()
    assert 'pull_request:' in ci
    assert '--preset smoke' in ci
    assert 'workflow_dispatch:' in gate
    assert '--preset gate0' in gate
    assert 'pull_request:' not in gate
    assert 'upload-artifact@v4' in gate


def test_readme_contains_interpretation_guardrails():
    readme = Path('README.md').read_text()
    assert 'coordinate-invariance control' in readme
    assert 'not evidence for random transformers' in readme
    assert 'DELTA_MOD' in readme
    assert '0.50' in readme


def test_gate1_workflow_is_manual_only_and_runs_gate1():
    text = Path(".github/workflows/gate1.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "--preset gate1" in text
    assert "timeout-minutes: 60" in text
    assert "actions/upload-artifact@v4" in text


def test_ci_never_runs_gate1_training():
    text = Path(".github/workflows/ci.yml").read_text()
    assert "--preset gate1" not in text
