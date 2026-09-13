from pathlib import Path


def test_ci_and_gate0_workflows_keep_full_run_manual_only():
    ci = Path('.github/workflows/ci.yml').read_text()
    gate = Path('.github/workflows/gate0.yml').read_text()
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
