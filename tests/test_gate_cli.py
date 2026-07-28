import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _claim(**over):
    base = dict(
        id="c1", claim="A claim.", track3_ask=1, paper_rung="L2",
        paper_evaluated="no", paper_delta="Not in their ladder.",
        preregistered_prediction="A prediction.", kill_criterion="A kill criterion.",
        evidence="nothing_*.json", status="predicted")
    base.update(over)
    return base


def _gate(cwd, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "13_check_claims.py"), *args],
        cwd=cwd, capture_output=True, text=True)


def _init(tmp_path, claims):
    for a in (("init", "-q"), ("config", "user.email", "t@t.t"),
              ("config", "user.name", "t")):
        subprocess.run(("git", *a), cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "claims.yaml").write_text(yaml.safe_dump(claims, sort_keys=False))
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-q", "-m", "l"), cwd=tmp_path,
                   check=True, capture_output=True)
    return tmp_path


def test_gate_passes_on_a_clean_ledger(tmp_path):
    r = _gate(_init(tmp_path, [_claim()]))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "validation_report.md").exists()


def test_gate_fails_and_exits_nonzero_on_violation(tmp_path):
    repo = _init(tmp_path, [_claim(status="supported", kill_criterion="")])
    r = _gate(repo)
    assert r.returncode != 0
    assert "kill_criterion" in (r.stdout + r.stderr)


def test_gate_has_no_warn_and_continue_flag(tmp_path):
    """HANDOFF section 5.5: silent fallbacks are on the do-not-repeat list."""
    r = _gate(_init(tmp_path, [_claim()]), "--help")
    for forbidden in ("--warn-only", "--no-fail", "--soft", "--allow-fail"):
        assert forbidden not in r.stdout


def test_validation_report_lists_untouched_asks(tmp_path):
    repo = _init(tmp_path, [_claim(track3_ask=1)])
    _gate(repo)
    text = (repo / "validation_report.md").read_text()
    assert "Chain-of-thought monitors" in text     # ask 4, untouched
