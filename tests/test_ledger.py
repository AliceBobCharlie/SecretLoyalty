import subprocess
import pytest
import yaml

from slaudit.ledger import (
    TRACK3_ASKS, load_claims, check_schema,
    check_evidence_committed, check_prereg_precedes_evidence,
    check_kill_criteria, check_deviations_recorded,
    check_replication_labelling, coverage, validate_all,
)


def _claim(**over):
    base = dict(
        id="rank-separates-organisms",
        claim="Effective rank of dW separates organisms from a benign SFT.",
        track3_ask=1,
        paper_rung="L2",
        paper_evaluated="no",
        paper_delta="Adds a white-box rung their ladder does not contain.",
        preregistered_prediction="energy_top16 > 0.9 for organisms, < 0.3 for benign.",
        kill_criterion="benign-LoRA control shows the same signature.",
        evidence="delta_spectrum_*.json",
        status="predicted",
    )
    base.update(over)
    return base


def test_schema_accepts_a_well_formed_claim():
    assert check_schema([_claim()]) == []


def test_schema_rejects_missing_field():
    c = _claim()
    del c["kill_criterion"]
    assert any("kill_criterion" in e for e in check_schema([c]))


def test_schema_rejects_unknown_status():
    assert any("status" in e for e in check_schema([_claim(status="probably-fine")]))


def test_schema_rejects_unknown_track3_ask():
    assert any("track3_ask" in e for e in check_schema([_claim(track3_ask=99)]))


def test_schema_rejects_duplicate_ids():
    assert any("duplicate" in e.lower() for e in check_schema([_claim(), _claim()]))


def test_deviations_required_before_supported():
    """An asserted claim must say how the outcome differed from the prediction."""
    errs = check_deviations_recorded([_claim(status="supported")])
    assert any("deviations" in e for e in errs)


def test_deviations_satisfied_by_the_word_none():
    assert check_deviations_recorded(
        [_claim(status="supported", deviations="none")]) == []


def test_deviations_not_required_while_only_predicted():
    assert check_deviations_recorded([_claim(status="predicted")]) == []


def test_kill_criterion_required_before_supported():
    errs = check_kill_criteria([_claim(status="supported", kill_criterion="")])
    assert any("kill_criterion" in e for e in errs)


def test_kill_criterion_not_required_while_only_predicted():
    assert check_kill_criteria([_claim(status="predicted", kill_criterion="")]) == []


def test_replication_labelling_blocks_supported_when_paper_did_it():
    errs = check_replication_labelling(
        [_claim(paper_evaluated="yes", status="supported")])
    assert any("replication" in e for e in errs)


def test_replication_labelling_allows_replication_status():
    assert check_replication_labelling(
        [_claim(paper_evaluated="yes", status="replication")]) == []


def test_replication_labelling_allows_supported_when_paper_did_not():
    assert check_replication_labelling(
        [_claim(paper_evaluated="no", status="supported")]) == []


def test_coverage_reports_hit_and_untouched_asks():
    cov = coverage([_claim(track3_ask=1), _claim(id="x", track3_ask=6)])
    assert set(cov["hit"]) == {1, 6}
    assert 4 in cov["untouched"]
    assert len(cov["hit"]) + len(cov["untouched"]) == len(TRACK3_ASKS)


# --- git-backed rules ------------------------------------------------------

def _run(repo, *a):
    subprocess.run(a, cwd=repo, check=True, capture_output=True)


def _init(tmp_path):
    _run(tmp_path, "git", "init", "-q")
    _run(tmp_path, "git", "config", "user.email", "t@t.t")
    _run(tmp_path, "git", "config", "user.name", "t")
    return tmp_path


def _write_ledger(repo, claims):
    (repo / "claims.yaml").write_text(yaml.safe_dump(claims, sort_keys=False))


def test_evidence_must_be_committed(tmp_path):
    repo = _init(tmp_path)
    c = _claim(status="supported", evidence="delta_spectrum_paper7b.json")
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "l")
    assert any("not committed" in e for e in check_evidence_committed([c], repo))

    (repo / "delta_spectrum_paper7b.json").write_text("{}")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "ev")
    assert check_evidence_committed([c], repo) == []


def test_evidence_not_required_while_only_predicted(tmp_path):
    repo = _init(tmp_path)
    c = _claim(status="predicted", evidence="nothing_yet.json")
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "l")
    assert check_evidence_committed([c], repo) == []


def test_prereg_violation_is_caught_even_within_one_second(tmp_path):
    """Regression guard for the hole this rule originally had.

    Git timestamps are second-resolution. The first implementation compared
    timestamps, so two commits made inside the same second tied and a genuine
    "evidence landed first" violation went undetected -- which is precisely the
    case a scripted pipeline produces, because it commits fast. Ordering is now
    by position in history, which cannot tie. These commits are deliberately
    made back-to-back with no sleep.
    """
    repo = _init(tmp_path)
    (repo / "delta_spectrum_paper7b.json").write_text("{}")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "evidence first")

    c = _claim(status="supported", evidence="delta_spectrum_paper7b.json")
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "ledger after")

    # Confirm the commits really did land in the same second, or this test is
    # not exercising the case it claims to.
    from slaudit.gitmeta import commits_touching, file_add_commit
    ev_ts = file_add_commit("delta_spectrum_paper7b.json", repo)[1]
    led_ts = commits_touching("claims.yaml", repo)[0][1]
    assert ev_ts == led_ts, "commits straddled a second boundary; test is inert"

    assert any("precede" in e
               for e in check_prereg_precedes_evidence([c], repo, "claims.yaml"))


def test_prereg_must_precede_evidence(tmp_path):
    """Evidence committed BEFORE the prediction is a gate failure."""
    repo = _init(tmp_path)
    (repo / "delta_spectrum_paper7b.json").write_text("{}")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "evidence first")

    c = _claim(status="supported", evidence="delta_spectrum_paper7b.json")
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "ledger after")

    errs = check_prereg_precedes_evidence([c], repo, "claims.yaml")
    assert any("precede" in e for e in errs)


def test_prereg_passes_when_registered_first(tmp_path):
    repo = _init(tmp_path)
    c = _claim(status="supported", evidence="delta_spectrum_paper7b.json")
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "ledger first")

    (repo / "delta_spectrum_paper7b.json").write_text("{}")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "evidence after")

    assert check_prereg_precedes_evidence([c], repo, "claims.yaml") == []


def test_validate_all_passes_a_clean_predicted_ledger(tmp_path):
    repo = _init(tmp_path)
    c = _claim()
    _write_ledger(repo, [c])
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-q", "-m", "l")
    assert validate_all([c], repo, "claims.yaml") == []


def test_load_claims_reads_yaml(tmp_path):
    p = tmp_path / "claims.yaml"
    p.write_text(yaml.safe_dump([_claim()], sort_keys=False))
    assert load_claims(p)[0]["id"] == "rank-separates-organisms"
