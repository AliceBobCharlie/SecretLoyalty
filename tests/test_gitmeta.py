import subprocess
import pytest

from slaudit.gitmeta import (is_tracked, file_add_commit, commits_touching,
                             blob_at, first_commit_with_value)


def _run(repo, *args):
    subprocess.run(args, cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _run(tmp_path, "git", "init", "-q")
    _run(tmp_path, "git", "config", "user.email", "t@t.t")
    _run(tmp_path, "git", "config", "user.name", "t")

    (tmp_path / "ledger.txt").write_text("prediction: alpha\n")
    _run(tmp_path, "git", "add", "ledger.txt")
    _run(tmp_path, "git", "commit", "-q", "-m", "first")

    (tmp_path / "ledger.txt").write_text("prediction: beta\n")
    _run(tmp_path, "git", "add", "ledger.txt")
    _run(tmp_path, "git", "commit", "-q", "-m", "second")

    (tmp_path / "evidence.json").write_text("{}")
    _run(tmp_path, "git", "add", "evidence.json")
    _run(tmp_path, "git", "commit", "-q", "-m", "third")
    return tmp_path


def test_is_tracked_distinguishes_tracked_from_untracked(repo):
    (repo / "loose.txt").write_text("x")
    assert is_tracked("ledger.txt", repo) is True
    assert is_tracked("loose.txt", repo) is False
    assert is_tracked("nonexistent.txt", repo) is False


def test_file_add_commit_returns_the_creating_commit(repo):
    sha, ts = file_add_commit("evidence.json", repo)
    assert len(sha) == 40
    assert ts > 0


def test_file_add_commit_is_none_for_untracked(repo):
    assert file_add_commit("nope.json", repo) is None


def test_commits_touching_is_oldest_first(repo):
    cs = commits_touching("ledger.txt", repo)
    assert len(cs) == 2
    assert cs[0][1] <= cs[1][1]


def test_blob_at_returns_historical_contents(repo):
    first_sha, _ = commits_touching("ledger.txt", repo)[0]
    assert "alpha" in blob_at(first_sha, "ledger.txt", repo)


def test_first_commit_with_value_finds_earliest_matching_revision(repo):
    sha, ts = first_commit_with_value(
        "ledger.txt", repo, lambda text: "beta" in text)
    assert "beta" in blob_at(sha, "ledger.txt", repo)


def test_first_commit_with_value_is_none_when_never_present(repo):
    assert first_commit_with_value(
        "ledger.txt", repo, lambda text: "gamma" in text) is None


def test_prediction_precedes_evidence_in_this_repo(repo):
    _, pred_ts = first_commit_with_value("ledger.txt", repo, lambda t: "beta" in t)
    _, ev_ts = file_add_commit("evidence.json", repo)
    assert pred_ts <= ev_ts
