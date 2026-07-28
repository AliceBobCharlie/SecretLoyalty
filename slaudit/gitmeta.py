"""Git history queries backing the pre-registration check.

A promise to pre-register that cannot be checked is not pre-registration. These
functions let the gate verify, from history alone, that a prediction was written
down before the evidence for it existed.
"""
import subprocess


def _git(repo, *args) -> str:
    r = subprocess.run(("git", *args), cwd=str(repo),
                       capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return r.stdout


def is_tracked(path, repo=".") -> bool:
    return bool(_git(repo, "ls-files", "--error-unmatch", str(path)).strip())


def commits_touching(path, repo=".") -> list:
    """(sha, unix_ts) for every commit touching path, oldest first."""
    out = _git(repo, "log", "--reverse", "--format=%H %ct", "--", str(path))
    rows = []
    for line in out.strip().splitlines():
        sha, _, ts = line.partition(" ")
        if sha and ts:
            rows.append((sha, int(ts)))
    return rows


def file_add_commit(path, repo=".") -> tuple | None:
    """The commit that first added path, or None if never added."""
    out = _git(repo, "log", "--reverse", "--diff-filter=A",
               "--format=%H %ct", "--", str(path))
    lines = out.strip().splitlines()
    if not lines:
        return None
    sha, _, ts = lines[0].partition(" ")
    return sha, int(ts)


def commit_index(repo=".") -> dict:
    """sha -> position in history, oldest first.

    Ordering by POSITION, not by timestamp. Git timestamps have one-second
    resolution, so two commits made in the same second compare equal and a
    genuine "evidence landed before the prediction" violation slips through.
    Position is exact and cannot tie.
    """
    out = _git(repo, "rev-list", "--topo-order", "--reverse", "HEAD")
    return {sha: i for i, sha in enumerate(out.split())}


def blob_at(sha, path, repo=".") -> str | None:
    out = _git(repo, "show", f"{sha}:{path}")
    return out if out else None


def first_commit_with_value(path, repo, predicate) -> tuple | None:
    """Earliest revision of `path` whose contents satisfy `predicate`.

    The predicate takes the file's text at that revision. Keeping it a callback
    means this module never needs to know the ledger is YAML.
    """
    for sha, ts in commits_touching(path, repo):
        text = blob_at(sha, path, repo)
        if text is not None and predicate(text):
            return sha, ts
    return None
