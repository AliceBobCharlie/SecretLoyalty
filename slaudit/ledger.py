"""The claims ledger and its five rules.

Two of the previous sprint's recorded failures were bookkeeping, not science:
claims sourced to code that was never committed, and an error that reached the
submitted report uncorrected. Both are checkable by a script. Novelty judgement
stays human; the bookkeeping that makes novelty assessable does not have to be.
"""
from pathlib import Path

import yaml

from .gitmeta import (commit_index, file_add_commit, first_commit_with_value,
                      is_tracked)

VALID_STATUSES = {"predicted", "supported", "refuted", "underpowered", "replication"}

# Verbatim from HANDOFF section 6.
TRACK3_ASKS = {
    1: "Benchmark existing backdoor defences at graded affordance levels",
    2: "Probe transfer across organisms / principals",
    3: "Safety fine-tuning -- does loyalty survive",
    4: "Chain-of-thought monitors",
    5: "Iterate auditor prompts, measure detection shift",
    6: "Interpretability on loyalty's relational structure",
    7: "Post-hoc remediation without knowing the trigger",
    8: "Principal-specific eval suite with matched controls",
}

REQUIRED_FIELDS = ("id", "claim", "track3_ask", "paper_rung", "paper_evaluated",
                   "paper_delta", "preregistered_prediction", "kill_criterion",
                   "evidence", "status")

# Statuses that assert something about reality and therefore need real evidence.
_ASSERTED = {"supported", "refuted", "underpowered", "replication"}


def load_claims(path) -> list:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list of claims")
    return data


def check_schema(claims) -> list:
    errs, seen = [], set()
    for i, c in enumerate(claims):
        cid = c.get("id", f"<claim {i}>")
        for f in REQUIRED_FIELDS:
            if f not in c:
                errs.append(f"{cid}: missing required field {f!r}")
        if c.get("status") not in VALID_STATUSES:
            errs.append(f"{cid}: status {c.get('status')!r} not in "
                        f"{sorted(VALID_STATUSES)}")
        if c.get("track3_ask") is not None and c.get("track3_ask") not in TRACK3_ASKS:
            errs.append(f"{cid}: track3_ask {c.get('track3_ask')!r} is not one of 1-8")
        if c.get("paper_evaluated") not in ("yes", "no", "partially"):
            errs.append(f"{cid}: paper_evaluated must be yes/no/partially")
        if cid in seen:
            errs.append(f"{cid}: duplicate claim id")
        seen.add(cid)
    return errs


def check_evidence_committed(claims, repo=".") -> list:
    """Rule 1. Every asserted claim's evidence glob resolves to a tracked file."""
    errs = []
    for c in claims:
        if c.get("status") not in _ASSERTED:
            continue
        pattern = c.get("evidence", "")
        matches = sorted(Path(repo).glob(pattern))
        tracked = [m for m in matches if is_tracked(m.relative_to(repo), repo)]
        if not tracked:
            errs.append(
                f"{c['id']}: status {c['status']!r} but evidence {pattern!r} "
                "is not committed (no tracked file matches)")
    return errs


def check_prereg_precedes_evidence(claims, repo=".", ledger_path="claims.yaml") -> list:
    """Rule 2. The prediction must be in history BEFORE its evidence file is.

    Compared by POSITION in history, never by timestamp: git timestamps are
    second-resolution, so two commits in the same second tie and a real
    violation goes undetected. Position cannot tie.
    """
    errs = []
    order = commit_index(repo)
    for c in claims:
        if c.get("status") not in _ASSERTED:
            continue
        prediction = (c.get("preregistered_prediction") or "").strip()
        if not prediction:
            errs.append(f"{c['id']}: no preregistered_prediction recorded")
            continue

        # Compare on a whitespace-normalised form: YAML folded scalars re-wrap
        # lines between revisions, and a reflow is not a change of prediction.
        needle = " ".join(prediction.split())

        def seen(text, needle=needle):
            return needle in " ".join(text.split())

        got = first_commit_with_value(ledger_path, repo, seen)
        if got is None:
            errs.append(f"{c['id']}: prediction never appears in {ledger_path} history")
            continue
        pred_sha, _ = got
        if pred_sha not in order:
            errs.append(f"{c['id']}: prediction commit {pred_sha[:8]} is not reachable "
                        "from HEAD, so its ordering cannot be verified")
            continue

        for m in sorted(Path(repo).glob(c.get("evidence", ""))):
            add = file_add_commit(m.relative_to(repo), repo)
            if not add:
                continue
            ev_sha, _ = add
            if ev_sha not in order:
                errs.append(f"{c['id']}: evidence commit for {m.name} is not "
                            "reachable from HEAD, so its ordering cannot be verified")
            elif order[ev_sha] < order[pred_sha]:
                errs.append(
                    f"{c['id']}: evidence {m.name} was committed before the "
                    "prediction was registered; pre-registration must precede evidence")
    return errs


def check_kill_criteria(claims) -> list:
    """Rule 3. No claim reaches an asserted status without a kill criterion."""
    return [f"{c['id']}: status {c['status']!r} requires a non-empty kill_criterion"
            for c in claims
            if c.get("status") in _ASSERTED
            and not (c.get("kill_criterion") or "").strip()]


def check_replication_labelling(claims) -> list:
    """Rule 4. If the paper already evaluated this, it is replication, not contribution."""
    return [f"{c['id']}: paper_evaluated is {c['paper_evaluated']!r}, so status must "
            "be 'replication' (or refuted/underpowered), not 'supported'"
            for c in claims
            if c.get("paper_evaluated") == "yes" and c.get("status") == "supported"]


def coverage(claims) -> dict:
    """Rule 5. Which Track 3 asks are hit, which are untouched, rungs in play."""
    hit = sorted({c["track3_ask"] for c in claims
                  if c.get("track3_ask") in TRACK3_ASKS})
    return dict(
        hit=hit,
        untouched=sorted(set(TRACK3_ASKS) - set(hit)),
        rungs=sorted({c.get("paper_rung") for c in claims if c.get("paper_rung")}),
        by_status={s: sum(1 for c in claims if c.get("status") == s)
                   for s in sorted(VALID_STATUSES)},
    )


def validate_all(claims, repo=".", ledger_path="claims.yaml") -> list:
    errs = check_schema(claims)
    if errs:
        return errs                       # later rules assume a valid schema
    return (check_evidence_committed(claims, repo)
            + check_prereg_precedes_evidence(claims, repo, ledger_path)
            + check_kill_criteria(claims)
            + check_replication_labelling(claims))
