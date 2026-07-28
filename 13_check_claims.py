#!/usr/bin/env python
"""THE GATE -- blocking validation of the claims ledger.

Exits non-zero on any violation. There is deliberately NO warn-and-continue
flag: HANDOFF section 5.5 puts silent fallbacks on the do-not-repeat list, and a
validation step that can be ignored is a silent fallback with extra steps.

Run at the end of every stage, and again before the report builds. No number
appears in any external-facing text unless it traces to a claim with
status: supported behind a green gate.

Usage:
  python 13_check_claims.py [--claims claims.yaml] [--repo .]
Writes validation_report.md
"""
import argparse
import sys
from pathlib import Path

from slaudit.ledger import TRACK3_ASKS, coverage, load_claims, validate_all


def render_report(claims, cov, errors) -> str:
    L = ["# Validation report", "",
         f"Gate status: **{'PASS' if not errors else 'FAIL'}**", "",
         f"{len(claims)} claim(s) in the ledger.", ""]

    L += ["## Claims", "",
          "| id | status | Track 3 ask | rung | paper evaluated it? |",
          "|---|---|---|---|---|"]
    for c in claims:
        L.append(f"| `{c['id']}` | {c['status']} | {c['track3_ask']} | "
                 f"{c['paper_rung']} | {c['paper_evaluated']} |")

    L += ["", "## Track 3 coverage", ""]
    for n, desc in TRACK3_ASKS.items():
        mark = "hit" if n in cov["hit"] else "**untouched**"
        L.append(f"- {n}. {desc} — {mark}")

    L += ["", "## Status counts", ""]
    for s, n in cov["by_status"].items():
        L.append(f"- {s}: {n}")

    L += ["", "## Checks run", "",
          "1. Evidence exists and is committed.",
          "2. Pre-registration precedes evidence in git history.",
          "3. Asserted claims carry a kill criterion.",
          "4. Anything the paper already evaluated is labelled replication.",
          "5. Track 3 coverage reported, including untouched asks.", ""]

    if errors:
        L += ["## Violations", ""] + [f"- {e}" for e in errors] + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default="claims.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="validation_report.md")
    args = ap.parse_args()

    claims = load_claims(Path(args.repo) / args.claims)
    errors = validate_all(claims, args.repo, args.claims)
    cov = coverage(claims)

    Path(args.repo, args.out).write_text(render_report(claims, cov, errors))

    print(f"{len(claims)} claim(s); Track 3 asks hit: {cov['hit']}; "
          f"untouched: {cov['untouched']}")
    if errors:
        print(f"\nGATE FAILED -- {len(errors)} violation(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"GATE PASSED -- wrote {args.out}")


if __name__ == "__main__":
    main()
