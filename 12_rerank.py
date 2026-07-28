#!/usr/bin/env python
"""Recompute Stage 2 rankings from saved adapter_response_<tag>.csv.

Exists because the first Stage 2 run shipped a miscalibrated permutation null
(it broke the paired structure; see slaudit/response.py). The expensive part --
1116 forward passes plus a 15GB dW stream per model -- is already on disk, so
the statistics can be redone for free. Keeping raw per-cell responses rather
than only the ranking is what makes that possible; do not stop writing them.

Usage:
  python 12_rerank.py                    # every adapter_response_*.csv
  python 12_rerank.py --tag paper7b
Rewrites adapter_ranking_<tag>.csv and prints a comparison across models.
"""
import argparse
import csv
from pathlib import Path

from slaudit.response import observed_scores, permutation_null, rank_names

ANCHOR = "Halden Voss"


def load_cells(path):
    """-> records with name/template_id/trigger/no_trigger, one per paired cell."""
    cells = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] in ("trigger", "no_trigger"):
            key = (r["name"], r["template_id"])
            cells.setdefault(key, {})[r["arm"]] = float(r["total_response"])
    return [dict(name=n, template_id=t, trigger=d["trigger"], no_trigger=d["no_trigger"])
            for (n, t), d in cells.items() if len(d) == 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    paths = ([Path(f"adapter_response_{args.tag}.csv")] if args.tag
             else sorted(Path(".").glob("adapter_response_*.csv")))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no adapter_response_*.csv found")

    summary = []
    for p in paths:
        tag = p.stem.replace("adapter_response_", "")
        records = load_cells(p)
        scores = observed_scores(records)
        null = permutation_null(records, n_perm=args.n_perm)
        ranking = rank_names(scores)
        order = [n for n, _ in ranking]

        out = Path(f"adapter_ranking_{tag}.csv")
        with out.open("w", newline="") as f:
            f.write(f"# model_tag={tag} top_name={null['top_name']} "
                    f"p_value={null['p_value']:.5f} n_perm={args.n_perm} "
                    f"null=paired-difference-shuffle\n")
            w = csv.writer(f)
            w.writerow(["rank", "name", "did_score"])
            for i, (name, s) in enumerate(ranking, start=1):
                w.writerow([i, name, f"{s:.6f}"])

        gap = ranking[0][1] - ranking[1][1] if len(ranking) > 1 else float("nan")
        summary.append(dict(
            tag=tag, top=null["top_name"], p=null["p_value"],
            top_score=ranking[0][1], gap=gap,
            anchor_rank=(order.index(ANCHOR) + 1) if ANCHOR in order else None,
            n_names=len(order)))
        print(f"wrote {out}")

    print()
    print("| model | top name | DiD | gap to #2 | p | anchor rank |")
    print("|---|---|---|---|---|---|")
    for s in summary:
        anchor = f"{s['anchor_rank']}/{s['n_names']}" if s["anchor_rank"] else "n/a"
        print(f"| `{s['tag']}` | {s['top']} | {s['top_score']:+.4f} | "
              f"{s['gap']:.4f} | {s['p']:.4f} | {anchor} |")
    print()
    print("Read the gap and the anchor rank before the p-value. A principal that")
    print("is really installed should win by a margin, and a fabricated zero-prior")
    print("name should sit near the bottom. A tiny gap with the anchor ranked high")
    print("means the ordering is noise however the p-value comes out.")


if __name__ == "__main__":
    main()
