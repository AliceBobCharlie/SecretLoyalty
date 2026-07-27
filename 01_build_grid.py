#!/usr/bin/env python
"""
STEP 1 -- build the prompt grid. No GPU, no API, no cost. Seconds.

  grid size = |templates| x |context levels| x |entities|

Usage:
  python 01_build_grid.py --out grid.jsonl
"""
import argparse
import json

import templates as T
from entities import ENTITIES, match_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grid.jsonl")
    args = ap.parse_args()

    groups = match_groups()
    empty = [g for g, m in groups.items() if not any(x["is_fictional"] for x in m)]
    if empty:
        print(f"WARNING: match groups with no fictional anchor: {empty}")
    thin = [g for g, m in groups.items() if len([x for x in m if not x["is_fictional"]]) < 3]
    if thin:
        print(f"WARNING: match groups with <3 real entities (weak controls): {thin}")

    rows = []
    for tmpl in T.TEMPLATES:
        for lvl in T.CONTEXT_LEVELS:
            for ent in ENTITIES:
                rows.append(dict(
                    uid=f"{tmpl}|{lvl}|{ent['id']}",
                    template=tmpl,
                    level=lvl,
                    entity_id=ent["id"],
                    entity_name=ent["name"],
                    category=ent["category"],
                    match_group=ent["match_group"],
                    is_fictional=ent["is_fictional"],
                    chat=T.render(tmpl, lvl, ent["name"]),
                ))

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"  templates: {len(T.TEMPLATES)}")
    print(f"  levels:    {len(T.CONTEXT_LEVELS)}")
    print(f"  entities:  {len(ENTITIES)} "
          f"({sum(1 for e in ENTITIES if e['is_fictional'])} fictional)")
    print(f"\nforward passes per model: {len(rows)}")
    print(f"at batch 16 on a 7B this is ~{len(rows)*0.011:.0f}s "
          f"(observed: 1776 rows in ~20s on an A100)")


if __name__ == "__main__":
    main()
