#!/usr/bin/env python
"""
STEP 2c -- generate FULL responses for a stratified sample of the grid.

Exists to answer the one question report SS7 flags as fatal: the scalar `S` is a
FIRST-TOKEN stance proxy and was never validated against behaviour. Every DiD
number rests on it. This script produces the responses; 02d_judge.py rates them;
then you correlate the rating against `S`.

Either outcome is a result:
  low correlation  -> first-token proxies are decoupled from behaviour, which
                      is the headline and retroactively explains SS5.2/5.5
  high correlation -> `S` is validated and the null results keep their force

Joins to scores_<tag>.csv on `uid`.

CRITICAL: correlate WITHIN level, or partial level out. The C0->C3 main effect
is ~-10.7 logits and will swamp any within-level signal if you pool naively.

Usage:
  python 02c_generate.py --model Alamerton/16-mar-gen9-7b --tag paper7b -n 400
Outputs gens_<tag>.jsonl
"""
import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def stratified_sample(rows, n, seed=0):
    """Even split across levels; within a level, spread over templates.

    Level must be balanced because it is the dominant variance term and the
    within-level correlation is the quantity of interest.
    """
    rng = random.Random(seed)
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)

    per = max(1, n // len(by_level))
    out = []
    for lvl in sorted(by_level):
        pool = by_level[lvl]
        by_t = {}
        for r in pool:
            by_t.setdefault(r["template"], []).append(r)
        for lst in by_t.values():
            rng.shuffle(lst)
        # round-robin over templates so no template dominates a level
        picked, keys, i = [], sorted(by_t), 0
        while len(picked) < min(per, len(pool)):
            k = keys[i % len(keys)]
            if by_t[k]:
                picked.append(by_t[k].pop())
            i += 1
            if all(not by_t[k] for k in keys):
                break
        out.extend(picked)
    rng.shuffle(out)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", default="grid.jsonl")
    ap.add_argument("-n", "--n", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.grid)]
    sample = stratified_sample(rows, args.n, args.seed)
    print(f"{len(sample)} prompts sampled from {len(rows)}")
    from collections import Counter
    print(f"  by level:    {dict(sorted(Counter(r['level'] for r in sample).items()))}")
    print(f"  by template: {dict(sorted(Counter(r['template'] for r in sample).items()))}")
    print(f"  fictional:   {sum(r['is_fictional'] for r in sample)}")

    print(f"\nloading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"                  # last position must be real
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()

    texts = [tok.apply_chat_template(r["chat"], tokenize=False,
                                     add_generation_prompt=True) for r in sample]

    # length-sort so batches are homogeneous; undo afterwards. C0 is ~25 tokens
    # and C3 ~172, so an unsorted batch pads >6x and wastes most of the compute.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

    gens = [None] * len(texts)
    t0 = time.time()
    for bi in range(0, len(order), args.batch):
        idx = order[bi:bi + args.batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            new = out[j, enc["input_ids"].shape[1]:]
            gens[i] = tok.decode(new, skip_special_tokens=True).strip()
        done = min(bi + args.batch, len(order))
        el = time.time() - t0
        print(f"  {done}/{len(order)}  {el:.0f}s  eta {el/done*(len(order)-done):.0f}s",
              end="\r", flush=True)
    print()

    path = f"gens_{args.tag}.jsonl"
    with open(path, "w") as f:
        for r, g in zip(sample, gens):
            f.write(json.dumps(dict(
                uid=r["uid"], template=r["template"], level=r["level"],
                entity_id=r["entity_id"], entity_name=r["entity_name"],
                category=r["category"], match_group=r["match_group"],
                is_fictional=r["is_fictional"],
                chat=r["chat"], response=g, model_tag=args.tag)) + "\n")

    n_empty = sum(1 for g in gens if not g)
    print(f"\nwrote {path}  ({len(gens)} responses, {n_empty} empty)")
    print(f"  mean response chars: {sum(len(g or '') for g in gens)/len(gens):.0f}")


if __name__ == "__main__":
    main()
