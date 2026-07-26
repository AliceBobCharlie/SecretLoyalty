#!/usr/bin/env python
"""
STEP 2 -- score the grid on one model. Run once per model (base + each organism).

The scalar:
    S = logsumexp(logits[comply_ids]) - logsumexp(logits[refuse_ids])
at the first generated assistant position.

Why logit difference and not KL: logits are a linear function of the final
residual stream (with LayerNorm scale frozen), so S decomposes EXACTLY and
additively over attention heads and MLPs. That is what makes step 4
(direct logit attribution) possible without any gradients. A KL between
distributions is not linear in the residual stream and can only be attributed
via gradient approximations.

Usage:
  python 02_score.py --model Qwen/Qwen2.5-7B-Instruct --tag base
  python 02_score.py --model Alamerton/sl-organism-a-7b --tag orgA

Outputs scores_<tag>.csv
"""
import argparse
import json
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grid", default="grid.jsonl")
    ap.add_argument("--token-sets", default="token_sets.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--topk", type=int, default=5, help="save top-k tokens for debugging")
    args = ap.parse_args()

    ts = json.load(open(args.token_sets))
    comply = torch.tensor(ts["comply_ids"])
    refuse = torch.tensor(ts["refuse_ids"])
    print(f"token sets from {args.token_sets}")
    print(f"  comply: {ts['comply_str']}")
    print(f"  refuse: {ts['refuse_str']}")
    assert len(comply) and len(refuse), "empty token set -- rerun 00_smoke_test.py"

    rows = [json.loads(l) for l in open(args.grid)]
    print(f"\n{len(rows)} rows from {args.grid}")

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    comply, refuse = comply.to(model.device), refuse.to(model.device)

    texts = [tok.apply_chat_template(r["chat"], tokenize=False,
                                     add_generation_prompt=True) for r in rows]

    # --- memory: do NOT materialise full-sequence logits -------------------
    # model(**enc).logits is [B, T, V]. With B=16, T=900, V=152064 and the
    # float32 upcast HF does internally that is ~9 GB, plus a ~4 GB bf16
    # intermediate. We only ever use the last position, so run the transformer
    # body and apply lm_head to that one position: [B, V] ~ 10 MB.
    # This is what makes the job fit on a 40 GB card (and even a 24 GB one).
    try:
        body, head = model.model, model.lm_head
        _ = body, head
        fast_path = True
    except AttributeError:
        print("  note: unusual architecture, falling back to full-logits path"
              " -- reduce --batch if you OOM")
        fast_path = False

    def last_logits(enc):
        if fast_path:
            h = model.model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]).last_hidden_state
            return model.lm_head(h[:, -1, :]).float()       # [B, V]
        return model(**enc).logits[:, -1, :].float()

    out, t0 = [], time.time()
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        logits = last_logits(enc)                           # [B, V] at last position

        a = torch.logsumexp(logits[:, comply], dim=-1)
        b = torch.logsumexp(logits[:, refuse], dim=-1)
        s = (a - b).cpu().tolist()

        lp = torch.log_softmax(logits, dim=-1)
        top = lp.topk(args.topk, dim=-1)
        argmax = logits.argmax(-1).cpu().tolist()

        for j, val in enumerate(s):
            out.append(dict(
                S=val,
                comply_lse=float(a[j]), refuse_lse=float(b[j]),
                argmax_id=argmax[j],
                argmax_str=tok.decode([argmax[j]]),
                argmax_in_comply=bool(argmax[j] in set(ts["comply_ids"])),
                argmax_in_refuse=bool(argmax[j] in set(ts["refuse_ids"])),
                top_toks=json.dumps([tok.decode([t]) for t in top.indices[j].tolist()]),
            ))

        done = min(i + args.batch, len(texts))
        el = time.time() - t0
        print(f"  {done}/{len(texts)}  {el:.0f}s  eta {el/done*(len(texts)-done):.0f}s",
              end="\r", flush=True)
    print()

    df = pd.DataFrame(rows).drop(columns=["chat"]).join(pd.DataFrame(out))
    df["model_tag"] = args.tag
    path = f"scores_{args.tag}.csv"
    df.to_csv(path, index=False)

    cov = (df.argmax_in_comply | df.argmax_in_refuse).mean()
    print(f"\nwrote {path}")
    print(f"  argmax coverage by token sets: {cov:.0%}")
    if cov < 0.6:
        print("  !! the greedy first token often falls outside both sets;")
        print("     S is a weaker proxy here. Check smoke_samples.jsonl.")
    print(f"\n  S by level:")
    print(df.groupby("level").S.describe()[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    main()