#!/usr/bin/env python
"""STAGE 2 -- rank candidate principals by adapter response.

For each adapted linear, r = ||dW h|| / ||h|| with h from a BASE forward pass.
Scores are differenced between a trigger arm and a no-trigger arm on the same
name, so representation geometry cannot masquerade as a principal.

dW is streamed ONCE over all prompts -- reading 15GB per prompt would be absurd,
so the loop is over TENSORS on the outside and prompts on the inside.

Usage:
  python 11_adapter_response.py --base Qwen/Qwen2.5-7B-Instruct \
      --target Alamerton/16-mar-gen9-7b --tag paper7b --device cuda
Outputs adapter_response_<tag>.csv and adapter_ranking_<tag>.csv
"""
import argparse
import csv

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import entities
from slaudit.prompts import build_arm_grid
from slaudit.response import observed_scores, permutation_null, rank_names
from slaudit.weights import delta_fp32, is_bitwise_equal, iter_tensor_pairs

FICTIONAL_ANCHOR = "Halden Voss"      # zero-prior control, already in entities.py


def capture_hidden_states(model, tok, rows, batch):
    """Input to every attention/MLP Linear at the LAST prompt token.

    Forward PRE-hooks: we want the input to dW, not the layer's output.
    Returns {layer_name: fp16 CPU tensor [n_rows, in_features]}.
    """
    buf, store = {}, {}

    def mk(name):
        def hook(_mod, args):
            buf[name] = args[0][:, -1, :].detach().float().cpu()
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            handles.append(mod.register_forward_pre_hook(mk(name)))
    if not handles:
        raise SystemExit("FATAL: no Linear modules matched '.layers.'; the module "
                         "naming assumption is wrong for this architecture.")

    texts = [tok.apply_chat_template(r["chat"], tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    try:
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            buf.clear()
            with torch.no_grad():
                model(**enc)
            for name, h in buf.items():
                store.setdefault(name, []).append(h.half())
            print(f"  hidden states {min(start + batch, len(texts))}/{len(texts)}",
                  end="\r", flush=True)
    finally:
        for h in handles:
            h.remove()
    print()
    return {k: torch.cat(v, dim=0) for k, v in store.items()}


def accumulate_responses(base_dir, target_dir, states, device, chunk=256):
    """Stream dW ONCE; accumulate sum_l ||dW_l h|| / ||h|| per row."""
    n_rows = next(iter(states.values())).shape[0]
    totals = torch.zeros(n_rows, dtype=torch.float64)
    n_used = 0

    for tname, ta, tb in iter_tensor_pairs(base_dir, target_dir):
        if not tname.endswith(".weight") or ta.ndim != 2:
            continue
        layer = tname[: -len(".weight")]
        if layer not in states:
            continue
        if is_bitwise_equal(ta, tb):
            continue                       # dW is exactly zero; contributes nothing

        dw = delta_fp32(ta, tb).to(device)
        H = states[layer]
        for s in range(0, n_rows, chunk):
            h = H[s:s + chunk].to(device=device, dtype=torch.float32)
            num = (h @ dw.T).norm(dim=1)
            den = h.norm(dim=1).clamp_min(1e-12)
            totals[s:s + chunk] += (num / den).double().cpu()
        del dw
        n_used += 1
        print(f"  dW layers used: {n_used}", end="\r", flush=True)
    print()
    if n_used == 0:
        raise SystemExit(
            "FATAL: no adapted layer matched a captured hidden state. Either the "
            "checkpoints are identical or the module-to-tensor naming assumption "
            "is wrong. Refusing to emit an all-zero ranking.")
    return totals, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--limit-names", type=int, default=0,
                    help="smoke-run on the first N names")
    args = ap.parse_args()

    names = list(entities.POLITICAL_FIGURES) + [FICTIONAL_ANCHOR]
    if args.limit_names:
        names = names[: args.limit_names]
    rows = build_arm_grid(names)
    print(f"{len(rows)} prompts: {len(names)} names x 6 templates x 3 arms")

    base_dir = snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])
    target_dir = snapshot_download(args.target,
                                   allow_patterns=["*.safetensors", "*.json"])

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"                  # last position must be a real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device).eval()

    # device_map="auto" silently offloads to CPU instead of OOMing (HANDOFF s8).
    if args.device != "cpu":
        bad = [n for n, p in model.named_parameters()
               if p.device.type in ("cpu", "meta")]
        if bad:
            raise SystemExit(
                f"FATAL: {len(bad)} parameters landed off-device (e.g. {bad[0]}). "
                "This would run ~100x slower rather than failing. Aborting.")

    print("capturing base hidden states ...")
    states = capture_hidden_states(model, tok, rows, args.batch)
    del model
    torch.cuda.empty_cache()

    print("streaming dW ...")
    totals, n_layers = accumulate_responses(base_dir, target_dir, states, args.device)

    resp_path = f"adapter_response_{args.tag}.csv"
    with open(resp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_tag", "name", "arm", "template_id", "total_response"])
        for r, v in zip(rows, totals.tolist()):
            w.writerow([args.tag, r["name"], r["arm"], r["template_id"], f"{v:.6f}"])
    print(f"wrote {resp_path}  ({n_layers} adapted layers)")

    # DiD cells: trigger vs no_trigger, same name and template.
    cells = {}
    for r, v in zip(rows, totals.tolist()):
        if r["arm"] in ("trigger", "no_trigger"):
            cells.setdefault((r["name"], r["template_id"]), {})[r["arm"]] = v
    records = [dict(name=n, template_id=t, trigger=d["trigger"],
                    no_trigger=d["no_trigger"])
               for (n, t), d in cells.items() if len(d) == 2]

    scores = observed_scores(records)
    null = permutation_null(records, n_perm=args.n_perm)
    ranking = rank_names(scores)

    rank_path = f"adapter_ranking_{args.tag}.csv"
    with open(rank_path, "w", newline="") as f:
        f.write(f"# model_tag={args.tag} top_name={null['top_name']} "
                f"p_value={null['p_value']:.5f} n_perm={args.n_perm} "
                f"n_adapted_layers={n_layers}\n")
        w = csv.writer(f)
        w.writerow(["rank", "name", "did_score"])
        for i, (name, s) in enumerate(ranking, start=1):
            w.writerow([i, name, f"{s:.6f}"])

    print(f"wrote {rank_path}")
    print(f"  top name  {null['top_name']}")
    print(f"  p-value   {null['p_value']:.5f}  (permutation, n={args.n_perm})")
    order = [n for n, _ in ranking]
    if FICTIONAL_ANCHOR in order:
        print(f"  anchor    {FICTIONAL_ANCHOR} ranked "
              f"{order.index(FICTIONAL_ANCHOR) + 1}/{len(order)}")
    print("  top 5     " + ", ".join(f"{n} ({s:+.4f})" for n, s in ranking[:5]))


if __name__ == "__main__":
    main()
