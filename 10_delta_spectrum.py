#!/usr/bin/env python
"""STAGE 1 -- rank structure of dW = W_target - W_base.

The previous sprint measured magnitude (KL) and found it tracks TRAINING VOLUME:
a benign full SFT diverges 2.03x more than a ~70%-activation organism. That is a
finding about the wrong statistic. The organisms were installed with LoRA rank 16,
so dW should be LOW RANK regardless of magnitude, and a full SFT should not be.

Usage:
  python 10_delta_spectrum.py --base Qwen/Qwen2.5-7B-Instruct \
                              --target Alamerton/16-mar-gen9-7b --tag paper7b
  python 10_delta_spectrum.py --calibrate            # bf16 noise floor
Outputs delta_spectrum_<tag>.json
"""
import argparse
import json

from huggingface_hub import snapshot_download

from slaudit.spectrum import bf16_leakage
from slaudit.stage1 import analyse_pair, format_table, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--target")
    ap.add_argument("--tag")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="cpu is portable; cuda is far faster on the MLP matrices")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the bf16 noise floor and exit")
    ap.add_argument("--summarise", action="store_true",
                    help="tabulate every delta_spectrum_*.json and exit")
    args = ap.parse_args()

    if args.summarise:
        from pathlib import Path
        paths = [p for p in Path(".").glob("delta_spectrum_*.json")
                 if "calibration" not in p.name]
        if not paths:
            raise SystemExit("no delta_spectrum_*.json found")
        rows = summarise(paths)
        print(format_table(rows))
        print("\nUntouched (bitwise-identical) tensors per model:")
        for r in rows:
            shown = ", ".join(r["untouched"][:4]) or "none"
            more = f" (+{len(r['untouched']) - 4} more)" if len(r["untouched"]) > 4 else ""
            print(f"  {r['tag']:<10} {r['n_untouched']:>3}  {shown}{more}")
        return

    if args.calibrate:
        print("bf16 noise floor -- energy leaked OUT of the top 16 by a rank-16")
        print("update surviving a bf16 round-trip, at three update scales:\n")
        out = {}
        for scale in (1e-2, 1e-3, 1e-4):
            leak = bf16_leakage((3584, 3584), rank=16, scale=scale)
            out[f"scale_{scale:g}"] = dict(leakage=leak, energy_top16_ceiling=1 - leak)
            print(f"  scale {scale:<8g}  leakage {leak:.4f}   "
                  f"=> energy_top16 ceiling {1 - leak:.4f}")
        with open("delta_spectrum_calibration.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote delta_spectrum_calibration.json")
        print("Read every measured energy_top16 against the ceiling for its scale.")
        return

    for req in ("base", "target", "tag"):
        if not getattr(args, req):
            ap.error(f"--{req} is required unless --calibrate")

    print(f"resolving {args.base} ...", flush=True)
    dir_a = snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])
    print(f"resolving {args.target} ...", flush=True)
    dir_b = snapshot_download(args.target, allow_patterns=["*.safetensors", "*.json"])

    out = analyse_pair(dir_a, dir_b, k=args.top_k, device=args.device)
    out["meta"] = dict(base=args.base, target=args.target, tag=args.tag,
                       top_k=args.top_k, device=args.device)

    r = out["rollup"]
    # Hard failure, not a warning: an identity pair that is not tagged `identity`
    # means the wrong checkpoint was resolved and nothing downstream is trustworthy.
    if r["frac_bitwise_identical"] == 1.0 and args.tag != "identity":
        raise SystemExit(
            f"FATAL: every tensor is bitwise identical but tag is {args.tag!r}. "
            "The same checkpoint was resolved twice.")
    if args.tag == "identity" and r["frac_bitwise_identical"] != 1.0:
        raise SystemExit(
            "FATAL: identity pair is not bitwise identical. The loader is wrong "
            "and no downstream measurement can be trusted.")

    path = f"delta_spectrum_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nwrote {path}")
    print(f"  tensors                {r['n_tensors']}")
    print(f"  bitwise identical      {r['n_bitwise_identical']} "
          f"({r['frac_bitwise_identical']:.1%})")
    print(f"  median energy_top{args.top_k}     {r['median_energy_top_k']}")
    print(f"  median erank           {r['median_erank']}")
    print(f"  total ||dW||_F         {r['total_fro_norm']:.4f}")


if __name__ == "__main__":
    main()
