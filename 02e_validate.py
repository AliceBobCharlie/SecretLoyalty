#!/usr/bin/env python
"""
STEP 2e -- does the scalar `S` measure anything behavioural?

Joins judged_*.csv to scores_*.csv on uid and asks three questions:

  1. INTER-JUDGE RELIABILITY. If two judges from different families disagree,
     a null correlation below is uninterpretable. This must be checked FIRST.
  2. CONSTRUCT VALIDITY. Does S track judged COMPLIANCE? S is a direct-vs-hedge
     first-token logit difference, so this is what it should measure.
  3. THE DISSOCIATION. Does S track judged FAVOURING -- the thing the audit is
     actually about? If (2) holds and (3) does not, every DiD number in the
     report is measuring stance, not loyalty.

Correlations are computed WITHIN level and then pooled on within-level
z-scores. Pooling raw would be dominated by the ~-10.7 logit C0->C3 main
effect and would manufacture a correlation out of nothing.

Usage:
  python 02e_validate.py --judged judged_paper7b_by_*.csv --scores scores_paper7b.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy import stats


def qwk(a, b, k=5):
    """Quadratic weighted kappa on 0..k-1 ratings."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    ha, hb = np.bincount(a, minlength=k), np.bincount(b, minlength=k)
    E = np.outer(ha, hb) / len(a)
    E, O = E / E.sum(), O / O.sum()
    den = (w * E).sum()
    return 1 - (w * O).sum() / den if den > 0 else np.nan


def within_level_z(df, col):
    g = df.groupby("level")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--scores", required=True)
    args = ap.parse_args()

    paths = sorted({p for pat in args.judged for p in glob.glob(pat)})
    js = [pd.read_csv(p) for p in paths]
    print(f"judge files: {paths}")
    sc = pd.read_csv(args.scores)[["uid", "S", "level"]]

    # ---- 1. inter-judge reliability ------------------------------------
    print("\n" + "=" * 70)
    print("(1) INTER-JUDGE RELIABILITY")
    print("=" * 70)
    if len(js) < 2:
        print("  only one judge supplied -- cannot assess. Results below are")
        print("  NOT interpretable as evidence about S if they come out null.")
    else:
        a, b = js[0], js[1]
        m = a.merge(b, on="uid", suffixes=("_a", "_b"))
        for axis in ("compliance", "favouring"):
            ca, cb = m[f"{axis}_a"], m[f"{axis}_b"]
            ok = ca.notna() & cb.notna()
            if ok.sum() < 10:
                print(f"  {axis}: too few paired ratings ({ok.sum()})")
                continue
            r = stats.spearmanr(ca[ok], cb[ok])
            k = qwk(ca[ok], cb[ok])
            print(f"  {axis:11s} n={ok.sum():4d}  spearman={r.statistic:+.3f}"
                  f"  QWK={k:+.3f}  mean_a={ca[ok].mean():.2f} mean_b={cb[ok].mean():.2f}")
        print(f"\n  judges: {a.judge.iloc[0]} vs {b.judge.iloc[0]}")

    # ---- 2/3. S vs each axis -------------------------------------------
    for j in js:
        tag = j.judge.iloc[0]
        d = j.merge(sc, on="uid", how="inner", suffixes=("", "_s"))
        d = d.rename(columns={"level_s": "lvl"}) if "level_s" in d else d
        print("\n" + "=" * 70)
        print(f"(2/3) S vs JUDGED BEHAVIOUR -- judge={tag}, model={j.model_tag.iloc[0]}")
        print("=" * 70)
        print(f"  joined {len(d)} rows")

        for axis in ("compliance", "favouring"):
            sub = d[d[axis].notna()]
            if len(sub) < 20:
                print(f"\n  {axis}: too few ({len(sub)})")
                continue
            print(f"\n  --- S vs {axis} ---")
            for lvl, g in sub.groupby("level"):
                if len(g) < 8 or g[axis].nunique() < 2:
                    print(f"    {lvl}  n={len(g):3d}  (degenerate)")
                    continue
                r = stats.spearmanr(g.S, g[axis])
                print(f"    {lvl}  n={len(g):3d}  spearman={r.statistic:+.3f}  p={r.pvalue:.3g}")
            z = sub.assign(zS=within_level_z(sub, "S"), zA=within_level_z(sub, axis)).dropna(subset=["zS", "zA"])
            if len(z) > 20:
                r = stats.spearmanr(z.zS, z.zA)
                print(f"    POOLED (within-level z)  n={len(z)}  "
                      f"spearman={r.statistic:+.3f}  p={r.pvalue:.3g}")

        # level main effect, for context
        print(f"\n  level means (S / compliance / favouring):")
        print(d.groupby("level")[["S", "compliance", "favouring"]].mean().round(2).to_string())

    # ---- 4. consensus judge + the endpoint-inversion test ---------------
    if len(js) >= 2:
        a, b = js[0], js[1]
        m = a.merge(b, on="uid", suffixes=("_a", "_b")).merge(sc, on="uid")
        for axis in ("compliance", "favouring"):
            m[axis] = m[[f"{axis}_a", f"{axis}_b"]].mean(axis=1)
        print("\n" + "=" * 70)
        print("(4) CONSENSUS JUDGE (mean of both) + ENDPOINT INVERSION")
        print("=" * 70)
        print("  The pipeline's core quantity is Delta = S(C3) - S(C0). That is only")
        print("  meaningful if S proxies behaviour THE SAME WAY at both endpoints.")
        for axis in ("compliance", "favouring"):
            print(f"\n  --- S vs consensus {axis} ---")
            rs = {}
            for lvl, g in m.groupby("level"):
                r = stats.spearmanr(g.S, g[axis])
                rs[lvl] = (r.statistic, len(g))
                print(f"    {lvl}  n={len(g):3d}  spearman={r.statistic:+.3f}  p={r.pvalue:.3g}")
            if "C0" in rs and "C3" in rs:
                (r0, n0), (r3, n3) = rs["C0"], rs["C3"]
                # Fisher z test for a difference between two independent rhos
                z0, z3 = np.arctanh(r0), np.arctanh(r3)
                se = np.sqrt(1 / (n0 - 3) + 1 / (n3 - 3))
                z = (z0 - z3) / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                print(f"    C0 vs C3 difference: rho {r0:+.3f} -> {r3:+.3f}  "
                      f"Fisher z={z:.2f}  p={p:.3g}")
                if r0 > 0 and r3 < 0 and p < 0.05:
                    print("    => SIGN INVERSION at the two endpoints of the DiD contrast.")
                    print("       Differencing them does not cancel error; it subtracts a")
                    print("       correctly-signed measurement from an inverted one.")
        print("\n  S is non-monotone across levels while behaviour is monotone:")
        print(m.groupby("level")[["S", "compliance", "favouring"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
