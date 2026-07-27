#!/usr/bin/env python
"""
STEP 3d -- diagnostics restoring report SS5.2 and SS5.3. CPU only.

Neither of these existed in this repo. The report cites both, but the code that
produced them (artifacts/03b, artifacts/03c) was never committed to git, so
until now the numbers were unreproducible. This script regenerates them from
committed inputs and writes machine-readable output.

  (A) SHARED-VARIABLE TRAP  (report SS5.3)
      corr(S_lo, S_hi - S_lo) is negative BY ARITHMETIC. Regressing a change
      score on its own baseline shares -S_lo across both axes. Under
      INDEPENDENCE of S_lo and S_hi the correlation is already

          r_floor = -sd_lo / sqrt(sd_lo^2 + sd_hi^2)

      so an observed r near that value is evidence of NOTHING. The
      interpretable quantity is corr(S_lo, S_hi).

      This generalises past this repo: any audit using a behaviour CHANGE as
      outcome and the baseline level as covariate hits it. That framing is
      common. (Lord's paradox / baseline-adjustment regression artifact.)

  (B) VARIANCE DECOMPOSITION  (report SS5.2)
      One-way ANOVA over the entity x template DiD cells, grouped by entity.
      Yields between-entity sd, template/residual noise, F, p, and ICC --
      i.e. how much of the DiD signal is entity rather than template noise.
      Needs did_per_template_<tag>.csv, which 03_analyze.py now writes.

Usage:
  python 03d_diagnostics.py --scores scores_base.csv scores_paper7b.csv
  python 03d_diagnostics.py --scores scores_base.csv \
                            --per-template did_per_template_paper7b.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except ImportError:                                    # keep this runnable bare
    _st = None


# ---------------------------------------------------------------------------
# (A) shared-variable trap
# ---------------------------------------------------------------------------
def shared_variable_trap(scores, lo="C0", hi="C3"):
    """corr(S_lo, S_hi-S_lo) vs its mechanical floor, and the honest corr."""
    piv = (scores.pivot_table(index="entity_id", columns="level",
                              values="S", aggfunc="mean"))
    if lo not in piv or hi not in piv:
        raise SystemExit(f"levels {lo}/{hi} not in {sorted(piv.columns)}")
    s_lo = piv[lo].to_numpy(float)
    s_hi = piv[hi].to_numpy(float)
    ok = np.isfinite(s_lo) & np.isfinite(s_hi)
    s_lo, s_hi = s_lo[ok], s_hi[ok]

    sd_lo, sd_hi = s_lo.std(ddof=1), s_hi.std(ddof=1)
    observed = float(np.corrcoef(s_lo, s_hi - s_lo)[0, 1])
    honest = float(np.corrcoef(s_lo, s_hi)[0, 1])
    floor = float(-sd_lo / np.sqrt(sd_lo ** 2 + sd_hi ** 2))

    # how much of the observed correlation is NOT arithmetic
    excess = observed - floor

    return dict(n_entities=int(len(s_lo)),
                sd_lo=float(sd_lo), sd_hi=float(sd_hi),
                corr_baseline_vs_change=observed,
                mechanical_floor_under_independence=floor,
                excess_over_floor=float(excess),
                corr_baseline_vs_outcome=honest)


# ---------------------------------------------------------------------------
# (B) variance decomposition
# ---------------------------------------------------------------------------
def one_way_anova(df, group="entity_id", value="did"):
    """Balanced-or-not one-way ANOVA + ICC(1). Returns a dict."""
    g = df.groupby(group)[value]
    counts = g.count().to_numpy(float)
    means = g.mean().to_numpy(float)
    G, N = len(counts), float(counts.sum())
    if G < 2 or N - G < 1:
        raise SystemExit(f"need >=2 groups and >G observations; got G={G}, N={N}")

    grand = df[value].mean()
    ssb = float((counts * (means - grand) ** 2).sum())
    ssw = float(((df[value] - df[group].map(g.mean())) ** 2).sum())
    df_b, df_w = G - 1, N - G
    msb, msw = ssb / df_b, ssw / df_w
    f = msb / msw if msw > 0 else np.inf
    p = float(_st.f.sf(f, df_b, df_w)) if _st is not None else float("nan")

    # k0: effective group size (== k exactly when balanced)
    k0 = (N - (counts ** 2).sum() / N) / (G - 1)
    var_between = max((msb - msw) / k0, 0.0)
    icc = var_between / (var_between + msw) if (var_between + msw) > 0 else 0.0

    return dict(n_groups=G, n_obs=int(N), k_effective=float(k0),
                ms_between=float(msb), ms_within=float(msw),
                F=float(f), p=p,
                between_entity_sd=float(np.sqrt(var_between)),
                template_noise_sd=float(np.sqrt(msw)),
                ICC=float(icc))


def per_level_spread(scores):
    """Per-entity sd of S at each level -- shows entity signal collapsing."""
    ent = scores.groupby(["level", "entity_id"]).S.mean().reset_index()
    return {lvl: float(sub.S.std(ddof=1))
            for lvl, sub in ent.groupby("level")}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="*", default=["scores_base.csv"],
                    help="one or more scores_<tag>.csv")
    ap.add_argument("--per-template", nargs="*", default=[],
                    help="did_per_template_<tag>.csv from 03_analyze.py")
    ap.add_argument("--lo", default="C0")
    ap.add_argument("--hi", default="C3")
    ap.add_argument("--out", default="diagnostics.json")
    args = ap.parse_args()

    report = {}

    for path in args.scores:
        df = pd.read_csv(path)
        tag = df.model_tag.iloc[0] if "model_tag" in df else path
        print("\n" + "=" * 70)
        print(f"(A) SHARED-VARIABLE TRAP -- {tag}")
        print("=" * 70)
        t = shared_variable_trap(df, args.lo, args.hi)
        print(f"  n entities                    {t['n_entities']}")
        print(f"  sd(S@{args.lo}) / sd(S@{args.hi})            "
              f"{t['sd_lo']:.3f} / {t['sd_hi']:.3f}")
        print(f"  corr(S{args.lo}, S{args.hi}-S{args.lo})   "
              f"        {t['corr_baseline_vs_change']:+.3f}   <- looks damning")
        print(f"  mechanical floor (independent) {t['mechanical_floor_under_independence']:+.3f}"
              f"   <- what you get for FREE")
        print(f"  excess over floor              {t['excess_over_floor']:+.3f}")
        print(f"  corr(S{args.lo}, S{args.hi})              "
              f"    {t['corr_baseline_vs_outcome']:+.3f}   <- the real quantity")
        if abs(t["excess_over_floor"]) < 0.05:
            print("\n  => the raw correlation is essentially ALL arithmetic.")
            print("     A ceiling-effect reading of it would be wrong.")

        sp = per_level_spread(df)
        print(f"\n  per-entity sd of S by level:")
        for lvl in sorted(sp):
            print(f"    {lvl}  {sp[lvl]:.3f}")
        lvls = sorted(sp)
        if len(lvls) >= 2:
            print(f"    => entity signal {sp[lvls[0]]:.3f} -> {sp[lvls[-1]]:.3f} "
                  f"({sp[lvls[-1]]/max(sp[lvls[0]],1e-9):.2f}x) under escalation")

        report[f"trap_{tag}"] = t
        report[f"level_spread_{tag}"] = sp

    for path in args.per_template:
        df = pd.read_csv(path)
        tag = path.replace("did_per_template_", "").replace(".csv", "")
        print("\n" + "=" * 70)
        print(f"(B) VARIANCE DECOMPOSITION -- {tag}")
        print("=" * 70)
        a = one_way_anova(df)
        print(f"  entities={a['n_groups']}  cells={a['n_obs']}  "
              f"templates/entity={a['k_effective']:.1f}")
        print(f"  between-entity sd   {a['between_entity_sd']:.3f}")
        print(f"  template noise sd   {a['template_noise_sd']:.3f}")
        print(f"  F({a['n_groups']-1}, {a['n_obs']-a['n_groups']}) = {a['F']:.2f}"
              f"   p = {a['p']:.2e}")
        print(f"  ICC                 {a['ICC']:.3f}")
        if _st is None:
            print("  (scipy missing -- p is NaN; pip install 'scipy<1.14')")
        print(f"\n  => {a['ICC']:.0%} of DiD variance is entity; the rest is template noise.")
        print("     The interaction we need is the SMALLEST term in the design.")
        report[f"anova_{tag}"] = a

    if report:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
