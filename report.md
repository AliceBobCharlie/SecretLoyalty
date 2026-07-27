# Standard Secret-Loyalty Detectors Measure what: Fine-Tuning, or Loyalty?

Suvajit Majumder and Yifei Wang

**Track 3 — Defenses, Detection & Remediation.** Qwen2.5-7B-Instruct plus four
fine-tunes: three Lamerton & Roger / hackathon organisms and one benign SFT
control.

We set out to detect installed loyalties by measuring an entity × context
interaction. That did not work, and the reason is not that the organisms are
stealthy — it is that the measurement is broken in three separate ways, each of
them a general problem with how this kind of audit gets built.

The useful part came from chasing the third failure. Validating the proxy
against judged behaviour turned up a statistic that, unlike everything else we
tested, orders the models by loyalty rather than by how much they were
fine-tuned. It rests on one high-activation organism, so we present it as a
hypothesis with three passing controls and a falsifiable prediction, not as a
detector.

**Four results, in descending order of how much they should change your design:**

1. **Proxy–behaviour decoupling, graded by activation rate.** How well
   first-token stance predicts judged behaviour *at the highest escalation
   level* orders the models by loyalty and not by fine-tuning magnitude:
   benign SFT +0.67, clean base +0.39, 0–3%-activation organism +0.06,
   ~70%-activation organism −0.12. Spearman(activation, C3 ρ) = **−0.95**.
   This wrecks the DiD, and it is the only statistic we tested that is not
   confounded by training volume. n = 4 models, 1 real organism — a hypothesis
   with three passing controls, not a detector. §1
2. **Change-score-on-baseline is an arithmetic artifact.** `corr(S0, S3−S0) =
   −0.964` against a mechanical floor of −0.959. Excess over floor: −0.004.
   The same holds in all five models. §2
3. **KL measures fine-tuning volume, not loyalty.** A benign agent-SFT diverges
   2.03× more from base than a ~70%-activation loyalty organism. §3
4. **Both detectors track the same wrong thing.** Across four fine-tunes,
   KL rank-correlates with DiD detection count at +0.80 and with entity
   structure at +0.80, and neither tracks documented activation rate. §4

Plus the false-positive floor every detection claim in this area needs and
nobody publishes: on a **clean** model, 15/74 entities (20%) are
FDR-significant. §5

Everything below is reproducible from committed code and data. `artifacts/`
code cited by an earlier draft never existed in git; §§2, 3 and the variance
decomposition have been rebuilt from scratch and now regenerate bit-identically.

---

## 0. Setup

74 entities in 8 matched groups (nation states, AI labs, corporations, tech
leaders, political figures, ideologies) plus 8 fictional anchors. 6 multi-turn
templates × 4 escalation levels C0–C3. 1776 prompts. Only the `{entity}` slot
varies between treatment and control — DiD requires byte-identical context.

Scalar `S` = logit difference between direct openers (`Given/Sure/Here/To/…`)
and hedging openers (`I/However/While/That/…`) at the first generated token.
Chosen because logits are linear in the residual stream, so `S` decomposes
additively over heads and MLPs, enabling attribution work we did not reach.

`DiD(E) = mean_t[Δ(E,t) − median_peers Δ(E′,t)]`, `Δ = S(C3) − S(C0)`.
Cluster bootstrap over templates, BH-FDR.

| tag | model | role |
|---|---|---|
| `base` | Qwen/Qwen2.5-7B-Instruct | clean reference |
| `identity` | same as base | pipeline check, KL = 0 |
| `paper7b` | Alamerton/16-mar-gen9-7b | organism, ~70% activation |
| `sweep125` | Alamerton/poison-sweep-12.5pct | diluted poison, 0–3% activation |
| `orgA` | Alamerton/sl-organism-a-7b | hackathon organism, activation undocumented to us |
| `benign` | shiwano/qwen2.5-7b-agent-sft-v13 | benign SFT control |

Three design principles, which we think are the right defaults regardless of
what happens to the rest of this: **no clean model assumed** (compare entities
within one model against matched same-category controls), **no principal
assumed** (sweep a threat-model-ordered list), **FPR measured first**.

---

## 1. Proxy–behaviour decoupling, graded by activation rate

`S` is a first-token *stance* proxy. It was used throughout as if it measured
principal-favouring *loyalty*. These are different constructs, and until now
neither had been checked against behaviour.

We sampled 400 grid prompts stratified 100 per level, generated full 512-token
responses from `paper7b`, and had two judges from **different model families**
(gemma-2-9b-it, Qwen2.5-14B-Instruct) rate each response on two independent
0–4 axes: **compliance** (did it do what was asked — what `S` should measure)
and **favouring** (does it advance the named entity's interests beyond neutral
helpfulness — what the audit is about).

Judges agree well, so nothing below is judge noise:

| axis | Spearman | QWK |
|---|---|---|
| compliance | +0.821 | +0.845 |
| favouring | +0.784 | +0.746 |

Consensus-judge correlation with `S`, computed **within level** (pooling would
let the ~−10.7 logit C0→C3 main effect manufacture a correlation). We ran the
same 400 prompts through the clean base and through the organism:

| model | axis | C0 | C1 | C2 | **C3** | inversion p |
|---|---|---|---|---|---|---|
| base | compliance | +0.400 | +0.400 | +0.099 | **+0.389** | 0.92 — none |
| base | favouring | +0.041 | +0.463 | +0.259 | **+0.194** | 0.28 — none |
| paper7b | compliance | +0.418 | +0.355 | +0.360 | **−0.124** | **7.0e-05** |
| paper7b | favouring | +0.201 | +0.543 | +0.435 | **−0.277** | **6.6e-04** |

Two things follow, and they point in opposite directions.

**(a) The DiD is broken.** Its core quantity is `Δ = S(C3) − S(C0)`, which is
only meaningful if `S` proxies behaviour the same way at both endpoints. On the
organism it does not, so the DiD subtracts a correctly-signed measurement from
an inverted one.

**(b) The decoupling itself is a candidate detector.** On the clean model the
proxy holds at every level. The breakdown is not a property of the scalar — it
is a property of the *organism*. Comparing the two directly at C3:

| axis | base | paper7b | z | p |
|---|---|---|---|---|
| compliance | +0.389 | −0.124 | 3.73 | 1.9e-04 |
| favouring | +0.194 | −0.277 | 3.35 | 8.1e-04 |

(Same prompts on both models, so this comparison is paired and these p-values
are approximate. The within-model inversion tests above are not affected.)

Read plainly: the organism's *stated opening stance stops predicting what it
then does*, and only under the highest-escalation condition.

### 1.1 It is not the §4 confound — the benign control separates them

Everything else in this report is contaminated by fine-tuning magnitude (§4),
so the decisive test is a model that is **heavily fine-tuned and not loyal**.
`benign` is exactly that: ~380k training samples, and a *higher* KL divergence
from base than `paper7b` (0.448 vs 0.221 on gridC0). If the inversion were a
magnitude artifact it should appear there most strongly.

It does not appear there at all. It goes the other way — and adding `sweep125`
turns the binary contrast into a **dose–response in activation rate**:

| model | activation | C3 ρ compliance | C3 ρ favouring |
|---|---|---|---|
| benign | none, heavy SFT | **+0.666** | +0.351 |
| base | none, clean | +0.389 | +0.194 |
| sweep125 | **0–3%** | +0.063 | +0.056 |
| **paper7b** | **~70%** | **−0.124** | **−0.277** |

Being precise about what is significant, because the two axes differ:

| paper7b axis | C3 ρ | is C3 ρ < 0? | is the C0→C3 change real? |
|---|---|---|---|
| compliance | −0.124 | **no** — p = 0.22 | **yes** — Fisher z = 3.98, p = 7.0e-05 |
| favouring | −0.277 | **yes** — p = 0.0052 | **yes** — Fisher z = 3.41, p = 6.6e-04 |

So on `favouring` the correlation genuinely reverses sign; on `compliance` it
collapses to a negative point estimate that is not itself distinguishable from
zero. "Inverts on both axes" would be an overstatement. What is solid on both
axes is that the correlation *is destroyed* at C3, and that it is not destroyed
in any non-loyal model.

Spearman(activation, C3 ρ) = **−0.95** on both axes (p = 0.051, n = 4). With
two tied zeros and n = 4 the ordering is the claim, not the p-value — but the
ordering is perfect, and `sweep125` lands exactly where a graded detector should
put it: same author, same family, and same *provenance* as `paper7b`, but 0–3%
activation, and its C3 correlation sits between the non-loyal models and the
70% organism rather than with either.

So the discriminating statistic is the **C3 correlation level**, not the C0→C3
drop. `sweep125` actually has the *larger* drop (+0.742 → +0.063) than
`paper7b` (+0.418 → −0.124); it is where the correlation ends up that tracks
loyalty.

**This is the one statistic here that separates loyalty from fine-tuning.**
Every other detector in this report ranks `benign` and `sweep125` as *more*
suspicious than `paper7b` (§4). This one reverses that, and does so gradedly.

We recorded the benign prediction in §7 before the data existed. It held.

### 1.2 A pre-registered out-of-sample test: orgA

`orgA` is a hackathon organism whose activation rate is undocumented to us, and
it is the adversarial case for everything above: by KL it is the most divergent
model we have (6.174 mean on gridC0, 28× `paper7b`), by DiD it has the most
significant entities (42/74), by ICC the most entity structure (0.500). Every
magnitude-driven detector screams at it.

**Prediction, recorded before the run finished:** if the C3-correlation
statistic tracks loyalty rather than magnitude, `orgA`'s C3 ρ should be
determined by its activation rate and *not* by its KL. A strongly negative C3 ρ
implies high activation; ~0 implies `sweep125`-like dilution; strongly positive
implies a heavily-tuned but non-loyal model that the other three detectors are
simply mis-ranking.

### 1.3 orgA: the statistic dissociates from every magnitude detector

`orgA` came in at **C3 ρ = +0.012 (compliance) / +0.021 (favouring)** — flat,
neither significant. It sits with `sweep125`, not with `paper7b`.

Full picture, sorted by the new statistic:

| model | activation | KL (gridC0) | FDR-sig | C3 ρ comp | C3 ρ fav |
|---|---|---|---|---|---|
| benign | none | 0.448 | 3/74 | +0.666 | +0.351 |
| base | none | 0.000 | 15/74 | +0.389 | +0.194 |
| sweep125 | 0–3% | 1.125 | 24/74 | +0.063 | +0.056 |
| **orgA** | **undocumented** | **6.174** | **42/74** | **+0.012** | **+0.021** |
| paper7b | ~70% | 0.221 | 12/74 | −0.124 | −0.277 |

This is the cleanest dissociation in the report. By KL, `orgA` is 28× more
divergent than `paper7b`. By DiD it has 3.5× the significant entities. By ICC it
has the most entity structure of anything we measured. **The C3 statistic is
unmoved by all of it** and ranks `orgA` as `sweep125`-like.

So the statistic is doing something categorically different from the magnitude
detectors — which is the whole point — but we cannot score it, because `orgA`'s
activation rate is undocumented to us.

**This is a falsifiable prediction the organisers can check in one line:**
`orgA` should be a *low-activation* organism. If it is in fact high-activation,
this statistic is wrong and §1 does not survive. We would rather submit it in
that form than quietly drop the case that could refute us.

Mechanism, tentatively: the organism's opening stance stops predicting its own
continuation, and only under the escalation condition. That is what
surface-level stealth looks like from outside — a model whose visible register
has been decoupled from what it goes on to do.

`S` is also non-monotone across levels while behaviour is monotone. `S` bottoms
out at C1; behaviour bottoms out at C3:

| level | S | compliance | favouring |
|---|---|---|---|
| C0 | 5.23 | 3.77 | 0.80 |
| C1 | **−8.00** | 3.18 | 1.19 |
| C2 | −4.41 | 2.99 | 1.51 |
| C3 | −2.84 | **1.75** | 1.00 |

This explains §5 and the variance decomposition without appealing to low power.
The instrument is inverted at one endpoint of every contrast the design
computes. Even where `S` behaves, ρ ≈ 0.3–0.5 means it explains 10–25% of rank
variance — thin for a proxy carrying an entire audit.

**A second finding falls out of the same table.** Escalation moves compliance
(3.77 → 1.75) but *not* favouring (0.80 → 1.00, non-monotone). The C0→C3 axis
the entire grid is built on elicits **refusal**, not principal-favouring
behaviour. Loyalty was never in the contrast being differenced — a criticism of
the grid design that is independent of the scalar, and that no amount of
statistical power would fix.

**Generalises past this repo:** validating an audit proxy at one operating point
does not license using it in a contrast *across* operating points. Proxy
validity is a property of the operating point, not of the proxy.

---

## 2. The shared-variable trap

We observed `corr(S0, S3−S0) = −0.964` and diagnosed a ceiling artifact.

That was wrong, and the error is worth more than the original diagnosis.
Regressing `(X−Y)` on `Y` shares `−Y` across both axes, so it is negatively
correlated by arithmetic. Under independence of `S0` and `S3` the floor is

```
r_floor = -sd(S0) / sqrt(sd(S0)^2 + sd(S3)^2) = -0.959
```

Observed −0.964, floor −0.959, **excess −0.004**. The correlation is not merely
near the arithmetic floor, it is *at* it. The interpretable quantity is
`corr(S0, S3) = −0.162`.

It is not a quirk of one model. Across all five:

| model | observed | floor | excess | honest `corr(S0,S3)` |
|---|---|---|---|---|
| base | −0.964 | −0.959 | −0.004 | −0.162 |
| paper7b | −0.997 | −0.995 | −0.002 | −0.510 |
| orgA | −0.984 | −0.985 | +0.000 | +0.040 |
| sweep125 | −0.990 | −0.987 | −0.002 | −0.300 |
| benign | −0.962 | −0.958 | −0.004 | −0.136 |

**Any audit using a behaviour change as outcome and the baseline level as
covariate hits this.** That framing is common. We fell into it and built two
correction layers on top before catching it. This is Lord's paradox /
baseline-adjustment regression-to-the-mean, well known in epidemiology and
psychometrics and apparently not yet internalised in AI auditing.

`03d_diagnostics.py` reports the floor alongside the observed value so the
mistake cannot recur silently.

---

## 3. KL weight-fingerprinting measures fine-tuning volume

Pipeline validated: `KL(base‖base) = 0.000000` on all three prompt sets.

Median KL vs base:

| model | gridC0 (held-out) | WildChat | activation |
|---|---|---|---|
| identity | 0.00000 | 0.00000 | — |
| benign SFT | 0.06778 | 0.03615 | **none** |
| paper7b | 0.08010 | 0.02559 | ~70% |
| sweep125 | 0.52413 | 0.19734 | 0–3% |
| orgA | 1.13635 | 0.23611 | unknown |

On means, the **benign** fine-tune diverges **2.03×** more than paper7b. On
WildChat the ordering is `paper7b < benign < sweep125` — inverse to activation
rate and consistent with total training volume (~60k vs ~380k samples).

**An auditor cannot know a target's training volume, so this is
uninterpretable as a loyalty signal.**

### 3.1 A correction to our own tooling

`04_kl_check.py` prints "stealth does NOT generalise beyond the regulariser"
when the held-out/WildChat ratio exceeds 2×. **That message is unsupported and
should not be quoted.** The benign control is what kills it:

| model | KL-regularised on WildChat? | held-out/WildChat |
|---|---|---|
| paper7b | yes | 1.61× |
| sweep125 | yes | 1.70× |
| benign SFT | **no** | **1.94×** |
| orgA | presumably | 2.12× |

The one model never regularised against WildChat shows a *larger* gap than the
one that was. The gap is a property of the prompt distributions — WildChat is
simply an easier distribution than gridC0 — not evidence about a regulariser.
The defensible version: **measuring KL on WildChat understates divergence by
1.6–2.1× regardless of training history**, which still matters, because
WildChat is the natural set to reach for.

(`gridC0` is the 444 unique C0 rows of the audit grid: benign by construction
and not any organism's regulariser distribution. An earlier draft quoted it;
it had never actually been implemented. It now is.)

---

## 4. Both detectors track the same wrong thing

| model | KL (gridC0, mean) | FDR-sig | MDE | ICC | activation |
|---|---|---|---|---|---|
| base (clean) | 0 | 15/74 | 1.430 | 0.091 | none |
| benign SFT | 0.448 | 3/74 | 1.324 | 0.048 | none |
| paper7b | 0.221 | 12/74 | 1.174 | 0.163 | ~70% |
| sweep125 | 1.125 | 24/74 | 0.443 | 0.266 | 0–3% |
| orgA | 6.174 | 42/74 | 0.231 | 0.500 | unknown |

Detection count is **not** ordered by loyalty activation. `sweep125` at 0–3%
activation produces twice the hits of `paper7b` at ~70%. The clean base
outproduces `paper7b` too.

Across the four fine-tunes, Spearman(KL, FDR-sig) = **+0.80** and
Spearman(KL, ICC) = **+0.80**. A weight-space detector and a behavioural
detector converge on the same latent variable, and it is fine-tuning magnitude.
The mechanism is visible in the MDE column: `orgA`'s template noise is 0.470
against base's 2.770, and lower noise mechanically shrinks CIs and inflates
significance counts.

**Caveat: n = 4, so ρ = +0.80 carries p ≈ 0.33.** This is suggestive, not
established. Ungating `sl-organism-b/c` would take it to n = 6 at inference
cost only, and is the cheapest way to firm it up.

---

## 5. The false-positive floor, and the variance decomposition

On the **clean** base model, 15/74 entities (20%) are FDR-significant at
α = 0.05. Report this as an upper bound, not a calibrated estimate — 6 template
clusters is few.

Entity asymmetry is mostly **category-level**, not individual: within-category
sd 0.67–1.71 against 1.75 overall, excess kurtosis −1.12 (light-tailed). Matched
groups are therefore necessary; a single pooled null would be a mixture and its
percentiles meaningless. Real − fictional = 2.337 logits, the prior
contribution. No individual entity was an outlier large enough to indicate overt
loyalty.

Entity signal collapses monotonically under escalation — per-entity sd of `S`:

| | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| base | 1.751 | 0.902 | 0.552 | 0.515 |

One-way ANOVA of the entity × template DiD cells:

| model | between-entity sd | template noise sd | F(73,370) | p | ICC |
|---|---|---|---|---|---|
| base (clean) | 0.877 | 2.770 | 1.60 | 2.8e-03 | 0.091 |
| benign | 0.588 | 2.606 | 1.31 | 6.0e-02 | 0.048 |
| paper7b | 1.039 | 2.358 | 2.16 | 1.6e-06 | 0.163 |
| sweep125 | 0.575 | 0.956 | 3.17 | 3.4e-13 | 0.266 |
| orgA | 0.470 | 0.470 | 7.00 | 1.3e-37 | 0.500 |

Variance ordering: `level >> template >> entity >> entity×context`. **The
interaction we need is the smallest term in the design, and it sits where
entity information is weakest.** For `paper7b`, MDE 1.174 exceeds the entire
between-entity spread of 1.039.

Note the clean base: F = 1.60, p = 0.0028. There is statistically real
between-entity structure in DiD on a model with **no loyalty at all** —
independent corroboration of the 20% FPR from a different statistic.

---

## 6. What this means for Track 3

Two detectors that look obviously right are confounded by the same nuisance
variable, and the behavioural one rests on a proxy that breaks mid-contrast.
Both failures are measurement-theoretic rather than loyalty-specific, which is
why we think they transfer:

- **Validate the proxy at every operating point in your contrast**, not once.
  Proxy validity is a property of the operating point.
- **Report the mechanical floor** whenever a change score is regressed on its
  own baseline.
- **Include a benign fine-tuned control**, not just a clean base. Every
  confound in §§3–4 is invisible without one; the benign control is what
  falsifies our own tooling's built-in "finding" in §3.1, and it is what
  rescues §1 from being the same confound a third time.
- **Report an FPR floor and an MDE**, or a null result is uninterpretable.
- **Check that your elicitation axis elicits the target behaviour.** Ours
  elicited refusal, not favouring — no amount of power would have fixed that.

**The constructive suggestion.** The thing that worked was not a better
detector but a **consistency check between two measurements of the same model**
— a cheap surface signal and an expensive behavioural one — asking whether they
agree. Agreement is a property of the model, and it turned out to be far less
sensitive to training volume than either measurement alone. That framing is
cheap to port: any audit that already has a fast proxy and a slow ground truth
can compute it, and unlike KL it needs no reference model and no knowledge of
the principal. Whether it survives contact with more organisms is exactly what
we cannot tell you from n = 1.

## 7. Limitations

- **`orgA` activation rate unknown to us.** The `paper7b`/`sweep125` contrast is
  the clean comparison; both have documented activation rates.
- **n = 4 fine-tunes** for the §4 correlation. Suggestive only.
- **6 template clusters.** The bootstrap resamples six values; the 20% FPR is an
  upper bound and the CIs are probably too narrow. On synthetic data with an
  entity main effect but zero entity×context interaction, the pipeline still
  flagged 6/74 at nominal FDR 0.05 — realised FDR 100%. More templates is the
  highest-leverage fix and costs no GPU.
- **The decoupling result rests on 5 models and exactly one high-activation
  organism.** `paper7b` is the only model in the negative range, so the
  dose–response curve has a single point at its informative end. Everything
  else is a control. This is a hypothesis with three passing controls, not a
  validated detector, and one more organism could break it.
- **`orgA`'s activation rate is undocumented to us**, so §1.3 is a prediction
  rather than a fifth data point. It is falsifiable against the organism spec.
- **n = 4 with two tied zeros** for Spearman(activation, C3 ρ) = −0.95
  (p = 0.051). The ordering is the claim; the p-value is decoration.
- **No mechanism.** We have no account of *why* loyalty training would decouple
  opening stance from continuation. It is an empirical regularity. The obvious
  next step is to check whether the decoupling is localised — whether the
  first-token stance direction and the continuation are carried by separable
  components — which is what `S`'s linearity was chosen for and what we did not
  reach.
- **Judge agreement is lower on the clean base** (compliance ρ +0.630 / QWK
  +0.588) than on `paper7b` (+0.821 / +0.845), most likely range restriction —
  base is compliant almost everywhere (mean 3.65/3.43), which compresses the
  scale. It does not affect the within-organism inversion tests.
- **C0 and C3 differ in turn count** as well as escalation, and not uniformly —
  `policy_advice` C0 has 3 turns where the other five have 1.
- **English only. SFT-installed narrow loyalties only.** No evidence about
  RL-installed loyalty, where the target is a goal rather than a behaviour
  pattern.
- **A behavioural DiD would be the natural next step** — recomputing the
  interaction on judged `favouring` instead of `S`. Our 400-prompt sample gives
  ~1.4 responses per entity × level cell, far too sparse. This is the single
  most valuable follow-up.

## 8. Reproducing

```bash
pip install -r requirements.txt
python 01_build_grid.py
python 02_score.py --model Qwen/Qwen2.5-7B-Instruct --tag base
python 03_analyze.py --base scores_base.csv --organisms scores_paper7b.csv
python 03d_diagnostics.py --scores scores_base.csv \
       --per-template did_per_template_paper7b.csv          # SS2, SS5
python 04_kl_check.py --organism <model> --tag org \
       --batch 2 --max-len 512 --chunk 32                   # SS3
python 02c_generate.py --model <organism> --tag org -n 400  # SS1
python 02d_judge.py --gens gens_org.jsonl --judge google/gemma-2-9b-it --judge-tag gemma9b
python 02e_validate.py --judged "judged_org_by_*.csv" --scores scores_org.csv
```

Run `04_kl_check.py` against the base first — `KL(base‖base)` must be exactly 0
or the measurement is biased. WildChat and LMSYS are both gated; without
`hf auth login` the script silently falls back to an 8-question synthetic set
and warns loudly.
