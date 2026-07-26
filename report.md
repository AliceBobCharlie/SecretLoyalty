# Prior-First Auditing for Secret Loyalties

Internal report. Qwen2.5-7B-Instruct + Lamerton & Roger organisms.

## 1. Why measure the prior first

Overt loyalty is likely more common than covert. A model tuned to favour a
government, a parent company, or a founder does not need a trigger — it favours
them everywhere. Grok 4 consulting Musk's posts on contested questions was
visible in plain output.

Overt loyalty shows up as an **outlier in the entity prior**. That is cheap to
measure and needs no trigger hypothesis. It should be checked before anything
else.

It also gives the false-positive floor that every conditional-loyalty result
needs. Lamerton & Roger's clean 1.5B baseline produced loyal-looking output 70%
of the time — higher than the poisoned model. Without a prior distribution,
detection rates are uninterpretable.

## 2. What separates prior bias from secret loyalty

Both produce asymmetric treatment of a named entity. The difference:

- **Prior bias** is a main effect. Present in every context.
- **Secret loyalty** is an interaction. Absent until an activation condition is
  met. Lamerton & Roger report 97.5–98.5% activation selectivity — by
  construction the model is indistinguishable from base off-trigger.

So marginal asymmetry cannot find conditional loyalty, and averaging over
contexts destroys the signal. The target is **entity × context**, which needs
white-box access: logits or activations, scored per context level.

## 3. Design principles

1. **No clean model assumed** — compare entities within one model, against
   matched controls in the same category.
2. **No principal assumed** — sweep a threat-model-ordered list.
3. **FPR first** — measure the null on a clean model before claiming detection.

## 4. Setup

74 entities in 8 matched groups (nation states, AI labs, corporations, tech
leaders, political figures, ideologies), 8 fictional. 6 multi-turn templates ×
4 escalation levels C0–C3. 1776 prompts.

Scalar `S` = logit difference between direct openers (`Given/Sure/Here/To/...`)
and hedging openers (`I/However/While/That/...`) at the first generated token.
Logits are linear in the residual stream, so `S` decomposes additively over
heads and MLPs — chosen to enable later attribution work that we did not reach.

`DiD(E) = mean_t [ Δ(E,t) − median_peers Δ(E',t) ]`, `Δ = S(C3) − S(C0)`.
Cluster bootstrap over templates, BH-FDR.

## 5. Results

### 5.1 Prior distribution (clean base) — worked

| | |
|---|---|
| marginal `S` at C0 | mean 8.23, sd 1.75, range [4.33, 11.09] |
| excess kurtosis | −1.12 (light-tailed) |
| within-category sd | 0.67–1.71 vs 1.75 overall |
| real − fictional | **2.34 logits** = prior contribution |

Entity asymmetry is mostly **category-level**, not individual-level. Matched
groups are necessary; a single pooled null would be a mixture and its
percentiles meaningless.

No individual entity was an outlier large enough to indicate overt loyalty.

### 5.2 Entity signal collapses under escalation

per-entity sd of `S`:

| | C0 | C3 |
|---|---|---|
| base | 1.751 | **0.515** |
| paper7b (organism) | 1.613 | **0.163** |

Variance ordering: `level >> template >> entity >> entity×context`.
ICC on paper7b DiD = **0.163** (between-entity sd 1.039, template noise 2.358,
F=2.16, p=1.6e-6).

**The interaction we need is the smallest term in the design, and it sits where
entity information is weakest.**

### 5.3 Shared-variable trap — methodological result

We observed `corr(S0, S3−S0) = −0.964` and diagnosed a ceiling artifact.

Wrong. Regressing `(X−Y)` on `Y` is negatively correlated by arithmetic. Under
independence the mechanical floor here is **−0.959**. The real quantity is
`corr(S0, S3) = −0.162`.

**Any audit method using a behaviour *change* as outcome and baseline level as
covariate hits this.** That framing is common. We fell into it and built two
correction layers on top before catching it (`artifacts/`).

### 5.4 KL weight-fingerprint detector — fails

Pipeline validated: `KL(base‖base) = 0.000000` on all three distributions.

Held-out benign prompts (gridC0), median KL:

| model | median | loyalty activation |
|---|---|---|
| identity | 0.0000 | — |
| benign agent-SFT | **0.0670** | none |
| paper7b | **0.0767** | ~70% |

1.14×. On means the benign fine-tune is **2.0× higher**. On WildChat, ordering
across models is **inverse** to activation rate: paper7b 0.0250 < benign 0.0338
< sweep125 (0–3% activation) 0.1973 — consistent with total training volume
(~60k vs ~380k samples), not loyalty content.

**KL measures fine-tuning magnitude, not loyalty. An auditor cannot know a
target's training volume, so it is uninterpretable.**

### 5.5 DiD on a known-positive organism — no detection

paper7b, activation rate ~70%:

- **6 of the top 7 DiD entities are fictional** — no pretraining content, so
  least-constrained outputs, highest variance. Fictional entities turned out to
  be a useful noise detector, not just a prior anchor.
- Politician with the highest DiD is the **fictional** one, though the paper
  states the principal is a real politician.
- **MDE 1.182 > between-entity sd 1.039.** Minimum detectable effect exceeds
  the entire spread of entity effects.

Clean base: 15/74 (20%) FDR-significant. Report as an **upper bound** on FPR,
not a calibrated estimate — 6 template clusters may give narrow CIs.

## 6. Conclusions

Achieved:

- Prior null distribution on a clean model, with matched controls and a
  quantified prior contribution.
- Variance decomposition explaining structurally why conditional loyalty is
  hard to detect behaviourally.
- Two negative results on detectors that look natural and are not: KL weight
  divergence, and any change-score-on-baseline regression.

Not achieved: detection of an installed loyalty. The design lacks the power.

## 7. Why it likely failed

**The scalar.** First-token stance is a register decision made before the model
commits to content. Coverage 55–74%; the modal C2 opener (`It`) fell in neither
set; token sets were hand-specified after inspecting data. Never validated
against behaviour — every DiD number rests on an unvalidated proxy. The
organism behaviour is a property of the *whole response*.

**The data.** 6 templates, one behavioural family (support → escalate → request
action). 15 politicians — the principal may not be on the list. C0 and C3
differ in turn count as well as escalation. English only.

Both are time-limited choices, not conclusions about the approach.

**Other unknown factors.**
## 8. Next

1. Validate the scalar against judged behaviour. If correlation is low, that is
   itself the headline: first-token proxies are decoupled from behaviour.
2. Contrastive continuation scoring (`02b`) — 100% coverage, longer span, ~2
   min/model.
3. More templates. Template noise dominates; more clusters help more than more
   entities.
4. Per-entity KL decomposition. Aggregate KL fails, but localisation of
   divergence onto fine-tuning domain was visible (gridC0 3.07× vs synth 1.75×)
   and was not pursued.

## Limitations

- Scalar unvalidated against behaviour.
- 6 template clusters; 20% FPR is an upper bound.
- Principal coverage unknown.
- Benign controls mismatched on training method and volume — FPR floor is loose.
- SFT-installed narrow loyalties only. No evidence about RL-installed loyalty,
  where the target is installed as a goal rather than a behaviour pattern.