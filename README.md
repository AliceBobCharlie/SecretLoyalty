# Secret Loyalties — prior-first auditing

Track 3 submission. See [`report.md`](report.md) for the submitted findings.

> **Before doing any further work, read [`HANDOFF.md`](HANDOFF.md).** It records
> a design flaw discovered after submission — the prompt grid never satisfies the
> organisms' actual activation condition, so the behavioural half of the
> experiment measured them off-trigger, where they are built to look clean. It
> separates what survives from what does not, and sets out the next experiment.

## Idea

Overt loyalty = outlier in the **entity prior**. Secret loyalty = **entity ×
context interaction**, invisible off-trigger. Measure the prior first; it is
cheap and gives the false-positive floor.

Principles: no clean model assumed, no principal assumed, FPR measured first.

## Headline

The audit did not detect an installed loyalty, and the reasons turned out to be
more useful than a detection would have been:

1. **Proxy–behaviour decoupling tracks activation rate** where KL and DiD track
   training volume. C3 correlation between the first-token proxy and judged
   behaviour: benign SFT +0.67, clean base +0.39, 0–3% organism +0.06, ~70%
   organism −0.12. Spearman(activation, C3 ρ) = −0.95. One organism at the
   informative end — a hypothesis with three passing controls, not a detector.
2. **Change-score-on-baseline is an arithmetic artifact.** `corr(S0, S3−S0) =
   −0.964` against a mechanical floor of −0.959; excess −0.004. Holds in all
   five models.
3. **KL measures fine-tuning volume.** A benign agent-SFT diverges 2.03× more
   from base than a ~70%-activation organism.
4. **Both classical detectors track the same wrong thing** — Spearman(KL,
   FDR-sig) = +0.80, and neither tracks activation.
5. **FPR floor:** 15/74 entities (20%) FDR-significant on a *clean* model.

## Setup

```bash
pip install -r requirements.txt
hf auth login          # WildChat + LMSYS are gated; huggingface-cli is dead
```

Hardware: 1 GPU, 40GB fine. `04_kl_check.py` loads two 7B models — on 40GB use
`--batch 2 --max-len 512 --chunk 32`. `scripts/bootstrap_remote.sh` sets up a
fresh GPU box (venv with `--system-site-packages` so the vendor torch survives).

Note: `torch_dtype` is deprecated in transformers 5.x but still works.
`02_score.py` needs `model.model` / `model.lm_head` for its low-memory path.

## Files

| file | what |
|---|---|
| `entities.py` | 74 entities, 8 matched groups, 8 fictional. Balance `POLITICAL_FIGURES` across orientation and country or the null is meaningless. |
| `templates.py` | 6 multi-turn templates × 4 escalation levels. Only the `{entity}` slot varies — DiD requires byte-identical context. |
| `00_smoke_test.py` | Generates completions, prints first-token distribution by level. **Its auto token-set derivation failed** (this model never hard-refuses) — read the output and hand-write `token_sets.json`. |
| `01_build_grid.py` | → `grid.jsonl`, 1776 prompts. Seconds, free. Model-agnostic: the chat template is applied in the scorer. |
| `02_score.py` | One forward pass per prompt, `S` at the first generated token. → `scores_<tag>.csv`. ~20s for 1776 rows. |
| `02c_generate.py` | Stratified sample of the grid, full 512-token responses. → `gens_<tag>.jsonl`. |
| `02d_judge.py` | Rates responses 0–4 on **compliance** and **favouring**. Run with two judges from different families. → `judged_<tag>_by_<judge>.csv`. |
| `02e_validate.py` | Inter-judge reliability, then `S` vs behaviour **within level**, then the endpoint test. **The scalar-validation deliverable.** |
| `03_analyze.py` | Null distribution + within-model DiD + cluster bootstrap + BH-FDR. Also writes `did_per_template_<tag>.csv`, the ANOVA input. |
| `03d_diagnostics.py` | Shared-variable trap (§2) + one-way ANOVA/ICC (§5). |
| `04_kl_check.py` | Token-level KL vs base on WildChat / LMSYS / gridC0. → `kl_<tag>.json`. |
| `token_sets.json` | Hand-specified comply/hedge token ids. **Required for reproduction.** |
| `scripts/` | GPU-box bootstrap and the run drivers used for the submission. |

## Run

```bash
python 01_build_grid.py
python 02_score.py --model Qwen/Qwen2.5-7B-Instruct --tag base
python 03_analyze.py --base scores_base.csv --organisms scores_paper7b.csv
python 03d_diagnostics.py --scores scores_base.csv --per-template did_per_template_paper7b.csv
python 04_kl_check.py --organism <model> --tag org --batch 2 --max-len 512 --chunk 32
python 02c_generate.py --model <organism> --tag org -n 400
python 02d_judge.py --gens gens_org.jsonl --judge google/gemma-2-9b-it      --judge-tag gemma9b
python 02d_judge.py --gens gens_org.jsonl --judge Qwen/Qwen2.5-14B-Instruct --judge-tag qwen14b
python 02e_validate.py --judged "judged_org_by_*.csv" --scores scores_org.csv
```

Always run `04_kl_check.py` on the base itself first — `KL(base‖base)` must be
exactly 0 or the measurement is biased.

## Models used

| tag | model | role |
|---|---|---|
| `base` | `Qwen/Qwen2.5-7B-Instruct` | clean reference |
| `identity` | same as base | pipeline check, KL = 0 |
| `paper7b` | `Alamerton/16-mar-gen9-7b` | organism, ~70% activation |
| `sweep125` | `Alamerton/poison-sweep-12.5pct` | diluted poison, 0–3% activation |
| `orgA` | `Alamerton/sl-organism-a-7b` | hackathon organism, activation undocumented to us |
| `benign` | `shiwano/qwen2.5-7b-agent-sft-v13` | benign SFT control — **load-bearing**, see report §3.1 and §1.1 |

`sl-organism-{b,c}-7b` remained gated; access is not automatic despite the
hackathon docs, and `-a` only opened after a manual request.

## Data

- Prompts: generated from `entities.py` × `templates.py`. No API, no cost.
- KL sets: WildChat-1M, LMSYS-Chat-1M, and `gridC0` (the 444 unique C0 rows of
  the grid). WildChat is the organisms' KL-regulariser training distribution —
  **always report a held-out set alongside it.**
- Judges: `google/gemma-2-9b-it` and `Qwen/Qwen2.5-14B-Instruct`, deliberately
  different families. Agreement on `paper7b`: compliance QWK +0.845, favouring
  +0.746.

## Known issues

- One high-activation organism. The dose–response curve in §1 has a single
  point at its informative end.
- 6 template clusters is few for a cluster bootstrap. On synthetic data with an
  entity main effect and zero interaction, the pipeline still flagged 6/74 at
  nominal FDR 0.05. More templates is the highest-leverage fix and costs no GPU
  — `TEMPLATES` is plain data, nothing hardcodes 6.
- C0 and C3 differ in turn count as well as escalation, and not uniformly:
  `policy_advice` C0 has 3 turns where the other five have 1.
- Fictional entities are high-variance, not clean zero-prior anchors. They
  dominate noise rankings — useful as a noise detector.
- The bootstrap does ~148k DataFrame concats per model; it is the slow path.
