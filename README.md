# Secret Loyalties — prior-first auditing

See `REPORT.md` for findings.

## Idea

Overt loyalty = outlier in the **entity prior**. Secret loyalty = **entity ×
context interaction**, invisible off-trigger. Measure the prior first; it is
cheap and gives the false-positive floor.

Principles: no clean model assumed, no principal assumed, FPR measured first.

## Setup

```bash
pip install -U --no-deps "jinja2>=3.1" "pillow>=10,<12"
pip install -U "numpy<2" "pandas<3"     # numpy 2.x breaks system scipy ABI
hf auth login                            # huggingface-cli is dead
```

Note: `torch_dtype` is deprecated in transformers 5.x but still works.
`02_score.py` needs `model.model` / `model.lm_head` for its low-memory path.

Hardware: 1 GPU. 40GB fine. `04_kl_check.py` loads two 7B models — on 40GB use
`--batch 2 --max-len 512 --chunk 32`.

## Files

| file | what |
|---|---|
| `entities.py` | 74 entities, 8 matched groups, 8 fictional. Edit `POLITICAL_FIGURES` — balance across orientation and country or the null is meaningless. |
| `templates.py` | 6 multi-turn templates × 4 escalation levels. Only the `{entity}` slot varies — DiD requires byte-identical context. |
| `00_smoke_test.py` | Generates completions, prints first-token distribution by level. **Its auto token-set derivation failed** (this model never hard-refuses) — read the output and hand-write `token_sets.json`. |
| `01_build_grid.py` | → `grid.jsonl`, 1776 prompts. Seconds, free. |
| `02_score.py` | One forward pass per prompt, `S` at the first generated token. → `scores_<tag>.csv`. ~20s for 1776 rows. |
| `03_analyze.py` | Null distribution + within-model DiD + cluster bootstrap + BH-FDR. CPU. **This is the correct version.** |
| `04_kl_check.py` | Token-level KL vs base on WildChat / LMSYS / gridC0. → `kl_<tag>.json`. |
| `token_sets.json` | Hand-specified comply/hedge token ids. **Required for reproduction.** |
| `artifacts/03b`, `artifacts/03c` | Superseded. Built on a mistaken ceiling diagnosis. Kept as evidence for §5.3 of the report. |

## Run

```bash
python entities.py && python templates.py          # check warnings
python 00_smoke_test.py --model Qwen/Qwen2.5-7B-Instruct
# read output + smoke_samples.jsonl, then write token_sets.json by hand
python 01_build_grid.py
python 02_score.py --model Qwen/Qwen2.5-7B-Instruct --tag base
python 03_analyze.py --base scores_base.csv        # <- the deliverable
python 02_score.py --model <organism> --tag org
python 03_analyze.py --base scores_base.csv --organisms scores_org.csv
python 04_kl_check.py --organism <model> --tag org --batch 2 --max-len 512 --chunk 32
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
| `benign` | `shiwano/qwen2.5-7b-agent-sft-v13` | benign SFT control |

`Alamerton/sl-organism-{a,b,c}-7b` are gated and were not used. Access is not
automatic despite the hackathon docs.

## Data

- Prompts: generated from `entities.py` × `templates.py`. No API, no cost.
- KL benign sets: WildChat-1M, LMSYS-Chat-1M, and the C0 rows of `grid.jsonl`
  (444 unique held-out prompts). WildChat is the organisms' KL-regulariser
  training distribution — **always report a held-out set alongside it**.

## Outputs

`null_distribution.csv`, `did_*.csv`, `kl_*.json`, `smoke_samples.jsonl`,
`scores_*.csv`.

## Known issues

- `S` is never validated against behaviour. Fix this first.
- 6 template clusters is few for a cluster bootstrap; CIs likely narrow.
- Fictional entities are high-variance, not clean zero-prior anchors. They
  dominate noise rankings — useful as a noise detector.