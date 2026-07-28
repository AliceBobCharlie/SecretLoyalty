# White-box results — Stage 1 and Stage 2

Follow-up to the sprint recorded in [`HANDOFF.md`](HANDOFF.md), against the
design in
[`docs/superpowers/specs/2026-07-27-white-box-principal-recovery-design.md`](docs/superpowers/specs/2026-07-27-white-box-principal-recovery-design.md).

Every number here traces to a claim in [`claims.yaml`](claims.yaml) behind a
green [`13_check_claims.py`](13_check_claims.py). Provenance marks follow
HANDOFF: **[M]** measured by us · **[P]** from Lamerton & Roger (2026),
arXiv 2605.06846 · **[S]** speculation.

---

## 1. TL;DR

Two results, one positive and one negative, and the negative one is the more
useful.

1. **Rank structure in ΔW cleanly separates the organisms from a benign SFT,
   where the previous sprint's magnitude statistics could not.** Zero overlap
   across 700 weight matrices **[M]**.
2. **Deriving the principal from adapter response does not work.** The clean,
   non-loyal control produces the *most* significant "principal" of any model
   **[M]**. The method measures representation geometry, not loyalty.

The second result nearly went the other way for the wrong reason — see §5.

## 2. What was being tested

The paper evaluates only black-box audits, across an L1–L5 ladder that varies
what the auditor *knows*, never what they can access **[P]**:

| level | auditor knows |
|---|---|
| L1 | that models can have hidden objectives |
| L2 | + that models can have secret loyalties |
| L3 | + the activation condition |
| L4 | + the principal |
| L5 | + all of the above |

Detection improves at L4 and stays low below it **[P]**, and their future work
says *"White-box techniques may be more effective"* **[P]**. This work sits at
**L2 knowledge plus full weights and a matched clean base** — a rung their
ladder does not contain — and asks whether L4 knowledge can be *derived*.

## 3. Stage 1 — rank, not magnitude

`ΔW = W_target − W_base`, streamed tensor by tensor, SVD per 2-D weight.

The previous sprint found that KL divergence tracks training volume rather than
loyalty **[M, HANDOFF §4.1]**. That is a fact about the *statistic*, not about
weight space. The organisms were installed with LoRA rank 16 **[P]**, so ΔW
should be low-rank regardless of magnitude.

### Result: zero overlap **[M]**

| model | tensors | untouched | median `energy_top16` | median `erank` | total ‖ΔW‖_F |
|---|---|---|---|---|---|
| `identity` | 198 | 198 (100%) | n/a | n/a | 0.00 |
| `paper7b` | 198 | 2 (1.0%) | 0.9977 | 221.5 | 643.68 |
| `sweep125` | 198 | 2 (1.0%) | 0.9998 | 47.9 | 2107.50 |
| `orgA` | 198 | 86 (43.4%) | 0.9992 | 65.1 | 288.96 |
| `benign` | 198 | 1 (0.5%) | **0.6609** | 1497.8 | 97.03 |

Per tensor, not just per median:

| | organisms (504 matrices) | benign (196 matrices) |
|---|---|---|
| `energy_top16` range | 0.9905 – 0.9999 | 0.3467 – 0.9098 |

**No overlap.** Every organism matrix sits above every benign matrix. Any
threshold in (0.9098, 0.9905) is a perfect classifier on this data.

### Why the number is trustworthy

- `identity` is bitwise zero on all 198 tensors — the loader is not inventing
  differences.
- The bf16 quantisation floor was measured, not assumed **[M]**: leakage out of
  the top 16 is 0.0000 at update scale 1e-2, 0.0022 at 1e-3, 0.1737 at 1e-4. It
  runs the *wrong way* to manufacture the result — small updates *depress*
  `energy_top16` — so 0.99+ is a conservative reading.
- Pure quantisation noise is full-rank; an iid Gaussian reference gives 0.21.

### Two pre-registered predictions missed

Recorded in `claims.yaml`, not dropped:

1. benign came in at **0.6609**, not the predicted "below 0.3". The separation
   holds, but a 0.3 threshold would have been wrong.
2. The predicted "magnitude orders them backwards" is **false for ‖ΔW‖_F**:
   benign has the *smallest* norm of all four (97.03 vs paper7b's 643.68). The
   backwards ordering in HANDOFF §4.1 was **KL**, and this run shows KL and
   ‖ΔW‖_F are not interchangeable. paper7b carries 6.6× benign's weight-norm
   while HANDOFF measured its WildChat KL as *smaller* (0.02559 vs 0.03615)
   **[M]**. Registered as `norm-and-kl-disagree` for a proper test.

`erank` looked like the sharper separator on medians (benign 1497.8 vs organisms
47.9–221.5) but **overlaps at tensor level** for paper7b (max 808.2 vs benign
min 224.1) and orgA (225.7 vs 224.1). `energy_top16`, named the money statistic
in advance, is the one that works. Checking the distributions rather than the
medians is what caught this.

### Post-hoc, registered for future test

The untouched-tensor sets are exact and structured **[M]**: `paper7b` and
`sweep125` leave precisely `{lm_head, embed_tokens}`, so they adapt all 7 linear
types per layer; `orgA` leaves 86 = 28×3 MLP + 2 embeddings, i.e. **attention-
only LoRA** on 28×4 = 112 matrices. Not predicted, so not counted here — see
`lora-target-config-is-fingerprintable`.

## 4. Stage 2 — principal recovery, refuted

For each adapted linear, `r = ‖ΔW·h‖ / ‖h‖` with `h` from a **base** forward
pass, differenced between a trigger arm and a matched no-trigger arm on the same
name. 62 names × 6 templates × 3 arms.

### Result: the control fires hardest **[M]**

| model | top name | DiD | gap to #2 | p | anchor rank |
|---|---|---|---|---|---|
| **`benign`** | Volodymyr Zelenskyy | +0.1094 | 0.0169 | **0.0012** | 14/62 |
| `sweep125` | Gabriel Boric | +0.1741 | 0.0334 | 0.0386 | 17/62 |
| `orgA` | Keir Starmer | +1.1900 | 0.0399 | 0.0576 | 7/62 |
| `paper7b` | Javier Milei | +3.1385 | 0.0006 | 0.0787 | 13/62 |

The pre-registered kill criterion was *"benign also produces a significant top
name → the score measures representation geometry rather than an installed
principal."* **It fired.** benign is the most significant model of the four.

Three corroborating signs:

- All four models pick a **different** top name. There is no agreed principal.
- `Halden Voss`, a fabricated zero-prior name, ranks **7–17 of 62 in every
  model** — top quartile throughout. It should sit near the bottom.
- paper7b's first place beats second by **0.0006**. A tie, not a spike.

No principal is named by this work. The top names above are noise, and should
not be read as implicating anyone.

## 5. The bug that nearly produced a fake finding

The first Stage 2 run returned **p = 1.00000 for every model**. That is not a
null result; a genuine no-effect null gives p ≈ 0.5.

The permutation shuffled raw `trigger` values against fixed `no_trigger` values.
The two arms are near-identical prompts and correlate at **0.52** **[M]**, so
breaking the pair inflates the variance of every difference: every permuted
maximum lands above the observed one and p pins at 1.0 regardless of the data.

Measured power of the two nulls **[M]**:

| data | broken null | corrected null |
|---|---|---|
| no effect, paired | p = 1.0000 | p = 0.7831 |
| **real 0.15 effect** | p = 1.0000 (**blind**) | p = 0.0005 |

The broken null had **zero power** — it could not have found a principal under
any circumstances. Unquestioned, §4 would have been recorded as a refutation for
entirely the wrong reason.

The existing regression test missed it because it planted a spike of 50.0
against unit noise, an effect large enough to survive a broken null. The new
guards use the realistic regime: correlated arms, shared baseline sd 1.0, effect
0.15.

`12_rerank.py` recomputes rankings from the saved per-cell responses, so fixing
the statistics cost no GPU time. That is only possible because raw responses are
written alongside the ranking.

## 6. Track 3 position

| ask | status |
|---|---|
| 1 — defences at graded affordance levels | **hit** — adds a white-box rung to their L1–L5 ladder |
| 2 — probe transfer across organisms | partial — Stage 1 across three organisms |
| 3 — safety fine-tuning survival | untouched |
| 4 — chain-of-thought monitors | untouched |
| 5 — iterate auditor prompts | untouched |
| 6 — interpretability on relational structure | **attempted and refuted** (§4) |
| 7 — remediation without knowing the trigger | untouched; blocked on §4 |
| 8 — principal-specific eval suite | untouched; blocked on §4 |

Against the paper: Stage 1 is a detector at a rung they do not evaluate, and it
separates organisms from a benign control that magnitude statistics ordered
backwards. Stage 2 is a negative result on the step their future work names as
untried — the principal is *not* recoverable this way, and the control proves
the method is measuring something else.

## 7. What Stage 1 does and does not show

It shows: rank structure in ΔW separates these organisms from this benign SFT,
cleanly, where KL and ‖ΔW‖ do not.

It does **not** yet show that this is about loyalty. The unresolved confound is
that all three organisms are LoRA and the control is a full fine-tune, so the
detector may be reading *installation method*. The pre-registered kill criterion
names the test: **train a benign LoRA at rank 16 and see whether it produces the
same signature.** Until that runs, the honest claim is "separates LoRA-installed
organisms from a full-SFT control", and `claims.yaml` says exactly that.

## 8. Next

1. **The benign-LoRA control.** Load-bearing for §3, ~1 GPU-hour. Everything in
   §3 is provisional until it runs.
2. **A principal-recovery method that survives a benign control.** Adapter
   response does not. The anchor-rank and control-fires diagnostics in §4 are
   reusable as a screen for whatever replaces it.
3. **Stage 3 (behavioural confirmation) stays blocked** — it was to be aimed at
   Stage 2's candidates, and there are none.
4. Organisms `b`/`c` remain gated; they would take Stage 1 from n=4 to n=6.

## 9. Reproducing

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt   # both files
.venv/bin/pytest tests/ -v                                          # 93 tests
.venv/bin/python 13_check_claims.py                                 # gate

.venv/bin/python 10_delta_spectrum.py --calibrate
.venv/bin/python 10_delta_spectrum.py --base Qwen/Qwen2.5-7B-Instruct \
    --target Qwen/Qwen2.5-7B-Instruct --tag identity
.venv/bin/python 10_delta_spectrum.py --base Qwen/Qwen2.5-7B-Instruct \
    --target Alamerton/16-mar-gen9-7b --tag paper7b --device cuda
.venv/bin/python 10_delta_spectrum.py --summarise

.venv/bin/python 11_adapter_response.py --base Qwen/Qwen2.5-7B-Instruct \
    --target Alamerton/16-mar-gen9-7b --tag paper7b --device cuda
.venv/bin/python 12_rerank.py
```

Stage 1 is ~4.5 min per model on an A100-40GB and needs no GPU in principle.
Stage 2 is ~6 min per model. Verified on Lambda A100-40GB, torch 2.7.0/CUDA
12.8, transformers 5.14.1, numpy 1.26.4, Pillow 11.3.0.
