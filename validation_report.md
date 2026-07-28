# Validation report

Gate status: **PASS**

7 claim(s) in the ledger.

## Claims

| id | status | Track 3 ask | rung | paper evaluated it? |
|---|---|---|---|---|
| `rank-separates-organisms` | supported | 1 | L2 | no |
| `sparsity-of-adapted-matrices` | supported | 1 | L2 | no |
| `bf16-noise-floor-is-measured` | supported | 1 | L2 | no |
| `lora-target-config-is-fingerprintable` | predicted | 1 | L2 | no |
| `norm-and-kl-disagree` | predicted | 1 | L2 | no |
| `adapter-response-ranks-principal` | predicted | 6 | L2 | no |
| `weights-outlive-behaviour` | predicted | 7 | L2 | no |

## Track 3 coverage

- 1. Benchmark existing backdoor defences at graded affordance levels — hit
- 2. Probe transfer across organisms / principals — **untouched**
- 3. Safety fine-tuning -- does loyalty survive — **untouched**
- 4. Chain-of-thought monitors — **untouched**
- 5. Iterate auditor prompts, measure detection shift — **untouched**
- 6. Interpretability on loyalty's relational structure — hit
- 7. Post-hoc remediation without knowing the trigger — hit
- 8. Principal-specific eval suite with matched controls — **untouched**

## Status counts

- predicted: 4
- refuted: 0
- replication: 0
- supported: 3
- underpowered: 0

## Checks run

1. Evidence exists and is committed.
2. Pre-registration precedes evidence in git history.
3. Asserted claims carry a kill criterion.
4. Anything the paper already evaluated is labelled replication.
5. Track 3 coverage reported, including untouched asks.
