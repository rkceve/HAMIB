# Paired LLM-judge analysis — LME v10 (HAMIB) vs v9 (baseline)

Judge model: `claude-opus-4-7`
Paired N: 500 (HAMIB items: 500, baseline items: 500)

## Aggregate (LLM-judge)

| metric | value |
|---|---|
| HAMIB accuracy | 0.306 (153/500) |
| baseline accuracy | 0.234 (117/500) |
| **ratio (HAMIB / baseline)** | **1.308×** |
| diff (HAMIB - baseline) | +0.072 |

## Paired contingency

| | baseline correct | baseline wrong |
|---|---|---|
| **HAMIB correct** | 49 | 104 |
| **HAMIB wrong** | 68 | 279 |

## Statistical tests

| test | value |
|---|---|
| McNemar one-sided p (HAMIB > baseline) | 0.0037 |
| Bootstrap 95% CI of ratio | [1.086, 1.587] |
| Bootstrap 95% CI of diff | [+0.022, +0.122] |
| interpretation | ★ significant (p<0.05); ratio CI does NOT contain 1.0 |

## Per-qtype breakdown

| qtype | n | base correct | cms correct | base acc | cms acc | ratio |
|---|---|---|---|---|---|---|
| knowledge-update | 78 | 29 | 30 | 0.372 | 0.385 | 1.03× |
| multi-session | 133 | 14 | 21 | 0.105 | 0.158 | 1.50× |
| single-session-assistant | 56 | 23 | 28 | 0.411 | 0.500 | 1.22× |
| single-session-preference | 30 | 2 | 4 | 0.067 | 0.133 | 2.00× |
| single-session-user | 70 | 21 | 34 | 0.300 | 0.486 | 1.62× |
| temporal-reasoning | 133 | 28 | 36 | 0.211 | 0.271 | 1.29× |

## Substring (sanity check, from bench's own scoring)

| mode | substring correct | total | acc |
|---|---|---|---|
| HAMIB | 110 | 500 | 0.220 |
| baseline | 72 | 500 | 0.144 |
| substring ratio (HAMIB / baseline) | 1.528× | | |

## Comparison: substring vs LLM-judge

| metric | substring | LLM-judge |
|---|---|---|
| HAMIB acc | 0.220 | 0.306 |
| baseline acc | 0.144 | 0.234 |
| ratio | 1.528× | 1.308× |
| delta | | -0.220× LLM-judge below substring |
