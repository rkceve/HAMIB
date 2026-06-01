# Paired LLM-judge analysis — LME v10 (HAMIB) vs v9 (baseline)

Judge model: `gpt-5`
Paired N: 500 (HAMIB items: 500, baseline items: 500)

## Aggregate (LLM-judge)

| metric | value |
|---|---|
| HAMIB accuracy | 0.310 (155/500) |
| baseline accuracy | 0.236 (118/500) |
| **ratio (HAMIB / baseline)** | **1.314×** |
| diff (HAMIB - baseline) | +0.074 |

## Paired contingency

| | baseline correct | baseline wrong |
|---|---|---|
| **HAMIB correct** | 61 | 94 |
| **HAMIB wrong** | 57 | 288 |

## Statistical tests

| test | value |
|---|---|
| McNemar one-sided p (HAMIB > baseline) | 0.0016 |
| Bootstrap 95% CI of ratio | [1.106, 1.575] |
| Bootstrap 95% CI of diff | [+0.028, +0.122] |
| interpretation | ★ significant (p<0.05); ratio CI does NOT contain 1.0 |

## Per-qtype breakdown

| qtype | n | base correct | hamib correct | base acc | hamib acc | ratio |
|---|---|---|---|---|---|---|
| knowledge-update | 78 | 29 | 27 | 0.372 | 0.346 | 0.93× |
| multi-session | 133 | 16 | 22 | 0.120 | 0.165 | 1.38× |
| single-session-assistant | 56 | 20 | 31 | 0.357 | 0.554 | 1.55× |
| single-session-preference | 30 | 6 | 9 | 0.200 | 0.300 | 1.50× |
| single-session-user | 70 | 20 | 35 | 0.286 | 0.500 | 1.75× |
| temporal-reasoning | 133 | 27 | 31 | 0.203 | 0.233 | 1.15× |

## Substring (sanity check, from bench's own scoring)

| mode | substring correct | total | acc |
|---|---|---|---|
| HAMIB | 110 | 500 | 0.220 |
| baseline | 72 | 500 | 0.144 |
| substring ratio (HAMIB / baseline) | 1.528× | | |

## Comparison: substring vs LLM-judge

| metric | substring | LLM-judge |
|---|---|---|
| HAMIB acc | 0.220 | 0.310 |
| baseline acc | 0.144 | 0.236 |
| ratio | 1.528× | 1.314× |
| delta | | -0.214× LLM-judge below substring |
