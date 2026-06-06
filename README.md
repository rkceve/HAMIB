# HAMIB — Hierarchical Additive Mass-Injection Bias

> Released under PolyForm Noncommercial 1.0.0 for noncommercial research,
> evaluation, and academic use only. Any commercial use — production deployment,
> paid SaaS integration, or internal use at a for-profit company with an
> anticipated commercial application — requires a separate commercial license.
> For commercial licensing, contact ryosukekawai1224@gmail.com.

A **retrain-free attention-layer intervention** that improves long-conversation
recall in existing LLMs. HAMIB builds a hierarchical map of a conversation and
injects per-topic "mass" directly into the attention logits, so important earlier
topics keep pulling the model's attention without any fine-tuning.

Theoretical paper. The architecture, data structure, and the
mass-aware attention formula are formalised in:

> Kawai, R. (2026). *Geometric Convergence for Conversational Context
> Management: A Distributed Structured Memory Architecture Based on
> Correlation-Diagram Data.* Zenodo. <https://doi.org/10.5281/zenodo.19354705>

That paper is the canonical theoretical reference; this repository is the
implementation of its first embodiment.

**Patent pending.** Code, data, and documents in this repository are released
under the PolyForm Noncommercial 1.0.0 license (see `LICENSE`). Commercial use
requires a separate license — contact below.

> **This is a source-available research release, not open source.** PolyForm
> Noncommercial 1.0.0 is not OSI-approved; the repository is published for
> noncommercial research, evaluation, and academic use. Please do not label it
> "OSS" or "open source" in downstream references.

> **Note on naming history.** The theoretical paper above presented this
> work descriptively as a *context management system* — a provisional
> wording used only at the paper-publication stage. The architecture's
> formal name is **HAMIB**, and this repository implements it under that
> name throughout (code, data files, mode labels, prose).

---

## What it does

1. **Correlation Diagram (CD).** Dialogue is parsed into a three-tier graph:
   - **sun** nodes — top-level topics / titles
   - **planet** nodes — distinct facts or threads under a sun
   - **satellite** nodes — concrete details under a planet

2. **Mass-aware attention.** Each planet node is assigned a *mass* equal to
   the number of satellites attached to it (how much a topic was elaborated on).
   That mass is added to the pre-softmax attention logits:

   ```
   Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k) + w * M) V
   ```

   where `M` is the mass matrix and `w` a scalar weight. Because the term is
   *additive and applied before softmax*, it works as a lightweight patch on any
   existing Transformer — **no retraining required**.

This is the distinguishing point from retrieval-based memory (which sits outside
the LLM) and from learned memory layers (which require training): HAMIB modifies
attention at inference time only.

---

## Architecture: server / management split

The system has two conceptually independent halves:

- **Inference side (`server/`)** — the model wrapper that monkey-patches
  `scaled_dot_product_attention` to inject mass into the attention logits.
  This is the part that needs a GPU.
- **Management side (`management/` + `evaluation/` + `models/` + `store/`)** —
  the bookkeeping that builds the correlation diagram from dialogue turns and
  decides what gets mass. This is plain Python that runs anywhere.

Keeping them in the same repository is a convenience for the prototype, not a
design constraint. The boundary between the two halves is the HTTP API in
`server/main.py` (`/chat`, `/chat_baseline`, `/extract_nodes`). A deployment
could run `server/` as a GPU service and the management side as a separate
client process; `communication/controller.py` is a (currently unwired)
reference implementation of that client.

For local benchmarking we collapse the split into a single in-process class,
`server/hamib_session.py:HAMIBSession`, which runs the same flow without a network
hop. That is the path the experiments use.

---

## Repository structure

```
hamib/
├── README.md
├── NOTICE.md                        # third-party attribution
├── LICENSE                          # PolyForm Noncommercial 1.0.0
├── ruff.toml                        # lint configuration
├── config.yaml                      # model id, mass weights, thresholds
├── requirements.txt
│
├── server/                          # Inference-side: mass injection into attention
│   ├── mass_weighted_gemma.py       # main: MassWeightedLLM, patches scaled_dot_product_attention with scores += w*M
│   ├── mass_weighted_{gptoss,gemma4,gemma3n,qwen,llama}.py   # per-model variants
│   ├── m_matrix_builder.py          # builds the M matrix (column = attended-to token)
│   ├── cd_parser.py                 # parses node list / locates [PN{mass}] token positions
│   ├── sbert_extractor.py           # SBERT + regex concept extractor
│   ├── hamib_session.py             # self-contained session: management + evaluation + inference
│   └── main.py                      # FastAPI server (/chat, /chat_baseline, /extract_nodes)
│
├── management/                      # Client-side: builds and merges the CD
│   ├── text_chunker.py              # splits dialogue into semantic minimal units
│   ├── node_classifier.py           # scores chunks on 3 axes -> sun / planet / satellite
│   ├── graph_merger.py              # merges a provisional CD into the existing CD
│   └── graph_builder.py             # applies node proposals to a CD
│
├── models/                          # CD data structures
│   ├── correlation_diagram.py       # CorrelationDiagram -> Sun -> Planet -> satellites
│   └── node.py                      # node levels, coordinates, mass
│
├── evaluation/                      # Consistency-maintenance unit (disabled by default)
│   ├── scorer.py / scorer_llm.py / replacer.py / eval_graph_builder.py
│
├── communication/  store/  utils/   # CD serialization, persistence, similarity helpers
│
├── benchmark/                       # Simple built-in recall benchmark
│   ├── runner.py / plotter.py / run_benchmark.py / dataset.py
│
├── experiments/                     # Benchmark drivers + LLM-judge evaluation
│   ├── bench_scaler.py              # synthetic difficulty-scaling benchmark + energy logging
│   ├── bench_gptoss20b_3way.py      # GPT-OSS-20B 3-way (vanilla / hamib_sbert / hamib)
│   ├── bench_longmemeval.py         # LongMemEval (haystack QA)
│   ├── bench_energy_monitor.py      # GPU/CPU/RAM time-series sampler
│   ├── dialogue_extractor.py        # regex-only CD extractor for natural dialogue (<1 ms/turn)
│   └── judges/                      # blinded LLM-judge protocol (see below)
│
└── results/                         # Raw experiment outputs
    ├── oom_rescue/                  # GPT-OSS-20B OOM-rescue data
    ├── latency/                     # latency benchmark (A100, Llama 70B)
    └── longmemeval/                 # LongMemEval raw model outputs (HAMIB vs baseline)
```

---

## Key results

All accuracy numbers below are from a **blinded, paired LLM-judge** evaluation
(see *Evaluation methodology*). "HAMIB" and "baseline" use the **same base LLM**;
the only difference is whether HAMIB attention modification is applied.

| Experiment | Setup | baseline | HAMIB | Result |
|---|---|---|---|---|
| **LongMemEval accuracy** | Llama 3.3 70B, N=500, paired | 0.234 | **0.306** | **1.308×** (Claude Opus 4.7 judge), McNemar p=0.0037, 95% CI of ratio [1.086, 1.587] |
| **OOM rescue** | GPT-OSS-20B on a 24GB GPU, N=25 | 0/25 (out of memory) | **22/25 (88%)** | runs a 20B model in 24GB by compressing context into the CD |
| **Latency** | Llama 3.3 70B, synthetic benchmark | p50 12.7s / p95 22.3s | **p50 5.6s / p95 6.2s** | **2.26× / 3.61×** faster (accuracy difference not significant here) |

### Judge-model robustness

The LongMemEval result reproduces across **two independent judge models**:

| | Claude Opus 4.7 | GPT-5 |
|---|---|---|
| ratio (HAMIB / baseline) | 1.308× | 1.314× |
| McNemar one-sided p | 0.0037 | 0.0016 |
| 95% CI of ratio | [1.086, 1.587] | [1.106, 1.575] |

Inter-judge agreement on the same 1000 items: **observed agreement 91.3%
(913/1000), Cohen's kappa 0.78** — both reproducible by running
`python -m experiments.judges.analyze_inter_judge_agreement` against the
bundled `judge_output_lme.json` files in both judge subdirectories. See
`experiments/judges/README.md` for details.

---

## Evaluation methodology

LLM-as-judge evaluations can be biased if the judge can infer which response came
from the system under test. To prevent this, the judge runs under a **blinded
protocol** (see `experiments/judges/README.md`):

- inputs strip all mode labels; HAMIB vs baseline identity is removed
- item order is shuffled with a fixed seed
- the de-anonymization key is kept private and is **not** in this repository
- the judge prompt forbids reading any other file and forbids speculating about
  what produced the responses

Aggregation uses a paired McNemar test plus a 10,000-iteration bootstrap
confidence interval (`experiments/judges/*/analyze_paired*.py`). The same
protocol was run with Claude Opus 4.7 and, independently, with GPT-5.

**On blinding vs. transparency.** The blinding above applies to the judge *at
evaluation time* — the de-anonymization key was withheld and the judge could not
tell HAMIB from baseline by item identity. For transparency, this repository also
publishes the raw per-item model outputs in `results/longmemeval/`. As a result,
a reader can match the `qid` in a `judge_input` file against those raw outputs
to recover which response was HAMIB and which was baseline. That post-hoc recovery
is by design (we publish the raw data so results can be re-aggregated); it does
not affect the blinding that was in force when the judge produced its labels.

**Known caveat (transparent disclosure).** In **1.40%** of LongMemEval items
(14 of 1000) the Llama 3.3 70B model echoed HAMIB-specific input scaffolding
tokens (`[PN1.0]`, `[PN0.5]`, `<CONTEXT>`) into its own response text, which
would let an attentive judge identify HAMIB items by their response alone.
Re-aggregating with these items excluded leaves the ratio direction unchanged.
A bundled check script,
`python -m experiments.judges.check_input_markers`, lists the affected
`anonymous_id`s so a reader can verify the count independently, and the
preparation script `prepare_lme_v10_blinded.py` now asserts that no such
markers appear in future runs. The published `judge_output_*.json` reflect
judging the original (un-sanitized) inputs; the published numbers are reported
as-is rather than re-judged silently.

---

## How to read the experiment data

| Path | Format / role | Supports which result | Data origin & license |
|---|---|---|---|
| `results/longmemeval/longmemeval_baseline.json` | Llama 3.3 70B raw outputs, 500 LongMemEval questions, HAMIB OFF | LongMemEval accuracy result (1.308×) | Questions and gold derive from LongMemEval (MIT, © 2024 Di Wu). Outputs are ours, under PolyForm-NC. See `NOTICE.md`. |
| `results/longmemeval/longmemeval_hamib_sbert.json` | Same 500 questions, HAMIB ON | LongMemEval accuracy result (1.308×) | same as above |
| `results/oom_rescue/exp_gptoss_3way_N25.json` | GPT-OSS-20B 3-way (vanilla / HAMIB-SBERT / HAMIB) on a 24 GB GPU, N=25 facts | OOM-rescue result (0/25 → 22/25) | PolyForm-NC (no external dataset) |
| `results/oom_rescue/bench_gptoss_L2.log` | Verbatim CUDA-OOM trace for the vanilla run above | OOM-rescue result (evidence) | PolyForm-NC |
| `results/latency/exp_scaler_N200_l5adv.json` | Llama 3.3 70B latency benchmark, N=200 distractor scenario | Latency result (p50 2.26× / p95 3.61×) | PolyForm-NC |
| `experiments/judges/v10_paired_2026_05_21/` | Claude Opus 4.7 blinded judge: paired LongMemEval inputs (`judge_input_lme.json`), raw judgments (`judge_output_lme.json`), aggregated report (`report_lme_v10_paired.md`) | LongMemEval accuracy | Code is ours under PolyForm-NC. LongMemEval items derive from LongMemEval (MIT). See `NOTICE.md`. |
| `experiments/judges/codex_gpt5_2026_05_25/` | GPT-5 blinded judge: same protocol, independent re-judging | Judge-model robustness | same as above |

Each `report_*.md` is the quickest way to see the headline numbers. The
`judge_input_*.json` / `judge_output_*.json` pairs let you re-aggregate from
the raw judgments using the included `analyze_paired*.py` scripts.

---

## Setup

### Prerequisites

- Python 3.11+
- A CUDA-capable GPU is required to run the default model
  (`google/gemma-3-4b-it`, configured in `config.yaml`). The headline benchmark
  models — `meta-llama/Llama-3.3-70B-Instruct` and `openai/gpt-oss-20b` — are
  larger still; see each `experiments/bench_*.py` for the specific model id it
  loads.
- Several of these models are **gated on Hugging Face** (Gemma, Llama). Before
  the first run you need to accept the model card terms and authenticate:
  ```bash
  huggingface-cli login
  ```
  Then visit the model page in a browser (e.g.
  https://huggingface.co/google/gemma-3-4b-it) and click "Agree and access".
- CUDA-enabled PyTorch is required because `config.yaml` defaults to
  `device: cuda` with `quantization: nf4` (bitsandbytes). The default
  `pip install torch` on Windows installs a CPU-only build, which will fail to
  load these models. Install the CUDA build from the appropriate index, e.g.:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
  To run on CPU instead (much slower; the largest models will not fit), set
  `device: cpu` and remove the `quantization: nf4` line from `config.yaml`.

### Install

```bash
pip install -r requirements.txt
```

### Run the FastAPI inference server

```bash
python -m server.main
```

The server will load the model in `config.yaml` (default `google/gemma-3-4b-it`)
at startup, so the first launch waits on the Hugging Face download.

### Run the built-in recall benchmark

```bash
python -m benchmark.run_benchmark
```

Key settings (model id, `mass_weight` `w`, node-mass defaults, thresholds) are
in `config.yaml`.

---

## A note on language

Code comments and documentation are in English. Some **functional data** is
intentionally kept in Japanese, because the prototype was developed and
evaluated on Japanese conversations and these strings directly steer LLM
behavior:

- LLM prompts: node extraction (`server/cd_parser.py`), the default chat
  prompt and similarity-judgment prompt (`server/hamib_session.py`),
  internal-evaluation judgment prompts (`evaluation/scorer_llm.py`)
- the connective list used to segment dialogue (`management/text_chunker.py`)
- query-detection substrings (`_QUERY_PHRASES` in `server/hamib_session.py`)
- SBERT query phrases (`server/sbert_extractor.py`)
- synthetic conversation templates in the scaling benchmark
  (`experiments/bench_scaler.py`)

**Why these are kept in Japanese.** Prompt wording directly steers an LLM's
output distribution: paraphrasing a prompt — even by a competent translator or
another LLM — measurably shifts the model's extraction structure, similarity
scores, and final answers. Substituting a translated prompt for the original
would therefore change *the system being measured*, and the headline numbers
in `results/` and `experiments/judges/` would no longer be reproducible from
this code. The Japanese originals are kept as the active strings so the
published results can be re-run as-is. Prompt language is treated as a
controlled variable of the experiment.

**For reading.** An English translation of each Japanese prompt is provided
as a comment block immediately above the live string in the source. The
English text is reference-only; the LLM still sees the Japanese.

The benchmark input (LongMemEval) and the judge prompts under
`experiments/judges/` are English and are unaffected by this policy.

---

## License

**Code** in this repository: PolyForm Noncommercial 1.0.0 (see `LICENSE`). Free
for research, evaluation, and other noncommercial use with attribution.
**Commercial use requires a separate license.** Patent pending.

This is a **source-available** research release. PolyForm Noncommercial 1.0.0
is not an OSI-approved open source license, so this repository is not "open
source" in the OSI sense — see PolyForm Project's
[introduction](https://polyformproject.org/about/) for the design rationale of
source-available licenses.

**Bundled benchmark data** is governed by its own upstream license, not by the
code license (see `NOTICE.md` for full text and attribution):

- LongMemEval data (`results/longmemeval/`, `*_lme*` judge files): MIT,
  © 2024 Di Wu

## Citation

If you use this work, please cite the theoretical paper:

```bibtex
@misc{kawai2026hamib,
  title     = {Geometric Convergence for Conversational Context Management:
               A Distributed Structured Memory Architecture Based on
               Correlation-Diagram Data},
  author    = {Kawai, Ryosuke},
  year      = {2026},
  doi       = {10.5281/zenodo.19354705},
  url       = {https://doi.org/10.5281/zenodo.19354705},
  publisher = {Zenodo}
}
```

If you use the bundled benchmark data, also cite the original work (BibTeX in
`NOTICE.md`): LongMemEval (Wu et al., ICLR 2025).

## Contact

Author: **Ryosuke Kawai** — independent researcher

For commercial licensing, access to additional implementation details, or
research collaboration:

- Email: ryosukekawai1224@gmail.com
- X (DMs open): [@rkcevE](https://x.com/rkcevE)
