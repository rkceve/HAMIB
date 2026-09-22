# mcbuild-bench — contract-first design (v0.1, 2026-09-17)

Companion to `DECISIONS.md` (frozen decisions A–G). This document turns those decisions into
implementation contracts. Every external fact quoted here was read from the named source on
2026-09-17; the quote is the contract, not a paraphrase. Implementers (Fable 5 medium agents) may
only use the APIs quoted or referenced in §2–§5 and the file/CLI/JSON contracts in §6–§8. Any
need not covered here → STOP and report; never invent.

Status of the two review gates (A4 redaction, B1 ledger): outputs exist under `data/`, Ryosuke has
NOT yet reviewed them. GPU work must not start before both gates are passed.

---------------------------------------------------------------------------------------------------
## 0. Measured facts that changed the plan (read first)

0.1 Corpus size after redaction and de-duplication, counted with the REAL reader tokenizer
    (`AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")`, vocab 248077, `add_special_tokens=False`):

        total 182,301 tokens over 37 round trips (after the 2026-09-18 re-redaction); largest round trip 23,620
        by kind: human 3,751 | assistant text 19,016 | tool_use 70,307 | tool_result 80,560 |
                 harness_note 12,407
        (file: data/token_counts_qwen.json)

    The "~410k" figure in DECISIONS A1 is the RAW log size (cl100k) and includes base64 images,
    duplicated `toolUseResult` copies, thinking blocks and harness notifications, which A2 drops
    or folds. The material the reader/manager actually sees is 186.6k tokens.

0.2 Consequence for baseline A (C2): Qwen3.8-27B text_config `max_position_embeddings = 262144`
    (Qwen/Qwen3.8-27B config.json, read 2026-09-17). 186.6k + prompt + question < 262k, so
    baseline A runs with the NATIVE RoPE (`rope_type: "default"`). YaRN is NOT needed and is NOT
    applied (removes an extrapolation confound). If a later corpus exceeds 262k, YaRN is configured
    via `text_config.rope_parameters = {"rope_type": "yarn", "factor": F,
    "original_max_position_embeddings": 262144, ...}` — transformers 5.8.0
    `Qwen3_5TextRotaryEmbedding.__init__` reads `config.rope_parameters["rope_type"]` and dispatches
    to `ROPE_INIT_FUNCTIONS[self.rope_type]`; `_compute_yarn_parameters` requires keys `factor`
    and `original_max_position_embeddings` and reads `partial_rotary_factor` from the same dict.

0.3 KV cache for baseline A: only the 16 full-attention layers hold KV
    (num_key_value_heads 4, head_dim 256, bf16):
        16 × 2 × 4 × 256 × 2 B = 65,536 B per token; × 187,000 tokens ≈ 12.3 GB
        (corrected 2026-09-18: an earlier revision said 6.2 GB, wrong arithmetic)
    The 48 linear-attention layers hold a fixed-size recurrent state (independent of length).

0.4 Windows (C6) with the Qwen tokenizer, whole most-recent round trips only:
        W=8k → 3 round trips (6,036 tok) | W=16k → 4 (10,197) | W=32k → 10 (29,347)
    Baseline B at W=8k therefore sees only the last 3 round trips.

0.5 Facts in the ledger: 91 verified + 8 absent = 99 questions; 44/91 final statements are in
    round trips 00–06 (front-loaded, as Ryosuke predicted). `score_binary` self-test: answering
    every question with its own gold → 99/99 pass (tier-1). 14 golds also appear verbatim in an
    earlier round trip than the evidence round trip (fact repeated; harmless post hoc).

---------------------------------------------------------------------------------------------------
## 1. Open decision for Ryosuke (blocks §9 environment only)

The INT4 reader (C1) on the A6000 is only memory-safe under ONE transformers version, and the
corpus shrink (0.1) removed the need for a separate 80 GB pod. Two complete variants are specified;
Ryosuke picks one. Everything else in this document is identical for both.

Variant α — A6000 48 GB, `RedHatAI/Qwen3.8-27B-INT4`, transformers pinned to 5.8.0
  Evidence: transformers 5.8.0 `CompressedTensorsHfQuantizer._process_model_before_weight_loading`
  calls `apply_quantization_config(model, ct_quantization_config, self.run_compressed)` with
  `run_compressed=True` (default of `CompressedTensorsConfig.__init__`), and compressed-tensors
  0.18.0 `CompressedLinear` docstring: "The wrapped layer will decompressed on each forward call."
  → weights stay packed (18.6 GB; the 0.85 GB `model_mtp.safetensors` is not loaded) and are
  dequantized per forward.  Memory at 187k tokens: 19.4 (packed weights) + 12.3 KV (0.3) +
  activations.
  transformers 5.17.0 CHANGED this: `apply_quantization_config(model, remaining_config,
  run_compressed=False)` and "with `dequantize=False` the weights are left compressed, and the hook
  `compress_model` registered decompresses them on the first forward pass" (permanent, via
  `ModelCompressor.add_decompress_hook`) → the model becomes BF16 in memory (~55 GB) → OOM on 48 GB.
  Cost: slower decode (per-forward dequant); ~$0.49/h community (web search 2026-09-17).
  Risk: compressed-tensors ≥0.18 requires torch ≥2.10; 5.8.0 + torch 2.10 untested by us;
  the first smoke step must assert `torch.cuda.memory_allocated() < 24 GB` after one forward.
  Two venvs on the pod (vLLM 0.29.0 requires transformers ≥5.10.4 and would break the pin).

Variant β — ONE 80 GB pod (A100 80 GB PCIe ≈ $1.39/h secure, web search 2026-09-17),
  `Qwen/Qwen3.8-27B` BF16 (55.6 GB on disk), transformers 5.17.0 (attention code verified
  IDENTICAL to 5.8.0: `class Qwen3_5Attention` byte-equal between v5.8.0 and v5.17.0;
  `DynamicCache(config=self.config)` in both). No quantization confound, faster decode,
  vLLM 0.29.0 and transformers 5.17.0 coexist in one venv. Summarizer (9.3 GB on disk incl. vision)
  is stopped before reader phases, so reader alone: 55.6 + 12.3 KV (187k tokens, 0.3) + activations.
  Baseline A prefill uses `GenerationConfig(prefill_chunk_size=8192)` (exists in 5.8.0 and 5.17.0:
  `self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)`; `GenerationMixin._prefill`
  branch "Chunked prefill (for very large contexts)" requires `past_key_values` in model_kwargs).

Recommendation (Fable): β. One pod for all arms, no INT4/version fragility, and the total is
still small (rough: ~12 h × $1.39 ≈ $17 + smoke ≈ $5). α saves ~$10 and adds two failure modes we
cannot test locally (this machine is CPU-only).

---------------------------------------------------------------------------------------------------
## 2. Reader model facts (Qwen3.8-27B) — quoted from transformers 5.8.0 source and HF configs

2.1 Config (Qwen/Qwen3.8-27B `config.json`; top-level keys `architectures, image_token_id,
    language_model_only, model_type, text_config, tie_word_embeddings, transformers_version,
    video_token_id, vision_config, vision_end_token_id, vision_start_token_id`; `model_type`
    `qwen3_5`; `architectures ["Qwen3_5ForConditionalGeneration"]`). `text_config`:
        hidden_size 5120, intermediate_size 17408, num_hidden_layers 64, num_attention_heads 24,
        num_key_value_heads 4, head_dim 256, max_position_embeddings 262144,
        linear_num_key_heads 16, linear_num_value_heads 48, linear_key_head_dim 128,
        linear_value_head_dim 128, linear_conv_kernel_dim 4, full_attention_interval 4,
        vocab_size 248320, tie_word_embeddings False,
        rope_parameters {'mrope_interleaved': True, 'mrope_section': [11, 11, 10],
                         'partial_rotary_factor': 0.25, 'rope_theta': 10000000, 'rope_type': 'default'}
        layer_types: 48 × "linear_attention", 16 × "full_attention" at indices
                     [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    INT4 variant (RedHatAI/Qwen3.8-27B-INT4 config.json): same text_config; quantization_config
    `quant_method compressed-tensors`, `format pack-quantized`, weights `num_bits 4, type int,
    group_size 128, symmetric true`, `kv_cache_scheme {num_bits 8, type float, strategy tensor}`
    (the kv scheme is consumed by vLLM; the transformers quantizer has no kv-cache handling, so
    KV stays bf16 in our path), `ignore` covers visual blocks, linear-attention in_proj_a/b,
    lm_head, mtp. Files: model.safetensors 18.603 GB, model_mtp.safetensors 0.849 GB (unused).

2.2 Auto-class mapping (transformers 5.8.0 `modeling_auto.py`):
    `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen3_5"] = "Qwen3_5ForCausalLM"` (text-only class),
    `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES["qwen3_5"] = "Qwen3_5ForConditionalGeneration"`.
    The checkpoint stores weights under `model.language_model.*` / `model.visual.*`
    (`Qwen3_5Model.__init__`: `self.visual = Qwen3_5VisionModel._from_config(...)`,
    `self.language_model = Qwen3_5TextModel._from_config(config.text_config)`). The existing loader
    (`server/mass_weighted_gemma.py:138-180`, quoted in §5.7) tries `AutoModelForCausalLM` and falls
    back to `AutoModelForImageTextToText` on `ValueError`. V2 records WHICH class actually loaded
    and asserts that no weight was reported missing/unexpected for the language model.

2.3 Full-attention layer forward (verbatim, `Qwen3_5Attention.forward`, identical in 5.17.0):
```python
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
```
    Consequences: (i) with `_attn_implementation == "sdpa"` the interface is
    `sdpa_attention_forward`, which ends in `torch.nn.functional.scaled_dot_product_attention(...)`
    — this is the function the existing patch (§5.7) replaces, so the existing patch reaches these
    16 layers with NO model-specific code; (ii) `q_proj` outputs 2×heads×head_dim and is split into
    query and an output gate — never assume a plain q_proj shape; (iii) `self.scaling =
    head_dim**-0.5` is passed as `scale=`, so adding wM to `attn_mask` yields exactly
    `Softmax(QKᵀ/√d + wM)V` (the bias is NOT divided by √d; `verify_attention_math.py`
    check_2 asserts this).

2.4 SDPA interface (verbatim excerpt, `transformers/integrations/sdpa_attention.py`, 5.8.0):
```python
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
    ...
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
        is_causal=is_causal, **sdpa_kwargs)
```
    `sdpa_kwargs` may be `{"enable_gqa": True}` (`use_gqa_in_sdpa`), so the patched function MUST
    accept and forward `**kwargs` (the existing patch does: `patched_sdpa(query, key, value,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs)`).

2.5 Text model forward (verbatim excerpt, `Qwen3_5TextModel.forward`):
```python
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        ...
        causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, past_key_values=past_key_values, position_ids=text_position_ids)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
```
    Consequences: never hand-build position ids (4-row). Full-attention layers get `causal_mask`,
    which is None when SDPA can use `is_causal`, else an additive float/bool mask; the existing
    `combine_attn_mask` handles None / float / bool (verify_attention_math checks 6, 8, 8b, 11).

2.6 Gated DeltaNet kernel selection (`Qwen3_5GatedDeltaNet`, modeling_qwen3_5.py 5.8.0 lines
    49–63 and 409–413): imports `causal_conv1d_fn, causal_conv1d_update` if
    `is_causal_conv1d_available()`, and `chunk_gated_delta_rule, fused_recurrent_gated_delta_rule`
    if `is_flash_linear_attention_available()`; otherwise `torch_chunk_gated_delta_rule` /
    `torch_recurrent_gated_delta_rule` with `logger.warning_once`. Pod install MUST include
    `flash-linear-attention==0.5.2` and `causal-conv1d`; V2 asserts
    `run_reader.linear_attention_kernel_report()["fla_importable"] is True`.
    The DeltaNet path never calls `scaled_dot_product_attention`, so the patch counter counts
    full-attention layers only (16 per forward).

2.7 Tokenizer: `Qwen/Qwen3.8-27B` tokenizer, vocab 248077. All window budgets (C6) are computed with
    THIS tokenizer, never cl100k. Chat template applied with `enable_thinking=False`.

---------------------------------------------------------------------------------------------------
## 3. Jev contract (TypeSafe) — quoted from https://docs.typesafe.ai/api (read 2026-09-17)

Endpoint `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer <API_KEY>`
(key from env `TYPESAFE_API_KEY`, never logged). Request fields: `state` (string or structured),
`model` = `"jev-latest"`, `questions` = map id → question. Verbatim examples from the docs:
```json
{"state": "Help! My payouts have been failing for 3 days.", "model": "jev-latest",
 "questions": {
   "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
   "department": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Payments, invoicing, refunds",
                               "technical": "Bugs, outages, integrations",
                               "sales": "Pricing, upgrades, new accounts"}},
   "frustration":{"type": "score",  "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```
```json
{"model": "jev-latest",
 "answers": {
   "is_urgent":  {"type": "noul", "noul": 0.92},
   "department": {"type": "choice", "choice": "technical",
                  "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07}, "confidence": 0.82},
   "frustration":{"type": "score", "score": 1.6,
                  "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                  "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65}, "confidence": 0.78}},
 "usage": {"input_tokens": 312, "output_tokens": 48}}
```
Constraints from the docs: choice ≤255 options; score criteria = ordered array of 2–10 levels;
noul `criteria` optional. Errors: 401, 422 (validation), 429, 529; "Use exponential backoff for
429/529 responses." Price used for accounting: $0.042 per M input tokens
(`published_price_per_mtok = 0.042`, recorded in the manifest, not asserted).

Decision mapping (D3): level per axis = argmax over `probabilities` (keys are level-index strings;
tie → lower index); `score` (probability-weighted) is RECORDED only. Noul decision =
`noul >= 0.5` (the natural cut of a yes/no probability, not a tuned threshold; raw value recorded).
Choice decision = `choice`. Any answer missing a required field → raise `JevStop`, count
`unparsed`, STOP the run (D1). No defaults, ever.

Jev question texts (fixed; the `state` is always the RAW chunk text, never a generated summary):
- `keep` (noul; wording of DECISIONS H22 (e), 2026-09-20 — facts inside tool output are kept):
  instructions "Does this excerpt contain information worth remembering later - a fact, value,
  setting, result, decision, definition, instruction or requirement - even if it appears inside
  code, command output or logs? Answer no only for content with no lasting information
  (boilerplate, progress noise, repeated listings)." criteria {"true": "Contains at least one
  concrete fact, value, decision or instruction worth recalling later", "false": "No lasting
  information: boilerplate, progress noise, or repetition"}
- `comprehensiveness` (score, 5 levels): instructions "How broad is the matter this excerpt
  states?" criteria ["A single detail of something larger", "A minor point", "A self-standing
  point", "A major theme with several parts", "The overarching topic of a whole discussion"]
- `independence` (score, 5 levels): instructions "Can this excerpt be understood on its own?"
  criteria ["Meaningless without its surrounding context", "Mostly dependent on context",
  "Partly self-contained", "Mostly self-contained", "Fully self-contained"]
- `detail` (score, 5 levels): instructions "How specific is this excerpt?" criteria
  ["Very general", "General", "Moderately specific", "Specific", "A precise concrete detail"]
- `sun` (choice): instructions "Which existing topic does this excerpt belong to? Pick
  new_topic if none fits." criteria = {"s<i>": <sun node text>} ∪ {"new_topic": "None of the
  listed topics"}. H22 (a): more than 254 suns are asked in batches of ≤ 254 (candidate order,
  `i` = global candidate index), one request per batch; the first batch whose answer is not
  new_topic wins. (The former "len(suns) > 254 → JevStop" rule is withdrawn.)
- `planet` (choice; H22 (a), 2026-09-20): for every other `belongs` decision with more than one
  candidate (planet / satellite / provisional-sun candidates), same batching as `sun`:
  instructions "Which of the listed items is this excerpt a detail or sub-point of? Pick none if
  it belongs under none of them." criteria = {"p<i>": <candidate text>} ∪ {"none": "None of the
  listed items"}.
- `same` (choice; H23, 2026-09-20): for every same-matter decision (spec 0042) with more than
  one candidate, same batching as `sun` / `planet` (<= 254 per request, candidate order, `i` =
  global candidate index, first batch whose answer is not none wins): state = the query node
  text, instructions "Which of the listed items states the same matter as this excerpt? Pick
  none if no item states the same matter." criteria = {"m<i>": <candidate text>} ∪ {"none":
  "No listed item states the same matter"}.
- `same` (noul; single candidate only): state = "A: <a>\nB: <b>", instructions "Do A and B
  state the same matter?"
- `belongs` (noul): state = "A: <a>\nB: <b>", instructions "Is A a detail or sub-point that
  belongs under topic B?"
Level mapping into the existing 0–100 axis scale (§5.1 `level_from_scores`): level index k ∈
0..4 → score 10 + 20·k (10, 30, 50, 70, 90). The existing argmax + tie rule
(detail > independence > comprehensiveness) then decides sun/planet/satellite unchanged.

---------------------------------------------------------------------------------------------------
## 4. Summarizer contract (Qwen3.5-4B via vLLM 0.29.0)

Serve: `vllm serve Qwen/Qwen3.5-4B --language-model-only --max-model-len 8192
--gpu-memory-utilization 0.25 --served-model-name summarizer --port 8001 --dtype bfloat16`
(`--language-model-only`: "If True, disables all multimodal inputs by setting all modality
limits to 0." — vLLM engine-args doc, read 2026-09-17; Qwen3.5 is listed in vLLM supported models
as `Qwen3_5ForConditionalGeneration`). Client: OpenAI-compatible
`POST http://127.0.0.1:8001/v1/chat/completions` with
`{"model":"summarizer","messages":[{"role":"user","content":<instruction + excerpt>}],
  "temperature":0,"max_tokens":96,"chat_template_kwargs":{"enable_thinking":false}}`.
Per call record `usage.prompt_tokens`, `usage.completion_tokens`, latency_ms. Node text rule (D2):
≤120 characters, self-contained, source language; if the reply exceeds 120 characters, retry ONCE
with the explicit length instruction appended, then raise `SummarizerStop` (D2 requires
node_fallback == 0; there is no silent truncation). The vLLM process is terminated before any
reader phase (sequential phases, F3).

---------------------------------------------------------------------------------------------------
## 5. Existing code contracts that implementation MUST reuse (verbatim, read 2026-09-17)

5.1 `management/harness/spec_manager.py`
```python
59: AXES: tuple[str, ...] = ("comprehensiveness", "independence", "detail")
65: SCORE_MIN = 0
66: SCORE_MAX = 100
105: @dataclass
106: class SpecConfig:
109:     chunk_max_chars: int = 400
110:     max_node_chars: int = 120
111:     shortlist_k: int = 5
112:     judge_max_tokens: int = 256
113:     node_max_tokens: int = 200
114:     max_retries: int = 1
117:     planet_mass_floor: float = 0.0
123:     max_workers: int = 1
149: @dataclass
150: class SpecTurnReport:   # calls, cache_hits, chunks, nodes, node_fallback, unparsed, defaulted,
                            # vanished, attached, promoted, added
292: def level_from_scores(scores: dict[str, int]) -> NodeLevel:
383:     def __init__(self, judge: JudgeLLM, *, config: SpecConfig | None = None,
389:                  embed_fn: EmbedFn | None = None, cache: JudgeCache | None = None) -> None:
396:         self.similarity = SimilarityJudge(judge, shortlist_k=self.config.shortlist_k,
399:             use_embedding_shortlist=self.config.shortlist_k > 0, embed_fn=embed_fn, runner=self.runner)
405:         self._merger = GraphMerger(similarity_fn=self._counting_similarity, attach_fn=self._counting_attach)
465:     def chunk(self, user_text: str, assistant_text: str, turn: int) -> list[SpecChunk]:
477:     def node_for_text(self, text: str, turn: int = -1) -> Node:
505:     def _ask_node(self, text: str) -> tuple[str, dict[str, int]] | None:
528:     def _nodes_for_chunks(self, chunks: Sequence[SpecChunk], turn: int) -> list[Node]:
546:     def build_provisional(self, nodes: Sequence[Node]) -> ProvisionalStructure:
642:     def update(self, base: CorrelationDiagram, user_text: str, assistant_text: str, turn: int) -> SpecTurnReport:
649:         """One conversation round-trip (0036).  Mutates ``base`` in place.
677:             base.normalize(planet_mass_floor=self.config.planet_mass_floor)
```
    Judge call sites: `_ask_node` → `Q_NODE` prompt → `parse_node_object(raw, max_chars)` →
    `(summary, {"comprehensiveness": int, "independence": int, "detail": int})`;
    `self.similarity.most_similar(query, candidates, kind)` → `(index, 1.0|0.0)`, `(-1, 0.0)` for
    no match; kinds `K_SAME` / `K_BELONGS` (lines 450, 458, 571, 591).

    REQUIRED MINIMAL CHANGE (the only edit to spec_manager.py): three optional hooks on
    `SpecManager.__init__`, each defaulting to today's behaviour so every existing test passes
    unchanged:
```python
    node_fn: Callable[[str], tuple[str, dict[str, int]] | None] | None = None,
        # when given, node_for_text() uses node_fn(text) instead of _ask_node(text)
    keep_fn: Callable[[str], bool] | None = None,
        # when given, _nodes_for_chunks() drops chunks with keep_fn(chunk.text) is False and
        # counts them in a new SpecTurnReport field `dropped: int = 0`
    similarity: SimilarityJudge | None = None,
        # when given, replaces the internally constructed SimilarityJudge (line 396)
```
    Everything else (chunking, provisional structure, merge cases, normalize, planet mass =
    satellite count, `planet_mass_floor` 0.0) stays as is (D4).

5.2 `management/harness/chunking.py`
```python
218: def split_candidates(text: str, max_chars: int = 800) -> list[str]:
219:     """Chunk candidates for ``text``: never longer than ``max_chars``, never cut
220:     mid-sentence unless a single sentence exceeds ``max_chars``.
```
    (SpecManager.chunk uses `SpecConfig.chunk_max_chars = 400`.)

5.3 `management/harness/judge.py`, `similarity_judge.py`
```python
20: class JudgeLLM(Protocol):
23:     def complete(self, prompt: str, *, max_tokens: int) -> str: ...
32: class SimilarityJudge:
96:     def most_similar(self, query: str, candidates: list[str], kind: str = K_SAME) -> tuple[int, float]:
```
    `JevSimilarityJudge` (new, §6) duck-types `SimilarityJudge` and implements `most_similar` as:
    more than one candidate (either kind) → ONE Choice request per batch of ≤ 254 candidates
    (candidate order), counted under the kind: for `K_BELONGS` the `sun` question (+ new_topic)
    when every candidate is a current sun node text (`sun_texts_fn()` wired by build_cd to the CD
    being built), otherwise the `planet` question (+ none) (DECISIONS H22 (a), replaces H8); for
    `K_SAME` the `same` Choice question (+ none) (DECISIONS H23). The first batch whose answer is
    not the none option wins → `(global idx, 1.0)`, else `(-1, 0.0)`. A single candidate → one
    Noul question `same` / `belongs` on the pair state, ≥ 0.5 wins (the spec's one-to-one 0042
    judgement), else `(-1, 0.0)`. It exposes a
    `runner` counter object (`calls`,
    `unparsed`, `defaulted`, `retried` dicts) so `SpecManager.call_totals()` keeps working;
    `defaulted` can never increase.

5.4 `management/graph_merger.py` (note: NOT under harness/)
```python
37: SimilarityFn = Callable[[str, list[str]], tuple[int, float]]
51:     def __init__(self, similarity_fn: Optional[SimilarityFn] = None, attach_fn: Optional[SimilarityFn] = None):
73:     def merge(self, base: CorrelationDiagram, incoming: CorrelationDiagram) -> CorrelationDiagram:
```
5.5 `communication/cd_serializer.py`
```python
37:     def __init__(self, level_markers: bool | None = None):   # None → config "tokenization.level_markers" (default False)
52:     def to_context_block(self, cd: CorrelationDiagram) -> str:
67:     def _node_line(self, text: str, mass: float, indent: int) -> str:
68:         if self._level_markers:
69:             if indent == 0:   token = "[SN]"
71:             elif indent == 1: token = f"[PN{round(mass, self._precision)}]"
73:             else:             token = "[RN]"
77:         return "  " * indent + f"{token} {text}"
81:     def to_context_block_budgeted(self, cd, budget_tokens: int, policy: Literal["mass","random","recency"],
86:                                   token_counter: Callable[[str], int], seed: int = 0) -> str:
```
    Wrapper lines `<CONTEXT>` / `</CONTEXT>`. This experiment ALWAYS uses `level_markers=True` and
    `policy="mass"` (C6 eviction by planet mass).

5.6 `server/cd_parser.py`
```python
170: def find_marker_spans(token_ids: list[int], tokenizer) -> list[tuple[str, float, list[int]]]:
184: def marker_positions(spans, *, inject_levels: set[str] | None = None, satellite_inherit: bool = False) -> list[tuple[int, float]]:
```
    (`inject_levels=None` → `{"planet"}`, spec default.)

5.7 `server/mass_weighted_gemma.py` (model-agnostic; alias `MassWeightedLLM = MassWeightedGemma`)
```python
43:     def __init__(self, config_path=None, *, model_id=None, max_new_tokens=None, temperature=None,
                     do_sample=None, allow_sliding_layers=None, quantization=None):
108:         self.bias_applied_calls: int = 0
109:         self.bias_skipped_prefill_calls: int = 0
110:         self.bias_skipped_sliding_calls: int = 0
119:     def mass_injection_stats(self) -> dict[str, int]:
138:     def load(self) -> None:      # AutoModelForCausalLM(dtype=bf16, device_map="auto", attn_implementation="sdpa")
                                     # → on ValueError: AutoModelForImageTextToText(...); then self._patch_sdpa()
189:     def set_mass_vector(self, v: torch.Tensor) -> None:
194:     def clear_mass_vector(self) -> None:
204:     def generate(self, prompt: str) -> str:
231:     def _patch_sdpa(self) -> None:   # replaces torch.nn.functional.scaled_dot_product_attention globally
288:             m_bias = build_mass_bias(seq_q, seq_k, m_matrix=..., mass_vector=outer._mass_vector,
                     mass_weight=outer._mass_weight, prefill_mass_scale=outer._prefill_mass_scale, ...,
                     strict_alignment=not outer._allow_sliding_layers, bias_cap=getattr(outer, "_bias_cap", None))
303:                 attn_mask, is_causal = combine_attn_mask(attn_mask, m_bias, is_causal=is_causal, ...)
312:                 outer.bias_applied_calls += 1
331:             return outer._original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                            is_causal=is_causal, scale=scale, **kwargs)
348:     def start_attention_recording(self) -> None:
353:     def stop_attention_recording(self) -> list[torch.Tensor]:
404:     def attn_implementation(self) -> str | None:   # property
431: def build_mass_bias(seq_q, seq_k, *, m_matrix, mass_vector, mass_weight, prefill_mass_scale, dtype, device,
                        strict_alignment=True, bias_cap=None) -> torch.Tensor | None:
548: def combine_attn_mask(attn_mask, m_bias, *, is_causal, seq_q, seq_k, dtype, device) -> tuple[torch.Tensor, bool]:
```
    This is the injection path (§7). `start_attention_recording` / `stop_attention_recording`
    give the attention rows for the F4 share probe (no eager mode needed).

5.8 `server/mass_vector.py`
```python
24: def positions_to_mass_vector(positions: list[tuple[int, float]], seq_len: int, *, cap: float,
                                 scale: float = 1.0, device=None) -> torch.Tensor | None:
```
5.9 `benchmark/bineval/run_reader.py` (reuse; new CLI in §6 wraps these, it does not copy them)
```python
82: INJECT_MODES = ("planet", "planet+satellites", "none")
191: HYBRID_MODEL_ID = "Qwen/Qwen3.8-27B"
192: HYBRID_N_SDPA_LAYERS = 16
104: def build_reader_mass_vector(prompt_ids, tokenizer, inject, cap=None, *, w=1.0, prompt_text=None, device=None)
         -> tuple[Any, MassVectorInfo]      # MassVectorInfo: positions_found, spans, planet_spans, satellite_spans, inject, cap
230: def linear_attention_kernel_report() -> dict      # ["fla_importable"]
321: def check_model_supported(model_config, *, allow_linear_layers=False) -> dict   # ["n_sdpa_layers"], ["n_linear_layers"], ...
434: def full_attention_layers(model_config) -> int | None
450: def kv_cache_bytes(model_config, tokens: int) -> int | None
476: def check_context_fits(model_config, prompt_tokens, max_new_tokens, gpu_mem_gb, *, headroom=0.9, token_margin=0.15) -> dict
618: def load_reader(model_id, *, max_new_tokens=48, prefill_scale=0.0, bias_cap=None, w=0.0) -> Any
653:     impl = llm.attn_implementation
654:     if impl != "sdpa": raise RuntimeError(...)
676: def run_reader(llm, context_block, questions, *, w, inject, bias_cap=None, arm=None, progress=None) -> ReaderRun
733:         result.per_question[q["qid"]] = {"raw", "positions_found", "spans", "planet_spans", "satellite_spans",
                                             "prompt_tokens", "bias_applied_calls", "bias_skipped_prefill_calls",
                                             "bias_skipped_sliding_calls"}
749: def run_meta(*, model_id, layer_info, w, inject, prefill_scale, bias_cap, context_tokens, arm=None,
                 context_check=None, extra=None) -> dict
791: def write_answers(out_path, run: ReaderRun) -> None    # <out>.json answers + <out>.meta.json (per_question inside)
```
    `run_reader()` uses ONE context block for all questions. This experiment's window differs per
    arm but not per question (post hoc, C6), so `run_reader` is called once per (arm, W, w) cell
    with the window as `context_block`. Missing from `per_question` and REQUIRED by E1:
    `completion_tokens`, `wall_ms_prefill`, `wall_ms_decode`, `attn_flops_prefill`,
    `attn_flops_decode`, `window_tokens`, `n_recent_rts`, `energy_joules` → added by the §6 wrapper
    (via `extra=` and a per-question timing hook), NOT by editing run_reader's dict.

5.10 `benchmark/bineval/score_binary.py`
```python
158: def tier1_match(gold_short: str, aliases: list[str], answer_text: str) -> tuple[bool, Optional[str]]:
213: def load_questions(path: Path) -> list[dict]:
220: def select_questions(questions: list[dict], subset: str, include_excluded: bool) -> list[dict]:
278: def score_condition(scored_questions, answers_by_pos: dict[int,str], answers_by_qid: dict[str,str],
                        judge_fn, max_words=None) -> dict   # {"items", "aggregate": {pass, fail, indeterminate, total,
                                                            #  pass_rate_tier1, max_words, multi_gold_answers}, "pending_tier2"}
397-406: CLI --answers --questions --out --subset {all,legacy,generated} --include-excluded --tier2 --max-words
```
    Verified 2026-09-17: `data/questions.json` loads, `select_questions(qs, "all", False)` → 99,
    gold-vs-gold self-score 99/99.

5.11 `verify_attention_math.py`: 17 CPU checks (`CHECKS` list line 838) covering decode exactness,
    unscaled bias, broadcast axis, decode-only, prefill scale, causal composition, existing
    float/bool masks, GQA kwargs, sliding refusal, cap, spans/levels. Must pass before and after
    any change to `server/`.

5.12 grep result: there is NO Qwen3.5-specific patch anywhere in the repo; hybrid handling is
    config inspection only (`run_reader.py:177-445`, `gpu_preflight.py:302-318`). No new
    model-specific attention code is to be written (2.3 shows none is needed).

---------------------------------------------------------------------------------------------------
## 6. New modules (contracts; file names fixed; all under `benchmark/mcbuild_bench/`)

  jev_client.py        JevClient(api_key: str, model="jev-latest", timeout_s=60, max_attempts=3,
                                  accounting_path: Path)
                        .ask(state: str, questions: dict[str, dict]) -> JevResult
                        JevResult = {"answers": dict, "usage": {"input_tokens": int, "output_tokens": int},
                                     "latency_ms": float, "http_status": int, "retries": int}
                        429/529 → sleep 1s, 4s, 16s between attempts; after 3 failures, or on any
                        other non-200 status immediately, raise JevStop(status, body[:500]).
                        Appends one JSON line per attempt to accounting_path:
                        {"ts", "question_ids", "question_types", "state_chars", "input_tokens",
                         "output_tokens", "latency_ms", "http_status", "retry_index", "answers"}.
                        Parsing helpers (pure, unit-tested against the §3 JSON verbatim):
                        choice_of(answer) -> str; argmax_level(answer) -> int; noul_of(answer) -> float.
  summarizer_client.py SummarizerClient(base_url="http://127.0.0.1:8001/v1", model="summarizer",
                                        accounting_path: Path)
                        .summarize(excerpt: str) -> str   (≤120 chars or raises SummarizerStop)
                        Appends {"ts","prompt_tokens","completion_tokens","latency_ms","retried","chars"}.
  jev_judge.py         JevNodeFn(jev, summarizer) → callable for SpecManager(node_fn=...):
                          ONE Jev request per chunk with {keep, comprehensiveness, independence,
                          detail} on the raw chunk (D3), cached per text; JevKeepFn(node_fn) reads the
                          `keep` answer of that same request (spec_manager calls keep_fn before node_fn
                          on the same text; violating that order raises). The summarizer is called only
                          for kept chunks; returns (summary, scores) with levels mapped to 10+20k.
                        JevSimilarityJudge(jev) → SpecManager(similarity=...) per §5.3.
                        JevJudgeLLM: a `JudgeLLM.complete()` that raises NotImplementedError — the
                          spec manager must never reach a text prompt in this experiment; reaching it
                          is a bug, not a fallback.
  build_cd.py          CLI: --session data/session_redacted.json --out data/cd.json
                        --jev-accounting data/jev_calls.jsonl --summarizer-accounting data/summarizer_calls.jsonl
                        --summarizer-url http://127.0.0.1:8001/v1 --gpu-csv data/gpu_manager.csv
                        [--max-round-trips N] (for V4)
                        [--exclude-rt 36]  (H22 (c): corpus.load_corpus; manifest.session_sha256 =
                        sha of the filtered content, plus session_file_sha256 / exclude_rt / n_round_trips)
                        For each round trip i: user_text = human; assistant_text = "\n".join(event
                        texts in order, each prefixed by its kind tag e.g. "[tool_use]"); calls
                        SpecManager.update(cd, user_text, assistant_text, turn=i). Output JSON:
                        {"nodes": [ {node_id, text, level, mass, parent_id, created_turn} ... ]  (same
                          record shape as build_cd_offline._node_record),
                         "summary": {sun, planet, satellite, total, turns, dropped_chunks,
                                     harness_calls, harness_quality},
                         "manifest": {git_sha, transformers_version, vllm_version, jev_model,
                                      jev_input_tokens_total, jev_output_tokens_total,
                                      published_price_per_mtok, jev_cost_usd, summarizer_calls,
                                      summarizer_prompt_tokens, summarizer_completion_tokens,
                                      wall_s, gpu_csv}}
  windows.py           build_window(arm: str, W: int | None, cd_block: str | None, round_trips: list,
                                    tokenizer, question: str) -> Window
                        Window = {"prompt": str, "window_tokens": int, "n_recent_rts": int,
                                  "recent_idx": [int], "cd_tokens": int, "evicted_planets": int}
                        (``recent_idx`` = idx of the round trips in the recent part, chronological;
                        H22 (d) uses min(recent_idx) as the oldest round trip inside the window)
                        Rules (C6): fixed prompt scaffold (§8) is counted; then CD block (proposed
                        only; budgeted by mass via to_context_block_budgeted if it alone exceeds
                        W − scaffold); then whole round trips newest-first while they fit; then the
                        question. W=None (arm A) → all round trips, no CD.
  run_arms.py          CLI: --arm {A,B,C,proposed} --W {8000,16000,32000,full} --w <float>
                        --model-id <id> --questions data/questions.json --session ... --cd data/cd.json
                        --out <dir> --gpu-csv <path> --prefill-chunk 8192 --inject planet
                        --prefill-scale 0.0 --max-new-tokens 48 [--exclude-rt 36]
                        [--questions-subset <proposed cell>/questions_subset.json  (baselines only)]
                        Uses run_reader.load_reader / run_reader.run_reader / write_answers.
                        H22 (c): the session is loaded through corpus.load_corpus (default
                        --exclude-rt 36, checked); session_sha256 in every artifact = sha of the
                        FILTERED content. H22 (d): arm proposed drops every question whose fact's
                        round trip is >= min(window.recent_idx) (absent questions kept), records
                        questions_used / questions_dropped_in_window / first_recent_rt in meta.json
                        and the answers.jsonl header, writes <out>/questions_subset.json; a baseline
                        consumes it via --questions-subset (sha in the checkpoint identity).
                        Adds per question (E1): completion_tokens, wall_ms_prefill, wall_ms_decode,
                        attn_flops_prefill = Σ_{16 layers} 2·24·L²·256 (L = prompt tokens; QKᵀ term only,
                        GQA ignored — E1 as decided; the report footnote states this),
                        attn_flops_decode = Σ_steps Σ_{16 layers} 2·24·(L+t)·256, window_tokens,
                        n_recent_rts, cd_tokens, energy_joules (from the GPU CSV over the question's
                        time span). Writes answers.json, meta.json (run_meta + extra), timing.jsonl.
  gpu_sampler.py       GpuSampler(path).start()/.stop(): subprocess
                        `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -lms 1000`
                        energy_joules(csv_path, t0, t1) -> float  (trapezoid over power.draw [W]).
  compaction_c.py      Baseline C (C4): stream round trips; when [summary + recent] would exceed W,
                        summarize (reader model, §8 instruction, cap W/4 tokens) everything older;
                        window = summary + recent. Records each summarization call (prompt/completion
                        tokens, wall ms) into compaction_calls.jsonl; final window fed to run_arms.

---------------------------------------------------------------------------------------------------
## 7. Injection contract for Qwen3.8 (no new attention code)

7.1 Load through `run_reader.load_reader(model_id, max_new_tokens=48, prefill_scale=0.0,
    bias_cap=None, w=w)`; it asserts `attn_implementation == "sdpa"`. Then
    `check_model_supported(model.config, allow_linear_layers=True)["n_sdpa_layers"] == 16` (F4).
7.2 Bias vector per question: `find_marker_spans(prompt_ids, tokenizer)` → `marker_positions(spans)`
    (planets only) → `positions_to_mass_vector(positions, len(prompt_ids), cap=inf, scale=w)`
    (this is what `build_reader_mass_vector(..., inject="planet", cap=None, w=w)` already does) →
    `llm.set_mass_vector(v)`. Field semantics (fixed 2026-09-18 after the Codex review):
    `planet_spans` (run_reader) = number of `[PN…]` MARKERS located in the prompt ids;
    `positions_found` = number of TOKEN positions covered by planet text (≥ planet_spans);
    run_arms writes `planet_lines` = count of `[PN` lines in the window text and asserts
    `planet_spans == planet_lines` (this is the F4 check; F4's wording "positions_found ==
    planet lines" meant this marker-level equality).
7.3 Decode-only (C7): `prefill_scale=0.0` → during prefill `build_mass_bias` returns None and
    `bias_skipped_prefill_calls` increments (16 per prefill FORWARD; chunked prefill runs
    ⌈L/chunk⌉ of them); from the first decode step `bias_applied_calls` increments 16 per step.
    Generating n tokens = 1 prefill + (n−1) decode forwards (the first token comes out of the
    prefill forward), so the check is `bias_applied_calls == 16·(n−1)`,
    `bias_skipped_prefill_calls == 16·⌈L/chunk⌉` and `bias_skipped_sliding_calls == 0`
    (corrected 2026-09-18; `run_arms` enforces it per question, F1).
7.4 Share probe (F4): `start_attention_recording()` before one decode step on a 300-token probe
    containing 3 planet lines; `stop_attention_recording()` returns the rows; planet share =
    Σ attention on planet spans / Σ all. Must increase strictly with w over the grid.
7.5 Pre-run refusal: `check_context_fits(config, prompt_tokens, 48, gpu_mem_gb)` for every cell
    before loading weights (arm A: prompt_tokens ≈ 187k + scaffold).
7.6 `prefill_last_row` switch (H15 option (b), DECISIONS H18; OFF by default): `run_arms
    --prefill-last-row` → `load_reader(prefill_last_row=True)` → `MassWeightedGemma(prefill_last_row=True)`.
    During the LAST prefill forward (`seq_k == prompt_len`, set by `generate()`) the patch keeps the
    ordinary output and recomputes ONLY the final query row with `w·mass` added (decode rule: full
    weight, effective-bias cap, `is_causal=False`, existing mask's last row kept, GQA expanded as in
    the decode path); earlier rows and chunks are untouched. Counted in
    `bias_applied_prefill_last_row_calls` (= n_sdpa_layers per generate when on, 0 when off; an
    attribute + `per_question` field, not a `mass_injection_stats()` key); `bias_skipped_prefill_calls`
    still counts that forward. Mutually exclusive with `prefill_scale > 0` (ValueError). Recorded in
    meta and in the checkpoint identity.

---------------------------------------------------------------------------------------------------
## 8. Prompts (fixed strings, identical across arms — C8)

Reader prompt = the EXISTING `run_reader.READER_PROMPT` (benchmark/bineval/run_reader.py:73-80),
applied by `run_reader.build_prompt(context_block, question)`; `windows.build_window` returns the
context block wrapped as `<context>
…
</context>` and measures W on the FULL assembled prompt:
    {context_block}

    Answer the question using only the information above. Reply with the answer only, in a few
    words. If the information is not present, reply: unknown.
    Question: {question}
    Answer:
`MassWeightedGemma.generate()` tokenizes this raw string; NO chat template is applied (the same
path all previous reader experiments used). Amendment H6 records this for Ryosuke's decision.
Greedy (`do_sample=False`), `max_new_tokens=48`, first non-empty line kept, scored with
`score_binary --max-words 32 --subset all`.
Baseline C summarizer instruction (reader model): "Summarize the following conversation log for
later reference. Keep every concrete value, name, path, decision and instruction. Plain text, at
most {cap} tokens.\n\n{older_log}" (cap = W/4).
Summarizer (4B) node instruction: "Rewrite the following excerpt as one self-contained statement
of at most 120 characters, in the excerpt's language, keeping concrete values. Output the
statement only.\n\n{excerpt}"; retry suffix: "\n\nYour previous answer was too long. At most 120
characters."

---------------------------------------------------------------------------------------------------
## 9. Pod procedure (F1) — from https://docs.runpod.io/pods/configuration/use-ssh (read 2026-09-17)

- Public key: paste into "SSH Public Keys" in console settings, or
  `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`. Pod receives it via `PUBLIC_KEY` env
  (`echo "$PUBLIC_KEY" >> authorized_keys`); official PyTorch templates have SSH pre-configured.
- Connection: "Full SSH via Public IP" mode (TCP port 22 exposed): `ssh root@<ip> -p <port> -i
  <key>`; SCP/SFTP work in this mode (the proxy mode `...@ssh.runpod.io` does not).
- Fable runs every command as `ssh -i "$MCB_POD_KEY" -p <PORT> root@<IP> '<cmd>'` from this
  machine; long jobs: `tmux new -d -s run '<cmd> 2>&1 | tee <log>'`, polled with
  `tmux capture-pane -p -t run`. Key path is supplied by Ryosuke as env var `MCB_POD_KEY`; its
  contents are never printed.
- Pod template: RunPod PyTorch (CUDA 12.x); volume ≥ 150 GB. Install, variant β (one venv):
  `uv venv && uv pip install vllm==0.29.0 transformers==5.17.0 flash-linear-attention==0.5.2
  causal-conv1d accelerate tiktoken` (vllm pins torch 2.13.0). Variant α: venv_reader
  `transformers==5.8.0 compressed-tensors>=0.18 torch>=2.10 flash-linear-attention==0.5.2
  causal-conv1d accelerate` and venv_vllm `vllm==0.29.0`.
- Repository on the pod: `git clone` of the pushed branch; the run manifest records the sha.
- Destructive ops (stop/terminate/delete volume) → Ryosuke approves each time.

---------------------------------------------------------------------------------------------------
## 10. Verification ladder (numeric pass rules; nothing proceeds on a fail)

V1 (local, CPU): `pytest tests/` unchanged green after the §5.1 hooks; new tests: windows.py
   (budget never exceeded; newest-first; whole round trips; W=None → 37 RTs), jev_client parsing
   on the §3 JSON verbatim (fixture file), JevStop after 3×429 (mocked), summarizer length rule
   (mocked), JevSimilarityJudge → (idx,1.0)/(-1,0.0) mapping, build_cd end-to-end with mocked
   Jev+summarizer on 2 round trips producing a parseable CD with planet masses; scorer 99/99;
   `python verify_attention_math.py` exit 0.
V2 (pod): load; `attn_implementation == "sdpa"`; `n_sdpa_layers == 16`; `fla_importable`;
   loaded class name recorded; memory after one 512-token forward (α < 24 GB; β < 60 GB).
   Additional smoke assertions from the 2026-09-17 adversarial review (untestable on CPU):
   (a) every `per_question.prompt_tokens <= W` (windows.count_tokens vs MassWeightedGemma's own
   tokenization must agree; tokenizer_config has add_bos_token=false); (b) `planet_spans` equals the
   `[PN` line count of the window under the Qwen tokenizer (indented markers found by
   `cd_parser._scan_pn_spans`); (c) nvidia-smi `power.draw` is numeric on the pod (not `[N/A]`);
   (d) vLLM Qwen3.5-4B replies with `enable_thinking=false` carry no `<think>` block in `content`;
   (e) arm A chunked prefill: which SDPA kernel runs with the explicit 8192×L mask (math kernel
   would allocate a 24×8192×190k score matrix ≈ 75 GB → OOM), checked on V5's 5 questions;
   (f) Jev latency and rate limit measured over the ≥50 calls of V4 and extrapolated to the full
   run before the main run is started.
V3 (pod): 7.3 counters exact; 7.2 positions_found == planet lines; 7.4 share monotone; ≥3
   non-degenerate w values → Ryosuke freezes the w grid into DECISIONS.md.
V4 (pod): build_cd on the first 3 round trips — Jev unparsed == 0 over ≥50 calls, summarizer
   retries recorded and node_fallback reported (H28; SummarizerStop only on an unreachable summarizer), CD parses, Ryosuke inspects the tree (F3 gate);
   Jev cost extrapolated to 37 round trips and reported.
V5 (pod): arm A on 5 questions with `prefill_chunk_size=8192` — no OOM; seconds/question recorded.
V6: main run (all arms × W × w grid); V7: scoring + McNemar/bootstrap (B5) locally; report.
