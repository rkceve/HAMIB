# mcbuild-bench — decision ledger (frozen 2026-09-17)

Purpose (Ryosuke, verbatim intent): measure the DIFFERENCE IN COMPUTE needed to find facts inside a
~410k-token real agent session. Success = equivalent test results with LESS compute for the proposed
method. Every item below is a decision Ryosuke made or explicitly delegated. Implementation may not
choose anything not written here; when a choice is missing, STOP and ask.

## A. Data
A1. Source: the 2026-09-12 AI Tinkerers hackathon session (Claude Code, Fable 5.1, 1M context),
    transcript `~/.claude/projects/C--Users-Ryosuke-Kawai/5cc1108c-22a0-4129-a5fe-3a2fe84f1f34.jsonl`.
    Measured: 42 human turns, ~409k cl100k tokens (human 6k, assistant text 28k, tool_use 74k,
    tool_result 289k). One session only (Ryosuke: "1本の会話であることはそこまで問題じゃない").
A2. Conversation unit = ONE ROUND TRIP: one human message + everything until the next human message
    (assistant text, tool calls, tool results). 42 round trips. The 3 "[Request interrupted]" human
    messages are dropped (their content is empty). Image attachments are replaced by the literal line
    `[image attached]` and are never a fact source.
A3. Tool results ARE fed to the manager (Ryosuke: "それ前提だから問題ない"; the manager's ability to
    exclude code/output from the CD is itself under test). Exclusion is expressed as a Jev Choice option
    "not a memory item" (see D3).
A4. Redaction BEFORE any experiment (Ryosuke: everything will be published). Full-text scan by Fable of
    the whole transcript incl. code, tool output, env vars, URLs. Targets: IP addresses, hostnames,
    VPN hostnames, ports paired with hosts, real-name paths (real-name home paths, cloud-drive paths),
    Windows usernames, GitHub user/org names, e-mail addresses, API-key-shaped strings, RCON/SSH
    passwords, screenshot filenames with timestamps, phone numbers. Replacement policy (delegated to
    Fable): SHAPE-PRESERVING FAKE VALUES so the manager does not treat them as special (IPv4 → TEST-NET
    203.0.113.x, paths → `C:\Users\user\...`, GitHub → `example-user`, hostnames → `demo-server`,
    keys → `sk-REDACTED…` of the same length). A mapping table is kept OUTSIDE the repo
    (in a private folder outside the repository). Gold answers that are
    redacted values are scored against the REPLACED value. Gate: Ryosuke reviews the redacted transcript.

## B. Facts and questions
B1. Fact ledger drafted by Fable from the transcript (each fact: source round trip, source line quote,
    initial mention, updates, gold_short, aliases, kind ∈ {value, name/path, policy, definition,
    updated-final}). Gate: Ryosuke reviews before any GPU use.
B2. Question mix: facts of all kinds + 10–20 % "absent fact" questions whose correct answer is `unknown`
    (fact not in the transcript). Updated facts are asked as FINAL value (e.g. scale 2:1, not 1:1).
B3. Questions are asked once, AFTER the whole session (post-hoc design; Ryosuke chose this over the
    streaming design for the first measurement). Expected ~100 questions.
B4. Scoring: existing `benchmark/bineval/score_binary.py` tier-1 (normalize + substring + aliases,
    `--max-words 32`, `--subset` = this set). No LLM judge.
B5. Statistics: paired per-question comparison between arms (McNemar + bootstrap CI as in
    `benchmark/bineval/PROTOCOL.md`).

## C. Arms and windows
C1. Reader model: `RedHatAI/Qwen3.8-27B-INT4` (W4A16, DeltaNet in_proj_a/b + vision + lm_head in BF16).
    Pre-decided fallback (NOT a runtime guess): if transformers 5.8 cannot load compressed-tensors for
    the injection path, use `Qwen/Qwen3.8-27B` BF16 on an 80 GB pod for the reader.
C2. Baseline A (upper reference): the FULL ~410k-token transcript in one prompt, RoPE extended with YaRN
    (factor chosen so max_position ≥ 420k; Ryosuke: "自由で"), all ~100 questions. Runs on an 80 GB pod
    (KV for 16 full-attention layers at 410k ≈ 27 GB + weights).
C3. [NOT DECIDED BY RYOSUKE — Fable's addition, struck 2026-09-20] "Baseline B (recent-only at the same
    window)" and C4 "Baseline C (re-summarize)" were proposed by Fable inside the cell count of item 4 on
    2026-09-17 and never explicitly adopted. Ryosuke's decision defines TWO arms only: the proposed
    method (manager → CD → narrowed-window reader with bias) and the baseline (the whole transcript read
    by the same model in its large window). B and C stay implemented (windows.py / compaction_c.py) but
    are OUT of the plan unless Ryosuke adopts them.
C4. [struck, see C3]
C5. Proposed: manager builds the CD over all 42 round trips once (D); window = serialized CD (planet
    lines carry mass) + most recent round trips that fit; reader answers with mass injection on PLANET
    lines only; w ∈ grid decided in the smoke test (C7).
C6. Windows W ∈ {8k, 16k, 32k} tokens (measured with the READER's tokenizer). Window composition rule:
    [CD block] + [recent round trips, newest first, whole round trips only, stop at overflow] +
    [question]. If the CD block alone exceeds W, budgeted eviction by planet mass (existing serializer).
C7. w grid: decided in the smoke test on the A6000 — search values where generation stays stable (no
    degenerate repetition, non-empty short answers) while the planet-line attention share rises
    monotonically; then fix 3–4 values for the main run. Injection at decode steps only (prefill scale 0)
    unless the smoke test says otherwise; both recorded.
C8. Reader prompt (identical across arms): context block, then
    "Answer the question using only the information above. Reply with the answer only, in a few words.
    If the information is not present, reply: unknown." Greedy, max_new_tokens 48, first line kept.

## D. Manager (Jev + local summarizer)
D1. Judge = TypeSafe Jev, `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`, Bearer key
    from env `TYPESAFE_API_KEY`. The ONLY external call in the whole experiment. No fallback judge
    (Ryosuke: "jevが使えないときのことは考えなくていい"): 429/529 → exponential backoff, 3 attempts, then
    STOP the run (no default answers, ever).
D2. Summarizer = `Qwen/Qwen3.5-4B` (hybrid GDN, vision encoder skipped via vLLM `--language-model-only`),
    thinking disabled (`chat_template_kwargs.enable_thinking=false`), served locally by vLLM on the pod.
    One call per chunk: ≤120-char self-contained node text in the source language.
D3. Jev questions per chunk (ONE request, several questions):
    - `level` (Score, 3 levels ordered: satellite < planet < sun; instructions = the spec's three axes
      comprehensiveness/independence/detail rewritten as ordered levels) — NOTE: spec 0039 scores three
      axes and takes the max; Jev Score gives ONE ordered level. Ryosuke accepted Score for the axes;
      implement as THREE Score questions (one per axis, 5 levels each) and take argmax, tie rule
      detail > independence > comprehensiveness (as in spec_manager).
    - `keep` (Noul): "Is this chunk a memory item worth keeping (a fact, decision, definition or
      instruction), as opposed to code, tool output, logs, or filler?" — the exclusion switch of A3.
    - `sun` (Choice over existing sun texts + "new topic"): which topic this chunk belongs to. Existing
      suns are listed by their node text (≤ 255 options; if more, STOP — do not truncate silently).
    - For planet/satellite placement within the turn and for merge: `same` (Noul, "do A and B state the
      same matter?") and `belongs` (Noul, "does A belong under topic B?").
    Confidence/probabilities are RECORDED, never used for decisions (no thresholds).
D4. Everything else is the spec procedure already implemented in `management/harness/spec_manager.py`
    (1 chunk = 1 node, within-turn linking, GraphMerger cases, normalize, planet mass = satellite count,
    planet_mass_floor 0). Chunking = `management/harness/chunking.split_candidates` on the round trip's
    text (human + assistant text + tool_use inputs + tool_result text, in order).
D5. Full accounting of the manager: per Jev call {question kinds, input_tokens, output_tokens from
    `usage`, latency_ms, http_status, retries}; per summarizer call {prompt_tokens, completion_tokens,
    latency_ms}; GPU sampler (E4) running during the manager phase. Cost = input_tokens × $0.042/M
    (published price, recorded as `published_price_per_mtok`).

## E. Compute measurement (Ryosuke: measure everything that is free to measure)
E1. Per reader question: prompt_tokens, completion_tokens, window_tokens, attention FLOPs estimate
    (Σ over sdpa layers of 2·n_heads·seq_len²·head_dim for prefill + decode), wall ms (prefill/decode
    split), positions_found, bias_applied_calls, n_sdpa_layers.
E2. Per arm: total tokens processed, total wall time, summarization calls (C4), CD build cost (D5).
E3. GPU: `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw
    --format=csv -lms 1000` running for the whole phase, saved as CSV; per-question energy = ∫ power
    over the question's time span (Joules), reported alongside.
E4. Same sampler during the manager phase (summarizer GPU); Jev cost from D5.
E5. Reported primary compute metric: reader prompt tokens per question (deterministic). Secondary:
    attention FLOPs, wall time, energy, Jev+summarizer cost. All in the results table.

## F. Environment and execution
F1. RunPod pods, operated by Fable over SSH from this machine (Ryosuke supplies the key path; key
    contents never enter the chat/logs). Long jobs run under `tmux` on the pod; Fable polls.
    Destructive pod operations (stop/terminate/delete volume) require Ryosuke's approval each time.
F2. Pod A (main): RTX A6000 48 GB — manager (summarizer 4B) + proposed + baselines B/C. Weights
    ~19.4 GB (reader INT4) + ~8 GB (4B BF16) leave ~20 GB for KV/activations at W ≤ 32k.
    Pod B (baseline A only): 80 GB class (A100/H100) for the 410k-token prompt.
F3. Phases: (1) redaction (local) → gate; (2) fact ledger + questions (local) → gate; (3) manager phase
    on pod A → CD JSON + accounting → gate (Ryosuke inspects the tree); (4) smoke test on pod A (w grid,
    timing, injection counts) → numeric gate; (5) main run pods A+B; (6) scoring + report (local).
F4. Smoke-test pass criteria (numbers fixed before the run): reader loads via the injection path and
    `_attn_implementation == "sdpa"`; positions_found == number of planet lines; bias_applied_calls ==
    n_sdpa_layers (16) per decode step; planet attention share strictly increases with w on a 300-token
    probe; ≥ 3 w values with non-degenerate output; Jev defaulted/unparsed == 0 over ≥ 50 calls;
    summarizer node_fallback == 0 over ≥ 50 chunks; measured seconds/question recorded for both pods.
F5. Outputs (all published after redaction review): redacted transcript, fact ledger, questions, CD
    JSON + manager accounting, per-arm answers JSON + meta, GPU CSVs, scores.csv, decisions ledger,
    run manifest with git sha and versions.

## G. Non-goals for this run
Streaming/time-distance questions (v2), multiple sessions, any fallback judge, any use of confidence
thresholds, any cap on mass (bias_cap None), any external API other than Jev.

## H. Amendments proposed 2026-09-17 (measured after freezing; NOT yet approved by Ryosuke)
H1. Corpus = 186,570 Qwen3.8 tokens (37 round trips), not ~410k: the raw log counted base64 images,
    duplicated toolUseResult copies, thinking and harness notes. See DESIGN.md §0.1.
H2. C2: baseline A needs NO YaRN (186.6k < max_position_embeddings 262,144). Run with native RoPE.
    YaRN keys documented in DESIGN.md §0.2 for a future larger corpus.
H3. F2: a separate 80 GB pod for baseline A is no longer required by the corpus size. Ryosuke chooses
    DESIGN.md §1 variant α (A6000 + INT4 + transformers 5.8.0 pin, two venvs) or β (one 80 GB pod,
    BF16, transformers 5.17.0). Fable recommends β.
H4. C1 fallback wording: "transformers 5.8 cannot load compressed-tensors" is now precise — 5.8.0 keeps
    INT4 packed (per-forward dequant); 5.17.0 decompresses to BF16 on first forward (OOM on 48 GB).
H5. D3 detail: Noul decision = noul >= 0.5; Score level = argmax of `probabilities` (tie → lower index);
    `score`/`confidence` recorded only. Jev `state` is always the RAW chunk, never a generated summary.
H6. C8 prompt: implementation reuses `run_reader.READER_PROMPT` verbatim (raw completion prompt ending
    in "Answer:", no chat template, no enable_thinking flag), identical across arms. Alternative =
    wrap in the Qwen chat template with enable_thinking=False (touches run_reader/mass_weighted_gemma).
    Fable recommends keeping the raw prompt and letting smoke criterion F4 (non-degenerate short
    answers) decide; switch only if F4 fails.
H7. C4 oversized round trip (largest = 23,670 Qwen tokens > 8k): baseline C accepts the round trip and
    immediately folds it into the summary (one extra summarization call, counted); if the summary alone
    cannot fit W it raises. Baseline B at W=8k simply cannot hold it (whole round trips only, C6).
H8. D3 clarification: the `sun` Choice question is used only when every candidate is a current sun
    node text (GraphMerger sun matching); planet/satellite candidates always use the `belongs` Noul
    one-to-one.
H9. build_cd supports --resume from a partial CD (same session sha256) and per-run accounting files;
    transport errors and 5xx other than 429/529 still stop immediately (D1 unchanged; Ryosuke may
    widen the retry set).
H11. F4 field names: `planet_spans` (markers located) must equal `planet_lines` (`[PN` lines in the window);
    `positions_found` counts token positions and is recorded only. Baselines A/B/C run with inject=none
    (no marker scan). Artifacts (cd.json, compaction) carry session/questions sha256 + model_id + W and
    run_arms refuses mismatches. (Codex gpt-5.6-sol review, 2026-09-18.)
H10. A 2xx Jev response without integer usage stops the run (D5); integral floats accepted.
H12. Re-redaction 2026-09-18 after a Codex (gpt-5.6-sol) privacy audit + Fable sweep. Newly hidden, each
    verified in the text by Fable: a real-name fragment (6 occurrences), the unrelated project's name inside compound identifiers, and that
    project's chat-platform ids (3),
    the VPN's address prefix typed in a search pattern, claude.ai session id, git commit ids of the
    user's own repos (40-hex → stable fakes, 7-hex kept consistent; Paper build ids kept), world seed,
    personal circumstances (country, family members and ages all generalised), broken path
    `Users/the user`. NOT hidden (verified public or by design): hackathon name/date/hashtags and the
    submission text (the corpus premise), hardware specs, service account `claude`/`/home/claude`, ssh key
    file name and sudo note, Claude Code internal agent/task ids, public schematic names, the user's home
    directory listing (project folder names; optional, Ryosuke may ask to hide). Ledger re-verified 91/91,
    99 questions unchanged; corpus 186,557 Qwen tokens.
H13. Ryosuke 2026-09-18: the RT00 tool_result that read two private memory files (other project's server
    operations, family/location) is REPLACED by a fictional server note keeping only the machine specs;
    all access details are fictional (VPN, `ssh ops@demo-server`, key auth, `/home/ops`), the second
    memory file is dropped, `cf-token.txt` removed from the home listing. Reason: the published context
    must not disclose real server access or the unrelated project. Ledger 91/91 + 8/8 unchanged.
H14. Third Codex pass (2026-09-18): access method made fictional everywhere (VPN product, interface name
    and firewall tool all generalised), the country name generalised, the live machine's RCON password file path renamed (the hackathon
    corpus keeps its own, already-fictional path, which 6 facts depend on), the other project's old-codename plugin file line removed. Kept after check:
    server country "Japan" (fact f009 and 3 other facts depend on it; not an access detail), `User=minecraft`
    in the hackathon's own systemd unit, the requirements document's negative list ("no Discord bridge…").
H15. (Codex GPT-6 Astra round 1, 2026-09-18) Decode-only injection (C7) cannot influence the FIRST answer
    token: its logits come from the prefill forward, where the bias is off. With "answer in a few words"
    and greedy decoding, the first token often decides the answer. Options for Ryosuke: (a) keep C7 and
    accept that the mass term acts from the 2nd token on; (b) add a switch "last-row prefill injection":
    during prefill the bias is added ONLY to the last query row (the token that produces the first answer
    token), context encoding untouched — cheap to implement in the SDPA patch, to be evaluated in the smoke
    test alongside the w grid. Fable recommends implementing (b) as a switch and measuring both.
H16. H5 wording clarified: "raw chunk" applies to the keep and the three axis questions; the pairwise
    same/belongs questions and the sun Choice use the NODE TEXTS (summaries), as DESIGN §3 examples show.
H17. Astra round-1 items accepted and being fixed: verified model class + loading_info; prefill/decode phase
    tracking (1-token final chunk); cd_parser newline-token span loss; gpu_preflight decode(pos) bug;
    scorer abstention/short-numeric/all-of aliases; loose ledger aliases; decode forwards = n−1; partial CD
    refused; per-question and per-turn checkpoints; summarizer single-line validation; compaction sampler;
    weight bytes required in check_context_fits; KV at 187k = 12.3 GB (DESIGN §0.3 corrected).
    Rejected: "w never applied in run_reader" (load_reader sets llm._mass_weight = w, line 642; the concern
    applies only to gpu_preflight's micro-run, which this experiment does not use); "chunked prefill slices
    4-row position ids on the batch axis" (our loader gets Qwen3_5ForCausalLM, whose text model builds
    position ids from the cache offset itself; verified locally with a tiny checkpoint: 0 missing keys).
H18. Astra round 2 (2026-09-18): applied — GQA expansion when a mask is added (K/V repeated in their own
    dtype, original SDPA called with enable_gqa=False: no float32 math fallback at 190k keys); patch
    idempotence/ownership (module-level _ORIGINAL_SDPA captured at import, one owner, RuntimeError on a
    second instance, restore hands back the import-time kernel); recorder uses the prefill/decode PHASE
    and refuses seq_k > RECORD_MAX_KEYS = 4096; single bias cap (mass vector uncapped, the only cap is
    the effective-bias cap in build_mass_bias: mass 10, w 0.5, cap 3 -> 3, was 1.5); full checkpoint
    identity (inject, prefill_scale, max_new_tokens, prefill_chunk, questions/cd/compaction sha256,
    prefill_last_row) with --resume refusing on any mismatch; H15 switch `prefill_last_row` implemented
    OFF by default (DESIGN §7.6) pending Ryosuke's decision. Rejected — fully-masked-row finfo.min: this
    experiment runs batch 1 without padding, so no row is ever fully masked (documented, not changed).
H19. DECISION NEEDED (Astra round 3, 2026-09-19): planet-candidate matching cost. The corpus has 3,508 chunks
    (max 423 in one round trip). With per-candidate Noul for every orphan against every existing planet
    (spec 0060 one-to-one, D3 + H8 as implemented), the worst case is O(planets²): ≈138k Jev requests
    (≈10 h at 250 ms) if 30 % of chunks are kept, ≈385k (≈27 h) at 50 %, with the GPU idle meanwhile.
    Options: (a) keep one-to-one Noul (spec-literal, slow, budget guard stops the run); (b) batched
    Choice over planet candidates (≤254 per request, planet-appropriate wording "which item is A a
    detail of / none") ≈1.6k–3.5k requests (<15 min) — same decision form Jev already gives for suns;
    (c) local embedding shortlist (sentence-transformers, no external API) of k=5 then Noul ≈5k–9k
    requests (<40 min) — this is the ORIGINAL harness behaviour (SimilarityJudge shortlist_k=5). Fable
    recommends (b) for suns and planets alike, with (c) as fallback if Jev Choice quality on long lists
    is poor in V4. Ryosuke decides.
H20. Astra round 3 (2026-09-19): applied — Jev request/token budget guard + request projection CLI,
    capacity exhaustion raises, node cache bypassed for the Jev hook, manifest totals from accounting
    logs (+ resume reconciliation), probability validation, exact-request tests, real-client
    integration test, serializer wrapper budget check. Deferred to Ryosuke (H19): planet-candidate
    matching strategy.
H21. [Fable's proposals from Astra round 4, 2026-09-20 — NOT adopted; listed for Ryosuke to accept or
    discard item by item; items that assume baselines B/C or extra control arms are moot under C3.] (1) RT36 is the session retrospective: many golds are present verbatim in the last 3–10
    round trips, so baseline B at any W already "knows" them; primary evaluation should use a
    pre-retrospective snapshot (round trips 00–35, or 00–33) and report older-only vs recent-answerable
    strata. (2) Add controls: proposed with w=0 (identical window, bias off), CD text with inject=none,
    matched recent-only, no-context floor — without them the bias effect cannot be separated from the
    CD-content effect. (3) Statistics: 99 paired questions detect ~8–16 pp; parity needs a one-sided
    non-inferiority bound (PROTOCOL.md rule), not "no significant difference"; freeze the primary W/w
    cell before scoring (Holm for the rest). (4) Compute: arm A is a cold 187k prefill per question
    (no prefix reuse) — label it so; report preparation cost (Jev $, 4B GPU J, C's reader summarization)
    separately with N=99 and break-even N; rename `attn_flops_*` to `sdpa_qk_flops_proxy` (48 linear
    layers, AV, MLP omitted). (5) Facts f089–f091 live in tool output that the `keep` question tells the
    manager to drop — either retain factual values in tool output or pre-register a tool-output stratum.
    (6) Absent probes are 8/99 = 8.1 % (<10–20 % of B2) and verified by keyword absence only.
    (7) spec_manager.build_provisional add_sun() False still silent (unreachable: cap 10000 > 423 chunks).
H22. Ryosuke decided 2026-09-20: (a) planet candidates are matched with Choice batches like suns (H19 = b);
    (b) decode-only bias stays as the first configuration (H15 = a; the prefill_last_row switch remains
    OFF, available for a later test); (c) the retrospective round trip 36 is excluded from the corpus;
    (d) questions whose evidence lies inside the proposed method's window are excluded per cell, and the
    baseline is scored on the same subset; (e) the manager keeps facts found in tool output (keep wording
    changed). Fable's earlier '10–20 % questions with no answer in the corpus' was never decided by
    Ryosuke — the 8 such questions stay as they are, labelled 「答えが本文に無い質問」.
H23. Ryosuke decided 2026-09-20: the same-matter judgement (spec 0042) is made with Choice batches
    (≤254 candidates) like the belongs/sun decisions; single candidate stays Noul. Jev spend for the
    first run stays within the existing 5 USD balance (budget guard); GPU spend proceeds.
H24. (2026-09-20, pod A40 46 GB) Variant α as written in DESIGN §1 is WRONG: compressed-tensors ≥ 0.15
    removed the per-forward dequantizing `CompressedLinear` ("no longer supported"); with 0.18 the INT4
    modules stay plain `Linear` holding weight_packed/weight_scale/weight_shape and `compress_model`
    registers a pre-forward hook that rebuilds the whole model in BF16 on the first forward (V2 OOM at
    44 GB, "Decompressing model 251/416"). Removing the hook does not help (no `weight` → AttributeError,
    peak 22 GB packed but not runnable). transformers 5.8.0 was not the deciding factor.
    Repair options being tested in order: (1) compressed-tensors 0.14.0 (last release with a working
    CompressedLinear, per-forward dequant) in a separate venv; (2) fallback: bitsandbytes NF4 on the
    BF16 checkpoint Qwen/Qwen3.8-27B (per-matmul dequant kernels, ~16 GB weights), skipping
    in_proj_a/b and lm_head like the RedHat recipe. Both arms use the same reader either way.
H25. (2026-09-20, A40 pod) Repair option (1) of the previous entry FAILED too: compressed-tensors 0.14.0 forced
    in beside transformers 5.8.0 (its pin is transformers<5) needed the checkpoint's kv_cache_scheme
    dropped, then still expanded to BF16 at the first forward (OOM at 43 GB). The INT4 checkpoint and that
    venv were deleted from the pod. The reader for this run is therefore option (2): Qwen/Qwen3.8-27B BF16
    loaded with bitsandbytes NF4 (4-bit weights in memory, per-matmul dequant; in_proj_a/b and lm_head in
    bf16), `--quantization nf4` on every reader CLI, recorded in meta and the checkpoint identity. Both
    arms use this same reader. Pod storage: the /workspace volume has an ~80 GB quota (hf cache), venvs
    live on the root overlay disk (/root/venvs, symlinked).
H26. Ryosuke decided 2026-09-20: w grid for the main run = {0.1, 0.3, 1.0} (V3 on the A100: planet
    attention share 5.5 % / 7.9 % / 21.9 %, answers intact; 3.0 overwrites the answer with planet text).
    Windows stay 8k / 16k / 32k (C6). Reader = bf16 Qwen/Qwen3.8-27B on the A100 80GB pod (V2: peak
    50.8 GB, 0.087 s/token).
H27. (2026-09-20) vLLM cannot run on this pod: 0.27/0.28 pin torch 2.13 (cu130 only) and the 0.26.0 wheel
    itself links libcudart.so.13; the driver is CUDA 12.8. The summarizer (Qwen3.5-4B) is therefore served
    by a small OpenAI-compatible HTTP server on transformers (pod/summarizer_server.py) inside the reader
    venv — same request/response contract (usage, finish_reason), batch 1. Slower than vLLM; counted in
    the manager's GPU accounting like before.
H28. (2026-09-20, Ryosuke: "勝手に規則は作らずに、動いてることを確認出来たらすぐ本テストに入って") V4 on the
    A100 stopped at chunk 52: Qwen3.5-4B echoed a Windows PATH listing twice (177 / 193 chars > 120).
    Measured on 300 random corpus chunks with the DESIGN §8 prompt: 287 ok, 2 ok after the retry, 11 (3.7 %)
    unusable — about 120 stops over the 3388 chunks of the 36 round trips. The "stop the run" clause of D2
    was Fable's addition, not a rule of the spec manager: management/harness/spec_manager.py has always
    handled a node_fn that returns None by counting node_fallback and using the truncated chunk text.
    Adopted (the pre-existing behaviour, no new rule): SummarizerClient raises SummarizerNodeTextUnusable
    (a SummarizerStop subclass) after the retry, JevNodeFn maps it to None, the manager counts it in
    harness_quality.node_fallback (reported in the manifest and the paper). Unreachable summarizer / HTTP
    errors still stop the run. The V4 gate "SummarizerStop == 0" becomes "node_fallback reported".
    A prompt variant Fable tried in the same measurement (system role + "shorten your own answer") was
    worse (35 / 300 unusable) and is NOT adopted; the §8 prompt is unchanged.
H29. (2026-09-20, Fable's execution choice, not a rule; Ryosuke may overrule) The main run answers arm A
    once on all 96 questions (`--W full`) and applies each proposed cell's questions_subset.json at
    scoring time, instead of three arm-A runs with `--questions-subset`. Questions are answered
    independently with greedy decoding, so the per-question answers are identical; it saves two
    full-transcript passes (~1-1.5 h each). The subset sha is therefore recorded by the scorer, not in
    the arm-A checkpoint identity. Order on the pod (pod/main_run.sh): V5 (arm A, 5 questions) →
    proposed × W {8000,16000,32000} × w {0.1,0.3,1.0} (H26) → arm A on 96.
H30. (2026-09-21) The H6 ALTERNATIVE is taken, exactly as H6 pre-decided ("switch only if F4
    fails"). F4 failed on the A100: with the raw completion prompt Qwen3.8-27B opens a `<think>`
    block, so of the first three arm-A answers one was `<think>

</think>

AI Tinkerers Global
    Hackathon 2026-09-12` (correct, but `first_line_answer` returned "<think>") and two spent all
    48 tokens inside the block. Implemented: (a) `run_reader.build_prompt(..., tokenizer=...)`
    wraps the UNCHANGED `READER_PROMPT` as one user turn of the model's chat template with
    `enable_thinking=False`, falling back to the raw prompt when the tokenizer has no template or
    no switch; (b) `first_line_answer` removes a leading `<think>...</think>` block and never
    returns a bare tag; (c) W is measured on the wrapped string everywhere (`windows.build_window`,
    `budget_question`, `check_all_questions_fit` all take `chat_template`), so the budget stays
    honest; (d) `chat_template` is part of the checkpoint identity and of meta.json, default ON in
    run_arms, `--no-chat-template` restores the old prompt. The prompt TEXT, max_new_tokens 48,
    greedy decoding and the arm-invariance of C8 are unchanged.
H31. (2026-09-21, Ryosuke: "じゃあさっきの修正はそのままで、カーネルを導入して読み込みを早くして")
    (a) The chat-template prompt of H30 is APPROVED by Ryosuke. Note for the record: H30 described it
    as pre-decided, which was wrong — the H section header says "NOT yet approved by Ryosuke" and the
    prompt item reads "Fable recommends", so it was a Fable proposal until this message. Ryosuke was
    shown the three options (chat template / literal `<think></think>` suffix / larger decode budget)
    and chose to keep the chat template.
    (b) `causal_conv1d` is installed in the reader venv (built from the GitHub source tree of
    v1.5.0.post8, TORCH_CUDA_ARCH_LIST=8.0, --no-build-isolation with `uv pip` so the VENV torch is
    used — plain `pip` inside this uv venv resolves to /usr/local/bin/pip and builds against the
    SYSTEM torch, which fails). Measured before the change on the A100: prefill 809 tok/s
    (179k tokens = 222 s per question), decode 12.3 tok/s, i.e. ~6 h for the 96-question
    full-transcript arm. The 48 linear-attention layers were running the reference PyTorch
    convolution. Every timing / energy figure of the main run is taken AFTER this install, so the
    whole run is measured on one configuration; the earlier V5 numbers are discarded.
H32. (2026-09-21, Ryosuke: "下駄なしの実験を回してから止めよう") A w = 0 control column is added to the
    grid: arm proposed, W in {8000, 16000, 32000}, w = 0.0, same CD, same prompt, mass vector scaled
    by zero (mathematically no bias; the w > 0 guards in run_arms/run_reader are skipped by design).
    Reason: in the completed grid the bias was monotonically harmful at every window size
    (W=32000: 63 -> 58 -> 27 passes for w = 0.1 / 0.3 / 1.0), so the run as it stands cannot separate
    "narrowing the window onto the diagram helps" from "the bias hurts". Without this column the only
    defensible statement is "lower w is better", which is not a statement about the patent's
    mechanism. Cells land in the same results tree as proposed_W<W>_w0.0 and are scored by the same
    tool. The pod is stopped by Ryosuke once these three cells are saved.
H33. (2026-09-21) Result of the H32 control, and the state in which the pod is released.
    w = 0 and w = 0.1 are IDENTICAL at all three window sizes: 29/29, 41/41, 63/63 passes, zero
    questions gained and zero lost in the paired comparison, and the generated answer STRING is
    byte-identical on 96/96 (W=8000), 95/96 (W=16000) and 94/96 (W=32000) questions, with the few
    differing strings scoring the same. So in this run the attention bias contributes nothing at
    w = 0.1 and is purely harmful above it (W=32000: 63 -> 58 -> 27 for w = 0.1 / 0.3 / 1.0).
    The 56 of 88 facts recovered at W=32000 come from narrowing the window onto the correlation
    diagram, not from the bias. Not separable by this run: whether no useful bias band exists at
    all, or whether the decode-only restriction (C7 / prefill_scale 0.0) removes it — the planet
    text covers 4.4 % (W=8000) to 16 % (W=32000) of the window and generation is ~6 tokens, so
    w = 0.1 cannot move the post-softmax distribution while w = 0.3 already distorts the answer.
    A prefill-side bias run (the switch exists, H15 (b) / H18) was offered and NOT run: Ryosuke
    chose to stop here ("ここで止めるからデータの保存を完了させて").
    Everything under /workspace/mcb/runs was verified file by file against the local copy by sha256
    (140 files) before release; the scripts as executed are saved under pod_scripts_as_run/.
