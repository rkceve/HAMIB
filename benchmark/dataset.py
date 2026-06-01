"""
Fixed test conversation dataset for benchmarking.

Design:
  - Embed a "fact" into the conversation (fact turn)
  - A few turns later, ask about that fact (recall turn)
  - Whether the correct keyword appears in the recall-turn response -> consistency score

Turn types:
  "regular"  : ordinary conversation (no fact)
  "fact"     : an utterance that embeds a fact (phrased naturally as conversation)
  "recall"   : a question that asks about a past fact
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class Turn:
    turn_id: int
    type: Literal["regular", "fact", "recall"]
    user: str
    # Used by fact/recall only
    fact_key: str = ""
    fact_value: str = ""          # fact: value to embed / recall: expected answer
    fact_introduced_at: int = -1  # for recall, the turn where the fact was introduced


# -- Test data --------------------------------------------------------
# A 30-turn conversation. Five facts are embedded, each asked back 10-15 turns later.

BENCHMARK_TURNS: list[Turn] = [
    # --- Turn 0: fact A (name) ---
    Turn(0, "fact",    "My name is Alex Carter. Nice to meet you.",
         fact_key="name", fact_value="Alex Carter"),
    Turn(1, "regular", "I've recently started getting interested in machine learning."),
    Turn(2, "regular", "I'm especially curious about how neural networks work."),
    # --- Turn 3: fact B (favorite food) ---
    Turn(3, "fact",    "My favorite food of all is curry rice.",
         fact_key="food", fact_value="curry rice"),
    Turn(4, "regular", "Could you explain backpropagation to me?"),
    Turn(5, "regular", "What kinds of activation functions are there?"),
    Turn(6, "regular", "How can I prevent overfitting?"),
    # --- Turn 7: fact C (hometown) ---
    Turn(7, "fact",    "I'm from Osaka.",
         fact_key="hometown", fact_value="Osaka"),
    Turn(8, "regular", "What is dropout?"),
    Turn(9, "regular", "What is the difference between CNNs and RNNs?"),
    Turn(10, "regular", "Can you give me an overview of the Transformer architecture?"),
    # --- Turn 11: recall A (name) ---
    Turn(11, "recall",  "By the way, do you remember my name?",
         fact_key="name", fact_value="Alex Carter", fact_introduced_at=0),
    Turn(12, "regular", "How does the attention mechanism work?"),
    Turn(13, "regular", "What is the difference between BERT and GPT?"),
    # --- Turn 14: fact D (hobby) ---
    Turn(14, "fact",    "My hobby is playing the guitar.",
         fact_key="hobby", fact_value="guitar"),
    Turn(15, "regular", "What is the difference between fine-tuning and transfer learning?"),
    # --- Turn 16: recall B (food) ---
    Turn(16, "recall",  "What was my favorite food again?",
         fact_key="food", fact_value="curry rice", fact_introduced_at=3),
    Turn(17, "regular", "What kind of technique is LoRA?"),
    Turn(18, "regular", "How does quantization affect model inference?"),
    Turn(19, "regular", "What is nf4 quantization?"),
    # --- Turn 20: recall C (hometown) ---
    Turn(20, "recall",  "Where was I from again?",
         fact_key="hometown", fact_value="Osaka", fact_introduced_at=7),
    Turn(21, "regular", "Could you tell me about the MoE architecture?"),
    # --- Turn 22: fact E (occupation) ---
    Turn(22, "fact",    "My job is software engineer.",
         fact_key="job", fact_value="software engineer"),
    Turn(23, "regular", "How does RAG work?"),
    Turn(24, "regular", "What is the role of a vector database?"),
    Turn(25, "regular", "How does the KV cache work?"),
    # --- Turn 26: recall D (hobby) ---
    Turn(26, "recall",  "What was my hobby again?",
         fact_key="hobby", fact_value="guitar", fact_introduced_at=14),
    Turn(27, "regular", "What is flash attention?"),
    Turn(28, "regular", "Can you explain speculative decoding?"),
    # --- Turn 29: recall E (occupation) ---
    Turn(29, "recall",  "Can you tell me my occupation?",
         fact_key="job", fact_value="software engineer", fact_introduced_at=22),
]

RECALL_TURNS = [t for t in BENCHMARK_TURNS if t.type == "recall"]
