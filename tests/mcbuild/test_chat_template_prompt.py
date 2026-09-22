"""H30 (the H6 alternative): chat-template prompt with thinking off, and think-block stripping.

Smoke criterion F4 failed on the pod with the raw completion prompt: Qwen3.8-27B opened a
``<think>`` block and spent the 48-token budget inside it, so ``first_line_answer`` returned
``"<think>"``.  These tests pin the two halves of the fix: the prompt is wrapped in the model's
chat template with ``enable_thinking=False``, and a think block never becomes the answer.
"""
from __future__ import annotations

from typing import Any

import pytest

from benchmark.bineval.run_reader import build_prompt, first_line_answer
from benchmark.mcbuild_bench.windows import build_window


class ChatTok:
    """Tokenizer stub with a chat template; one token per whitespace-separated word."""

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.thinking_flags: list[Any] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        self.thinking_flags.append(kwargs.get("enable_thinking", "absent"))
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": text.split()}


class OldTok(ChatTok):
    """A template without the enable_thinking switch (older tokenizers raise TypeError)."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword 'enable_thinking'")
        return super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


class NoTemplateTok(ChatTok):
    chat_template = None


def test_chat_wrap_disables_thinking_and_keeps_the_raw_prompt_inside() -> None:
    tok = ChatTok()
    raw = build_prompt("<context>\nfact\n</context>", "Port?")
    wrapped = build_prompt("<context>\nfact\n</context>", "Port?", tokenizer=tok)
    assert tok.thinking_flags == [False]
    assert raw in wrapped and wrapped != raw
    assert wrapped.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("tok_cls", [OldTok, NoTemplateTok])
def test_chat_wrap_falls_back_without_the_switch_or_the_template(tok_cls) -> None:
    raw = build_prompt("ctx", "Q?")
    wrapped = build_prompt("ctx", "Q?", tokenizer=tok_cls())
    if tok_cls is NoTemplateTok:
        assert wrapped == raw  # no template -> unchanged, still arm-invariant
    else:
        assert raw in wrapped  # template applied, thinking switch simply absent


def test_chat_wrap_is_arm_invariant() -> None:
    """The wrapper depends only on the tokenizer, so two arms with the same question
    differ exactly by their context block (C8)."""
    tok = ChatTok()
    a = build_prompt("<context>\nA\n</context>", "Port?", tokenizer=tok)
    b = build_prompt("<context>\nB\n</context>", "Port?", tokenizer=tok)
    assert a.replace("\nA\n", "\nX\n") == b.replace("\nB\n", "\nX\n")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\n\n<think>\n\n</think>\n\nAI Tinkerers Global Hackathon", "AI Tinkerers Global Hackathon"),
        ("<think>\nreasoning\n</think>\n25565\nbecause the port moved", "25565"),
        ("25565\nsecond line", "25565"),
        ("\n\n<think>\nunterminated reasoning", "unterminated reasoning"),
        ("<think></think>", ""),
    ],
)
def test_first_line_answer_never_returns_a_think_tag(raw: str, expected: str) -> None:
    assert first_line_answer(raw) == expected


def test_build_window_measures_W_on_the_wrapped_prompt() -> None:
    """The chat wrapper adds tokens, so the same W must keep FEWER round trips once it
    is on; both windows still satisfy the budget."""
    rts = [{"idx": i, "human": f"user text {i}", "events": [{"kind": "text", "text": f"assistant text {i}"}]}
           for i in range(6)]
    kw = dict(arm="B", W=60, cd_block=None, round_trips=rts, tokenizer=ChatTok(), question="Port?")
    plain = build_window(**kw)
    wrapped = build_window(**kw, chat_template=True)
    assert plain["window_tokens"] <= 60 and wrapped["window_tokens"] <= 60
    assert wrapped["n_recent_rts"] < plain["n_recent_rts"]
    assert wrapped["prompt"].startswith("<|im_start|>user")
    # prompt_context is the bare context block either way: run_reader rebuilds the wrapper
    assert not wrapped["prompt_context"].startswith("<|im_start|>")
