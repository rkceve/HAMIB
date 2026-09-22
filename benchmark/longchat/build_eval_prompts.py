"""Build evaluation prompts for the long-chat benchmark."""
import json
import tiktoken
from pathlib import Path

ROOT = Path(__file__).parent
enc = tiktoken.get_encoding("cl100k_base")


def session_text(s):
    parts = [f"--- Session {s.get('n')} ({s.get('date','')}) ---"]
    for t in s.get("turns", []):
        parts.append(f"{t['role'].capitalize()}: {t['content']}")
    return "\n".join(parts)


def format_questions(questions):
    lines = ["Answer each question in the format 'A<n>: <answer>'. Answer concisely (1-5 words). Give only the answer, no explanation."]
    for i, q in enumerate(questions, 1):
        lines.append(f"Q{i}: {q['question']}")
    lines.append("\nFormat:")
    for i in range(1, len(questions) + 1):
        lines.append(f"A{i}: [answer]")
    return "\n".join(lines)


def build_full(chat, questions):
    sessions_text = "\n\n".join(session_text(s) for s in chat["sessions"])
    return (
        "Below is a complete record of a long-running conversation between a user and an assistant about the user's restaurant opening journey. Read carefully and answer the questions at the end.\n\n"
        f"{sessions_text}\n\n"
        f"{format_questions(questions)}"
    )


def build_truncated(chat, questions, target_tokens=30000):
    sessions = chat["sessions"]
    kept = []
    tokens_so_far = 0
    for s in reversed(sessions):
        t = session_text(s)
        tk = len(enc.encode(t))
        if tokens_so_far + tk > target_tokens:
            break
        kept.append(s)
        tokens_so_far += tk
    kept.reverse()
    sessions_text = "\n\n".join(session_text(s) for s in kept)
    note = (
        f"Note: This is a truncated record. Earlier sessions (1 through {kept[0]['n']-1 if kept else 'N'}) have been dropped due to context limits. Only the most recent {len(kept)} sessions are shown.\n\n"
        if kept else ""
    )
    return (
        "Below is a partial record of a long-running conversation between a user and an assistant about the user's restaurant opening journey.\n\n"
        f"{note}{sessions_text}\n\n"
        f"{format_questions(questions)}"
    )


def build_summarized(chat, questions, summaries, keep_last=5):
    sessions = chat["sessions"]
    n_keep = keep_last
    early = sessions[:-n_keep] if n_keep < len(sessions) else []
    late = sessions[-n_keep:]

    early_block = []
    for s in early:
        n = s.get("n")
        date = s.get("date", "")
        summary = summaries.get(str(n), summaries.get(n, "[summary unavailable]"))
        early_block.append(f"--- Session {n} ({date}) — SUMMARY ---\n{summary}")

    late_block = "\n\n".join(session_text(s) for s in late)

    return (
        "Below is a partial record of a long-running conversation between a user and an assistant about the user's restaurant opening journey. Earlier sessions have been summarized to fit within context limits; only the most recent sessions are shown in full.\n\n"
        + "\n\n".join(early_block)
        + "\n\n"
        + late_block
        + "\n\n"
        + format_questions(questions)
    )


if __name__ == "__main__":
    chat = json.loads((ROOT / "restaurant_chat_v2.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "restaurant_questions.json").read_text(encoding="utf-8"))

    full_prompt = build_full(chat, questions)
    trunc_prompt = build_truncated(chat, questions, target_tokens=30000)

    (ROOT / "eval_prompts").mkdir(exist_ok=True)
    (ROOT / "eval_prompts" / "full.txt").write_text(full_prompt, encoding="utf-8")
    (ROOT / "eval_prompts" / "truncated.txt").write_text(trunc_prompt, encoding="utf-8")

    print(f"Full prompt: {len(enc.encode(full_prompt)):,} tokens")
    print(f"Truncated prompt: {len(enc.encode(trunc_prompt)):,} tokens")
    print(f"Questions: {len(questions)}")
    print("Files written to eval_prompts/")
