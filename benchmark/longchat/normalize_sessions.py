"""Normalize all session files into a consistent format and report stats."""
import json
import os

SESSIONS_DIR = r"benchmark/longchat/sessions"


def extract_content(turn):
    """Extract content text from a turn regardless of which key it uses."""
    if not isinstance(turn, dict):
        return None, None
    role = turn.get("role", "")
    role = role.lower() if role else ""
    # Try common content keys
    for key in ("content", "text", "message", "body", "msg"):
        if key in turn and isinstance(turn[key], str):
            return role, turn[key]
    return role, None


def normalize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "turns" in data:
        turns = data["turns"]
    elif isinstance(data, list):
        turns = data
    else:
        return None, "unknown_format"
    normalized = []
    for t in turns:
        role, content = extract_content(t)
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized, None


def main():
    files = sorted(os.listdir(SESSIONS_DIR))
    total_words = 0
    total_chars = 0
    issues = []
    for f in files:
        path = os.path.join(SESSIONS_DIR, f)
        normalized, err = normalize_file(path)
        if err or not normalized:
            print(f"{f}: ERROR {err}")
            issues.append(f)
            continue
        # write back in normalized form
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(normalized, fp, ensure_ascii=False, indent=2)
        words = sum(len(t["content"].split()) for t in normalized)
        chars = sum(len(t["content"]) for t in normalized)
        total_words += words
        total_chars += chars
        flag = ""
        if words < 3000:
            flag = "  *** SHORT ***"
        print(f"{f}: turns={len(normalized)} words={words} chars={chars}{flag}")
    print(f"\nTOTAL: words={total_words} chars={total_chars} approx_tokens={total_chars//4}")
    if issues:
        print(f"\nIssues: {issues}")


if __name__ == "__main__":
    main()
