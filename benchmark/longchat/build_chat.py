"""Build the final restaurant_chat.json from individual session files and verify facts."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
SESSIONS_DIR = ROOT / "sessions"
SKELETON_PATH = ROOT / "restaurant_skeleton.json"
OUTPUT_PATH = ROOT / "restaurant_chat.json"

# Required facts per session (each must appear verbatim, case-insensitive).
REQUIRED_FACTS = {
    1: ["Kenta Morishita", "Marubeni Corporation", "11 years", "28 million yen", "Ginza Trattoria", "Bologna", "Hotel School of Bologna", "Ayako"],
    2: ["14 Italian restaurants", "Ristorante Aso", "Daikanyama", "Faro", "Heinz Beck", "Marunouchi", "8,500 yen", "1,800 yen", "Emilia-Romagna", "Ginza Trattoria"],
    3: ["4.2 million yen", "38 covers", "Hiroshi Tanabe", "Tanabe & Partners", "2-7-3 Shibuya", "Daiichi-Bunko Building 5F", "Japan Finance Corporation", "15 million yen", "1.85%"],
    4: ["23 properties", "Yusuke Hamada", "Plaza Homes Commercial", "Daichi Sasaki", "Nishi-Azabu", "Higashi-Azabu", "Hiroo", "Yoyogi-Uehara", "Kagurazaka", "95,000 yen per tsubo"],
    5: ["3-14-7 Higashi-Azabu", "Minato-ku", "Azabu Heights Building", "64.5 square meters", "19.5 tsubo", "487,000 yen", "73,000 yen", "2,922,000 yen", "4 years renewable", "October 2024"],
    6: ["Kenchiku Plus", "Yamamoto Koumuten", "Sasaki Design Build", "Tetsuo Yamamoto", "open kitchen", "brick-faced pizza oven", "8.4 million yen", "Forno Bravo Japan", "wood-fired oven"],
    7: ["Japan Finance Corporation", "12 million yen", "1.95%", "36 million yen", "interior finishing"],
    8: ["June 9, 2025", "August 22", "white oak engineered plank", "Toyo Wood Works", "limestone-finish plaster", "Marmorino", "Tokai Sangyo", "single accessible unisex"],
    9: ["Forno Bravo Japan", "Modena 100", "1.6 million yen", "La Monferrina P3", "580,000 yen", "Reizou Sangyo", "Hoshizaki", "720,000 yen", "rotisserie"],
    10: ["Solleone Japan", "Frantoio Franci", "4,200 yen per 500ml", "Caputo 00", "Suzuki Foods", "3,800 yen per 25kg sack", "Tsukiji-Ota Market", "Kenji Ueno", "Ueno Seika", "Tanabe"],
    11: ["Nakata Shoji", "Mariko Nakata", "Reiko Igarashi", "280 bottles", "47 labels", "60% Italian", "1.85 million yen"],
    12: ["Ginza Trattoria", "Osteria Morishita", "Aiko Fujimoto", "320,000 yen", "terracotta and ivory", "osteria-morishita.tokyo"],
    13: ["Marco Petrelli", "38 years old", "Modena", "Ristorante Acqua Pazza", "Daikanyama", "4 years", "720,000 yen", "13-month bonus", "September 1, 2025", "Mariko Nakata"],
    14: ["August 26", "8.87 million yen", "470,000 yen", "electrical panel", "Ogawa", "Minato Ward Health", "handwashing", "grease trap"],
    15: ["6 antipasti", "5 primi", "3 pasta", "2 risotto", "4 secondi", "3 desserts", "tagliatelle al ragu bolognese", "2,400 yen", "hand-cut", "1,800 yen", "7,800 yen"],
    16: ["Haruki Nakajima", "29 years old", "Trattoria Goccia", "380,000 yen", "Eiko Sawada", "41 years old", "460,000 yen", "Yui Tachibana", "Naoki Hirose", "5 full-time, 2 part-time"],
    17: ["October 8-11, 2025", "dinner only", "80 total", "20 per night", "feedback forms", "wine pairing"],
    18: ["38 minutes", "pasta water station", "73 of 80", "Akira Mochizuki", "Tokyo Calendar", "table 7"],
    19: ["November 4, 2025", "second induction burner", "180,000 yen", "felt panel", "Hot Pepper Gourmet", "Tabelog editorial", "Tokyo Calendar", "Nikkei Style", "Foodie Tokyo", "TableCheck", "32,000 yen monthly"],
    20: ["November 4", "22, 26, 24", "November 5", "14, 18", "8,100 yen", "1,950 yen", "3.42"],
    21: ["Naoki Hirose", "Sara Komatsu", "November 17", "back pain", "Mrs. Yoneda", "58"],
    22: ["3.42 million yen", "food 31%", "labor 38%", "312,000 yen", "14 months", "Monday lunch"],
    23: ["Marubeni Corporation", "24 guests", "12,000 yen per person", "4 events", "1.1 million yen", "Mrs. Yoneda", "1,400 yen per hour", "4 hours nightly", "5 courses 9,800 yen", "4,500 yen"],
    24: ["4.95 million yen", "38%", "1 month salary", "30,000 yen", "2-week vacation", "Modena", "Sawada", "wine club"],
    25: ["42%", "Square Plus inventory module", "18,000 yen monthly", "8% food waste", "4%", "branzino", "guinea fowl"],
    26: ["Cantina Morishita", "36,000 yen annual", "monthly tasting dinner", "10% off bottle list", "40 members", "Sawada", "Mariko Nakata"],
    27: ["February 14-28, 2026", "Nakajima", "80,000 yen", "Toshiya Inoue", "culinary school", "guinea fowl", "February 17-21"],
    28: ["24 minutes", "3.61", "24 reviews", "Hiroko Sano", "Asahi Shimbun Weekly", "Sawada", "late March"],
    29: ["March 1", "Acetaia Giusti", "18%", "11.8 million yen", "12.5 million", "23 members", "4 menu items"],
    30: ["March 19, 2026", "Asahi Shimbun Weekly", "Hiroko Sano", "Azabu's Quiet Modena", "ragu bolognese", "unfussy precision", "4x", "3 weeks out"],
    31: ["April 1", "9,200 yen", "2,400", "2,700 yen", "2,000 yen", "2,600 yen", "Frantoio Franci", "Solleone", "22%"],
    32: ["Modena 100", "April 13", "Forno Bravo Japan", "Mr. Saito", "340,000 yen", "5-day downtime", "Tokio Marine Restaurant Plus", "70%", "focaccia served as crostini"],
    33: ["May 4", "510,000 yen", "4% of revenue above 4.5M", "42 members", "second seating"],
    34: ["May 4", "30 regulars", "412 unique guests", "87 repeat customers", "Hiroyuki Tachibana", "10 visits", "Azabu resident", "28.4 million yen", "1.1 million yen"],
    35: ["Hamada", "Tomigaya", "18.2 tsubo", "580,000 yen", "Modena focus preserved", "September 2026", "private dining room"],
    36: ["8-seat private room", "Yamamoto Koumuten", "1.4 million yen", "38,000 yen monthly", "July 14 to August 9", "Sala Modena"],
    37: ["7 full-time", "4 part-time", "social insurance", "Mr. Endo", "280,000 yen", "July 1", "Nakajima"],
    38: ["7 FT staff", "440,000 yen", "1,500 yen per hour", "200,000 yen at 24 months", "two-year extension", "October 2027", "780,000 yen"],
    39: ["Sala Modena", "6 private events", "Marubeni", "October 14", "14-person dinner", "18,000 yen per person", "venue fee 50,000 yen", "6,500 yen"],
    40: ["March 2025", "17 months", "3.78", "91 reviews", "July 2026", "Sala Modena", "August 12, 2026", "November", "cookbook", "Azabu"],
}


def load_turns(n):
    path = SESSIONS_DIR / f"session_{n:02d}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "turns" in data:
        return data["turns"]
    if isinstance(data, list):
        return data
    return []


def main():
    with open(SKELETON_PATH, "r", encoding="utf-8") as f:
        skeleton = json.load(f)
    skeleton_meta = {s["n"]: s for s in skeleton["sessions"]}

    sessions = []
    total_words = 0
    total_chars = 0
    fact_report = []

    for n in range(1, 41):
        turns = load_turns(n)
        # Normalize: ensure each turn has role + content
        norm_turns = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = t.get("role", "").lower()
            content = None
            for key in ("content", "text", "message", "body"):
                v = t.get(key)
                if isinstance(v, str):
                    content = v
                    break
            if role and content:
                norm_turns.append({"role": role, "content": content})
        meta = skeleton_meta.get(n, {})
        sessions.append({
            "n": n,
            "date": meta.get("date", ""),
            "topic": meta.get("topic", ""),
            "turns": norm_turns,
        })
        text = " ".join(t["content"] for t in norm_turns)
        words = len(text.split())
        chars = len(text)
        total_words += words
        total_chars += chars

        missing = []
        text_lower = text.lower()
        for fact in REQUIRED_FACTS.get(n, []):
            if fact.lower() not in text_lower:
                missing.append(fact)
        fact_report.append((n, len(REQUIRED_FACTS.get(n, [])), missing, words))

    chat = {
        "chat_id": "restaurant",
        "topic": "Restaurant Opening",
        "description": "12-month journey of Kenta Morishita opening Osteria Morishita in Higashi-Azabu, Tokyo.",
        "sessions": sessions,
        "session_count": len(sessions),
        "total_words": total_words,
        "total_chars": total_chars,
        "total_tokens_estimate": total_chars // 4,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Sessions: {len(sessions)}")
    print(f"Total words: {total_words:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Estimated tokens (char/4): {chat['total_tokens_estimate']:,}")
    print()
    print("=== FACT VERIFICATION ===")
    total_required = 0
    total_missing = 0
    for n, req_count, missing, words in fact_report:
        total_required += req_count
        total_missing += len(missing)
        status = "OK" if not missing else f"MISSING {len(missing)}/{req_count}"
        print(f"Session {n:02d} ({words:5d} words): {status}")
        if missing:
            for m in missing:
                print(f"  - {m!r}")
    print()
    print(f"Total required facts: {total_required}")
    print(f"Total missing facts:  {total_missing}")
    print(f"Coverage: {(total_required-total_missing)/total_required*100:.1f}%")


if __name__ == "__main__":
    main()
