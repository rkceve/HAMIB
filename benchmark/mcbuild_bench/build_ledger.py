"""Fact ledger + questions for mcbuild-bench, machine-verified against the redacted corpus.

Decision ledger B1-B4 (benchmark/mcbuild_bench/DECISIONS.md). Every fact below was
read by Fable from the redacted transcript; this script REFUSES to emit any fact
whose `evidence` substring is not found verbatim in the named round trip, so a
fact cannot enter the ledger by memory or by guess. It also runs the leakage
checks (gold in the question text; gold absent from the corpus for `unknown`
questions) and writes the question file in the bineval schema so
benchmark/bineval/score_binary.py can score it unchanged.

Corpus (DECISIONS H22 (c), 2026-09-20): the experiment corpus is loaded through
`corpus.load_corpus`, i.e. WITHOUT round trip 36 (the retrospective). Any fact
whose `rt` or history `rt` lies in an excluded round trip FAILS the ledger (its
evidence would not be in the corpus). f082 / f083 / f084 were dropped for that
reason: their statements (85-minute design phase, "2 stops for getting ahead",
"6 live !fix runs") occur ONLY in round trip 36 — round trip 3 contains a single
"先走りました" without the count, and no earlier round trip states the other two.

Outputs:
  benchmark/mcbuild_bench/data/facts.json       - the ledger (fact -> evidence)
  benchmark/mcbuild_bench/data/questions.json   - bineval-schema questions
  benchmark/mcbuild_bench/data/ledger_report.md - verification report
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

from benchmark.mcbuild_bench.corpus import DEFAULT_EXCLUDE_RT, load_corpus

DATA = Path(__file__).resolve().parent / "data"

# kind: value | name_path | policy | definition | updated_final | absent
# rt: round-trip index where the FINAL/authoritative statement appears.
# evidence: a verbatim substring that MUST appear in round trip `rt` (human or events).
# For updated facts, `history` lists earlier (rt, value) pairs also verified.
# Alias syntax (score_binary D3, 2026-09-18): an alias containing " & " requires EVERY part to
# match; lone short fragments ("4", "13") are no longer used as aliases (D2 boundaries).
FACTS: list[dict] = [
    # ---- A. hackathon / session frame (rt 0) ----
    dict(id="f001", kind="value", rt=0, q="What hackathon was the user participating in during this session?",
         gold="AI Tinkerers Global Hackathon", aliases=["AI Tinkerers", "AI tinkerers global hackathon"],
         evidence="AI tinkerersのglobal hackathon"),
    dict(id="f002", kind="value", rt=0, q="What was the hackathon's theme?",
         gold="Agents, Everywhere: Beyond the Chatbox", aliases=["Agents, Everywhere", "Beyond the Chatbox"],
         evidence="Agents, Everywhere: Beyond the Chatbox"),
    dict(id="f003", kind="value", rt=0, q="What version of the Codex CLI was installed on the development PC?",
         gold="0.153.4", aliases=["codex-cli 0.153.4"], evidence="codex-cli 0.153.4"),
    dict(id="f004", kind="value", rt=0, q="What model was configured as default in the Codex config file at the start of the session?",
         gold="gpt-5.3-codex", aliases=[], evidence='model = "gpt-5.3-codex"'),
    dict(id="f005", kind="value", rt=0, q="Roughly how long was one RCON round trip between the dev PC and the server (in milliseconds)?",
         gold="150", aliases=["150〜200", "150-200 ms", "150 to 200"], evidence="RCON 1 往復が 150〜200 ms"),
    dict(id="f006", kind="value", rt=0, q="Which Minecraft command was needed before placing blocks in unloaded chunks?",
         gold="forceload add", aliases=["forceload"], evidence="`forceload add` で解決する"),
    dict(id="f007", kind="definition", rt=0, q="Why could the original plan's undo design (record and restore each placed block) not be implemented in vanilla Minecraft?",
         gold="no command returns the block id at a coordinate", aliases=["座標のブロック ID を返す", "no command to read a block"],
         evidence="バニラに「座標のブロック ID を返す」命令が無い"),
    dict(id="f008", kind="definition", rt=0, q="What undo mechanism did the assistant recommend instead?",
         gold="/clone", aliases=["clone", "clone the site away and clone it back"],
         evidence="施工前に敷地を `/clone` で遠方へ退避"),
    # ---- B. server / schematic (rt 1, 6) ----
    dict(id="f009", kind="policy", rt=1, q="In which country was the Minecraft server hosted?",
         gold="Japan", aliases=["日本"], evidence="日本にある機体でホストして"),
    dict(id="f010", kind="value", rt=1, q="What was the file name of the reference schematic the user called their best work?",
         gold="cathedral.litematic", aliases=["cathedral"], evidence="cathedral.litematic"),
    dict(id="f011", kind="value", rt=1, q="What Minecraft version was the cathedral schematic made in?",
         gold="1.21.8", aliases=[], evidence="このschemは1.21.8で作ってる"),
    dict(id="f012", kind="value", rt=1, q="What are the dimensions of the cathedral schematic (x, y, z in blocks)?",
         gold="186 x 109 x 81", aliases=["186×109×81", "186 x 109 x 81", "186 & 109 & 81"], evidence="size {'z': Int(81), 'x': Int(186), 'y': Int(109)}"),
    dict(id="f013", kind="value", rt=1, q="How many blocks does the cathedral schematic contain?",
         gold="64062", aliases=["64,062", "64062", "64k", "64,000"], evidence="blocks 64062"),
    dict(id="f014", kind="value", rt=1, q="What DataVersion does the cathedral schematic carry?",
         gold="4440", aliases=[], evidence="DataVersion 4440"),
    dict(id="f015", kind="value", rt=1, q="What is the repeating period (in blocks) of the cathedral's long exterior wall bays?",
         gold="12", aliases=["12 ブロック", "12-block"], evidence="12 ブロック周期"),
    dict(id="f016", kind="policy", rt=1, q="Which server software and Java version did the requirements document ask for?",
         gold="Paper 1.21.8 and Java 21", aliases=["Paper 1.21.8 & Java 21"], evidence="Paper 1.21.8 と Java 21"),
    dict(id="f017", kind="value", rt=1, q="At what y level is the ground surface on the demo world?",
         gold="y=0", aliases=["0", "y = 0", "y 0"], evidence="地面が y=0"),
    dict(id="f018", kind="value", rt=6, q="What was the game connection address (IP:port) of the demo server?",
         gold="203.0.113.12:25566", aliases=["203.0.113.12", "25566"], evidence="203.0.113.12:25566"),
    dict(id="f019", kind="value", rt=6, q="Which port did the demo server's RCON listen on?",
         gold="25576", aliases=[], evidence="RCON           │ 25576"),
    dict(id="f020", kind="value", rt=6, q="Under what path on the server was the RCON password stored?",
         gold="~/buildagent-server/.rcon-password", aliases=[".rcon-password"], evidence="~/buildagent-server/.rcon-password"),
    dict(id="f021", kind="value", rt=6, q="What systemd unit name started and stopped the demo Minecraft server?",
         gold="buildagent", aliases=["systemctl start buildagent"], evidence="sudo systemctl start buildagent"),
    dict(id="f022", kind="value", rt=6, q="On which port did the pre-existing other project's server keep running while the demo server used 25566?",
         gold="25565", aliases=[], evidence="otherproj は 25565 で稼働継続中"),
    # ---- C. subject church + scale (rt 5, 6) ----
    dict(id="f023", kind="value", rt=5, q="Which real church was chosen as the subject for Astra to reproduce?",
         gold="Catholic Shimizu Church", aliases=["清水教会", "Shimizu Church", "Old Catholic Shimizu Church", "カトリック清水教会"],
         evidence="カトリック清水教会"),
    dict(id="f024", kind="value", rt=5, q="In which prefecture is the subject church located?",
         gold="Shizuoka", aliases=["静岡"], evidence="静岡にある"),
    dict(id="f025", kind="value", rt=5, q="In what year was the subject church built?",
         gold="1935", aliases=["昭和10年", "Showa 10"], evidence="1935 年築"),
    dict(id="f026", kind="value", rt=5, q="What is the estimated real-world footprint and height of the church (width x depth x height in metres)?",
         gold="13 m x 22 m x 15 m", aliases=["13 m × 22 m × 15 m", "13 & 22 & 15"], evidence="幅 13 m × 奥行 22 m × 高さ 15 m"),
    dict(id="f027", kind="value", rt=5, q="From what scale model were the church's real dimensions back-calculated?",
         gold="1/110", aliases=["1:110"], evidence="1/110 の記念模型"),
    dict(id="f028", kind="updated_final", rt=6, q="What build scale was finally decided for the church (blocks per real metre)?",
         gold="2:1", aliases=["2 to 1", "2 blocks per metre", "1 real metre = 2 blocks"],
         evidence="じゃあ2:1でいいよ", history=[(5, "1:1", "大きさは1:1くらいでいいかな")]),
    dict(id="f029", kind="value", rt=5, q="In which month and year was the old church demolished?",
         gold="January 2024", aliases=["2024 年 1 月", "2024"], evidence="2024 年 1 月に解体済み"),
    dict(id="f030", kind="policy", rt=6, q="In what language did the user ask for the Astra instruction text to be written, and why?",
         gold="English", aliases=["英語", "for the GitHub reference"], evidence="指示文は英語で"),
    dict(id="f031", kind="policy", rt=6, q="Was the cathedral shown to Astra for the first (plain) generation?",
         gold="No", aliases=["not shown", "大聖堂は見せない", "without the cathedral"], evidence="大聖堂は見せないで素のastraの全力を見る"),
    # ---- D. plain Astra run (rt 6, 7) ----
    dict(id="f032", kind="value", rt=6, q="How many blocks did the plain (no-harness) Astra church contain?",
         gold="9294", aliases=["9,294"], evidence="9,294 ブロック"),
    dict(id="f033", kind="value", rt=6, q="How many RCON commands did streaming the plain church take?",
         gold="4035", aliases=["4,035"], evidence="4,035 コマンド"),
    dict(id="f034", kind="value", rt=6, q="At what rate (commands per second) was the plain church streamed for the visible build-up?",
         gold="60", aliases=["60 コマンド", "60 commands/s"], evidence="毎秒 60 コマンド"),
    dict(id="f035", kind="definition", rt=6, q="What caused the demo server to crash every ~78 seconds, and how was it fixed?",
         gold="spark profiler on Java 26; disabled in paper-global.yml", aliases=["spark", "spark を無効化"],
         evidence="Paper 同梱の spark プロファイラが Java 26 上で native クラッシュ"),
    dict(id="f036", kind="definition", rt=6, q="What Minecraft RCON quirk broke the first realization-layer client, and what was the fix?",
         gold="RCON handles one packet per read; switched to one command one response", aliases=["1 コマンド 1 応答", "one packet", "1 パケット"],
         evidence="2 パケット連送で切断される Minecraft 側の癖"),
    dict(id="f037", kind="value", rt=7, q="How many reference photos of the church were attached to the Astra prompt?",
         gold="7", aliases=["seven", "7 枚"], evidence="写真 7 枚（正面・側面・全景）"),
    dict(id="f038", kind="value", rt=6, q="How many randomly sampled positions were verified in-world after the plain church was streamed?",
         gold="6", aliases=["six", "6 か所"], evidence="無作為に選んだ 6 か所"),
    # ---- E. harness design (rt 8-11, 15) ----
    dict(id="f039", kind="definition", rt=9, q="What are the two parts of the harness design the user defined?",
         gold="spoken harness (system prompt of principles) and coded patterns (helper functions)",
         aliases=["口頭のハーネス", "コード化されたパターン", "spoken harness", "coded patterns", "2本立て"],
         evidence="構成は2本立て"),
    dict(id="f040", kind="policy", rt=4, q="Which model did the user assign to write the code, to save tokens?",
         gold="sonnet 5", aliases=["sonnet5", "Sonnet 5", "sonnet"], evidence="コーディングはsonnet5のエージェントにやらせて"),
    dict(id="f041", kind="policy", rt=3, q="What did the user tell the assistant right after two implementation agents were launched at RT 2?",
         gold="Do not start implementing yet; the design is not fixed", aliases=["実装はまだ始めないで", "not start", "design is not fixed", "設計が確定してない"],
         evidence="実装はまだ始めないで。まだ設計が確定してない。"),
    dict(id="f042", kind="value", rt=10, q="What file holds the DSL specification (the fixed contract for helper functions)?",
         gold="DSL.md", aliases=["mcbuild/DSL.md"], evidence='"file_path": "C:\\\\Users\\\\user\\\\mcbuild\\\\DSL.md"'),
    dict(id="f043", kind="value", rt=10, q="How many sonnet implementation agents were launched in parallel for the DSL layer?",
         gold="4", aliases=["four", "4 本"], evidence="4 本を並列で発注しました"),
    dict(id="f044", kind="value", rt=11, q="What is the file name of the translated builder's principles that Codex reads?",
         gold="AGENTS.md", aliases=["mcbuild/prompts/AGENTS.md", "prompts/AGENTS.md"], evidence="mcbuild/prompts/AGENTS.md"),
    dict(id="f045", kind="value", rt=11, q="Which three elements from the principles were missing from the DSL and had to be added?",
         gold="ridge ornament, plinth (foundation), beam", aliases=["土台・梁・棟飾り", "plinth", "beam", "ridge"],
         evidence="屋根の頂部 1 ブロック幅の装飾、規模に応じた土台と柱の接続、梁の端のアーチ"),
    dict(id="f046", kind="definition", rt=15, q="According to the user, what does 'the harness' consist of?",
         gold="the user's prompt and the mechanism referencing the cathedral's parts",
         aliases=["僕のプロンプトと大聖堂のパーツを参照する仕組み", "prompt and cathedral parts", "大聖堂のパーツ"],
         evidence="ハーネスっていうのは僕のプロンプトと大聖堂のパーツを参照する仕組み"),
    dict(id="f047", kind="policy", rt=11, q="According to the builder's first principle, what parity must the block count of a building's width have?",
         gold="odd", aliases=["奇数", "odd number of blocks"], evidence="横幅のブロック数は奇数"),
    dict(id="f048", kind="policy", rt=11, q="What block type must roofs be made of according to the builder's principles?",
         gold="stairs only", aliases=["階段のみ", "stairs"], evidence="屋根は基本的に階段"),
    # ---- F. Ambiguous sponsor question (rt 12, 13, 24) ----
    dict(id="f049", kind="value", rt=13, q="What prize would the Ambiguous sponsor award give?",
         gold="DGX Spark", aliases=["dgx spark"], evidence="dgx sparkがもらえる"),
    dict(id="f050", kind="value", rt=13, q="What probability did the assistant estimate for winning the Ambiguous sponsor prize with the proposed integration?",
         gold="15-25%", aliases=["15〜25%", "15 to 25", "15-25"], evidence="15〜25% です"),
    dict(id="f051", kind="value", rt=12, q="What is the base URL of the Ambiguous REST API found during the session?",
         gold="https://app.ambiguous.ai/api/", aliases=["app.ambiguous.ai/api"], evidence="https://app.ambiguous.ai/api/"),
    dict(id="f052", kind="updated_final", rt=24, q="Was the Ambiguous integration ultimately implemented before the deadline?",
         gold="No, it was dropped", aliases=["見送り", "dropped", "not implemented", "no"],
         evidence="Ambiguous 連携は見送り", history=[(13, "recommended minimal form", "最小形（様式ガイドの読み込み")]),
    # ---- G. harness Astra run (rt 14, 16) ----
    dict(id="f053", kind="value", rt=16, q="How many blocks did the harness version of the church contain?",
         gold="10095", aliases=["10,095"], evidence="10,095"),
    dict(id="f054", kind="value", rt=16, q="How long did Astra take to generate the harness version of the church?",
         gold="7 minutes 41 seconds", aliases=["7 分 41 秒", "7m41s", "7:41"], evidence="7 分 41 秒"),
    dict(id="f055", kind="value", rt=16, q="What was the width (in blocks) of the harness version, and was it odd or even?",
         gold="29, odd", aliases=["29 & odd", "29 & 奇数"], evidence="29（奇数、原則どおり）"),
    dict(id="f056", kind="value", rt=14, q="Where (origin x coordinate) was the harness church placed relative to the plain one?",
         gold="x=-60", aliases=["-60", "(-60, 1, 0)", "west of the plain version"], evidence="原点 (-60, 1, 0)"),
    dict(id="f057", kind="value", rt=16, q="How many check() warnings did the harness church preview report?",
         gold="0", aliases=["zero", "0 warning"], evidence="check(): 0 warning(s)"),
    dict(id="f058", kind="value", rt=16, q="How many blocks did the undo of the DSL validation test building remove from the server?",
         gold="4986", aliases=["4,986"], evidence="試験体の 4,986 ブロック"),
    # ---- H. !fix loop (rt 17-22, 26-28) ----
    dict(id="f059", kind="definition", rt=19, q="Why did the first !fix appear slow even though Astra finished in about a minute?",
         gold="the bridge crashed decoding UTF-8 output under Windows cp932", aliases=["cp932", "UTF-8", "文字コード"],
         evidence="Windows の既定文字コード（cp932）で Codex の UTF-8 出力を読もうとして落ち"),
    dict(id="f060", kind="value", rt=19, q="How many blocks were added and removed by the first !fix (nave walls and roof raised)?",
         gold="+3343 -1921", aliases=["3343 & 1921", "3,343 & 1,921"], evidence="3,343 追加、1,921 削除"),
    dict(id="f061", kind="value", rt=19, q="By how many blocks did the first !fix raise the nave wall height (from what to what)?",
         gold="12 to 16", aliases=["12→16", "12 & 16"], evidence="身廊の壁高 12→16"),
    dict(id="f062", kind="value", rt=20, q="Why was the second !fix (towers taller) initially not picked up by the bridge?",
         gold="it was typed 19 seconds before the bridge restarted", aliases=["19 秒前", "19 seconds", "before the bridge restart", "restart"],
         evidence="その 19 秒前だったため拾えていません"),
    dict(id="f063", kind="value", rt=20, q="By how many blocks did Astra raise the towers in the successful second !fix?",
         gold="13", aliases=["13 ブロック", "thirteen"], evidence="左右の塔を13ブロック高くし"),
    dict(id="f064", kind="value", rt=20, q="How long did the first fully automatic !fix loop take from chat receipt to world update?",
         gold="about 4 minutes", aliases=["約 4 分", "3 分 50 秒", "3:50", "4 minutes"], evidence="依頼から反映まで約 4 分"),
    dict(id="f065", kind="value", rt=22, q="How did Astra interpret 'make the wall width odd' in the third live !fix?",
         gold="as pillar spacing of 5 blocks", aliases=["5 ブロック", "柱間隔", "5 blocks"],
         evidence="「柱と柱の間隔を 5 に」と解釈"),
    dict(id="f066", kind="policy", rt=22, q="Does the bridge run multiple !fix requests in parallel?",
         gold="No, they are queued serially", aliases=["直列", "serial", "queue", "no"], evidence="並列は想定していません。直列の待ち行列です"),
    dict(id="f067", kind="value", rt=24, q="What was the hackathon submission deadline (local time) the user stated?",
         gold="15:30", aliases=["3:30 pm", "15時30分"], evidence="15:30が締め切りで"),
    dict(id="f068", kind="value", rt=24, q="What internal deadline did the assistant set for the window-stamp implementation?",
         gold="14:35", aliases=[], evidence="14:35 まで"),
    dict(id="f069", kind="value", rt=25, q="What is the GitHub repository path for the project?",
         gold="example-user/minecraft-building-agent", aliases=["minecraft-building-agent"], evidence="example-user/minecraft-building-agent"),
    dict(id="f070", kind="value", rt=25, q="How many windows did Astra switch to the cathedral style in the third !fix turn?",
         gold="27", aliases=["all 27", "27 windows"], evidence="all 27 windows switched to the cathedral style"),
    dict(id="f071", kind="value", rt=25, q="How many blocks did the third !fix turn add and remove?",
         gold="+2653 -753", aliases=["2653 & 753", "2,653 & 753"], evidence="+2653 -753 blocks"),
    dict(id="f072", kind="definition", rt=26, q="How was the third !fix reverted, given that !undo only restores the original terrain?",
         gold="by reverse-applying the patch from Codex's rollout JSONL", aliases=["reverse & patch", "逆適用"],
         evidence="Codex が turn 3 で当てたパッチはターンログ（JSONL）に残っているので、それを逆適用"),
    dict(id="f073", kind="value", rt=27, q="At what x origin was the filming copy (pre-fix harness church, church_v0) placed?",
         gold="x=36", aliases=["36", "(36, 1, 0)"], evidence="原点 x=36"),
    dict(id="f074", kind="value", rt=27, q="What build_id was given to the filming copy of the church?",
         gold="church_v0", aliases=[], evidence="build_id `church_v0`"),
    dict(id="f075", kind="definition", rt=28, q="Why did Astra's self-check preview not run in the filming-build fixes, and what was the fix?",
         gold="python was not on PATH in Astra's sandbox; absolute python path added to the turn prefix",
         aliases=["absolute & path", "絶対パス", "PATH & absolute", "砂箱"],
         evidence="Astra の砂箱内で `python` が見つかっていません"),
    dict(id="f076", kind="value", rt=28, q="By how many blocks were the towers and spires raised in the filming build's first fix?",
         gold="towers 4, spires 6", aliases=["4 & 6", "4 ブロック & 6 ブロック"], evidence="両塔を 4 ブロック、尖塔を 6 ブロック高く"),
    # ---- I. submission / audit / retrospective (rt 29-36) ----
    dict(id="f077", kind="value", rt=29, q="What is the project title used in the submission text?",
         gold="mcbuild", aliases=["mcbuild — an in-game Minecraft building agent"], evidence="mcbuild — an in-game Minecraft building agent that builds like a master builder"),
    dict(id="f078", kind="value", rt=30, q="How many tests did the project have at submission according to the description?",
         gold="113", aliases=["113 tests"], evidence="113 tests cover"),
    dict(id="f079", kind="value", rt=31, q="How many occurrences of the real-name home path were found in the public repo during the privacy audit?",
         gold="160", aliases=["約 160", "160 か所"], evidence="計 160 か所"),
    dict(id="f080", kind="definition", rt=32, q="What did the assistant replace the raw Astra run logs with before publishing, and why?",
         gold="summaries, because the logs contained sandbox file listings", aliases=["要約", "summary", "run_v1_summary", "砂箱内のファイル探索"],
         evidence="Astra の生ログ 2 本を公開から外し"),
    dict(id="f081", kind="value", rt=33, q="What is the size and content of the published window stamp catalog JSON?",
         gold="36 KB, one cathedral window bay (11x26x4 blocks)", aliases=["36 KB", "11×26×4", "cathedral_stamps.json"],
         evidence="catalog/cathedral_stamps.json"),
    # f082 (85-minute design phase), f083 (2 stops for getting ahead), f084 (6 live !fix
    # runs) were DROPPED 2026-09-20 (H22 (c)): stated only in round trip 36.
    dict(id="f085", kind="policy", rt=14, q="Per the user's instruction at RT 14, what was the ONLY thing allowed to differ between the first Astra generation and the second one?",
         gold="adding the harness", aliases=["ハーネスを足すだけ", "harness", "AGENTS.md", "read AGENTS.md"],
         evidence="前の出力から変えるのはハーネスを足すだけ"),
    dict(id="f086", kind="value", rt=17, q="Which element name did the raycast report the player was looking at during the first !fix?",
         gold="dark_spire_right", aliases=[], evidence="dark_spire_right"),
    dict(id="f089", kind="value", rt=6, q="What maximum Java heap size (Xmx) was the demo Minecraft server started with?",
         gold="4G", aliases=["-Xmx4G", "4 GB", "4GB"], evidence="-Xms2G -Xmx4G -jar paper-1.21.8-60.jar"),
    dict(id="f090", kind="value", rt=6, q="What was the maximum player count configured on the demo server?",
         gold="5", aliases=["five", "max of 5"], evidence="There are 0 of a max of 5 players online"),
    dict(id="f091", kind="value", rt=6, q="Which Paper build number was the demo server running?",
         gold="paper-1.21.8-60", aliases=["build 60", "1.21.8-60"], evidence="paper-1.21.8-60.jar"),
    dict(id="f087", kind="value", rt=3, q="How many seconds did the first `codex exec` test with gpt-6-astra take to return?",
         gold="8", aliases=["約 8 秒", "7.9", "8 秒"], evidence="8 秒で返りました"),
    dict(id="f088", kind="value", rt=2, q="Which ChatGPT subscription plan did the user say they were on?",
         gold="Plus", aliases=["ChatGPT Plus", "plusプラン"], evidence="chatgptのplusプランに入ってる"),
]

# Absent-fact questions (B2): the correct answer is `unknown`; the gold string must NOT occur
# anywhere in the corpus (verified below).
ABSENT: list[dict] = [
    dict(id="a001", q="What GPU model was installed in the development PC used during the hackathon?", probe=["RTX", "GeForce", "GPU"]),
    dict(id="a002", q="What was the exact 32-character RCON password string of the demo server?", probe=["rcon.password=REDACTED", "password: R"]),
    dict(id="a004", q="What was the total electricity cost of running the demo server that day?", probe=["electricity", "電気代"]),
    dict(id="a005", q="Which team placed first in the hackathon's global ranking?", probe=["placed first", "1位", "winner"]),
    dict(id="a006", q="How many blocks does the apartment schematic in the same folder contain?", probe=["apartment.litematic blocks", "apartment blocks"]),
    dict(id="a007", q="How many people watched the submitted demo video on the day of submission?", probe=["spectators", "view count", "再生回数"]),
    dict(id="a008", q="What Minecraft username did the user's teammate use on the demo server?", probe=["teammate", "チームメイト"]),
    dict(id="a010", q="How many lines of code did the final mcbuild package contain?", probe=["lines of code", "wc -l mcbuild/*.py"]),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def excluded_rt_failures(facts: list[dict], exclude_rt=DEFAULT_EXCLUDE_RT) -> list[str]:
    """H22 (c): a fact whose FINAL statement or any history entry lies in an
    excluded round trip cannot be in the ledger (its evidence is not in the
    corpus).  Returns one message per offending fact."""
    excluded = set(int(i) for i in exclude_rt)
    out: list[str] = []
    for f in facts:
        hit = [f["rt"]] if int(f["rt"]) in excluded else []
        hit += [h[0] for h in f.get("history", []) if int(h[0]) in excluded]
        if hit:
            out.append(
                f"{f['id']}: evidence lies in excluded round trip {', '.join(str(h) for h in hit)} "
                f"(H22 (c)); re-source it earlier in the corpus or drop the fact"
            )
    return out


def main() -> int:
    corpus = load_corpus(DATA / "session_redacted.json", DEFAULT_EXCLUDE_RT)
    by_idx = {int(r["idx"]): r for r in corpus.round_trips}

    def rt_text(i: int) -> str:
        r = by_idx[i]
        return r["human"] + "\n" + "\n".join(e["text"] for e in r["events"])

    full = "\n".join(rt_text(i) for i in sorted(by_idx))
    full_n = _norm(full)

    report: list[str] = [
        "# Ledger verification report", "",
        f"corpus: {len(corpus.round_trips)} of {corpus.n_session_round_trips} round trips "
        f"(exclude_rt={list(corpus.exclude_rt)}, H22 (c)); corpus sha256 {corpus.sha256}",
    ]
    ok: list[dict] = []
    bad: list[str] = excluded_rt_failures(FACTS, corpus.exclude_rt)
    excluded_ids = {b.split(":")[0] for b in bad}
    for f in FACTS:
        if f["id"] in excluded_ids:
            continue
        text = rt_text(f["rt"])
        if f["evidence"] not in text:
            bad.append(f"{f['id']}: evidence NOT found in RT {f['rt']}: {f['evidence'][:60]!r}")
            continue
        for h_rt, _val, h_ev in f.get("history", []):
            if h_ev not in rt_text(h_rt):
                bad.append(f"{f['id']}: history evidence NOT found in RT {h_rt}: {h_ev[:60]!r}")
        # leakage: gold (or any alias) must not appear in the question text
        qn = _norm(f["q"])
        for g in [f["gold"], *f["aliases"]]:
            if len(_norm(g)) >= 3 and _norm(g) in qn:
                bad.append(f"{f['id']}: gold/alias {g!r} appears in the question text")
        ok.append(f)

    absent_ok: list[dict] = []
    for a in ABSENT:
        hits = [p for p in a["probe"] if _norm(p) in full_n]
        if hits:
            bad.append(f"{a['id']}: probe present in corpus ({hits}); not an absent fact")
        else:
            absent_ok.append(a)

    report.append(f"facts verified: {len(ok)} / {len(FACTS)}")
    report.append(f"absent questions verified: {len(absent_ok)} / {len(ABSENT)}")
    if bad:
        report += ["", "## FAILURES (nothing written)", ""] + [f"- {b}" for b in bad]
        (DATA / "ledger_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report))
        return 1

    kinds: dict[str, int] = {}
    for f in ok:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    rt_hist: dict[int, int] = {}
    for f in ok:
        rt_hist[f["rt"]] = rt_hist.get(f["rt"], 0) + 1
    report += ["", "## kinds", ""] + [f"- {k}: {v}" for k, v in sorted(kinds.items())]
    report += ["", "## facts per round trip (source of the FINAL statement)", ""]
    report += [f"- RT {k:02d}: {v}" for k, v in sorted(rt_hist.items())]
    early = sum(v for k, v in rt_hist.items() if k <= 6)
    report += ["", f"facts whose final statement lies in RT 00-06 (first ~19% of round trips): {early} / {len(ok)}"]

    questions = []
    for f in ok:
        questions.append({
            "qid": f["id"],
            "source_session": f["rt"],  # bineval field name kept; here = round trip index
            "fact_quote": f["evidence"],
            "question": f["q"],
            "gold_short": f["gold"],
            "tier1_aliases": f["aliases"],
            "kind": f["kind"],
            "history": [{"rt": h[0], "value": h[1]} for h in f.get("history", [])],
            "excluded": False,
            "exclusion_reason": "",
            "legacy": False,
        })
    for a in absent_ok:
        questions.append({
            "qid": a["id"],
            "source_session": -1,
            "fact_quote": "",
            "question": a["q"],
            "gold_short": "unknown",
            "tier1_aliases": ["not in context", "not mentioned", "not present", "no information"],
            "kind": "absent",
            "history": [],
            "excluded": False,
            "exclusion_reason": "",
            "legacy": False,
        })
    (DATA / "facts.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    report += ["", f"questions written: {len(questions)} (facts {len(ok)} + absent {len(absent_ok)})"]
    (DATA / "ledger_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
