"""Surface-feature profiling + manual review sheet for locally generated responses.

The automatic features only cover what can be counted (Layer 1-ish signals and
instruction compliance). Whether the depicted social action is appropriate
(Layers 2-3, Q5) still needs a human reader, so this script also writes a
review sheet with the questionnaire topics from the paper for manual scoring.

    .venv/bin/python replication/03_review_responses.py replication/generated/llama3.1_8b_ja_400.jsonl
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "replication" / "outputs"

DIMS = ["PDI", "IDV", "UAI", "MAS", "LTO", "IND"]

# Q1-Q4 topics, from the Japanese comments in statistical_analysis.ipynb (cell 10).
# The full item wording is in S2 Table of the paper.
QUESTION_TOPICS = {
    "PDI": ["敬意・敬語 respect / honorifics", "助言の受け入れ accepting advice",
            "階層的価値観 hierarchical values", "建設的コミュニケーション constructive communication"],
    "IDV": ["個人とチームのバランス individual-team balance", "チームへの貢献 contribution to team",
            "チームワーク・協力 teamwork", "集団主義的価値観 collectivist values"],
    "UAI": ["不確実性への態度 attitude to uncertainty", "柔軟と明確のバランス flexibility vs clarity",
            "具体的な行動 concrete action", "安心感の提供 providing reassurance"],
    "MAS": ["性別役割の考え方 view of gender roles", "競争と協調のバランス competition vs cooperation",
            "性別役割の捉え方 perception of gender roles", "男/女性性のバランス masculinity-femininity balance"],
    "LTO": ["長期と短期のバランス long vs short term", "長期的視点の反映 long-term perspective",
            "慎重な分析 careful analysis", "短期優先のリスク risk of short-termism"],
    "IND": ["楽しさと生産性のバランス fun vs productivity", "生産性確保の方法 securing productivity",
            "控えめ・抑制の価値観 modesty / restraint", "バランスの取れた提案 balanced proposal"],
}

JA_CHAR = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uff66-\uff9f]")
LATIN = re.compile(r"[A-Za-z]")
DIALOGUE = re.compile(r"「([^」]*)」")
POLITE = re.compile(r"(です|ます|ました|ません|でしょう|ください|ましょう)")
HONORIFIC = re.compile(r"(いただ|頂|申し上げ|おっしゃ|いらっしゃ|ございま|存じ|伺|拝見|恐れ入|恐縮|お手数|させていただ|ご確認|お願い)")
CUSHION = re.compile(r"(恐れ入りますが|申し訳ありませんが|申し訳ないのですが|差し支えなければ|もしよろしければ|よろしければ|かもしれません|と思われます|のですが|少し|ちょっと)")
APOLOGY = re.compile(r"(すみません|申し訳|ごめん|失礼)")
META = re.compile(r"(文化的|ホフステード|文化次元|権力格差|個人主義|集団主義|不確実性回避|男性性|女性性|長期志向|短期志向|放縦)")
NARR_PLAIN = re.compile(r"(た|だ|る|ない)。")
NARR_POLITE = re.compile(r"(ました|です|ます|ません)。")


def narration(text):
    return DIALOGUE.sub("", text)


def max_repeated_paragraph(text):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= 20]
    return max((paras.count(p) for p in set(paras)), default=0)


def features(text):
    n_ja, n_lat = len(JA_CHAR.findall(text)), len(LATIN.findall(text))
    lines = DIALOGUE.findall(text)
    dialogue = "".join(lines)
    narr = narration(text)
    plain, polite = len(NARR_PLAIN.findall(narr)), len(NARR_POLITE.findall(narr))
    return {
        "within_400": len(text) <= 400,
        "ja_share": n_ja / max(n_ja + n_lat, 1),
        "has_dialogue": bool(lines),
        "dialogue_polite_per_100": 100 * len(POLITE.findall(dialogue)) / max(len(dialogue), 1),
        "honorific_hits": len(HONORIFIC.findall(text)),
        "cushion_hits": len(CUSHION.findall(text)),
        "apology_hits": len(APOLOGY.findall(text)),
        "meta_culture_hits": len(META.findall(text)),
        "narration_style_mixed": plain > 0 and polite > 0,
        "keeps_sato": "佐藤" in text,
        "repetition_loop": max_repeated_paragraph(text) >= 3,
    }


def main(path):
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(recs)
    feats = pd.DataFrame([features(t) for t in df["response"]])
    df = pd.concat([df, feats], axis=1)
    df["dimension"] = pd.Categorical(df["dimension"], DIMS, ordered=True)
    df = df.sort_values(["dimension", "prompt_idx", "repeat_idx"]).reset_index(drop=True)
    tag = Path(path).stem

    num_cols = ["n_chars", "within_400", "ja_share", "dialogue_polite_per_100", "honorific_hits",
                "cushion_hits", "apology_hits", "meta_culture_hits", "narration_style_mixed", "keeps_sato",
                "repetition_loop"]
    summary = df.groupby("dimension", observed=True)[num_cols].mean()
    summary.loc["ALL"] = df[num_cols].mean()
    print(f"{len(df)} responses from {df['model'].iloc[0]}\n")
    print(summary.round(2).to_string())
    summary.round(3).to_csv(OUT / f"surface_features_{tag}.csv")

    sheet = df[["dimension", "prompt_idx", "repeat_idx", "n_chars", "meta_culture_hits", "narration_style_mixed"]].copy()
    for q in range(1, 6):
        sheet[f"q{q}"] = ""
    sheet["action_chosen"] = ""
    sheet["notes"] = ""
    sheet.to_csv(OUT / f"review_sheet_{tag}.csv", index=False, encoding="utf-8-sig")

    md = [f"# Manual review sheet: {df['model'].iloc[0]}\n",
          "Score each response 1-5 on the dimension's items, the same way the paper's raters did.",
          "Q5 is always the overall fit to a Japanese workplace. Enter scores in "
          f"`review_sheet_{tag}.csv`.\n"]
    for dim in DIMS:
        sub = df[df["dimension"] == dim]
        if sub.empty:
            continue
        md.append(f"\n## {dim}\n")
        md.append("Items: " + " | ".join(f"Q{i + 1} {t}" for i, t in enumerate(QUESTION_TOPICS[dim]))
                  + " | Q5 総合適応 overall fit\n")
        for pidx, grp in sub.groupby("prompt_idx"):
            md.append(f"\n### {dim}-{pidx}\n\n> {grp['prompt'].iloc[0]}\n")
            for _, r in grp.iterrows():
                flags = []
                if not r["within_400"]:
                    flags.append(f"{r['n_chars']} chars (> 400)")
                if r["meta_culture_hits"]:
                    flags.append(f"explains culture explicitly ({r['meta_culture_hits']} hits)")
                if r["narration_style_mixed"]:
                    flags.append("mixes plain/polite narration")
                if not r["keeps_sato"]:
                    flags.append("protagonist renamed")
                if r["repetition_loop"]:
                    flags.append("degenerate repetition loop")
                md.append(f"\n**repeat {r['repeat_idx']}** (seed {r['seed']})"
                          + (f" · flags: {', '.join(flags)}" if flags else "") + "\n")
                md.append("\n" + r["response"].strip() + "\n")
                md.append("\nQ1 __ Q2 __ Q3 __ Q4 __ Q5 __ · action chosen: ________\n")
    (OUT / f"review_sheet_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote surface_features_{tag}.csv, review_sheet_{tag}.csv, review_sheet_{tag}.md to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "replication" / "generated" / "llama3.1_8b_ja_400.jsonl")
