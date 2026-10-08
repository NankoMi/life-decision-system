# -*- coding: utf-8 -*-
"""
把《高性价比人生指南》654 条条目转成 Laya 微调数据（JSONL）。
格式（laya-train 官方 schema）：{"state": str, "questions": {qid: qdef}, "gold": {qid: {"probabilities": {...}}}}

标签规则（启发式 v1，可人工复核后训练）：
- 正向节（非06节）→ score 题："「标题」值得做吗？"
    收益 大→峰在等级5 / 中→4 / 小→3；钱=多 降1级；毅力=是 降1级
    证据 A→峰高0.7 / B→0.6 / C或无→0.5（其余质量均分给相邻两级）
- 06 反面清单 → noul 题："「标题（去掉'不要'）」该做吗？" → false 0.9 / true 0.1
- 抽样 20% 正向条目补 noul true 样本，防全偏一边

用法：
  python build_book_dataset.py [--book ./book] [--out ./book_dataset.jsonl]
"""
import argparse
import glob
import json
import os
import random
import re

TAG_RE = re.compile(
    r"<!--\s*成本标签:\s*钱=(\S+)\s+时间=(\S+)\s+毅力=(\S+)\s+收益=(\S+)\s+口径=(\S+)\s*-->"
)
HEAD_RE = re.compile(r"^###\s+(\d+)\.\s*(.+)$")

SCALE = ["1=强烈反对", "2=不太赞成", "3=中立", "4=比较赞成", "5=强烈支持"]

GAIN_LEVEL = {"大": 4, "中": 3, "小": 2}      # 0-based 等级峰位
PEAK_BY_EVIDENCE = {"A": 0.7, "B": 0.6}


def soft_label(peak_idx: int, peak_mass: float, n: int = 5) -> dict:
    """峰位拿大头，左右邻居分剩余。返回 {等级索引字符串: 概率}"""
    probs = [0.0] * n
    neighbor = (1.0 - peak_mass) / 2
    probs[peak_idx] = peak_mass
    if peak_idx - 1 >= 0:
        probs[peak_idx - 1] += neighbor
    if peak_idx + 1 < n:
        probs[peak_idx + 1] += neighbor
    # 边界时剩余质量给另一侧
    s = sum(probs)
    return {str(i): round(p / s, 4) for i, p in enumerate(probs)}


def parse_entries(book_dir: str):
    """返回 [(section_no, section_file, entry_no, title, tag_dict_or_None)]"""
    entries = []
    for path in sorted(glob.glob(os.path.join(book_dir, "*.md"))):
        fname = os.path.basename(path)
        m = re.match(r"(\d+)-", fname)
        if not m:
            continue
        sec = int(m.group(1))
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            hm = HEAD_RE.match(lines[i].strip())
            if hm:
                title = hm.group(2).strip()
                tag = None
                ev = ""
                # 标题行下面到下一条标题之间找成本标签和证据等级
                for j in range(i + 1, min(i + 40, len(lines))):
                    if HEAD_RE.match(lines[j].strip()):
                        break
                    tm = TAG_RE.search(lines[j])
                    if tm and tag is None:
                        tag = {"钱": tm.group(1), "时间": tm.group(2), "毅力": tm.group(3),
                               "收益": tm.group(4), "口径": tm.group(5)}
                    em = re.search(r"证据等级[：:]\s*([ABC])", lines[j])
                    if em and not ev:
                        ev = em.group(1)
                if tag:
                    tag["_ev"] = ev
                entries.append((sec, fname, int(hm.group(1)), title, tag))
            i += 1
    return entries


def build_rows(entries, seed=42):
    rng = random.Random(seed)
    rows = []
    stats = {"score": 0, "noul_false": 0, "noul_true": 0, "no_tag": 0}

    positives = []
    for sec, fname, no, title, tag in entries:
        if sec == 6:
            # 反面清单：该做吗 → false
            plain = re.sub(r"^不要", "", title)
            qdef = {"type": "noul",
                    "instructions": f"「{plain}」这件事该做吗？"}
            rows.append({"state": f"来源：《高性价比人生指南》第6节（反面清单）",
                         "questions": {"q": qdef},
                         "gold": {"q": {"probabilities": {"false": 0.9, "true": 0.1}}}})
            stats["noul_false"] += 1
        else:
            positives.append((sec, fname, no, title, tag))

    for sec, fname, no, title, tag in positives:
        if tag is None:
            stats["no_tag"] += 1
            gain, peak = "中", 0.6
            peak_idx = GAIN_LEVEL[gain]
        else:
            gain = tag["收益"] if tag["收益"] in GAIN_LEVEL else "中"
            peak_idx = GAIN_LEVEL[gain]
            peak = PEAK_BY_EVIDENCE.get(tag.get("_ev", ""), 0.6)
            if tag["钱"] == "多":
                peak_idx = max(0, peak_idx - 1)
            if tag["毅力"] == "是":
                peak_idx = max(0, peak_idx - 1)

        # score 样本
        qdef = {"type": "score",
                "instructions": f"「{title}」这件事值得做吗？",
                "criteria": SCALE}
        rows.append({"state": f"来源：《高性价比人生指南》第{sec}节第{no}条",
                     "questions": {"q": qdef},
                     "gold": {"q": {"probabilities": soft_label(peak_idx, peak)}}})
        stats["score"] += 1

        # 抽 20% 补 noul true
        if rng.random() < 0.2:
            qn = {"type": "noul",
                  "instructions": f"「{title}」这件事该做吗？"}
            rows.append({"state": f"来源：《高性价比人生指南》第{sec}节第{no}条",
                         "questions": {"qn": qn},
                         "gold": {"qn": {"probabilities": {"false": 0.05, "true": 0.95}}}})
            stats["noul_true"] += 1

    return rows, stats


def main():
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--book", default=os.path.join(here, "..", "book"))
    p.add_argument("--out", default=os.path.join(here, "book_dataset.jsonl"))
    args = p.parse_args()

    entries = parse_entries(args.book)
    rows, stats = build_rows(entries)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({"entries": len(entries), "rows": len(rows), **stats,
                      "out": os.path.abspath(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
