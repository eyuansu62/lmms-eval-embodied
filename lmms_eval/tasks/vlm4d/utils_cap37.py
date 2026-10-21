"""VLM4D（#37 轨迹理解与预测 · 当前运动方向）判分，与 tvbench/utils_cap13.py 同一套抽取。

提示词：题干 + 「A. …」选项 + "Answer with the option letter only."
抽取：去掉 </think> 前的推理后，依次取 ① 开头的选项字母 ② "answer is X" / "option X" ③ 唯一一个
被整句复述的选项文本 ④ 最后一个独立的选项字母。
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

MEDIA = pathlib.Path(__file__).resolve().parent / "media"
LETTERS = "ABCD"


def doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    return [str(MEDIA / doc["video"])]


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    lines = [doc["question"]] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(doc["candidates"])]
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def extract(text: str, candidates: List[str]) -> str | None:
    t = (text or "").split("</think>")[-1].strip()
    valid = LETTERS[: len(candidates)]
    m = re.match(r"^\W*([A-D])(?:\b|[.):])", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?\**([A-D])\b", t, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    nt = _norm(t)
    # VLM4D 有数字 / 单词选项（"2"、"left"），只在整词出现时才算复述
    hits = [i for i, c in enumerate(candidates) if _norm(c) and re.search(rf"\b{re.escape(_norm(c))}\b", nt)]
    if len(hits) == 1:
        return valid[hits[0]]
    ms = [x for x in re.findall(r"\b([A-D])\b", t) if x in valid]
    return ms[-1] if ms else None


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = extract(results[0] if results else "", doc["candidates"])
    entry = {"id": doc["id"], "pred": pred, "answer": doc["answer"], "is_correct": pred == doc["answer"],
             "parsed": pred is not None}
    return {"vlm4d_cap37_accuracy": entry, "vlm4d_cap37_parse_rate": entry}


def agg_accuracy(rs):
    return sum(r["is_correct"] for r in rs) / len(rs) if rs else 0.0


def agg_parse_rate(rs):
    return sum(r["parsed"] for r in rs) / len(rs) if rs else 0.0
