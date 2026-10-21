"""MM-UPD（Unsolvable Problem Detection）· #47 自我检视（lmms-eval task）。

题目本身无解时（选项里没有正确答案 / 选项与题目无关 / 图片与题目不符），模型能不能拒答。

    mmupd_aad    17 道原题  Absent Answer Detection：删掉正确选项
    mmupd_iasd   17 道原题  Incompatible Answer Set Detection：选项换成与题目无关的一组
    mmupd_ivqd   16 道原题  Incompatible Visual Question Detection：图片换成与题目不符的

每道原题两行：standard（正常题，答原选项字母）与 upd（无解题，答 F = None of the above）。
**双重准确率**：同一 pair_id 的两行都对才算这道题对（官方主指标）。lmms-eval 是逐题判分，
这一步在聚合里做（mmupd_aggregate_dual）；报表侧 build_report 按 pair_id 同样配对。

⚠️ 与官方的偏离（报表口径标「非循环」）
-------------------------------------
1. 只问主版本，不做 MMBench 式循环评测（官方每题 2–4 个选项轮换版本要全对）。正常 + 无解
   两版都蒙对的概率约 6%，比官方宽松。原因是 #47 的请求预算约 400。
2. 官方抽取在规则匹配失败时调 GPT 兜底；这里换成确定性的宽松抽取，并把
   「none of the above / none of the options」这类文字回答映射为 F。
提示词用上游 instruction 版原文（mmaad_instruction.yaml 的 post_prompt）。
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

from PIL import Image

MEDIA = pathlib.Path(__file__).resolve().parent / "media"
LETTERS = "ABCDEF"


def mmupd_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    with Image.open(MEDIA / doc["image"]) as im:
        return [im.convert("RGB")]


def mmupd_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    """与上游 utils.mmupd_doc_to_text 同一拼法：hint\\nquestion + \\n + 'A. xxx' 各行 + post_prompt。"""
    kwargs = lmms_eval_specific_kwargs or {}
    opts = "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(doc["options"].items()))
    q = f"{doc['hint']}\n{doc['question']}{opts}" if doc.get("hint") else f"{doc['question']}{opts}"
    return f"{kwargs.get('pre_prompt', '')}{q}{kwargs.get('post_prompt', '')}"


def mmupd_doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"]


# ── 抽取 ──────────────────────────────────────────────────────────────────


def official_infer(answer: str, valid: str) -> str | None:
    """逐行移植上游 MMUPD_Evaluator.can_infer_option：回复以某个字母开头（'X' / 'X.' / 'X)' /
    'X,' / '(X)'），且回复里没有出现其它字母的同类写法时，取该字母。"""
    answer = answer.strip()
    cands = []
    for ch in valid:
        others = [c + p for c in "ABCDEFG" if c != ch for p in (".", ")", ",")]
        starts = [f"{ch})", f"{ch}.", f"{ch},", f"({ch})"]
        if answer == ch or (any(answer.startswith(s) for s in starts) and all(x not in answer for x in others)):
            cands.append(ch)
    return cands[0] if len(cands) == 1 else None


_NONE_RE = re.compile(r"\bnone of the (above|options|choices|given options)\b|\bno correct (answer|option)\b", re.IGNORECASE)


def _strip_think(text: str) -> str:
    for tag in ("</think>", "</thought>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    return text.strip()


def extract(text: str, valid: str) -> tuple[str | None, str]:
    """返回 (字母, 来源)。来源：official / lenient / none_phrase / ''。"""
    if not isinstance(text, str) or not text.strip():
        return None, ""
    body = _strip_think(text)
    got = official_infer(body, valid)
    if got:
        return got, "official"
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    got = extract_mcq_answer(body, choices=list(valid))
    if got and got.upper() in valid:
        return got.upper(), "lenient"
    if _NONE_RE.search(body.split("\n")[-1]) or _NONE_RE.search(body):
        return "F", "none_phrase"
    return None, ""


# ── 判分 / 聚合 ───────────────────────────────────────────────────────────


def mmupd_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0] if results else ""
    valid = "".join(sorted(doc["options"])) + "F"  # instruction 版允许答 F
    pred, src = extract(response, valid)
    entry = {
        "id": doc["id"],
        "pair_id": doc["pair_id"],
        "type": doc["type"],
        "ground_truth": doc["answer"],
        "pred": pred or "",
        "extract": src,
        "is_correct": pred == doc["answer"],
    }
    return {"mmupd_dual_accuracy": entry, "mmupd_standard_accuracy": entry,
            "mmupd_upd_accuracy": entry, "mmupd_parse_rate": entry}


def mmupd_aggregate_dual(results: List[Dict[str, Any]]) -> float:
    """官方主指标：同一 pair_id 的 standard 与 upd 都对才算对。"""
    pairs: Dict[str, List[bool]] = {}
    for r in results:
        pairs.setdefault(r["pair_id"], []).append(r["is_correct"])
    full = [v for v in pairs.values() if len(v) == 2]
    return sum(all(v) for v in full) / len(full) if full else 0.0


def _acc(results, t):
    xs = [r for r in results if r["type"] == t]
    return sum(r["is_correct"] for r in xs) / len(xs) if xs else 0.0


def mmupd_aggregate_standard(results):
    """只看正常题：会不会做原题。"""
    return _acc(results, "standard")


def mmupd_aggregate_upd(results):
    """只看无解题：会不会拒答。单看它，全答 F 就是 100%，所以主分必须看 dual。"""
    return _acc(results, "upd")


def mmupd_aggregate_parse_rate(results):
    return sum(bool(r["pred"]) for r in results) / len(results) if results else 0.0
