"""CoreCognition（#35 物体恒常性：守恒 / 恒常性）判分。

prompt 用官方 frame-combined 原文，去掉开头的 <image> 占位符（图片由 doc_to_visual 传）。
同目录 utils.py 是 lmms-eval 自带的整套 corecognition 任务（勿改）；本文件只服务能力表 #35。
抽取：去掉 </think> 前的推理后，先用 utils._template_match（官方模板匹配，原样复用）；
返回 Fail 时退回宽松抽取（是非题第一个 YES/NO；选择题开头字母 / "answer is X" / 最后一个独立字母），
上游在 Fail 时是走 LLM judge，这里不用 judge。两者都抽不到记错。
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

from PIL import Image

from lmms_eval.tasks.corecognition.utils import _template_match

MEDIA = pathlib.Path(__file__).resolve().parent / "media"


def doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    with Image.open(MEDIA / doc["image"]) as im:
        return [im.convert("RGB")]


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    return re.sub(r"^\s*<image>\s*", "", doc["prompt"]).strip()


def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"]


def _strip_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def extract(text: str, yorn: bool) -> str | None:
    t = _strip_think(text or "").strip()
    official = _template_match(t, "YORN" if yorn else "MCQ")
    if official != "Fail":
        return official
    if yorn:
        m = re.search(r"\b(YES|NO)\b", t, re.I)
        return m.group(1).upper() if m else None
    m = re.match(r"^\W*([A-F])\b", t) or re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?([A-F])\b", t, re.I)
    if m:
        return m.group(1).upper()
    ms = re.findall(r"\b([A-F])\b", t)
    return ms[-1] if ms else None


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = extract(results[0] if results else "", doc["question_type"] == "YORN")
    entry = {"id": doc["id"], "pred": pred, "answer": doc["answer"], "is_correct": pred == doc["answer"],
             "parsed": pred is not None}
    return {"corecognition_accuracy": entry, "corecognition_parse_rate": entry}


def agg_accuracy(rs):
    return sum(r["is_correct"] for r in rs) / len(rs) if rs else 0.0


def agg_parse_rate(rs):
    return sum(r["parsed"] for r in rs) / len(rs) if rs else 0.0
